import re
import json
import logging
import os
import math

logger = logging.getLogger()
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

TOOL_CALL_PATTERN = re.compile(r'<tool_call>\s*({.*?})\s*</tool_call>', re.DOTALL)


class RewardScheduler:
    """Dynamic scheduler for adjusting tau during training."""

    def __init__(
        self,
        start: float = 0.9,
        end: float = 0.1,
        default: float = 0.5,
        total_epochs: int = 10,
        mode: str = "fix", # fix, linear
    ):
        self.start = start
        self.end = end
        self.default = default
        self.total_epochs = max(total_epochs, 1)
        self.mode = mode

    def get_ratio(self, epoch: int) -> float:
        if self.mode == "fix":
            return self.default
        elif self.mode == "linear":
            ratio = min(epoch / self.total_epochs, 1.0)
            return self.start + (self.end - self.start) * ratio
        else:
            raise NotImplementedError(f"Reward scheduler mode: {self.mode} is not implemented.")


def to_JSON(obj: str | dict | list) -> dict | list:
    """Convert text to JSON object"""
    if isinstance(obj, dict) or isinstance(obj, list):
        return obj

    try:
        return json.loads(obj)
    except Exception as e:
        logger.debug(f"Debug: Error in JSON loading: {e}")
        return obj

def extract_tool_calls(text: str) -> tuple[list[str], int]:
    """Extract tool_call from text"""
    tool_calls = []
    fail_tool_calls_count = 0
    
    matches = TOOL_CALL_PATTERN.findall(text)
    for match in matches:
        try:
            tool_call = json.loads(match)
            tool_calls.append({"name": tool_call["name"], "arguments": tool_call["arguments"]})
        except Exception as e:
            logger.debug(f"Debug: Error in JSON loading: {e}")
            fail_tool_calls_count += 1

    return tool_calls, fail_tool_calls_count

def get_mcp_servers(gt_calls: list[str]) -> list[str]:
    mcp_servers = set()
    for gt_call in gt_calls:
        mcp_server = gt_call["name"].split("-")[0]
        mcp_servers.add(mcp_server)
    return list(mcp_servers)

def _check_value_match(v1, v2) -> bool:
    """
    Compare two values with number sensitivity handling.
    Returns True if values are numerically equal (1 == 1.0).
    """
    if v1 == v2:
        return True
    
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        return math.isclose(v1, v2, rel_tol=1e-6)

    return False

def _check_tool_call_match(sol_call: dict, gt_call: dict) -> bool:
    """
    Check if a solution call matches a ground truth call, respecting masked_arguments.
    """
    # 1. Check Name
    if sol_call.get("name") != gt_call.get("name"):
        return False

    sol_args = sol_call.get("arguments", {})
    gt_args = gt_call.get("arguments", {})
    masked_keys = set(gt_call.get("masked_arguments", [])) 

    # 2. Iterate through GT arguments constraints
    for key, gt_val in gt_args.items():
        # If argument is masked, skip
        if key in masked_keys:
            continue
        
        # If gt has key but sol not, return False
        if key not in sol_args:
            return False
        
        sol_val = sol_args[key]
        
        # If argument is not masked, compare
        if not _check_value_match(sol_val, gt_val):
            return False

    return True

def _compute_trace_score(solution: list[dict], ground_truth: list[dict]) -> float:
    """
    Compute trace reward score with support for masked arguments and number tolerance.
    Return 1.0 if ground truth tool calls is a subset of permutation of solution tool calls.
    """
    # If no solution, return 0
    if not solution:
        return 0.0
    
    # If no ground_truth, return 1 (this is unlike to happen)
    if not ground_truth:
        return 1.0

    # Create a solution index list to mark which call has been matched
    # Avoid two same call in GT matching to the same call in Solution
    solution_matched_indices = set()

    # Check if EVERY call in ground truth has a corresponding match in solution
    for gt_call in ground_truth:
        match_found = False

        for idx, sol_call in enumerate(solution):
            # If this solution call has been matched by previous GT call, skip
            if idx in solution_matched_indices:
                continue
            
            # Check if match
            if _check_tool_call_match(sol_call, gt_call):
                solution_matched_indices.add(idx)
                match_found = True
                break
        
        # If current GT call not found any match in solution, return 0
        if not match_found:
            return 0.0

    return 1.0

def _compute_state_score(solution: str | dict, ground_truth: str | dict, mcp_servers: list[str]) -> float:
    """
    Compute state reward score by comparing the final MCP server configuration.
    Return partial score for each of correctly matched MCP server configuration.
    """
    solution = to_JSON(solution)
    solution = {k: to_JSON(v) for k, v in solution.items()}

    ground_truth = to_JSON(ground_truth)
    ground_truth = {k: v for k, v in ground_truth.items() if k in mcp_servers} # filter out non-used MCP servers

    matches = sum(1 for server, config in ground_truth.items() if solution.get(server) == config)
    return matches / len(ground_truth) if ground_truth else 0.0

def _compute_length_penalty(solution: list[dict], ground_truth: list[dict]) -> float:
    """Penalize each extra tool calls."""
    if not ground_truth or not solution:
        return 0.0
    
    penalty = max(len(solution)-len(ground_truth), 0) * 0.05
    return min(penalty, 0.5)

def _compute_format_penalty(fail_tool_calls_count: int) -> float:
    """Penalize each failed tool calls."""
    penalty = fail_tool_calls_count * 0.05
    return min(penalty, 0.5)

def _compute_score(solution: str, ground_truth: str, extra_info: dict = {}) -> dict:
    """
    Calculate reward for MCPFactory.
    
    Args:
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional):
            - sol_final_config (dict): The solution's final MCP server configuration.
            - gts_final_config (dict): The ground truth's final MCP server configuration.
            - reward_scheduler (dict): The reward scheduler config.
    """
    # Compute tau
    current_epoch = extra_info.get("epoch", 1)
    total_epochs = extra_info.get("total_epochs", 2)
    scheduler_cfg = extra_info["reward_scheduler"]
    scheduler = RewardScheduler(
        start=scheduler_cfg.get("start_ratio", 0.9),
        end=scheduler_cfg.get("end_ratio", 0.1),
        default=scheduler_cfg.get("default_ratio", 0.5),
        total_epochs=total_epochs,
        mode=scheduler_cfg.get("mode", "linear")
    )
    tau = scheduler.get_ratio(current_epoch)

    try:
        # Prepare
        gt_calls = to_JSON(ground_truth)
        sol_calls, fail_tool_calls_count = extract_tool_calls(solution)
        mcp_servers = get_mcp_servers(gt_calls)

        # Compute trace score
        trace_score = _compute_trace_score(
            solution=sol_calls,
            ground_truth=gt_calls,
        )

        # Compute state score
        state_score = _compute_state_score(
            solution=extra_info.get('sol_final_config', {}),
            ground_truth=extra_info.get('gts_final_config', {}),
            mcp_servers=mcp_servers,
        )

        # Compute penalty
        length_penalty = _compute_length_penalty(
            solution=sol_calls,
            ground_truth=gt_calls,
        )
        format_penalty = _compute_format_penalty(
            fail_tool_calls_count = fail_tool_calls_count,
        )

        score = max(0, tau * trace_score + (1-tau) * state_score - length_penalty - format_penalty)
        return {
            "score": score,
            "trace_score": trace_score,
            "state_score": state_score,
            "penalty": length_penalty + format_penalty,
            "tau": tau,
        }

    except Exception as e:
        logger.debug(f"Debug: Error in compute_score: {e}")
        return {
            "score": 0.0,
            "trace_score": 0.0,
            "state_score": 0.0,
            "penalty": 0.0,
            "tau": tau,
        }
    
def compute_tool_reward(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    res = _compute_score(solution_str, ground_truth, extra_info)

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float):
        return float(res)
    else:
        return float(res[0])