#!/usr/bin/env python
"""
Drop-in wrapper around `tau2 run` that makes the NL-assertions *evaluator* (the
LLM judge used during reward computation) configurable, with a custom
third-party OpenAI-compatible endpoint.

Relationship with run_tau2_bench.sh
-----------------------------------
run_tau2_bench.sh is the shell orchestrator: it parses CLI args, exports the
endpoint config for the agent / user / evaluator as env vars, and then launches
this wrapper instead of the bare `tau2` console script:

    uv run python tau2_eval_patch.py run --domain retail --agent-llm ...

This wrapper does ONE extra thing before handing off to tau2's CLI `main()`: it
rewrites the NL-judge globals so the evaluator uses your third-party endpoint.
The two files communicate only through env vars (TAU2_EVAL_*).

WHY THIS EXISTS
---------------
tau2-bench hard-codes the NL-assertions judge to ``gpt-4.1-2025-04-14``
(``tau2/config.py``: ``DEFAULT_LLM_NL_ASSERTIONS``) and the judge's ``generate()``
call passes NO ``api_base``/``api_key`` (see
``tau2/evaluator/evaluator_nl_assertions.py``). So the judge always hits
``api.openai.com`` reading ``OPENAI_API_KEY`` from the environment.

When you point the *user simulator* at a third-party aggregator such as yibuapi
(DeepSeek-V3.2), ``OPENAI_API_KEY`` holds that aggregator's key, and the judge
ships it to the real OpenAI endpoint:

    OpenAIException - Incorrect API key provided: sk-REDACTED.

The ``tau2 run`` CLI exposes ``--agent-llm`` / ``--user-llm`` but NOT the
evaluator, and ``TextRunConfig`` has no evaluator field either (tau2's own
experiment scripts likewise rely on the gpt-4.1 default). So we redirect the
judge by rewriting the two module globals its call site reads at *call time*:

    tau2.evaluator.evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS       # model
    tau2.evaluator.evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS  # kwargs

We mutate module attributes (never the vendored source), so this patch survives
a ``git checkout`` / re-clone of the tau2-bench checkout. The injected
``api_base``/``api_key`` flow through ``generate()`` -> ``litellm.completion()``
as kwargs -- the same proven mechanism the local agent already uses (it passes
its own ``api_base=localhost`` the same way).

Difference from a single-global-endpoint setup (e.g. the One-Eval bridge, where
agent == user == evaluator share one ``OPENAI_API_BASE``): here the agent is a
*local* SGLang server, so the agent keeps its explicit ``api_base`` and only the
judge is repointed.

Env vars (all optional; unset => stock gpt-4.1 NL evaluator is untouched):
  TAU2_EVAL_LLM           judge model name, provider-prefix optional.
                           "deepseek-v3.2" -> "openai/deepseek-v3.2".
  TAU2_EVAL_API_BASE      OpenAI-compatible base URL for the judge.
  TAU2_EVAL_API_KEY       API key for the judge.
  TAU2_EVAL_TEMPERATURE    optional (default 0.0; deterministic grading).
"""

from __future__ import annotations

import os
import sys

# Import the evaluator module so we can rewrite the globals its call site reads.
# This triggers tau2's normal imports but does NOT start a run.
import json as _json

import tau2.evaluator.evaluator_nl_assertions as _nla
from tau2.utils.llm_utils import extract_json_from_llm_response

# Capture the judge's generate() before we wrap it (see apply_judge_json_patch).
_orig_judge_generate = _nla.generate


def _ensure_provider_prefix(model: str) -> str:
    """Force an OpenAI-compatible provider prefix for bare model names.

    litellm needs a provider prefix to honour a custom ``api_base``; without one
    a bare name like ``deepseek-v3.2`` may be routed through litellm's own
    DeepSeek provider (wrong auth/endpoint) instead of our aggregator.
    """
    if "/" in model:
        return model
    return f"openai/{model}"


def _mask_key(key: str | None) -> str:
    if not key:
        return "<unset>"
    if len(key) <= 12:
        return "*" * len(key)
    return f"{key[:6]}{'*' * 6}{key[-4:]}"


def apply_evaluator_patch() -> bool:
    """Redirect the NL-assertions judge if TAU2_EVAL_LLM is set.

    Returns True if the patch was applied, False if left at stock defaults.
    """
    model = os.environ.get("TAU2_EVAL_LLM", "").strip()
    api_base = os.environ.get("TAU2_EVAL_API_BASE", "").strip() or None
    api_key = os.environ.get("TAU2_EVAL_API_KEY", "").strip() or None
    if not model:
        return False  # Nothing requested -> stock gpt-4.1 behavior.

    # Start from the stock args ({"temperature": 0.0}) and add endpoint config.
    args = dict(_nla.DEFAULT_LLM_NL_ASSERTIONS_ARGS)
    args["temperature"] = float(
        os.environ.get("TAU2_EVAL_TEMPERATURE", args.get("temperature", 0.0))
    )
    if api_base:
        args["api_base"] = api_base
    if api_key:
        args["api_key"] = api_key

    _nla.DEFAULT_LLM_NL_ASSERTIONS = _ensure_provider_prefix(model)
    _nla.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args

    print(
        "[tau2_eval_patch] NL-assertions evaluator redirected:\n"
        f"    model:    {_nla.DEFAULT_LLM_NL_ASSERTIONS}\n"
        f"    api_base: {api_base or '<litellm default>'}\n"
        f"    api_key:  {_mask_key(api_key)}\n"
        f"    temp:     {args['temperature']}",
        file=sys.stderr,
    )
    return True


def apply_judge_json_patch() -> None:
    """Make the NL judge robust to markdown-fenced JSON.

    evaluator_nl_assertions.py does a raw ``json.loads(assistant_message.content)``.
    gpt-4.1 (the stock judge) returns clean JSON, but many judge models -- e.g.
    deepseek-v3.2 -- wrap it in ```json ... ``` fences, so ``json.loads`` fails
    with "Expecting value: line 1 column 1 (char 0)", which aborts reward
    computation and triggers task retries (wrong/missing NL rewards on retail).

    We wrap the judge's ``generate()`` (the only generate() call in this module)
    to run the response through tau2's own ``extract_json_from_llm_response``
    before the evaluator parses it. No-op when content is already clean JSON.
    """
    def _judge_generate_json(*args, **kwargs):
        msg = _orig_judge_generate(*args, **kwargs)
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content:
            try:
                _json.loads(content)  # already clean JSON?
            except Exception:
                cleaned = extract_json_from_llm_response(content)
                try:
                    msg.content = cleaned
                except Exception:
                    # Pydantic model might be frozen; copy instead.
                    msg = msg.model_copy(update={"content": cleaned})
        return msg

    _nla.generate = _judge_generate_json
    print("[tau2_eval_patch] NL judge JSON: fenced responses will be unwrapped",
          file=sys.stderr)


def main() -> None:
    apply_evaluator_patch()
    apply_judge_json_patch()
    # Same entry point as the `tau2` console script; parses sys.argv[1:].
    from tau2.cli import main as tau2_main

    tau2_main()


if __name__ == "__main__":
    main()
