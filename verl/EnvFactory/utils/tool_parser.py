import os
import json
import logging
import regex

from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import rollout_trace_op
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "WARN"))


@ToolParser.register("qwen3")
class Qwen3ToolParser(ToolParser):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_regex = regex.compile(r"<tool_call>(.*?)</tool_call>", regex.DOTALL)

    @rollout_trace_op
    async def extract_tool_calls(
        self, responses_ids: list[int], tools: list[OpenAIFunctionToolSchema] = None
    ) -> tuple[str, list[FunctionCall]]:
        loop = get_event_loop()
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)
        if self.tool_call_start_token not in text or self.tool_call_end_token not in text:
            return text, []

        matches = self.tool_call_regex.findall(text)
        function_calls = []
        for match in matches:
            try:
                function_call = json.loads(match)
                name, arguments = function_call["name"], function_call["arguments"]
            except Exception as e:
                logger.error(f"⚠️ Failed to decode tool call: {e}")
                name, arguments = "Invalid", "Invalid tool call format: {e}. Raw content: {match}"

            function_calls.append(FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))

        # remaining text excluding tool call tokens
        content = self.tool_call_regex.sub("", text)

        return content, function_calls