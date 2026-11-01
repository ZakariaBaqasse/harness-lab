"""Tests for preserving tool calls and matching results in follow-up messages."""

import copy
import importlib
import os
import unittest
from unittest.mock import patch

from mistralai.client.models import ChatCompletionResponse


class ToolMessageTests(unittest.TestCase):
    """Test preservation of tool calls and matching results."""

    def test_follow_up_contains_all_tool_calls_and_matching_results(self) -> None:
        """Verify follow-up messages preserve every tool call and its result."""
        with patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"}):
            main = importlib.import_module("harness_lab.main")

        def response(message: dict, finish_reason: str) -> ChatCompletionResponse:
            return ChatCompletionResponse.model_validate(
                {
                    "id": "response-id",
                    "object": "chat.completion",
                    "model": "test-model",
                    "created": 0,
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    "choices": [
                        {"index": 0, "finish_reason": finish_reason, "message": message}
                    ],
                }
            )

        responses = iter(
            [
                response(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "lookup_status",
                                    "arguments": '{"project_name": "Atlas"}',
                                },
                            },
                            {
                                "id": "call-2",
                                "type": "function",
                                "function": {
                                    "name": "lookup_status",
                                    "arguments": {"project_name": "Borealis"},
                                },
                            },
                        ],
                    },
                    "tool_calls",
                ),
                response({"role": "assistant", "content": "Done"}, "stop"),
            ]
        )
        requests = []

        def complete(**kwargs):
            requests.append(copy.deepcopy(kwargs["messages"]))
            return next(responses)

        with (
            patch.object(main.client.chat, "complete", side_effect=complete),
            patch.object(main, "save_trace"),
            patch.object(main, "MAX_ITERATIONS", 2),
        ):
            main.harness_lab_main()

        self.assertEqual(len(requests), 2)
        messages = requests[1]
        self.assertEqual(
            [message["role"] for message in messages],
            ["user", "assistant", "tool", "tool"],
        )
        self.assertEqual(
            [call["id"] for call in messages[1]["tool_calls"]], ["call-1", "call-2"]
        )
        self.assertEqual(
            [message["tool_call_id"] for message in messages[2:]], ["call-1", "call-2"]
        )
        self.assertEqual(messages[2]["content"], "Project: Atlas Status: OK")
        self.assertEqual(messages[3]["content"], "Project: Borealis Status: OK")


if __name__ == "__main__":
    unittest.main()
