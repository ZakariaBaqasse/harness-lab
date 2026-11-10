"""Tests for preserving tool calls and matching results in follow-up messages."""

import pytest
from mistralai.client.models import ChatCompletionResponse


@pytest.fixture()
def response():
    """Build a chat completion response for mocked client calls."""

    def _response(message: dict, finish_reason: str) -> ChatCompletionResponse:
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

    return _response


def test_follow_up_contains_all_tool_calls_and_matching_results(
    response, main, mistral_client
) -> None:
    """Verify follow-up messages preserve every tool call and its result."""
    mistral_client_mock, requests = mistral_client(
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
    main.harness_lab_main(mistral_client_mock)

    assert len(requests) == 2
    messages = requests[1]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert [call["id"] for call in messages[1]["tool_calls"]] == [
        "call-1",
        "call-2",
    ]
    assert [message["tool_call_id"] for message in messages[2:]] == [
        "call-1",
        "call-2",
    ]
    assert messages[2]["content"] == "Project: Atlas Status: OK"
    assert messages[3]["content"] == "Project: Borealis Status: OK"


def test_runtime_error_raised_on_max_iterations(
    mocker, main, mistral_client, response
) -> None:
    """Verify a runtime error is raised when max iterations are exceeded."""
    mocker.patch.object(main, "MAX_ITERATIONS", 1)
    mistral_client_mock, _ = mistral_client(
        [
            response(
                {
                    "role": "assistant",
                    "content": "Doing something",
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
            )
        ]
    )
    with pytest.raises(RuntimeError, match="Max iterations reached"):
        main.harness_lab_main(mistral_client_mock)


def test_value_error_raised_for_invalid_response_shape(main, mistral_client) -> None:
    """Verify a value error is raised when the response shape is invalid."""
    mistral_client_mock, _ = mistral_client(
        [
            ChatCompletionResponse.model_validate(
                {
                    "id": "response-id",
                    "object": "chat.completion",
                    "model": "test-model",
                    "created": 0,
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    "choices": [],
                },
            ),
        ]
    )
    with pytest.raises(
        ValueError,
        match="The response is malformed or contains no assistant message",
    ):
        main.harness_lab_main(mistral_client_mock)


def test_runtime_error_raised_on_compaction_needed(
    mocker,
    main,
    mistral_client,
    response,
    token_counter_module,
    tokenizer,
) -> None:
    """Verify a runtime error is raised when compaction is needed."""
    mocker.patch.object(token_counter_module, "EFFECTIVE_TOKEN_LIMIT", 1000)
    mistral_client_mock, _ = mistral_client(
        [
            response(
                {
                    "role": "assistant",
                    "content": "Doing something",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "lookup_status",
                                "arguments": '{"project_name": "Atlas"}',
                            },
                        }
                    ],
                },
                "tool_calls",
            )
        ]
    )
    with pytest.raises(RuntimeError, match="Compaction needed"):
        main.harness_lab_main(mistral_client_mock, tokenizer)
