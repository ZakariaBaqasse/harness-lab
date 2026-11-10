"""Tests for the token_counter module."""

import pytest
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.tool_calls import (
    Function,
    FunctionCall,
    Tool,
    ToolCall,
)

from harness_lab.token_counter import estimate_tokens


@pytest.fixture
def messages_and_tools():
    """Fixture for a list of user messages and tools."""
    messages = [UserMessage(content="Check the status of project Atlas")]
    tools = [
        Tool(
            function=Function(
                name="lookup_status",
                description="Get status of a project",
                parameters={
                    "type": "object",
                    "properties": {
                        "project_name": {
                            "type": "string",
                            "description": "The project name.",
                        }
                    },
                    "required": ["project_name"],
                },
            )
        )
    ]
    return messages, tools


def test_user_message_and_tools_estimate_tokens(tokenizer, messages_and_tools):
    """Test that estimate_tokens returns the correct token count."""
    messages, tools = messages_and_tools
    estimated_tokens = estimate_tokens(messages, tokenizer, tools=tools)
    assert abs(estimated_tokens - 90) / 90 < 1e-1


def test_user_message_with_tools_and_assistant_estimate_tokens(
    tokenizer, messages_and_tools
):
    """Test that estimate_tokens returns the correct token count when no tools are provided."""
    messages, tools = messages_and_tools
    messages.append(
        AssistantMessage(
            prefix=True,
            content="",
            tool_calls=[
                ToolCall(
                    id="m7JqYr4Da",
                    function=FunctionCall(
                        name="lookup_status", arguments='{"project_name": "Atlas"}'
                    ),
                )
            ],
        )
    )

    estimated_tokens = estimate_tokens(messages, tokenizer, tools=tools)
    assert abs(estimated_tokens - 102) / 102 < 1e-1


def test_trigger_compaction(
    tokenizer, messages_and_tools, token_counter_module, mocker
):
    """Test that check_compaction_needed returns the correct compaction status."""
    mocker.patch.object(
        token_counter_module,
        "EFFECTIVE_TOKEN_LIMIT",
        1000,
    )
    messages, tools = messages_and_tools
    result = token_counter_module.check_compaction_needed(
        messages, tokenizer, tools=tools
    )
    assert "compaction_needed" in result
    assert "estimated_tokens" in result
    assert result["compaction_needed"] == True
    # 1104 is the sum of the estimated tokens for all messages in the fixture above (which equals 90) plus the estimated tokens for the assistant message with tool call (which equals 1024)
    assert abs(result["estimated_tokens"] - 1104) / 1104 < 1e-1


def test_trigger_compaction_no_compaction_needed(
    tokenizer, messages_and_tools, token_counter_module, mocker
):
    """Test that check_compaction_needed returns the correct compaction status when no compaction is needed."""
    mocker.patch.object(
        token_counter_module,
        "EFFECTIVE_TOKEN_LIMIT",
        2000,
    )
    messages, tools = messages_and_tools
    result = token_counter_module.check_compaction_needed(
        messages, tokenizer, tools=tools
    )
    assert "compaction_needed" in result
    assert "estimated_tokens" in result
    assert result["compaction_needed"] == False
    # 1104 is the sum of the estimated tokens for all messages in the fixture above (which equals 90) plus the estimated tokens for the assistant message with tool call (which equals 1024)
    assert abs(result["estimated_tokens"] - 1104) / 1104 < 1e-1
