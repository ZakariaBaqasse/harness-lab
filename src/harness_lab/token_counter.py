"""Module for estimating tokens before processing."""

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import (
    MistralTokenizer,
)

EFFECTIVE_TOKEN_LIMIT = 64000
ALLOWED_CONTEXT_PERCENTAGE = 0.8
ESTIMATED_LLM_RESPONSE_SIZE = 1024


def estimate_tokens(
    messages: list, tokenizer: MistralTokenizer, tools: list | None = None
):
    """Estimate the number of tokens in the input text."""
    token_count_request = ChatCompletionRequest(messages=messages, tools=tools)
    tokenized = tokenizer.encode_chat_completion(token_count_request)
    return len(tokenized.tokens)


def check_compaction_needed(
    messages: list, tokenizer: MistralTokenizer, tools: list | None = None
):
    """Check if compaction of the input messages is needed by estimating tokens."""
    estimated_tokens = estimate_tokens(messages, tokenizer, tools)
    return {
        "compaction_needed": estimated_tokens + ESTIMATED_LLM_RESPONSE_SIZE
        > EFFECTIVE_TOKEN_LIMIT * ALLOWED_CONTEXT_PERCENTAGE,
        "estimated_tokens": estimated_tokens + ESTIMATED_LLM_RESPONSE_SIZE,
    }
