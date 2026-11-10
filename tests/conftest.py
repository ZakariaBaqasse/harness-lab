"""Pytest fixtures for importing the app and mocking Mistral responses."""

import copy
import importlib

import pytest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from harness_lab.constants import TOKENIZER_REPO_ID


@pytest.fixture()
def main(mocker):
    """Import harness_lab.main with trace saving mocked out."""
    module = importlib.import_module("harness_lab.main")
    mocker.patch.object(module, "save_trace")
    return module


@pytest.fixture()
def token_counter_module():
    """Import harness_lab.token_counter with trace saving mocked out."""
    module = importlib.import_module("harness_lab.token_counter")
    return module


@pytest.fixture()
def mistral_client(mocker):
    """Build a mocked Mistral client that replays responses in order.

    Returns a factory yielding (client_mock, requests), where requests holds a
    deep copy of the messages passed to each chat.complete call.
    """

    def _build(responses):
        replies = iter(responses)
        requests = []

        def complete(**kwargs):
            requests.append(copy.deepcopy(kwargs["messages"]))
            return next(replies)

        client = mocker.Mock()
        client.chat.complete.side_effect = complete
        return client, requests

    return _build


@pytest.fixture
def tokenizer():
    """Fixture for the Mistral tokenizer."""
    return MistralTokenizer.from_hf_hub(repo_id=TOKENIZER_REPO_ID)
