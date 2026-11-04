"""Pytest fixtures for importing the app and mocking Mistral responses."""

import copy
import importlib

import pytest


@pytest.fixture()
def main(mocker):
    """Import harness_lab.main with trace saving mocked out."""
    module = importlib.import_module("harness_lab.main")
    mocker.patch.object(module, "save_trace")
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
