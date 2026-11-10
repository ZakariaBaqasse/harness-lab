"""Tests for the harness console entry point."""

import os
import subprocess
import sys
import sysconfig
from importlib.metadata import entry_points
from pathlib import Path

from mistralai.client.models import ChatCompletionRequest, ChatCompletionResponse


def test_console_entry_point_constructs_client_and_runs_harness(mocker, main):
    """The installed command can be called without arguments."""
    client = mocker.patch.object(main, "Mistral").return_value
    run_harness = mocker.patch.object(main, "harness_lab_main")
    entry_point = next(iter(entry_points(group="console_scripts", name="harness-lab")))

    entry_point.load()()

    main.Mistral.assert_called_once_with(api_key=main.api_key)
    run_harness.assert_called_once_with(client=client)


def test_generated_command_starts_without_client_argument(tmp_path):
    """Run the generated command with an offline client and harness."""
    (tmp_path / "sitecustomize.py").write_text(
        "import harness_lab.main as app\n"
        "app.Mistral = lambda **kwargs: object()\n"
        "app.harness_lab_main = lambda client: print('harness-started')\n",
        encoding="utf-8",
    )
    command = Path(sysconfig.get_path("scripts")) / (
        "harness-lab.exe" if sys.platform == "win32" else "harness-lab"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    env["MISTRAL_API_KEY"] = "test-key"

    result = subprocess.run(
        [str(command)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "harness-started"


def test_harness_sends_sdk_compatible_messages(mocker, main):
    """The harness sends messages accepted by the Mistral SDK."""
    client = mocker.Mock()

    def response(message, finish_reason):
        return ChatCompletionResponse.model_validate(
            {
                "id": "response-id",
                "object": "chat.completion",
                "model": "test-model",
                "created": 0,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": message,
                    }
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
                        }
                    ],
                },
                "tool_calls",
            ),
            response({"role": "assistant", "content": "Done"}, "stop"),
        ]
    )

    def complete(**kwargs):
        ChatCompletionRequest.model_validate(kwargs)
        return next(responses)

    client.chat.complete.side_effect = complete
    main.harness_lab_main(client)

    assert client.chat.complete.call_count == 2
    assert main.save_trace.call_args_list[0].args[1]["task"] == (
        "Check the status of project Atlas"
    )
