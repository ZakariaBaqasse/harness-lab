"""Provide a fake project-status tool for harness testing."""

from collections.abc import Callable

from mistralai.client.models import ChatCompletionRequestToolTypedDict


def lookup_status(project_name: str):
    """Provide a fake project-status lookup tool and its tool definition."""
    return "Project: " + project_name + " Status: OK"


tools: list[ChatCompletionRequestToolTypedDict] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_status",
            "description": "Get status of a project",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "The project name.",
                    }
                },
                "required": ["project_name"],
            },
        },
    }
]

names_to_functions: dict[str, Callable] = {"lookup_status": lookup_status}
