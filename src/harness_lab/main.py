"""Run the project-status harness and record execution traces."""

import json
import os
from datetime import datetime
from uuid import uuid4

from dotenv import load_dotenv
from mistralai.client import Mistral
from mistralai.client.models import (
    ChatCompletionResponse,
)

from harness_lab.fake_tool import names_to_functions, tools
from harness_lab.trace import save_trace

load_dotenv()


MAX_ITERATIONS = 5
MODEL = "mistral-small-2603"

api_key = os.environ["MISTRAL_API_KEY"]


def harness_lab_main() -> None:
    """Run the project-status harness and record its trace."""
    client = Mistral(api_key=api_key)
    messageslist: list = [
        {"role": "user", "content": "Check the status of project Atlas"}
    ]
    run_id = uuid4()
    iterations = 0
    save_trace(
        str(run_id),
        {
            "event_type": "run_started",
            "run_id": str(run_id),
            "sequence": iterations,
            "timestamp": datetime.now().isoformat(),
            "task": messageslist[0].get("content"),
        },
    )
    response: ChatCompletionResponse | None = None
    total_input_tokens = 0
    total_output_tokens = 0
    start_time = datetime.now()
    while True:
        try:
            save_trace(
                str(run_id),
                {
                    "event_type": "llm_request",
                    "run_id": str(run_id),
                    "sequence": iterations,
                    "model": MODEL,
                    "tools": tools,
                    "messages": messageslist,
                    "message_count": len(messageslist),
                    "timestamp": datetime.now().isoformat(),
                },
            )
            response = client.chat.complete(
                model=MODEL,
                messages=messageslist,
                tools=tools,
                tool_choice="auto",
            )

            if not response.choices or response.choices[0].message is None:
                raise ValueError("The response contains no assistant message")

            message = response.choices[0].message
            tool_calls = (
                message.tool_calls if isinstance(message.tool_calls, list) else []
            )

            messageslist.append(message.model_dump())
            save_trace(
                str(run_id),
                {
                    "event_type": "llm_response",
                    "run_id": str(run_id),
                    "sequence": iterations,
                    "timestamp": datetime.now().isoformat(),
                    "input_tokens": response.usage.prompt_tokens
                    if response.usage.prompt_tokens
                    else 0,
                    "output_tokens": response.usage.completion_tokens
                    if response.usage.completion_tokens
                    else 0,
                    "tool_calls": [
                        {
                            "function_name": tool_call.function.name,
                            "function_arguments": tool_call.function.arguments,
                        }
                        for tool_call in tool_calls
                    ],
                    "messages": [m for m in messageslist],
                    "finish_reason": response.choices[0].finish_reason,
                    "final_answer": response.choices[0].message.content
                    if response.choices and response.choices[0].message
                    else None,
                },
            )

            if not tool_calls and response.choices[0].finish_reason == "stop":
                break

            for tool_call in tool_calls:
                if not tool_call.id:
                    raise ValueError("The assistant returned a tool call without an ID")

                function_name = tool_call.function.name
                arguments = tool_call.function.arguments
                function_arguments = (
                    json.loads(arguments) if isinstance(arguments, str) else arguments
                )
                if (
                    not isinstance(function_arguments, dict)
                    or "project_name" not in function_arguments
                ):
                    raise ValueError(
                        "Function arguments must be a dict containing 'project_name'"
                    )
                function_results = names_to_functions[function_name](
                    **function_arguments
                )
                messageslist.append(
                    {
                        "role": "tool",
                        "name": function_name,
                        "content": function_results,
                        "tool_call_id": tool_call.id,
                    }
                )

                save_trace(
                    str(run_id),
                    {
                        "event_type": "tool_execution",
                        "run_id": str(run_id),
                        "sequence": iterations,
                        "function_name": function_name,
                        "function_arguments": function_arguments,
                        "function_results": function_results,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

            total_input_tokens += (
                response.usage.prompt_tokens if response.usage.prompt_tokens else 0
            )
            total_output_tokens += (
                response.usage.completion_tokens
                if response.usage.completion_tokens
                else 0
            )
            iterations += 1
            if iterations >= MAX_ITERATIONS:
                save_trace(
                    str(run_id),
                    {
                        "event_type": "max_iterations_reached",
                        "run_id": str(run_id),
                        "sequence": iterations,
                        "timestamp": datetime.now().isoformat(),
                        "status": "max_iterations_reached",
                        "total_input_tokens": total_input_tokens,
                        "total_output_tokens": total_output_tokens,
                        "messages": messageslist,
                    },
                )
                raise RuntimeError("Max iterations reached")

        except Exception as e:
            print(f"An error occurred: {e}")
            save_trace(
                str(run_id),
                {
                    "event_type": "run_failed",
                    "run_id": str(run_id),
                    "sequence": iterations,
                    "timestamp": datetime.now().isoformat(),
                    "status": "failed",
                    "final_answer": None,
                    "error": str(e),
                },
            )
            raise

    print(f"Run completed for run_id: {run_id} with total iterations: {iterations}")
    save_trace(
        str(run_id),
        {
            "event_type": "run_completed",
            "run_id": str(run_id),
            "sequence": iterations,
            "status": "success",
            "final_answer": response.choices[0].message.content
            if response and response.choices[0].message
            else None,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_duration_ms": int(
                (datetime.now() - start_time).total_seconds() * 1000
            ),
        },
    )


if __name__ == "__main__":
    harness_lab_main()
