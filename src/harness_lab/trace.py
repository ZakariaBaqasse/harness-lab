"""Utilities for saving run event traces."""

import json
from pathlib import Path


def save_trace(run_id: str, event_payload: dict) -> None:
    """Save the event trace for a given run ID.

    Args:
        run_id (str): The unique identifier for the run.
        event_payload (dict): The event data to be saved.

    Returns:
        None
    """
    traces_folder = Path("traces")
    traces_folder.mkdir(exist_ok=True)
    with open(traces_folder / f"{run_id}.jsonl", "a") as f:
        json.dump(event_payload, f)
        f.write("\n")
    print(f"Saving trace for run_id: {run_id} with event_payload: {event_payload}")
