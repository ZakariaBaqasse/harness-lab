"""Tests for saving trace events to a Windows-safe JSONL file."""

import json
import os
import tempfile
from pathlib import Path

from harness_lab.trace import save_trace


class TestSaveTrace:
    """Tests for saving trace events to a Windows-safe JSONL file."""

    def test_windows_safe_filename_and_jsonl_append(self) -> None:
        """Verify trace filenames are Windows-safe and events append as JSONL."""
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)

                save_trace("run-id", {"event": "start"})
                save_trace("run-id", {"event": "end"})

                trace = Path("traces") / "run-id.jsonl"
                events = [json.loads(line) for line in trace.read_text().splitlines()]
                assert events == [{"event": "start"}, {"event": "end"}]
            finally:
                os.chdir(original_directory)
