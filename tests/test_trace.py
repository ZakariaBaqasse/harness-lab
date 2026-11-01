"""Tests for saving trace events to a Windows-safe JSONL file."""

import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from harness_lab.trace import save_trace


class SaveTraceTests(unittest.TestCase):
    """Tests for saving trace events to a Windows-safe JSONL file."""

    def test_windows_safe_filename_and_jsonl_append(self) -> None:
        """Verify trace filenames are Windows-safe and events append as JSONL."""
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                with patch("harness_lab.trace.datetime") as mock_datetime:
                    mock_datetime.now.return_value = datetime(
                        2026, 10, 31, 21, 18, 18, 504082
                    )
                    save_trace("run-id", {"event": "start"})
                    save_trace("run-id", {"event": "end"})

                trace = Path("traces") / "2026-10-31T21-18-18.504082-run-id.jsonl"
                events = [json.loads(line) for line in trace.read_text().splitlines()]
                self.assertEqual(events, [{"event": "start"}, {"event": "end"}])
            finally:
                os.chdir(original_directory)


if __name__ == "__main__":
    unittest.main()
