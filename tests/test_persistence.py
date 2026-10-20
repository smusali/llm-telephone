import builtins
import contextlib
import csv
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from llm_tel.persistence import Recorder


@contextlib.contextmanager
def cp1252_default():
    """Emulate a Windows text-file default without overriding explicit encodings."""
    real_open = builtins.open

    def open_with_default(file, mode="r", *args, **kwargs):
        if "b" not in mode and len(args) < 2 and kwargs.get("encoding") is None:
            kwargs["encoding"] = "cp1252"
        return real_open(file, mode, *args, **kwargs)

    with patch("builtins.open", side_effect=open_with_default):
        yield


class PersistenceTests(unittest.TestCase):
    def test_unicode_row_with_cp1252_default(self):
        row = {
            "cycle": 1,
            "phase": "A→Q",
            "model": "offline-fake",
            "text": "café, 中文\nsecond line",
            "score": 8,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with cp1252_default(), Recorder(path) as recorder:
                recorder.log(**row)
            with open(path / "chain.csv", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows, [{key: str(value) for key, value in row.items()}])
            lines = (path / "chain.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(line) for line in lines], [row])

    def test_complete_offline_run_with_cp1252_default(self):
        # Use a synthetic key only to import the existing configuration. Every
        # completion is replaced locally before the CLI is invoked.
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-test"}, clear=True):
            from llm_tel import client, config
            from telephone import main

        calls = []

        def complete(*, model, messages, temperature):
            calls.append(model)
            text = (
                '{"score": 8}'
                if model == config.JUDGE_MODEL
                else "Offline café 中文 response"
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
            )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=complete))
        )
        with tempfile.TemporaryDirectory() as directory:
            with (
                contextlib.chdir(directory),
                cp1252_default(),
                patch.object(client, "_client", return_value=fake_client),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main(["--start", "Synthetic seed"])
            experiment = next((Path(directory) / "experiments").iterdir())
            with open(experiment / "chain.csv", encoding="utf-8", newline="") as file:
                csv_rows = list(csv.DictReader(file))
            json_rows = [
                json.loads(line)
                for line in (experiment / "chain.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(json_rows), config.MAX_CYCLES)
            self.assertEqual(len(csv_rows), config.MAX_CYCLES)
            self.assertEqual(len(calls), config.MAX_CYCLES * 2)
            for index, (csv_row, json_row) in enumerate(zip(csv_rows, json_rows), 1):
                self.assertEqual(json_row["cycle"], index)
                self.assertEqual(json_row["phase"], "A→Q" if index % 2 else "Q→A")
                self.assertEqual(json_row["text"], "Offline café 中文 response")
                self.assertEqual(json_row["score"], 8)
                self.assertEqual(
                    csv_row, {key: str(value) for key, value in json_row.items()}
                )


if __name__ == "__main__":
    unittest.main()
