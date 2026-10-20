import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from llm_tel import config
from llm_tel.config import get_api_key


class ConfigApiKeyTests(unittest.TestCase):
    def test_get_api_key_success(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-v1-secret-test-key"}):
            self.assertEqual(get_api_key(), "sk-or-v1-secret-test-key")

    def test_get_api_key_missing_exits_with_code_2_and_message(self):
        stdout_buf = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(stdout_buf):
            with self.assertRaises(SystemExit) as ctx:
                get_api_key()
            self.assertEqual(ctx.exception.code, 2)
        self.assertIn(
            "OPENROUTER_API_KEY is not set; export it or add it to .env",
            stdout_buf.getvalue(),
        )

    def test_get_api_key_empty_exits_with_code_2_and_message(self):
        stdout_buf = io.StringIO()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}), contextlib.redirect_stdout(stdout_buf):
            with self.assertRaises(SystemExit) as ctx:
                get_api_key()
            self.assertEqual(ctx.exception.code, 2)
        self.assertIn(
            "OPENROUTER_API_KEY is not set; export it or add it to .env",
            stdout_buf.getvalue(),
        )

    def test_lazy_module_attribute_access(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-v1-lazy-key"}):
            self.assertEqual(config.OPENROUTER_API_KEY, "sk-or-v1-lazy-key")

        with patch.dict(os.environ, {}, clear=True):
            stdout_buf = io.StringIO()
            with contextlib.redirect_stdout(stdout_buf):
                with self.assertRaises(SystemExit) as ctx:
                    _ = config.OPENROUTER_API_KEY
                self.assertEqual(ctx.exception.code, 2)

    def test_cli_help_without_api_key(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        result = subprocess.run(
            [sys.executable, os.path.join(repo_root, "telephone.py"), "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage: telephone.py", result.stdout)
        self.assertIn("--start", result.stdout)

    def test_cli_run_without_api_key_exits_with_code_2_and_message(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        env["PYTHONPATH"] = repo_root
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, os.path.join(repo_root, "telephone.py"), "--start", "Test seed phrase"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(result.returncode, 2)
            combined_output = result.stdout + result.stderr
            self.assertIn(
                "OPENROUTER_API_KEY is not set; export it or add it to .env",
                combined_output,
            )
            self.assertNotIn("sk-or-v1", combined_output)


if __name__ == "__main__":
    unittest.main()
