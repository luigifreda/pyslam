"""Run with python -m unittest discover -s docs/images -p 'test_*.py'."""

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import render_diagrams


class RenderDiagramsTest(unittest.TestCase):
    def run_renderer(self, source, returncode):
        def fake_render(cmd, **kwargs):
            # A renderer can write output before reporting a failure.
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"new PNG" if returncode == 0 else b"partial PNG")
            return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr="render failed" if returncode else "")

        with (
            mock.patch.object(sys, "argv", ["render_diagrams.py", str(source)]),
            mock.patch.object(render_diagrams, "find_mmdc", return_value=["mmdc"]),
            mock.patch.object(render_diagrams, "find_chrome", return_value=None),
            mock.patch.object(render_diagrams.subprocess, "run", side_effect=fake_render),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            render_diagrams.main()

    def test_failed_render_preserves_output_and_remains_outdated(self):
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / "diagram.mmd"
                source.write_text("graph TD; A --> B;")
                png = source.with_suffix(".png")
                if existing:
                    png.write_bytes(b"old PNG")
                    os.utime(png, (1, 1))
                with self.assertRaises(SystemExit) as error:
                    self.run_renderer(source, returncode=1)
                self.assertEqual(error.exception.code, 1)
                self.assertTrue(render_diagrams.is_outdated(source))
                if existing:
                    self.assertEqual(png.read_bytes(), b"old PNG")
                else:
                    self.assertFalse(png.exists())
                self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                                 ["diagram.mmd", "diagram.png"] if existing else ["diagram.mmd"])

    def test_successful_render_replaces_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "diagram.mmd"
            source.write_text("graph TD; A --> B;")
            png = source.with_suffix(".png")
            png.write_bytes(b"old PNG")
            os.utime(png, (1, 1))
            self.run_renderer(source, returncode=0)
            self.assertEqual(png.read_bytes(), b"new PNG")
            self.assertFalse(render_diagrams.is_outdated(source))
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["diagram.mmd", "diagram.png"])


if __name__ == "__main__":
    unittest.main()
