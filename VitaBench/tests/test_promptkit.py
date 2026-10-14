import json
import re
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
import contextlib
import io

from promptkit.client import complete
from promptkit.formal import verify
from promptkit.records import extract_dafny, normalize_aggregation, normalize_semantic, parse_json_response
from promptkit.render import render
import run_sample

ROOT = Path(__file__).resolve().parents[1]


class PromptKitTests(unittest.TestCase):
    def setUp(self):
        self.action = json.loads((ROOT / "tests/fixtures/action.json").read_text())
        self.semantic = json.loads((ROOT / "tests/fixtures/semantic_response.json").read_text())
        self.formal = json.loads((ROOT / "tests/fixtures/formal_response.json").read_text())

    def test_formal_prompt_contains_context_and_real_template_sections(self):
        p = render("formal", self.action)
        for text in ("Verification Layers", "2026-11-10", "2026-11-11", "Available Tools", "Dafny"):
            self.assertIn(text, p)
        self.assertNotIn("{task_context}", p)

    def test_semantic_prompt_preserves_ten_layers_and_current_action(self):
        p = render("semantic", self.action)
        for n in range(10):
            self.assertIn(f"layer_{n}_", p)
        self.assertIn("fixture_book_hotel", p)
        self.assertIn("2026-11-11", p)

    def test_formal_render_removes_template_double_braces(self):
        p = render("formal", self.action)
        for code in re.findall(r"```dafny\n(.*?)```", p, flags=re.S):
            self.assertNotIn("{{", code)
            self.assertNotIn("}}", code)
        self.assertIn("predicate Layer0_FormatValid(s: AgentState) {", p)

    def test_formal_render_preserves_user_braces_literally(self):
        self.action["user_intent"] = 'Keep the literal marker {{keep_me}} and {task_context}.'
        p = render("formal", self.action)
        self.assertIn(self.action["user_intent"], p)

    def test_aggregation_requires_actual_input_records(self):
        with self.assertRaises(ValueError):
            render("aggregation", self.action)

    def test_aggregation_includes_both_evidence_lanes(self):
        p = render("aggregation", self.action, semantic=self.semantic, formal=self.formal)
        self.assertIn("UNPROVEN", p)
        self.assertIn("layer_5_temporal: 2.0/10", p)
        self.assertIn("Make the Final Verdict", p)

    def test_semantic_score_is_computed_from_layers(self):
        self.semantic["overall_score"] = 0
        self.assertEqual(normalize_semantic(self.semantic)["overall_score"], 80)

    def test_invalid_scores_are_not_converted_to_decisions(self):
        for value in [True, -1, 11, float("nan")]:
            self.semantic["tool_layer_scores"]["layer_5_temporal"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_semantic(self.semantic)

    def test_missing_aggregation_decision_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_aggregation({"confidence": 0.9})

    def test_json_fence_accepted_but_extra_text_rejected(self):
        self.assertEqual(parse_json_response('```json\n{"a": 1}\n```'), {"a": 1})
        with self.assertRaises(ValueError):
            parse_json_response('Some prose {"a": 1}')

    def test_dafny_extraction_requires_one_code_block(self):
        self.assertEqual(extract_dafny('Analysis\n```dafny\nmethod M() {}\n```'), 'method M() {}\n')
        with self.assertRaises(ValueError):
            extract_dafny('```dafny\nmethod A() {}\n```\n```dafny\nmethod B() {}\n```')

    def test_api_failure_cannot_return_a_verdict_or_leak_server_body(self):
        class Broken:
            def open(self, *args, **kwargs):
                raise HTTPError("https://example.invalid", 429, "sensitive server message", {}, None)
        with patch.dict("os.environ", {"FORMALJUDGE_BASE_URL": "https://example.invalid/v1", "FORMALJUDGE_MODEL": "example"}), patch("promptkit.client.build_opener", return_value=Broken()):
            with self.assertRaises(RuntimeError) as cm:
                complete("example prompt")
        self.assertIn("429", str(cm.exception))
        self.assertNotIn("sensitive", str(cm.exception))

    def test_local_adapter_uses_real_request_shape(self):
        captured = {}
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, size): return b'{"choices":[{"message":{"content":"example response"}}]}'
        class Endpoint:
            def open(self, request, **kwargs):
                captured["url"] = request.full_url
                captured["payload"] = json.loads(request.data)
                return Response()
        with patch.dict("os.environ", {"FORMALJUDGE_BASE_URL": "http://127.0.0.1:8000/v1", "FORMALJUDGE_MODEL": "local-model"}), patch("promptkit.client.build_opener", return_value=Endpoint()):
            text, model = complete("test prompt")
        self.assertEqual(text, "example response")
        self.assertEqual(model, "local-model")
        self.assertEqual(captured["url"], "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(captured["payload"]["messages"], [{"role": "user", "content": "test prompt"}])

    def test_solver_process_status_is_separate_from_decision(self):
        for code in [0, 4, 2]:
            with self.subTest(code=code), patch("promptkit.formal.subprocess.run", return_value=SimpleNamespace(returncode=code, stdout="test output", stderr="")):
                result = verify(ROOT / "examples/check_date.dfy")
                self.assertEqual(result["status"], "VERIFIED" if code == 0 else "UNPROVEN")
                self.assertNotIn("decision", result)
                self.assertEqual(bool(result["error_type"]), code == 2)

    def test_formal_process_error_stops_aggregation(self):
        self.formal["error_type"] = "execution_error"
        with self.assertRaises(ValueError):
            render("aggregation", self.action, semantic=self.semantic, formal=self.formal)

    def test_solver_timeout_is_not_a_negative_label(self):
        with patch("promptkit.formal.subprocess.run", side_effect=subprocess.TimeoutExpired("dafny", 1)):
            with self.assertRaises(RuntimeError):
                verify(ROOT / "examples/check_date.dfy")

    def test_cli_offline_aggregation_writes_a_real_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "prompt.txt"
            argv = ["run_sample.py", "render", "--stage", "aggregation", "--input", str(ROOT / "tests/fixtures/action.json"), "--semantic", str(ROOT / "tests/fixtures/semantic_response.json"), "--formal", str(ROOT / "tests/fixtures/formal_response.json"), "--output", str(out)]
            with patch("sys.argv", argv), patch("run_sample.complete") as client, contextlib.redirect_stdout(io.StringIO()):
                run_sample.main()
            client.assert_not_called()
            self.assertIn("Average Score", out.read_text())

    def test_cli_semantic_call_preserves_input_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "semantic.json"
            argv = ["run_sample.py", "infer", "--stage", "semantic", "--input", str(ROOT / "tests/fixtures/action.json"), "--output", str(out)]
            with patch("sys.argv", argv), patch("run_sample.complete", return_value=(json.dumps(self.semantic), "fixture-model")) as client, contextlib.redirect_stdout(io.StringIO()):
                run_sample.main()
            client.assert_called_once()
            result = json.loads(out.read_text())
            self.assertEqual("test_fixture", result["input_origin"])
            self.assertEqual(result["overall_score"], 80)

    def test_cli_does_not_call_model_when_output_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "existing.json"
            out.write_text("preserve me")
            argv = ["run_sample.py", "infer", "--stage", "semantic", "--input", str(ROOT / "tests/fixtures/action.json"), "--output", str(out)]
            with patch("sys.argv", argv), patch("run_sample.complete") as client, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run_sample.main()
            client.assert_not_called()
            self.assertEqual(out.read_text(), "preserve me")

    def test_cli_invalid_model_response_leaves_no_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "semantic.json"
            argv = ["run_sample.py", "infer", "--stage", "semantic", "--input", str(ROOT / "tests/fixtures/action.json"), "--output", str(out)]
            with patch("sys.argv", argv), patch("run_sample.complete", return_value=("not json", "fixture-model")), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run_sample.main()
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
