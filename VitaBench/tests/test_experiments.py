from copy import deepcopy
import contextlib
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experimentkit.attacks import make_attacks
from experimentkit.cli import main
from experimentkit.design import load_design
from experimentkit.runner import execute, plan
from experimentkit.summary import summarize, summarize_directory

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [ROOT / "experiments" / name / "config.json" for name in
           ["01_multi_agent", "02_adversarial", "03_cross_verifier"]]


def result(job, prediction="PASS", status="complete"):
    row = {**job, "status": status}
    if status == "complete":
        row["prediction"] = prediction
    return row


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.design = load_design(CONFIGS[0])
        self.env = {"FORMALJUDGE_BASE_URL": "https://example.invalid/v1", "FORMALJUDGE_MODEL": "fixture-model"}

    def test_all_three_matrices_cover_their_intended_variable(self):
        designs = [load_design(p) for p in CONFIGS]
        self.assertEqual([len(d.jobs()) for d in designs], [8, 8, 16])
        self.assertEqual([d.max_calls for d in designs], [16, 16, 32])
        self.assertEqual({j["group"] for j in designs[0].jobs()}, {"gpt-4o", "claude-sonnet-4-5-20250929"})
        self.assertEqual({j["group"] for j in designs[1].jobs()}, {"L1", "L2", "L3", "L4"})
        self.assertEqual({j["group"] for j in designs[2].jobs()}, {"judge_a", "judge_b"})

    def test_cross_verifier_uses_identical_samples_for_every_profile(self):
        d = load_design(CONFIGS[2])
        by_profile = {p["id"]: {(j["sample_id"], j["method"]) for j in d.jobs() if j["verifier_id"] == p["id"]} for p in d.profiles}
        self.assertEqual(by_profile["judge_a"], by_profile["judge_b"])

    def test_data_example_index_matches_exported_actions(self):
        index = {x["sample_id"]: x for x in json.loads((ROOT / "examples/index.json").read_text())["examples"]}
        self.assertEqual(set(index), {x["sample_id"] for x in self.design.samples})
        for sample in self.design.samples:
            entry = index[sample["sample_id"]]
            self.assertEqual(entry["source_agent"], sample["source_agent"])
            self.assertEqual(entry["task_id"], sample["action"]["task_id"])
            self.assertEqual(entry["tool_name"], sample["action"]["tool_calls"][0]["name"])
            self.assertEqual(sample["action"]["example_origin"], "vitabench_trace_excerpt")
            self.assertNotIn("raw_data", sample["action"])
            self.assertNotIn("headers", sample["action"])

    def test_attack_variants_preserve_seed_request_and_lineage(self):
        seed = json.loads((CONFIGS[1].parent / "seed.json").read_text())
        for row in make_attacks(seed):
            self.assertEqual(row["seed_id"], seed["sample_id"])
            self.assertEqual(row["action"]["user_intent"], seed["action"]["user_intent"])
            self.assertEqual(row["action"]["history"][:len(seed["action"]["history"])], seed["action"]["history"])
            self.assertNotEqual(row["action"]["tool_calls"][0]["arguments"], seed["action"]["tool_calls"][0]["arguments"])
            self.assertEqual(row["action"]["example_origin"], "generated_attack_variant")

    def test_attack_builder_matches_packaged_variants_without_mutating_seed(self):
        p = CONFIGS[1].parent
        seed = json.loads((p / "seed.json").read_text())
        before = deepcopy(seed)
        built = make_attacks(seed)
        saved = [json.loads(line) for line in (p / "samples.jsonl").read_text().splitlines()]
        self.assertEqual(built, saved)
        self.assertEqual(seed, before)
        self.assertEqual(len({x["attack_level"] for x in built}), 4)

    def test_plan_does_not_run_a_model_or_include_label_metadata(self):
        with tempfile.TemporaryDirectory() as tmp, patch("experimentkit.runner.complete") as caller:
            out = Path(tmp) / "plan"
            plan(self.design, out)
            caller.assert_not_called()
            for p in out.glob("job_*/*prompt.txt"):
                text = p.read_text()
                self.assertNotIn("gpt-4o", text)
                self.assertNotIn("saved_judge_annotations", text)
            self.assertFalse((out / "results.jsonl").exists())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "plan"
            plan(self.design, out)
            before = (out / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                plan(self.design, out)
            self.assertEqual(before, (out / "manifest.json").read_bytes())

    def test_insufficient_budget_stops_before_output_and_calls(self):
        with tempfile.TemporaryDirectory() as tmp, patch("experimentkit.runner.complete") as caller:
            out = Path(tmp) / "run"
            with self.assertRaises(ValueError):
                execute(self.design, out, max_api_calls=1)
            self.assertFalse(out.exists())
            caller.assert_not_called()

    def test_live_cli_requires_explicit_execution_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with patch("sys.argv", ["run.py", "run", "--output", str(out)]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(CONFIGS[0])
            self.assertFalse(out.exists())

    def test_errors_and_missing_jobs_do_not_become_negative_predictions(self):
        jobs = self.design.jobs()
        s = summarize(self.design, [result(jobs[0], status="error")])
        self.assertEqual(sum(g["execution_errors"] for g in s["groups"]), 1)
        self.assertEqual(sum(g["not_run"] for g in s["groups"]), 7)
        self.assertEqual(sum(g["scored"] for g in s["groups"]), 0)
        self.assertTrue(all(g["accuracy"] is None for g in s["groups"]))

    def test_summary_uses_labels_only_after_prediction(self):
        rows = [result(j, self.design.labels[j["sample_id"]]) for j in self.design.jobs()]
        s = summarize(self.design, rows)
        self.assertTrue(all(g["accuracy"] == 1 for g in s["groups"]))
        self.assertTrue(all(g["coverage"] == 1 for g in s["groups"]))
        self.assertEqual("saved_judge_annotations", s["label_origin"])

    def test_duplicate_and_mismatched_rows_are_rejected(self):
        row = result(self.design.jobs()[0])
        with self.assertRaises(ValueError):
            summarize(self.design, [row, row])
        bad = {**row, "verifier_id": "different"}
        with self.assertRaises(ValueError):
            summarize(self.design, [bad])

    def test_stale_input_fingerprints_reject_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "plan"
            plan(self.design, out)
            changed = replace(self.design, fingerprints={"samples_sha256": "changed"})
            with self.assertRaises(ValueError):
                summarize_directory(changed, out)

    def test_mocked_hybrid_and_control_follow_real_stage_order(self):
        small = replace(self.design, samples=self.design.samples[:1])
        semantic = json.loads((ROOT / "tests/fixtures/semantic_response.json").read_text())
        final = {"decision": "BLOCK", "confidence": 0.8, "reasoning": "fixture", "formal_assessment": "fixture", "semantic_assessment": "fixture", "consistency_check": "fixture"}
        calls = []
        def caller(prompt, **kwargs):
            calls.append(prompt)
            if "Make the Final Verdict" in prompt: return json.dumps(final), "fixture-model"
            if "Your mission: Verify" in prompt: return "```dafny\nmethod M() { assert true; }\n```", "fixture-model"
            return json.dumps(semantic), "fixture-model"
        def solver(path, **kwargs):
            self.assertIn("assert true", path.read_text())
            return {"status": "VERIFIED", "reasoning": "fixture", "violations": [], "error_type": None}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, self.env):
            rows = execute(small, Path(tmp) / "run", max_api_calls=4, caller=caller, solver=solver)
        self.assertEqual(len(calls), 4)
        self.assertEqual([r["status"] for r in rows], ["complete", "complete"])
        self.assertEqual([r["prediction"] for r in rows], ["BLOCK", "PASS"])

    def test_model_failure_is_recorded_without_prediction(self):
        small = replace(self.design, samples=self.design.samples[:1], methods=["hybrid"])
        def failed(*args, **kwargs): raise RuntimeError("fixture transport failure")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, self.env):
            rows = execute(small, Path(tmp) / "run", max_api_calls=3, caller=failed)
        self.assertEqual(rows[0]["status"], "error")
        self.assertNotIn("prediction", rows[0])
        self.assertEqual(rows[0]["api_call_attempts"], 1)

    def test_identical_resolved_cross_verifier_profiles_are_rejected(self):
        d = load_design(CONFIGS[2])
        env = {name: value for suffix in ["A", "B"] for name, value in [(f"JUDGE_{suffix}_BASE_URL", "https://example.invalid/v1"), (f"JUDGE_{suffix}_MODEL", "same-model")]}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env), self.assertRaises(ValueError):
            execute(d, Path(tmp) / "run", max_api_calls=32)

    def test_label_leakage_fields_and_embedded_credentials_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            samples = deepcopy(self.design.samples)
            samples[0]["action"]["expected_label"] = "BLOCK"
            (p / "samples.jsonl").write_text("\n".join(json.dumps(s) for s in samples))
            config = {"experiment": "multi_agent", "samples": "samples.jsonl", "methods": ["hybrid"], "verifiers": self.design.profiles}
            (p / "config.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError): load_design(p / "config.json")
            samples[0]["action"].pop("expected_label")
            (p / "samples.jsonl").write_text("\n".join(json.dumps(s) for s in samples))
            config["verifiers"] = [{**self.design.profiles[0], "api_key": "not-allowed"}]
            (p / "config.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError): load_design(p / "config.json")


if __name__ == "__main__":
    unittest.main()
