import ast
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReleaseContractTests(unittest.TestCase):
    def test_source_layout_is_consolidated(self):
        for legacy in ("causal", "diffsynth", "train"):
            self.assertFalse((ROOT / legacy).exists(), f"legacy root directory remains: {legacy}")
        for package in (
            "viddub/foundation",
            "viddub/full_context",
            "viddub/causal",
        ):
            self.assertTrue((ROOT / package / "__init__.py").is_file(), package)
        source = "\n".join(
            path.read_text(errors="replace")
            for path in (ROOT / "viddub").rglob("*.py")
        )
        self.assertNotIn("from diffsynth", source)
        self.assertNotIn("from causal.", source)
        self.assertNotIn("from dub_core", source)

    def test_final_training_contract(self):
        dmd = read_json("configs/full_context/dmd2.json")
        self.assertEqual(dmd["stage"], "dmd")
        self.assertEqual(dmd["student_steps"], 2)
        self.assertEqual(dmd["max_steps"], 100)
        self.assertEqual(dmd["teacher_cfg"], 2)
        self.assertEqual(dmd["fake_updates"], 5)
        self.assertEqual(dmd["seed"], 20261008)
        self.assertEqual(dmd["rollout_quality_weight"], 0.987912666033765)

        fake = read_json("configs/full_context/fake2.json")
        self.assertEqual(fake["max_steps"], 200)
        self.assertEqual(fake["student_steps"], 2)
        self.assertEqual(fake["teacher_cfg"], 2)

    def test_selected_manifest_is_paper_final(self):
        manifest = read_json("weights/manifest.json")
        self.assertEqual(manifest["selected_model"], "viddub-full2-dmd-r4-step100")
        selected = [item for item in manifest["artifacts"] if item.get("selected")]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["name"], "viddub-full2-dmd-r4-step100")
        self.assertEqual(selected[0]["source_run"], "FULL2_DMD_R4")
        self.assertFalse(selected[0]["causal"])
        self.assertEqual(selected[0]["student_steps"], 2)
        self.assertEqual(selected[0]["inference_audio_cfg"], 1)

    def test_inference_uses_device_generator_and_single_audio_branch(self):
        source = (ROOT / "scripts/infer.py").read_text()
        tree = ast.parse(source)
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Generator"
        ]
        self.assertTrue(calls, "scripts/infer.py must construct a torch.Generator")
        self.assertTrue(
            any(any(keyword.arg == "device" for keyword in call.keywords) for call in calls),
            "CUDA inference must create its RNG on the latent device",
        )
        self.assertIn('float(config["inference_audio_cfg"]) != 1.0', source)
        self.assertIn('config.get("paper_selected") is not True', source)
        self.assertIn('config.get("causal") is not False', source)
        self.assertIn('"FULL2_DMD_R4"', source)
        self.assertIn('"google/umt5-xxl"', source)

    def test_full_video_inference_has_landmark_fallback(self):
        detector = (ROOT / "data/s01_analyze.py").read_text()
        inference = (ROOT / "viddub/full_context/inference.py").read_text()
        self.assertIn("min_detection_confidence=0.5", detector)
        self.assertIn("min_tracking_confidence=0.5", detector)
        self.assertIn("min_detection_confidence=0.05", inference)
        self.assertIn("min_tracking_confidence=0.1", inference)
        self.assertIn("if not np.isfinite(lm).any()", inference)

    def test_paper_long_video_protocol_is_the_release_default(self):
        cli = (ROOT / "scripts/infer.py").read_text()
        inference = (ROOT / "viddub/full_context/inference.py").read_text()
        runtime = (ROOT / "viddub/full_context/runtime.py").read_text()
        self.assertIn('"--mode", choices=("auto", "clip", "windowed")', cli)
        self.assertIn('"--overlap", type=int, default=17', cli)
        self.assertIn("PAPER_OVERLAP = 17", inference)
        self.assertIn("window_seed = seed + window_index", inference)
        self.assertIn("harmonize_lower_face(gen, crops, mask > 127)", inference)
        self.assertIn("crop_output[start:start + ov]", inference)
        self.assertIn('"-af", "apad"', runtime)

    def test_exporter_defaults_to_final_model(self):
        source = (ROOT / "scripts/export_hf_models.py").read_text()
        self.assertIn('"repo": "VidDub"', source)
        self.assertIn('"source_run": "FULL2_DMD_R4"', source)
        self.assertIn('default=["final"]', source)
        self.assertNotIn("VidDub-2Step", source)
        self.assertIn('default="other"', source)
        self.assertIn(
            "VidDub Individual Commercial and Research License 1.0", source)
        self.assertIn('ARXIV_ID = "2611.03404"', source)
        self.assertIn("@article{{liu2026viddub", source)
        self.assertIn("AIPeanutman/VidDub-Bench", source)
        self.assertIn('("--paper-url", args.paper_url)', source)
        self.assertIn('REPO_ROOT / "weights" / "staged"', source)
        self.assertIn('REPO_ROOT / "export"', source)

    def test_public_example_uses_release_inference_path(self):
        source = (ROOT / "examples/infer_bidirectional.sh").read_text()
        self.assertIn("python scripts/infer.py", source)
        self.assertIn("--overlap 17", source)
        self.assertNotIn("--steps 30", source)

    def test_benchmark_runner_reuses_loaded_model_and_frozen_pairs(self):
        inference = (ROOT / "scripts/infer.py").read_text()
        runner = (ROOT / "scripts/run_benchmark.py").read_text()
        validator = (ROOT / "scripts/validate_benchmark.py").read_text()
        self.assertIn("def load_inference_runtime(", inference)
        self.assertIn("def infer_file(", inference)
        self.assertEqual(runner.count("load_inference_runtime("), 1)
        self.assertIn('"audio_same" if condition == "same" else "audio_cross"', runner)
        self.assertIn('seed=int(row["seed"])', runner)
        self.assertIn('mode="clip"', runner)
        self.assertIn('row["audio_cross"] != target["audio_same"]', validator)
        self.assertIn("manifest/checksums.sha256", validator)

    def test_training_dataset_release_contract_is_explicit(self):
        stage = (ROOT / "scripts/stage_h3_dataset.py").read_text()
        validator = (ROOT / "scripts/validate_training_dataset.py").read_text()
        card = (ROOT / "data/TRAINING_DATA_CARD.md").read_text()
        self.assertIn('"license": "CC-BY-NC-4.0"', stage)
        self.assertIn('"source_disjoint_from_benchmark": True', stage)
        self.assertIn("45_221", validator)
        self.assertIn("85_435", validator)
        self.assertIn("license: cc-by-nc-4.0", card)
        self.assertIn("approximately **211.2 GiB**", card)

    def test_cli_help_smoke(self):
        for script in (
            "scripts/infer.py",
            "scripts/run_benchmark.py",
            "scripts/validate_benchmark.py",
            "scripts/validate_training_dataset.py",
            "scripts/export_hf_models.py",
            "scripts/audit_release.py",
            "scripts/stage_checkpoint.py",
        ):
            result = subprocess.run(
                [sys.executable, str(ROOT / script), "--help"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, f"{script}: {result.stderr}")
            self.assertIn("usage:", result.stdout.lower())

    def test_exported_package_when_available(self):
        configured = os.environ.get("VIDDUB_MODEL_PACKAGE")
        package = Path(configured) if configured else ROOT.parents[1] / "weight/VidDub"
        if not package.is_dir():
            self.skipTest("portable VidDub package has not been generated")
        config = json.loads((package / "config.json").read_text())
        receipt = json.loads((package / "checksums.json").read_text())
        self.assertTrue(config["paper_selected"])
        self.assertEqual(config["source_run"], "FULL2_DMD_R4")
        self.assertFalse(config["causal"])
        self.assertEqual(config["student_steps"], 2)
        self.assertEqual(config["inference_audio_cfg"], 1)
        for entry in receipt["files"].values():
            path = package / entry["path"]
            self.assertEqual(path.stat().st_size, entry["bytes"])
            self.assertEqual(sha256(path), entry["sha256"])


if __name__ == "__main__":
    unittest.main()
