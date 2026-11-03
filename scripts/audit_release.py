"""Audit a VidDub source tree or exported model directory before release."""
import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".sh", ".md", ".json", ".jsonl", ".txt", ".toml", ".yaml", ".yml"}
SKIP_PARTS = {".git", "__pycache__"}
SKIP_FILES = {Path(__file__).resolve()}
RULES = {
    "internal absolute path": re.compile(r"/apdcephfs|/root/|/home/[^/]+/"),
    "internal host": re.compile(r"\b28\.7\.\d+\.\d+\b"),
    "secret assignment": re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[=:]\s*['\"][^'\"]+"),
    "old project spelling": re.compile(r"\b(?:CausalDub|CasualDub)\b"),
    "unresolved placeholder": re.compile(
        r"<(?:HF_NAMESPACE|VIDDUB_GITHUB_URL)>|REPLACE_ME|replace-me"
    ),
}


def scan_tree(root, enforce_git_size=False):
    errors = []
    for path in root.rglob("*"):
        if (path.resolve() in SKIP_FILES or any(part in SKIP_PARTS for part in path.parts)
                or not path.is_file()):
            continue
        rel = path.relative_to(root)
        if enforce_git_size and path.stat().st_size > 100 * 1024 * 1024:
            errors.append(f"large Git file: {rel} ({path.stat().st_size} bytes)")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError as exc:
            errors.append(f"unreadable: {rel}: {exc}")
            continue
        for label, pattern in RULES.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                errors.append(f"{label}: {rel}:{line}")
    return errors


def audit_source(root):
    errors = []
    manifest_path = root / "weights/manifest.json"
    if not manifest_path.is_file():
        return ["missing release selection contract: weights/manifest.json"]
    manifest = json.loads(manifest_path.read_text())
    selected = [item for item in manifest.get("artifacts", []) if item.get("selected")]
    if manifest.get("selected_model") != "viddub-full2-dmd-r4-step100" or len(selected) != 1:
        errors.append("weights/manifest.json must select exactly viddub-full2-dmd-r4-step100")
    elif selected[0].get("source_run") != "FULL2_DMD_R4":
        errors.append("selected model is missing final paper run provenance")
    for relative in (
        "AGENTS.md",
        "viddub/__init__.py",
        "viddub/foundation/NOTICE",
        "viddub/full_context/runtime.py",
        "viddub/causal/model.py",
        "configs/sft-r2a.json",
        "configs/full_context/ode2.json",
        "configs/full_context/fake2.json",
        "configs/full_context/dmd2.json",
        "data/TRAINING_DATA_CARD.md",
        "dependencies/licenses/CC-BY-NC-4.0.txt",
        "docs/BENCHMARK.md",
        "docs/EVALUATION.md",
        "examples/infer_bidirectional.sh",
        "scripts/run_benchmark.py",
        "scripts/validate_benchmark.py",
        "scripts/validate_training_dataset.py",
        "tests/test_release_contract.py",
    ):
        if not (root / relative).is_file():
            errors.append(f"missing release contract: {relative}")
    return errors


def audit_model(root):
    errors = []
    for relative in ("README.md", "config.json", "checksums.json"):
        if not (root / relative).is_file():
            errors.append(f"missing model-package file: {relative}")
    if errors:
        return errors
    config = json.loads((root / "config.json").read_text())
    receipt = json.loads((root / "checksums.json").read_text())
    if config.get("model_type") != "viddub" or not config.get("paper_selected"):
        errors.append("model config does not identify the paper-selected VidDub model")
    if config.get("source_run") != "FULL2_DMD_R4" or config.get("causal") is not False:
        errors.append("model config is missing selected-run provenance")
    if config.get("student_steps") != 2 or config.get("inference_audio_cfg") != 1:
        errors.append("model config violates the final two-step, single-audio-branch contract")
    if receipt.get("source_selection", {}).get("source_run") != "FULL2_DMD_R4":
        errors.append("checksum receipt is missing selected-run provenance")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--staging", action="store_true",
                        help="Warn, rather than fail, for owner-controlled license metadata")
    args = parser.parse_args()
    root = args.root.resolve()
    is_source = (root / "configs/full_context/dmd2.json").is_file()
    errors = scan_tree(root, enforce_git_size=is_source)
    errors.extend(audit_source(root) if is_source else audit_model(root))

    license_errors = []
    if is_source:
        if not (root / "LICENSE").is_file():
            license_errors.append("missing owner-approved project LICENSE")
        if (root / "LICENSE_PENDING.md").exists():
            license_errors.append("LICENSE_PENDING.md remains in release tree")
    else:
        readme = (root / "README.md")
        if readme.is_file() and not re.search(r"(?m)^license:\s*\S+", readme.read_text()):
            license_errors.append("model card has no owner-approved license identifier")
    if args.staging:
        for message in license_errors:
            print(f"WARNING: {message}", file=sys.stderr)
    else:
        errors.extend(license_errors)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f"release audit passed: {root}")


if __name__ == "__main__":
    main()
