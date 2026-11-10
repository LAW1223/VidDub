import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "package_training_clips",
    ROOT / "scripts/package_training_clips.py",
)
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


class PackageTrainingClipsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.clips = self.root / "clips"
        self.output = self.root / "output"
        self.selection = self.root / "selection.jsonl"
        rows = []
        for index in range(3):
            clip_id = f"synthetic_{index:03d}_000"
            relative = f"{index:03d}/{clip_id}"
            folder = self.clips / relative
            folder.mkdir(parents=True)
            for filename in PACKAGE.MEDIA_FILES:
                (folder / filename).write_bytes(
                    f"{clip_id}/{filename}".encode()
                )
            rows.append(
                {
                    "clip_id": clip_id,
                    "id": clip_id.rsplit("_", 1)[0],
                    "dir": relative,
                    "lang": "en",
                }
            )
        self.selection.write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_packages_complete_resumable_zip_shards(self):
        release = PACKAGE.package(
            self.selection,
            self.clips,
            self.output,
            shard_size=2,
            workers=2,
            expected_clips=3,
        )
        self.assertEqual(release["clips"], 3)
        self.assertEqual(release["zip_shards"], 2)
        first = self.output / "zips/clips-00000.zip"
        original = first.read_bytes()
        with zipfile.ZipFile(first) as archive:
            self.assertEqual(len(archive.namelist()), 9)
            self.assertIn(
                "clips/000/synthetic_000_000/video.mp4",
                archive.namelist(),
            )
            self.assertEqual(
                archive.getinfo("manifest/shard-items-00000.jsonl").date_time,
                PACKAGE.FIXED_TIME,
            )
        PACKAGE.package(
            self.selection,
            self.clips,
            self.output,
            shard_size=2,
            workers=2,
            expected_clips=3,
        )
        self.assertEqual(first.read_bytes(), original)
        self.assertTrue((self.output / "README.md").is_file())
        self.assertTrue((self.output / "LICENSE").is_file())
        self.assertEqual(
            len((self.output / "manifest/shards.jsonl").read_text().splitlines()),
            2,
        )

    def test_rejects_unsafe_clip_directory(self):
        row = json.loads(self.selection.read_text().splitlines()[0])
        row["dir"] = "../outside"
        self.selection.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "unsafe clip directory"):
            PACKAGE.read_selection(self.selection)


if __name__ == "__main__":
    unittest.main()
