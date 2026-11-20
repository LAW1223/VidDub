import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from viddub.causal.sampling import rollout, schedule
from viddub.full_context import inference
from viddub.full_context.color_harmonization import HarmonizationConfig, harmonize_lower_face
from viddub.full_context.runtime import Mp4StreamWriter, write_mp4


class _FullContextConfig:
    num_ref = 1

    @staticmethod
    def video_blocks(frames):
        return [(1, frames)]


class _FullContextModel(torch.nn.Module):
    causal = False
    config = _FullContextConfig()

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        latents,
        timesteps,
        context,
        vace_context,
        whisper_hs,
        num_frames,
        drop_audio=None,
        **kwargs,
    ):
        self.calls.append((latents.shape, timesteps.shape, drop_audio))
        return torch.ones_like(latents)


class FullContextRolloutTests(unittest.TestCase):
    def setUp(self):
        self.model = _FullContextModel()
        self.reference = torch.zeros(1, 2, 4, 2, 2)
        self.conditions = {
            "context": torch.zeros(1, 1, 1),
            "vace_context": torch.zeros(1, 3, 4, 2, 2),
            "whisper_hs": torch.zeros(1, 1, 1),
            "num_frames": 13,
        }

    def test_bidirectional_model_denoises_the_complete_sequence_each_step(self):
        output = rollout(
            self.model,
            self.reference,
            self.conditions,
            schedule(2),
            noise=torch.zeros_like(self.reference),
            use_cache=False,
        )

        self.assertEqual(len(self.model.calls), 2)
        self.assertTrue(all(call[0] == self.reference.shape for call in self.model.calls))
        self.assertTrue(all(call[1] == (1, 4) for call in self.model.calls))
        self.assertTrue(torch.equal(output, -torch.ones_like(output)))

    def test_bidirectional_model_rejects_causal_cache(self):
        with self.assertRaisesRegex(ValueError, "cannot use a history cache"):
            rollout(
                self.model,
                self.reference,
                self.conditions,
                schedule(2),
                noise=torch.zeros_like(self.reference),
                use_cache=True,
            )


class _DummyFrameSource:
    def __init__(self, frames, chunk_size=37):
        self.frames = frames
        self.chunk_size = chunk_size
        self.num_frames, self.height, self.width = frames.shape[:3]

    def iter_chunks(self):
        for start in range(0, self.num_frames, self.chunk_size):
            yield self.frames[start:start + self.chunk_size]

    def get_frame(self, index):
        return self.frames[int(index)]


class _MemoryWriter:
    def __init__(self):
        self.chunks = []

    def write(self, chunk):
        self.chunks.append(chunk.copy())

    def result(self):
        return np.concatenate(self.chunks, axis=0)


class StreamingInferenceEquivalenceTests(unittest.TestCase):
    @staticmethod
    def _synthetic_landmarks(num_frames, height, width, face_size):
        lm = np.full((num_frames, 468, 2), np.nan, dtype=np.float32)
        cx, cy = width / 2.0, height / 2.0
        half = face_size / 2.0
        lm[:, :, 0] = cx
        lm[:, :, 1] = cy
        lm[:, 234, 0] = cx - half
        lm[:, 454, 0] = cx + half
        lm[:, 10, 1] = cy - half
        lm[:, 152, 1] = cy + half
        lm[:, 2, 1] = cy - 0.1 * half
        lm[:, 13, 1] = cy + 0.1 * half
        lm[:, 14, 1] = cy + 0.25 * half
        return lm, np.ones(num_frames, dtype=np.int8)

    @staticmethod
    def _fake_sampler(pipe, whisper, adapter, zero_latents, clip, audio,
                      steps=2, audio_cfg=1.0, seed=0, fps=24):
        frames = clip["frames"].copy()
        offset = (seed % 5) + 1
        frames[:, 180:360, 160:352] = np.clip(
            frames[:, 180:360, 160:352].astype(np.int16) + offset * 7, 0, 255
        ).astype(np.uint8)
        return frames

    def test_streaming_matches_in_memory_dub_video(self):
        rng = np.random.default_rng(1234)
        for num_frames, face_size, harmonize, window_box, overlap, use_sample_fn in (
            (150, 160.0, True, False, 17, True),
            (220, 420.0, False, False, 17, True),
            (150, 160.0, True, True, 0, False),
        ):
            frames = rng.integers(30, 220, size=(num_frames, 320, 320, 3), dtype=np.uint8)
            audio = np.zeros(int(num_frames / 24.0 * 16000), dtype=np.float32)
            lm, nfaces = self._synthetic_landmarks(num_frames, 320, 320, face_size)
            models = (None, None, None, None)

            with mock.patch.object(
                inference, "detect_landmarks", return_value=(lm.copy(), nfaces.copy())
            ), mock.patch.object(
                inference, "detect_landmarks_iter", return_value=(lm.copy(), nfaces.copy())
            ), mock.patch.object(
                inference, "sample", side_effect=self._fake_sampler
            ):
                sample_fn = self._fake_sampler if use_sample_fn else None
                expected = inference.dub_video(
                    models,
                    frames,
                    audio,
                    steps=2,
                    seed=7,
                    window_box=window_box,
                    overlap=overlap,
                    harmonize=harmonize,
                    log=lambda *_: None,
                    sample_fn=sample_fn,
                )
                writer = _MemoryWriter()
                written = inference.dub_video_stream(
                    models,
                    _DummyFrameSource(frames),
                    audio,
                    writer,
                    steps=2,
                    seed=7,
                    window_box=window_box,
                    overlap=overlap,
                    harmonize=harmonize,
                    log=lambda *_: None,
                    sample_fn=sample_fn,
                )
            self.assertEqual(written, num_frames)
            self.assertTrue(np.array_equal(expected, writer.result()))

    def test_missing_face_frames_pass_through_and_all_empty_window_is_skipped(self):
        rng = np.random.default_rng(42)
        num_frames = 170
        frames = rng.integers(30, 220, size=(num_frames, 320, 320, 3), dtype=np.uint8)
        audio = np.zeros(int(num_frames / 24.0 * 16000), dtype=np.float32)
        lm, nfaces = self._synthetic_landmarks(num_frames, 320, 320, 160.0)
        # Frames 0..84 are B-roll (no face), so window 0 (0..81) has no face at all.
        # Frame 120 has a 1-frame detector blink that should be bridged.
        lm[:85] = np.nan
        nfaces[:85] = 0
        lm[120] = np.nan
        nfaces[120] = 0
        models = (None, None, None, None)

        for window_box, overlap, use_sample_fn in (
            (False, 17, True),
            (True, 0, False),
        ):
            calls = []

            def tracking_sampler(*args, **kwargs):
                calls.append(kwargs.get("seed"))
                return self._fake_sampler(*args, **kwargs)

            with mock.patch.object(
                inference, "detect_landmarks", return_value=(lm.copy(), nfaces.copy())
            ), mock.patch.object(
                inference, "detect_landmarks_iter", return_value=(lm.copy(), nfaces.copy())
            ), mock.patch.object(
                inference, "sample", side_effect=tracking_sampler
            ):
                sample_fn = tracking_sampler if use_sample_fn else None
                expected = inference.dub_video(
                    models,
                    frames,
                    audio,
                    steps=2,
                    seed=10,
                    window_box=window_box,
                    overlap=overlap,
                    harmonize=True,
                    log=lambda *_: None,
                    sample_fn=sample_fn,
                )
                in_mem_calls = len(calls)
                calls.clear()
                writer = _MemoryWriter()
                written = inference.dub_video_stream(
                    models,
                    _DummyFrameSource(frames),
                    audio,
                    writer,
                    steps=2,
                    seed=10,
                    window_box=window_box,
                    overlap=overlap,
                    harmonize=True,
                    log=lambda *_: None,
                    sample_fn=sample_fn,
                )

            self.assertEqual(written, num_frames)
            self.assertTrue(np.array_equal(expected, writer.result()))
            # Window 0 (0..81) has zero face frames and must be skipped.
            self.assertEqual(in_mem_calls, 2)
            self.assertEqual(len(calls), 2)
            # B-roll frames (0..84) remain pixel-identical to source.
            self.assertTrue(np.array_equal(expected[:85], frames[:85]))
            # Bridged 1-frame drop (frame 120) still gets dubbed.
            self.assertFalse(np.array_equal(expected[120], frames[120]))

    def test_no_face_error_includes_size_hint(self):
        frames = np.zeros((81, 128, 128, 3), dtype=np.uint8)
        audio = np.zeros(int(81 / 24.0 * 16000), dtype=np.float32)
        empty_lm = np.full((81, 468, 2), np.nan, dtype=np.float32)
        empty_nf = np.zeros(81, dtype=np.int8)
        with mock.patch.object(
            inference, "detect_landmarks", return_value=(empty_lm, empty_nf)
        ):
            with self.assertRaisesRegex(RuntimeError, "too small"):
                inference.dub_video(
                    (None, None, None, None),
                    frames,
                    audio,
                    log=lambda *_: None,
                )

    def test_too_few_skin_anchors_falls_back_to_feathered_blending(self):
        rng = np.random.default_rng(77)
        num_frames = 150
        frames = rng.integers(30, 220, size=(num_frames, 320, 320, 3), dtype=np.uint8)
        audio = np.zeros(int(num_frames / 24.0 * 16000), dtype=np.float32)
        lm, nfaces = self._synthetic_landmarks(num_frames, 320, 320, 160.0)
        models = (None, None, None, None)

        tiny_mask = np.zeros((512, 512), dtype=np.uint8)
        tiny_mask[250:254, 250:254] = 255
        logs_mem = []
        logs_stream = []

        with mock.patch.object(
            inference, "detect_landmarks", return_value=(lm.copy(), nfaces.copy())
        ), mock.patch.object(
            inference, "detect_landmarks_iter", return_value=(lm.copy(), nfaces.copy())
        ), mock.patch.object(
            inference, "lower_face_mask", return_value=tiny_mask.copy()
        ):
            expected_unharmonized = inference.dub_video(
                models,
                frames,
                audio,
                steps=2,
                seed=3,
                overlap=17,
                harmonize=False,
                log=lambda *_: None,
                sample_fn=self._fake_sampler,
            )
            fallback_mem = inference.dub_video(
                models,
                frames,
                audio,
                steps=2,
                seed=3,
                overlap=17,
                harmonize=True,
                log=logs_mem.append,
                sample_fn=self._fake_sampler,
            )
            writer = _MemoryWriter()
            written = inference.dub_video_stream(
                models,
                _DummyFrameSource(frames),
                audio,
                writer,
                steps=2,
                seed=3,
                overlap=17,
                harmonize=True,
                log=logs_stream.append,
                sample_fn=self._fake_sampler,
            )

        self.assertEqual(written, num_frames)
        self.assertTrue(np.array_equal(fallback_mem, expected_unharmonized))
        self.assertTrue(np.array_equal(writer.result(), expected_unharmonized))
        self.assertTrue(any("color harmonization skipped" in msg for msg in logs_mem))
        self.assertTrue(any("color harmonization skipped" in msg for msg in logs_stream))

    def test_harmonization_ignores_source_teeth_inside_mouth_interior(self):
        # Matching dark skin tone across source and generated frames, but
        # source has bright white teeth inside the central mouth while
        # generated has closed warm lips.
        skin = np.array([115, 78, 62], dtype=np.uint8)
        lips = np.array([145, 72, 74], dtype=np.uint8)
        teeth = np.array([245, 245, 245], dtype=np.uint8)
        source = np.broadcast_to(skin, (5, 256, 256, 3)).copy()
        generated = np.broadcast_to(skin, (5, 256, 256, 3)).copy()
        mask = np.zeros((256, 256), dtype=bool)
        mask[70:210, 50:206] = True
        # Central mouth patch (inside mask, outside cheek/upper anchors)
        source[:, 120:170, 95:161] = teeth
        generated[:, 120:170, 95:161] = lips

        out, report = harmonize_lower_face(generated, source, mask)
        lip_region = out[:, 130:160, 105:151]
        # Closed lips must stay close to their natural color rather than
        # getting desaturated toward white/gray teeth.
        max_lip_drift = np.abs(lip_region.astype(np.float32) - lips.astype(np.float32)).max()
        self.assertLess(max_lip_drift, 3.0)
        self.assertAlmostEqual(report["chroma_field_strength"], 0.90)

        _, custom_report = harmonize_lower_face(
            generated, source, mask, config=HarmonizationConfig(chroma_field_strength=0.35)
        )
        self.assertAlmostEqual(custom_report["chroma_field_strength"], 0.35)

    def test_mp4_stream_writer_produces_decodable_video(self):
        import decord

        frames = np.full((12, 64, 64, 3), 120, dtype=np.uint8)
        audio = np.zeros(8000, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            stream_path = Path(tmp) / "stream.mp4"
            batch_path = Path(tmp) / "batch.mp4"
            with Mp4StreamWriter(stream_path, 64, 64, audio=audio, fps=24) as writer:
                writer.write(frames[:5])
                writer.write(frames[5:])
            write_mp4(batch_path, frames, audio=audio, fps=24)
            self.assertEqual(len(decord.VideoReader(str(stream_path))), 12)
            self.assertEqual(len(decord.VideoReader(str(batch_path))), 12)

    def test_resolve_padding_frames_validation(self):
        self.assertEqual(inference.resolve_padding_frames(0.0, 0.0, fps=24.0), (0, 0))
        self.assertEqual(inference.resolve_padding_frames(0.5, 0.25, fps=24.0), (12, 6))
        with self.assertRaisesRegex(ValueError, "lead_in_seconds"):
            inference.resolve_padding_frames(-0.1, 0.0, fps=24.0)
        with self.assertRaisesRegex(ValueError, "lead_out_seconds"):
            inference.resolve_padding_frames(0.0, float("nan"), fps=24.0)
        with self.assertRaisesRegex(ValueError, "fps"):
            inference.resolve_padding_frames(0.5, 0.0, fps=0.0)

    def test_padded_source_and_trimming_writer_match_in_memory_padding(self):
        rng = np.random.default_rng(2026)
        num_frames = 150
        fps = 24.0
        lead_in_frames, lead_out_frames = 12, 10
        frames = rng.integers(30, 220, size=(num_frames, 320, 320, 3), dtype=np.uint8)
        audio = np.linspace(-0.5, 0.5, round(num_frames / fps * 16000), dtype=np.float32)
        lead_in_samples = round(lead_in_frames / fps * 16000)
        lead_out_samples = round(lead_out_frames / fps * 16000)
        padded_audio = np.pad(audio, (lead_in_samples, lead_out_samples), mode="constant")
        padded_frames = np.pad(
            frames,
            ((lead_in_frames, lead_out_frames), (0, 0), (0, 0), (0, 0)),
            mode="edge",
        )
        lm, nfaces = self._synthetic_landmarks(len(padded_frames), 320, 320, 160.0)
        models = (None, None, None, None)

        seen_window0_audio = []

        def recording_sampler(pipe, whisper, adapter, zero_latents, clip, window_audio,
                              steps=2, audio_cfg=1.0, seed=0, fps=24):
            if seed == 5:
                seen_window0_audio.append(window_audio.copy())
            return self._fake_sampler(
                pipe, whisper, adapter, zero_latents, clip, window_audio,
                steps=steps, audio_cfg=audio_cfg, seed=seed, fps=fps,
            )

        with mock.patch.object(
            inference, "detect_landmarks", return_value=(lm.copy(), nfaces.copy())
        ), mock.patch.object(
            inference, "detect_landmarks_iter", return_value=(lm.copy(), nfaces.copy())
        ):
            mem_full = inference.dub_video(
                models,
                padded_frames,
                padded_audio,
                steps=2,
                seed=5,
                fps=fps,
                overlap=17,
                harmonize=True,
                log=lambda *_: None,
                sample_fn=recording_sampler,
            )
            mem_trimmed = mem_full[lead_in_frames:lead_in_frames + num_frames]

            base_writer = _MemoryWriter()
            trim_writer = inference.TrimmingFrameWriter(
                base_writer, skip_start=lead_in_frames, keep_frames=num_frames
            )
            padded_source = inference.PaddedFrameSource(
                _DummyFrameSource(frames, chunk_size=29),
                lead_in_frames=lead_in_frames,
                lead_out_frames=lead_out_frames,
            )
            self.assertEqual(padded_source.num_frames, len(padded_frames))
            self.assertTrue(np.array_equal(padded_source.get_frame(0), frames[0]))
            self.assertTrue(np.array_equal(padded_source.get_frame(lead_in_frames - 1), frames[0]))
            self.assertTrue(np.array_equal(padded_source.get_frame(lead_in_frames), frames[0]))
            self.assertTrue(np.array_equal(padded_source.get_frame(padded_source.num_frames - 1), frames[-1]))

            written = inference.dub_video_stream(
                models,
                padded_source,
                padded_audio,
                trim_writer,
                steps=2,
                seed=5,
                fps=fps,
                overlap=17,
                harmonize=True,
                log=lambda *_: None,
                sample_fn=recording_sampler,
            )

        self.assertEqual(written, len(padded_frames))
        self.assertEqual(trim_writer.written, num_frames)
        self.assertEqual(len(base_writer.result()), num_frames)
        self.assertTrue(np.array_equal(mem_trimmed, base_writer.result()))
        # Window 0 starts with pure silence for lead_in_samples, followed by exact original audio.
        self.assertEqual(len(seen_window0_audio), 2)
        for window_audio in seen_window0_audio:
            self.assertTrue(np.all(window_audio[:lead_in_samples] == 0.0))
            tail_len = len(window_audio) - lead_in_samples
            self.assertTrue(np.allclose(window_audio[lead_in_samples:], audio[:tail_len]))

    def test_infer_file_lead_in_out_in_memory_and_streaming(self):
        from scripts import infer as infer_cli

        infer_cli.import_runtime()

        class _FakeArray:
            def __init__(self, arr):
                self._arr = arr

            def asnumpy(self):
                return self._arr

        class _FakeReader:
            def __init__(self, frames, fps=24.0):
                self._frames = frames
                self._fps = fps

            def __len__(self):
                return len(self._frames)

            def get_avg_fps(self):
                return self._fps

            def __getitem__(self, item):
                return _FakeArray(self._frames[item])

            def get_batch(self, indices):
                return _FakeArray(self._frames[indices])

        rng = np.random.default_rng(77)
        fps = 24.0
        for num_frames in (120, 170):
            frames = rng.integers(30, 220, size=(num_frames, 320, 320, 3), dtype=np.uint8)
            raw_audio = np.linspace(-0.25, 0.25, round(num_frames / fps * 16000), dtype=np.float32)
            lead_in_s, lead_out_s = 0.5, 0.25
            lead_in_f, lead_out_f = 12, 6
            padded_n = num_frames + lead_in_f + lead_out_f
            lm, nfaces = self._synthetic_landmarks(padded_n, 320, 320, 160.0)
            runtime = {
                "models": (None, None, None, None),
                "config": {},
                "steps": 2,
                "audio_cfg": 1.0,
                "sample_fn": self._fake_sampler,
            }
            with tempfile.TemporaryDirectory() as tmp:
                out_path = Path(tmp) / f"out_{num_frames}.mp4"
                with mock.patch.object(
                    infer_cli.decord, "VideoReader", return_value=_FakeReader(frames, fps=fps)
                ), mock.patch.object(
                    infer_cli.librosa, "load", return_value=(raw_audio.copy(), 16000)
                ), mock.patch.object(
                    inference, "detect_landmarks", return_value=(lm.copy(), nfaces.copy())
                ), mock.patch.object(
                    inference, "detect_landmarks_iter", return_value=(lm.copy(), nfaces.copy())
                ):
                    with self.assertRaisesRegex(ValueError, "--mode clip"):
                        infer_cli.infer_file(
                            runtime, "in.mp4", "in.wav", out_path,
                            mode="clip", lead_in_seconds=lead_in_s,
                        )
                    infer_cli.infer_file(
                        runtime, "in.mp4", "in.wav", out_path,
                        seed=0, mode="auto", overlap=17, harmonize=True,
                        lead_in_seconds=lead_in_s, lead_out_seconds=lead_out_s,
                    )
                import decord
                self.assertEqual(len(decord.VideoReader(str(out_path))), num_frames)

    def test_full_range_roi_landmark_fallback_maps_coordinates(self):
        import types
        from data import s01_analyze

        width, height = 1920, 1088
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Simulated detection centered at (960, 544) with 120x120 px box
        bb = types.SimpleNamespace(
            xmin=900 / width,
            ymin=484 / height,
            width=120 / width,
            height=120 / height,
        )
        det = types.SimpleNamespace(
            location_data=types.SimpleNamespace(relative_bounding_box=bb)
        )
        fake_landmarks = [types.SimpleNamespace(x=0.5, y=0.5) for _ in range(468)]
        fake_landmarks[10] = types.SimpleNamespace(x=0.5, y=0.3)
        fake_landmarks[2] = types.SimpleNamespace(x=0.5, y=0.48)
        fake_landmarks[152] = types.SimpleNamespace(x=0.5, y=0.7)
        fake_landmarks[234] = types.SimpleNamespace(x=0.32, y=0.5)
        fake_landmarks[454] = types.SimpleNamespace(x=0.68, y=0.5)

        class _FakeDetector:
            def __init__(self, detections):
                self._detections = detections

            def process(self, img):
                return types.SimpleNamespace(detections=self._detections)

        class _FakeRoiMesh:
            def __init__(self):
                self.crops = []

            def process(self, crop):
                self.crops.append(crop.shape)
                return types.SimpleNamespace(
                    multi_face_landmarks=[types.SimpleNamespace(landmark=fake_landmarks)]
                )

        roi_mesh = _FakeRoiMesh()
        pts, count = s01_analyze._detect_face_via_full_range_roi(
            frame, width, height, _FakeDetector([det]), roi_mesh
        )
        self.assertEqual(count, 1)
        self.assertEqual(pts.shape, (468, 2))
        self.assertEqual(roi_mesh.crops, [(300, 300, 3)])
        # Center of 300x300 crop around (960, 544) maps back to (960, 544)
        self.assertAlmostEqual(float(pts[0, 0]), 960.0, places=1)
        self.assertAlmostEqual(float(pts[0, 1]), 544.0, places=1)
        self.assertAlmostEqual(float(pts[152, 1] - pts[10, 1]), 120.0, places=1)

        # No detections (B-roll frame) -> returns (None, 0)
        none_pts, zero_count = s01_analyze._detect_face_via_full_range_roi(
            frame, width, height, _FakeDetector([]), roi_mesh
        )
        self.assertIsNone(none_pts)
        self.assertEqual(zero_count, 0)

    def test_clean_landmark_track_rejects_outliers_without_corrupting_motion(self):
        num_frames = 100
        lm, _ = self._synthetic_landmarks(num_frames, 1088, 1920, 130.0)
        # Smooth real downward lean of 140 px across the clip
        shifts = np.linspace(0.0, 140.0, num_frames, dtype=np.float32)
        lm[:, :, 1] += shifts[:, None]
        # Inject three outlier frames: poster jump, chest jump, inverted face
        lm[15, :, 0] += 550.0
        lm[15, :, 1] -= 300.0
        lm[50, :, 1] += 320.0
        lm[80, 152, 1] = lm[80, 10, 1] - 20.0  # chin above forehead

        cleaned, rejected = inference.clean_landmark_track(lm, 1088, 1920)
        self.assertEqual(rejected, 3)
        self.assertFalse(np.isfinite(cleaned[15]).any())
        self.assertFalse(np.isfinite(cleaned[50]).any())
        self.assertFalse(np.isfinite(cleaned[80]).any())
        # Legitimate smooth leaning frames remain intact
        self.assertTrue(np.isfinite(cleaned[0]).all())
        self.assertTrue(np.isfinite(cleaned[99]).all())

    def test_stabilized_tracking_follows_moving_speaker_in_memory_and_streaming(self):
        rng = np.random.default_rng(999)
        num_frames = 140
        H, W = 720, 960
        frames = rng.integers(40, 200, size=(num_frames, H, W, 3), dtype=np.uint8)
        audio = np.zeros(int(num_frames / 24.0 * 16000), dtype=np.float32)

        # Stationary clip should NOT trigger stabilized tracking
        static_lm, _ = self._synthetic_landmarks(num_frames, H, W, 140.0)
        static_box = inference.crop_box(static_lm, H, W)
        self.assertFalse(inference._needs_stabilized_tracking(static_lm, static_box))

        # Moving clip: speaker leans down 150 px and closer (+30% scale)
        moving_lm = np.full((num_frames, 468, 2), np.nan, dtype=np.float32)
        for t in range(num_frames):
            frac = t / (num_frames - 1)
            cx = W * 0.5 + 40.0 * np.sin(2.0 * np.pi * frac)
            cy = 240.0 + 150.0 * frac
            half = (120.0 + 36.0 * frac) / 2.0
            moving_lm[t, :, 0] = cx
            moving_lm[t, :, 1] = cy
            moving_lm[t, 234, 0] = cx - half
            moving_lm[t, 454, 0] = cx + half
            moving_lm[t, 10, 1] = cy - half
            moving_lm[t, 152, 1] = cy + half
            moving_lm[t, 2, 1] = cy - 0.1 * half
            moving_lm[t, 13, 1] = cy + 0.1 * half
            moving_lm[t, 14, 1] = cy + 0.25 * half
        moving_nf = np.ones(num_frames, dtype=np.int8)
        self.assertTrue(
            inference._needs_stabilized_tracking(
                moving_lm, inference.crop_box(moving_lm, H, W)
            )
        )

        # Fake sampler modifies every pixel inside the edit mask noticeably
        def mark_mask_sampler(pipe, whisper, adapter, zero_latents, clip, window_audio,
                              steps=2, audio_cfg=1.0, seed=0, fps=24):
            gen = clip["frames"].copy()
            m = clip["mask"]
            gen[:, m] = np.clip(gen[:, m].astype(np.int16) + 40, 0, 255).astype(np.uint8)
            return gen

        models = (None, None, None, None)
        logs_mem, logs_stream = [], []
        with mock.patch.object(
            inference, "detect_landmarks", return_value=(moving_lm.copy(), moving_nf.copy())
        ), mock.patch.object(
            inference, "detect_landmarks_iter", return_value=(moving_lm.copy(), moving_nf.copy())
        ):
            out_mem = inference.dub_video(
                models,
                frames,
                audio,
                steps=2,
                seed=1,
                overlap=17,
                harmonize=False,
                log=logs_mem.append,
                sample_fn=mark_mask_sampler,
            )
            writer = _MemoryWriter()
            written = inference.dub_video_stream(
                models,
                _DummyFrameSource(frames, chunk_size=31),
                audio,
                writer,
                steps=2,
                seed=1,
                overlap=17,
                harmonize=False,
                log=logs_stream.append,
                sample_fn=mark_mask_sampler,
            )

        self.assertEqual(written, num_frames)
        self.assertTrue(np.array_equal(out_mem, writer.result()))
        self.assertTrue(any("stabilized per-frame face tracking" in msg for msg in logs_mem))
        self.assertTrue(any("stabilized per-frame face tracking" in msg for msg in logs_stream))

        # Verify on frame 0 (top position) and frame -1 (bottom position) that ONLY
        # the real lower face was edited — never the chest far below frame 0's chin!
        for t in (0, num_frames // 2, num_frames - 1):
            diff = np.abs(out_mem[t].astype(np.float32) - frames[t].astype(np.float32)).mean(axis=2)
            ys, _ = np.nonzero(diff > 10.0)
            self.assertGreater(len(ys), 100)
            chin_y = float(moving_lm[t, 152, 1])
            nose_y = float(moving_lm[t, 2, 1])
            face_h = float(moving_lm[t, 152, 1] - moving_lm[t, 10, 1])
            # Must not paint onto chest below chin + generous feather margin
            self.assertLess(ys.max(), chin_y + 0.35 * face_h)
            # Must cover near nose_y (upper mouth not clipped off)
            self.assertLess(ys.min(), nose_y + 0.15 * face_h)

    def test_full_frame_mesh_hallucination_overridden_by_full_range_roi(self):
        import types
        from data import s01_analyze

        H, W = 1088, 1920
        frames = np.zeros((3, H, W, 3), dtype=np.uint8)

        def make_norm_face(cx_px, cy_px, size_px, span_w, span_h):
            half = size_px / 2.0
            pts = np.tile(np.array([cx_px, cy_px], dtype=np.float32), (468, 1))
            pts[234, 0] = cx_px - half * 0.8
            pts[454, 0] = cx_px + half * 0.8
            pts[10, 1] = cy_px - half
            pts[152, 1] = cy_px + half
            pts[2, 1] = cy_px - half * 0.05
            pts[13, 1] = cy_px + half * 0.15
            pts[14, 1] = cy_px + half * 0.30
            return types.SimpleNamespace(
                landmark=[
                    types.SimpleNamespace(x=float(p[0] / span_w), y=float(p[1] / span_h))
                    for p in pts
                ]
            )

        # Frame 0: FaceMesh misses -> ROI finds real 120px face at (960, 540).
        # Frame 1: FaceMesh hallucinates a giant 390px face at (1050, 740) on chest ->
        #          full-range ROI must override it back to the real face at (962, 544)!
        # Frame 2: FaceMesh misses -> ROI finds real 120px face at (964, 548).
        full_mesh_calls = [0]

        class _FakeFaceMesh:
            def __init__(self, static_image_mode=False, **kwargs):
                self.static = static_image_mode

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def close(self):
                pass

            def process(self, img):
                if not self.static:
                    idx = full_mesh_calls[0]
                    full_mesh_calls[0] += 1
                    if idx == 1:
                        return types.SimpleNamespace(
                            multi_face_landmarks=[make_norm_face(1050.0, 740.0, 390.0, W, H)]
                        )
                    return types.SimpleNamespace(multi_face_landmarks=None)
                ih, iw = img.shape[:2]
                return types.SimpleNamespace(
                    multi_face_landmarks=[make_norm_face(iw * 0.5, ih * 0.5, 120.0, iw, ih)]
                )

        roi_calls = [0]

        class _FakeFaceDetection:
            def __init__(self, **kwargs):
                pass

            def close(self):
                pass

            def process(self, img):
                idx = roi_calls[0]
                roi_calls[0] += 1
                cx = (960.0 + 2.0 * idx) / W
                cy = (540.0 + 4.0 * idx) / H
                bw, bh = 120.0 / W, 120.0 / H
                bb = types.SimpleNamespace(xmin=cx - bw / 2, ymin=cy - bh / 2, width=bw, height=bh)
                return types.SimpleNamespace(
                    detections=[
                        types.SimpleNamespace(
                            location_data=types.SimpleNamespace(relative_bounding_box=bb)
                        )
                    ]
                )

        with mock.patch.object(
            s01_analyze.mp.solutions.face_mesh, "FaceMesh", _FakeFaceMesh
        ), mock.patch.object(
            s01_analyze.mp.solutions.face_detection, "FaceDetection", _FakeFaceDetection
        ):
            lm, nfaces = s01_analyze.detect_landmarks_iter([frames], 3, H, W)

        self.assertTrue(np.isfinite(lm).all())
        self.assertTrue((nfaces == 1).all())
        # Frame 1 must match the real face (~544), NOT the 390px chest hallucination (~740)
        for t in range(3):
            cy = float((lm[t, :, 1].min() + lm[t, :, 1].max()) / 2.0)
            sz = float(max(np.ptp(lm[t, :, 0]), np.ptp(lm[t, :, 1])))
            self.assertAlmostEqual(cy, 540.0 + 4.0 * t, delta=4.0)
            self.assertAlmostEqual(sz, 120.0, delta=6.0)

    def test_stabilized_edge_pad_keeps_edge_framed_moving_speaker_centered(self):
        # Vertical 9:16 video where speaker starts very near the top edge (cy=90, face_h=160 ->
        # cy - side/2 = 90 - 144 = -54 < 0) and moves down 180 px.
        H, W, T = 1280, 720, 90
        lm = np.full((T, 468, 2), np.nan, dtype=np.float32)
        for t in range(T):
            cx = W * 0.5
            cy = 90.0 + 180.0 * (t / (T - 1))
            half = 80.0
            lm[t, :, 0] = cx
            lm[t, :, 1] = cy
            lm[t, 234, 0] = cx - half
            lm[t, 454, 0] = cx + half
            lm[t, 10, 1] = cy - half
            lm[t, 152, 1] = cy + half
            lm[t, 2, 1] = cy - 0.1 * half
            lm[t, 13, 1] = cy + 0.1 * half
            lm[t, 14, 1] = cy + 0.25 * half

        pad = inference._stabilized_edge_pad(lm, H, W)
        self.assertGreaterEqual(pad, 54)
        boxes = inference._stabilized_crop_boxes(lm + pad, H + 2 * pad, W + 2 * pad)
        mask, _ = inference._stabilized_lower_face_mask(lm + pad, boxes, feather=15)
        ys = np.nonzero(mask > 127)[0]
        # Mask height in 512x512 crop space must stay tight (~0.6 * 512/1.8 ≈ 170 px), never bloated
        self.assertLess(int(ys.max() - ys.min()), 210)


if __name__ == "__main__":
    unittest.main()
