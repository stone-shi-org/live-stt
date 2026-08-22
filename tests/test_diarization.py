"""live_stt/diarization.py: the pure mapping (pyannote Annotation ->
house JSON) and text-merge logic are unit-tested directly, offline, against
a fake ``itertracks``-shaped stand-in -- no pyannote.audio install, network,
or real model needed (see tests/conftest.py's offline-safety philosophy).
``load_pipeline``/``diarize_file``'s actual model-loading path is only
covered for its error handling (missing dependency, missing token): running
the real pyannote pipeline is out of scope for a unit test.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from live_stt.config import Settings
from live_stt.diarization import (
    DiarizationError,
    annotation_to_house_json,
    assign_text,
    load_pipeline,
)
from live_stt.pb.livestt.v1 import asr_pb2


@dataclass
class FakeSegment:
    start: float
    end: float


class FakeAnnotation:
    """Stands in for pyannote.core.Annotation: only needs itertracks(yield_label=True)."""

    def __init__(self, tracks: list[tuple[FakeSegment, str]]) -> None:
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False):
        assert yield_label
        for i, (segment, label) in enumerate(self._tracks):
            yield segment, f"track_{i}", label


class FakePipelineOutput:
    """Stands in for pyannote 4.x's top-level pipeline output wrapper,
    which exposes the classic Annotation via .speaker_diarization."""

    def __init__(self, annotation: FakeAnnotation) -> None:
        self.speaker_diarization = annotation


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


class TestAnnotationToHouseJson:
    def test_maps_segments_and_speaker_stats(self):
        annotation = FakeAnnotation(
            [
                (FakeSegment(0.0, 1.2), "SPEAKER_00"),
                (FakeSegment(1.2, 2.0), "SPEAKER_01"),
                (FakeSegment(2.0, 3.5), "SPEAKER_00"),
            ]
        )
        result = annotation_to_house_json(annotation, model="pyannote/speaker-diarization-community-1")

        assert result["task"] == "diarize"
        assert result["num_speakers"] == 2
        assert [s["speaker"] for s in result["segments"]] == [
            "SPEAKER_00",
            "SPEAKER_01",
            "SPEAKER_00",
        ]
        assert result["segments"][0]["start"] == 0.0
        assert result["segments"][0]["end"] == 1.2
        assert result["segments"][0]["text"] == ""

        speaker_00 = next(s for s in result["speakers"] if s["id"] == "SPEAKER_00")
        assert speaker_00["segment_count"] == 2
        assert speaker_00["total_speech_duration"] == pytest.approx(1.2 + 1.5)

    def test_accepts_the_4x_pipeline_output_wrapper(self):
        annotation = FakeAnnotation([(FakeSegment(0.0, 1.0), "SPEAKER_00")])
        wrapped = FakePipelineOutput(annotation)
        result = annotation_to_house_json(wrapped, model="m")
        assert result["num_speakers"] == 1

    def test_bare_label_gets_speaker_prefix(self):
        annotation = FakeAnnotation([(FakeSegment(0.0, 1.0), "0")])
        result = annotation_to_house_json(annotation, model="m")
        assert result["segments"][0]["speaker"] == "SPEAKER_0"
        assert result["segments"][0]["label"] == "0"

    def test_already_prefixed_label_is_not_double_prefixed(self):
        annotation = FakeAnnotation([(FakeSegment(0.0, 1.0), "SPEAKER_02")])
        result = annotation_to_house_json(annotation, model="m")
        assert result["segments"][0]["speaker"] == "SPEAKER_02"

    def test_empty_annotation_raises(self):
        with pytest.raises(DiarizationError, match="no segments"):
            annotation_to_house_json(FakeAnnotation([]), model="m")

    def test_speaker_order_is_deterministic_by_first_appearance(self):
        annotation = FakeAnnotation(
            [
                (FakeSegment(0.0, 1.0), "SPEAKER_01"),
                (FakeSegment(1.0, 2.0), "SPEAKER_00"),
            ]
        )
        result = annotation_to_house_json(annotation, model="m")
        assert [s["id"] for s in result["speakers"]] == ["SPEAKER_01", "SPEAKER_00"]


class TestAssignText:
    def _house_json(self):
        annotation = FakeAnnotation(
            [
                (FakeSegment(0.0, 1.0), "SPEAKER_00"),
                (FakeSegment(1.0, 2.0), "SPEAKER_01"),
            ]
        )
        return annotation_to_house_json(annotation, model="m")

    def test_words_are_assigned_by_midpoint(self):
        words = [
            asr_pb2.Word(text="hello", start_sec=0.1, end_sec=0.3),
            asr_pb2.Word(text="there", start_sec=0.4, end_sec=0.6),
            asr_pb2.Word(text="world", start_sec=1.2, end_sec=1.4),
        ]
        result = assign_text(self._house_json(), words)
        assert result["segments"][0]["text"] == "hello there"
        assert result["segments"][1]["text"] == "world"

    def test_a_word_in_a_gap_between_segments_is_dropped_not_guessed(self):
        annotation = FakeAnnotation(
            [
                (FakeSegment(0.0, 1.0), "SPEAKER_00"),
                (FakeSegment(2.0, 3.0), "SPEAKER_01"),
            ]
        )
        house_json = annotation_to_house_json(annotation, model="m")
        words = [asr_pb2.Word(text="orphan", start_sec=1.4, end_sec=1.6)]
        result = assign_text(house_json, words)
        assert result["segments"][0]["text"] == ""
        assert result["segments"][1]["text"] == ""

    def test_no_words_leaves_empty_text_and_is_not_an_error(self):
        result = assign_text(self._house_json(), [])
        assert all(s["text"] == "" for s in result["segments"])


class TestLoadPipelineErrors:
    def test_missing_dependency_raises_diarization_error(self, monkeypatch: pytest.MonkeyPatch):
        # pyannote.audio is deliberately NOT in requirements.txt (see
        # requirements-diarization.txt) -- but it IS installed in this dev
        # venv now, for the real end-to-end run against NOTSOFAR-1 audio (see
        # CLAUDE.md). This test asserts the CODE PATH, not this venv's
        # current state, which is exactly why it broke the moment the real
        # dependency got installed for that run: `sys.modules[name] = None`
        # is the standard way to force `import pyannote.audio` to raise
        # ImportError regardless of whether the package is actually present.
        import sys

        monkeypatch.setitem(sys.modules, "pyannote.audio", None)
        with pytest.raises(DiarizationError, match="pyannote.audio is not installed"):
            load_pipeline(_settings(diarization_hf_token="fake-token"))

    def test_missing_token_is_checked_before_importing_torch(self, monkeypatch: pytest.MonkeyPatch):
        # Even if pyannote.audio somehow were importable, a missing token
        # should fail fast with a clear message rather than a deep pyannote
        # traceback -- verified by making the import succeed with a stub.
        import sys
        import types

        fake_pyannote_audio = types.ModuleType("pyannote.audio")
        fake_pyannote_audio.Pipeline = object()
        fake_pyannote = types.ModuleType("pyannote")
        fake_pyannote.audio = fake_pyannote_audio
        monkeypatch.setitem(sys.modules, "pyannote", fake_pyannote)
        monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote_audio)

        with pytest.raises(DiarizationError, match="LSTT_DIARIZATION_HF_TOKEN"):
            load_pipeline(_settings(diarization_hf_token=None))


class TestLoadPipelineDevice:
    """load_pipeline's device handling, stubbed against fake pyannote.audio
    and fake torch modules -- so this stays hermetic regardless of whether
    torch/pyannote.audio are actually installed (they are, in this dev venv,
    for the real end-to-end run recorded in CLAUDE.md, but this test must not
    depend on that, and must not require an actual CUDA GPU to run at all).
    """

    def _stub_pyannote(self, monkeypatch: pytest.MonkeyPatch, fake_pipeline: object) -> None:
        import sys
        import types

        fake_pyannote_audio = types.ModuleType("pyannote.audio")

        class FakePipelineClass:
            @staticmethod
            def from_pretrained(model, token):
                return fake_pipeline

        fake_pyannote_audio.Pipeline = FakePipelineClass
        fake_pyannote = types.ModuleType("pyannote")
        fake_pyannote.audio = fake_pyannote_audio
        monkeypatch.setitem(sys.modules, "pyannote", fake_pyannote)
        monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote_audio)

    def _stub_torch_cuda(self, monkeypatch: pytest.MonkeyPatch, *, available: bool) -> None:
        import sys
        import types

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: available)
        fake_torch.device = lambda name: f"device:{name}"
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def test_unknown_device_is_rejected(self, monkeypatch: pytest.MonkeyPatch):
        self._stub_pyannote(monkeypatch, object())
        with pytest.raises(DiarizationError, match="unknown LSTT_DIARIZATION_DEVICE"):
            load_pipeline(_settings(diarization_hf_token="tok", diarization_device="tpu"))

    def test_cpu_default_never_touches_torch_cuda(self, monkeypatch: pytest.MonkeyPatch):
        # cpu is the default -- must not call .to() or even import torch's
        # cuda check, so a plain CPU deployment never needs a CUDA-capable
        # torch build at all.
        class FakePipeline:
            def to(self, device):
                raise AssertionError("must not call .to() on the cpu default")

        self._stub_pyannote(monkeypatch, FakePipeline())
        pipeline = load_pipeline(_settings(diarization_hf_token="tok"))
        assert pipeline is not None

    def test_cuda_requested_but_unavailable_fails_loudly(self, monkeypatch: pytest.MonkeyPatch):
        class FakePipeline:
            def to(self, device):
                pass

        self._stub_pyannote(monkeypatch, FakePipeline())
        self._stub_torch_cuda(monkeypatch, available=False)
        with pytest.raises(DiarizationError, match="torch.cuda.is_available"):
            load_pipeline(_settings(diarization_hf_token="tok", diarization_device="cuda"))

    def test_cuda_available_moves_the_pipeline_onto_it(self, monkeypatch: pytest.MonkeyPatch):
        moved = {}

        class FakePipeline:
            def to(self, device):
                moved["device"] = device

        self._stub_pyannote(monkeypatch, FakePipeline())
        self._stub_torch_cuda(monkeypatch, available=True)
        load_pipeline(_settings(diarization_hf_token="tok", diarization_device="cuda"))
        assert moved["device"] == "device:cuda"


class TestSettingsDefaults:
    def test_diarization_defaults(self):
        settings = _settings()
        assert settings.diarization_model == "pyannote/speaker-diarization-community-1"
        assert settings.diarization_hf_token is None
        assert settings.diarization_num_speakers == 2
        assert settings.diarization_device == "cpu"

    def test_token_overridable_via_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LSTT_DIARIZATION_HF_TOKEN", "hf_abc123")
        settings = Settings(_env_file=None)
        assert settings.diarization_hf_token == "hf_abc123"
