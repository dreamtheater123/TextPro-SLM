"""Boundary tests for duration handling and the released segmentation policy."""

from unittest.mock import patch
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from textpro_slm.audio import WhisperProEncoder, segment_audio


@pytest.mark.parametrize("samples", [1, 16000, 480000])
def test_short_audio_keeps_the_complete_waveform(samples):
    # Short clips must not require the optional VAD dependency or discard silence.
    assert segment_audio(np.zeros(samples, dtype=np.float32), None) == [(0, samples)]


def test_no_detected_speech_still_splits_the_complete_long_clip():
    y = np.zeros(71 * 16000, dtype=np.float32)
    fake = SimpleNamespace(get_speech_timestamps=lambda *a, **kw: [])
    with patch.dict(sys.modules, {"silero_vad": fake}):
        assert segment_audio(y, None) == [(0, 480000), (480000, 960000), (960000, len(y))]


def test_vad_keeps_the_start_margin_and_caps_an_uninterrupted_span():
    y = np.zeros(65 * 16000, dtype=np.float32)
    timestamps = [{"start": 1., "end": 8.}, {"start": 29., "end": 64.}]
    fake = SimpleNamespace(get_speech_timestamps=lambda *a, **kw: timestamps)
    with patch.dict(sys.modules, {"silero_vad": fake}):
        spans = segment_audio(y, None)
    assert spans == [(0, 128000), (461600, 941600), (941600, len(y))]
    assert all(0 < b-a <= 480000 for a,b in spans)


@pytest.mark.parametrize("wave", [np.array([]), np.array([np.nan]),
                                  np.array([np.inf]), np.zeros((10, 2))])
def test_invalid_waveform_rejected_before_model_execution(wave):
    encoder = WhisperProEncoder.__new__(WhisperProEncoder)
    encoder.long_audio = False
    with pytest.raises(ValueError, match="finite mono waveform"):
        encoder.encode_waveform(wave)


def test_original_model_rejects_long_audio_without_truncation():
    encoder = WhisperProEncoder.__new__(WhisperProEncoder)
    encoder.long_audio = False
    with pytest.raises(ValueError, match="at most 30 seconds"):
        encoder.encode_waveform(np.zeros(480001, dtype=np.float32))


def test_30plus_rejects_token_level_interleaving():
    encoder = WhisperProEncoder.__new__(WhisperProEncoder)
    encoder.long_audio = True
    with pytest.raises(ValueError, match="global prosody"):
        encoder.encode_waveform(np.zeros(16000, dtype=np.float32), token_level=True)
