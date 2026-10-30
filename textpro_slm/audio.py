"""Single-utterance and 30Plus speech encoding."""

import json
from pathlib import Path

import librosa
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import WhisperProcessor

from .whisperpro import WhisperPro


ENCODER_REVISIONS = {
    "qqjz/WhisperPro": "28c276609246ae3b5cd5b48a0fe239d88c479b0a",
    "qqjz/WhisperPro-30Plus": "899d2aa3dcf6070184bf78bbd1d5b3819bcec7b9",
}


def resolve_model(model, revision=None, cache_dir=None):
    path = Path(model).expanduser()
    if path.is_dir():
        return path
    return Path(snapshot_download(str(model), revision=revision, cache_dir=cache_dir))


def segment_audio(y, vad):
    """Preserve the 30Plus VAD, greedy merging, and long-span splitting policy."""
    if len(y) <= 480000:
        return [(0, len(y))]
    from silero_vad import get_speech_timestamps

    duration = len(y) / 16000
    timestamps = get_speech_timestamps(
        torch.from_numpy(y).float(), vad, sampling_rate=16000, return_seconds=True
    )
    adjusted = []
    for i, ts in enumerate(timestamps):
        start = 0 if i == 0 else max(0, ts["start"] - .15)
        end = duration if i == len(timestamps) - 1 else ts["end"]
        adjusted.append((start, end))
    if not adjusted:
        adjusted = [(0, duration)]
    merged = []
    start, end = adjusted[0]
    for next_start, next_end in adjusted[1:]:
        if next_end - start <= 30:
            end = next_end
        else:
            merged.append((start, end))
            start, end = next_start, next_end
    merged.append((start, end))
    spans = []
    for start, end in merged:
        a, b = round(start * 16000), min(len(y), round(end * 16000))
        while b - a > 480000:
            spans.append((a, a + 480000))
            a += 480000
        if b > a:
            spans.append((a, b))
    if not spans or any(not 0 < b - a <= 480000 for a, b in spans):
        raise ValueError("Invalid audio segmentation")
    return spans


class WhisperProEncoder:
    """Extract transcript-aligned states or global prosody from 16-kHz speech."""

    def __init__(self, model="qqjz/WhisperPro", device="cuda", revision=None,
                 long_audio=None, cache_dir=None):
        self.path = resolve_model(model, revision or ENCODER_REVISIONS.get(str(model)), cache_dir)
        self.device = torch.device(device)
        self.long_audio = "30Plus" in str(model) if long_audio is None else long_audio
        self.model = WhisperPro.from_pretrained(
            str(self.path), torch_dtype=torch.float32
        ).to(self.device).eval()
        codec = json.loads((self.path / "codec_config.json").read_text())
        self.model.prosody_layer_idx = codec["prosody_layer_idx"]
        self.processor = WhisperProcessor.from_pretrained(str(self.path))
        self.vad = None

    @torch.inference_mode()
    def extract_segment(self, audio, previous_text=None):
        inputs = self.processor([audio], sampling_rate=16000, return_tensors="pt")
        kwargs = {}
        if previous_text:
            ids = self.processor.tokenizer.encode(previous_text, add_special_tokens=False)[-224:]
            text = self.processor.tokenizer.decode(ids, skip_special_tokens=True)
            kwargs["prompt_ids"] = self.processor.get_prompt_ids(
                text, return_tensors="pt"
            ).to(self.device)
        codes = self.model.extract_speech_codes(
            inputs.input_features.to(self.device, dtype=torch.float32),
            max_length=448, **kwargs,
        )
        tokens, hidden = codes["text_tokens"][0], codes["prosody_features"][0]
        pad = self.processor.tokenizer.pad_token_id
        if pad is None:
            pad = self.processor.tokenizer.eos_token_id
        matches = (tokens == pad).nonzero(as_tuple=True)[0]
        length = int(matches[0]) if len(matches) else len(tokens)
        if length == 0:
            raise ValueError("WhisperPro produced no valid prosody positions")
        # Preserve the training code's interval, including its special-token policy.
        features = hidden[:length].float()
        if not torch.isfinite(features).all():
            raise ValueError("WhisperPro produced non-finite prosody")
        text = self.processor.tokenizer.decode(tokens[:length], skip_special_tokens=True)
        return features, text

    def encode_waveform(self, waveform, token_level=False):
        """Encode a finite mono float waveform sampled at 16 kHz."""
        y = np.ascontiguousarray(waveform, dtype=np.float32)
        if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
            raise ValueError("Expected a nonempty, finite mono waveform at 16 kHz")
        if len(y) > 480000 and not self.long_audio:
            raise ValueError("Original WhisperPro supports at most 30 seconds; use the paired 30Plus models")
        if self.long_audio and token_level:
            raise ValueError("30Plus uses global prosody rather than interleaving")
        if len(y) > 480000:
            if self.vad is None:
                from silero_vad import load_silero_vad
                self.vad = load_silero_vad()
            spans = segment_audio(y, self.vad)
        else:
            spans = [(0, len(y))]
        vectors, texts = [], []
        for a, b in spans:
            previous = " ".join(texts) if self.long_audio else None
            features, text = self.extract_segment(y[a:b], previous)
            vectors.append(features.cpu().numpy() if token_level else features.mean(0).cpu().numpy())
            texts.append(text)
        prosody = vectors[0] if token_level else np.stack(vectors).mean(0)
        return {
            "transcript": " ".join(t for t in texts if t),
            "prosody": prosody,
            "duration": len(y) / 16000,
            "segment_spans": spans,
            "segment_transcripts": texts,
        }

    def encode(self, audio, token_level=False):
        waveform, _ = librosa.load(str(audio), sr=16000, mono=True)
        return self.encode_waveform(waveform, token_level=token_level)
