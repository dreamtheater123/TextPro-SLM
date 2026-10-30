"""TextPro-SLM inference: speech as transcript tokens plus compact prosody."""

from .audio import WhisperProEncoder
from .model import TextProSLM

__all__ = ["TextProSLM", "WhisperProEncoder"]
