"""Inference-only WhisperPro: transcript generation and decoder prosody states."""

import torch
from transformers import WhisperForConditionalGeneration


class WhisperPro(WhisperForConditionalGeneration):
    """Use the same generation and decoder pass as the original implementation."""

    def __init__(self, config):
        super().__init__(config)
        self.prosody_layer_idx = config.decoder_layers // 2

    @torch.inference_mode()
    def extract_speech_codes(self, input_features, max_length=448, **kwargs):
        tokens = self.generate(input_features, max_length=max_length, **kwargs)
        encoded = self.model.encoder(input_features, return_dict=True)
        decoded = self.model.decoder(
            input_ids=tokens,
            encoder_hidden_states=encoded.last_hidden_state,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
        return {
            "text_tokens": tokens,
            "prosody_features": decoded.hidden_states[self.prosody_layer_idx],
        }
