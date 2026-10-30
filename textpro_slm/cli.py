"""Command-line inference for WhisperPro and TextPro-SLM."""

import argparse
import json

import numpy as np
import torch

from .audio import WhisperProEncoder
from .model import SYSTEM_PROMPTS, TextProSLM


def main():
    # No --instruction: the question/instruction is spoken in the audio.
    # With --instruction: append the supplied text after the ASR transcript.
    parser = argparse.ArgumentParser(description="Speech understanding with TextPro-SLM")
    parser.add_argument("--model", default="qqjz/TextPro-SLM-7B-Global")
    parser.add_argument("--encoder", help="Optional paired local WhisperPro directory")
    parser.add_argument("--revision", help="Optional model Hub revision; defaults to the verified release")
    parser.add_argument("--cache-dir", help="Hugging Face model cache directory")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--instruction", default="", help="Optional text instruction; omit for audio-only QA")
    parser.add_argument("--task", choices=SYSTEM_PROMPTS, default="semantic",
                        help="System prompt: semantic for spoken QA, emotion for AIR-Bench emotion, audio for other audio QA")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum reply tokens (paper: 512)")
    parser.add_argument("--temperature", type=float, default=.7, help="Sampling temperature (paper: 0.7)")
    parser.add_argument("--top-p", type=float, default=.9, help="Nucleus sampling threshold (paper: 0.9)")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of the paper's sampling policy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = TextProSLM(model=args.model, encoder=args.encoder, device=args.device,
                       revision=args.revision, cache_dir=args.cache_dir)
    # The paper uses sampling. Greedy is an optional user-selected alternative.
    result = model.respond(
        audio=args.audio, instruction=args.instruction, task=args.task,
        max_new_tokens=args.max_new_tokens, do_sample=not args.greedy,
        temperature=args.temperature, top_p=args.top_p,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def encode_main():
    parser = argparse.ArgumentParser(description="Extract a transcript and prosody with WhisperPro")
    parser.add_argument("--model", default="qqjz/WhisperPro")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir")
    parser.add_argument("--token-level", action="store_true", help="Return token-aligned states for original WhisperPro")
    parser.add_argument("--prosody-output", help="Optional output .npy file")
    args = parser.parse_args()
    torch.set_num_threads(1)
    encoder = WhisperProEncoder(args.model, args.device, args.revision, cache_dir=args.cache_dir)
    result = encoder.encode(args.audio, args.token_level)
    prosody = result.pop("prosody")
    if args.prosody_output:
        np.save(args.prosody_output, prosody)
    result["prosody_shape"] = list(prosody.shape)
    print(json.dumps(result, ensure_ascii=False, indent=2))
