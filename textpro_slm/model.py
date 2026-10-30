"""The released TextPro-SLM fusion policies and single-audio inference."""

import json

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from .audio import ENCODER_REVISIONS, WhisperProEncoder, resolve_model


MODEL_REVISIONS = {
    "qqjz/TextPro-SLM-7B-Global": "736729f730d9a9124f200a3718b1e4e094bbbb50",
    "qqjz/TextPro-SLM-7B-Interleaving": "19ea9981c07bbc6554668d071b9a8e01b96eb63f",
    "qqjz/TextPro-SLM-3B-Global": "ffe30a88b174f6f7f5005359335a60ca71a2eb05",
    "qqjz/TextPro-SLM-7B-30Plus": "3cb941f89a7c8cefd55b7dba106423659e61269e",
}
SYSTEM_PROMPTS = {
    "semantic": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant to answer the user questions.",
    "audio": "You are a helpful assistant that performs audio-based QAs. You will be provided the prosody embedding and transcript of the audio.",
    "emotion": "Act as an expert Acoustic Linguist to classify speech emotion by analyzing prosodic metadata and transcripts.",
}


class ProsodyProjector(torch.nn.Module):
    """Map WhisperPro's 1280-dimensional prosody into the LLM embedding space."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim), torch.nn.GELU(),
            torch.nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class TextProSLM:
    """Speech QA, with either a spoken instruction or an added text instruction."""

    def __init__(self, model="qqjz/TextPro-SLM-7B-Global", encoder=None,
                 device="cuda", revision=None, cache_dir=None):
        path = resolve_model(model, revision or MODEL_REVISIONS.get(str(model)), cache_dir)
        self.config = json.loads((path / "textpro_config.json").read_text())
        self.long_audio = "segment_max_seconds" in self.config
        self.device = torch.device(device)
        self.dtype = torch.float32 if self.device.type == "cpu" else torch.bfloat16
        # Pair each speech LLM with the encoder used for its training.
        self.encoder = WhisperProEncoder(
            encoder or self.config["speech_encoder"], device,
            revision=None if encoder else ENCODER_REVISIONS.get(
                self.config["speech_encoder"], self.config.get("speech_encoder_revision")
            ),
            long_audio=self.long_audio, cache_dir=cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(path / "llm"))
        self.llm = AutoModelForCausalLM.from_pretrained(
            str(path / "llm"), torch_dtype=self.dtype, attn_implementation="sdpa"
        ).to(self.device).eval()
        if self.long_audio:
            # The original 30Plus inference loader explicitly enables KV caching.
            self.llm.config.use_cache = True
            self.llm.generation_config.use_cache = True
        self.projector = ProsodyProjector(
            self.config["prosody_dim"], self.llm.config.hidden_size
        ).to(device=self.device, dtype=self.dtype)
        self.projector.load_state_dict(load_file(str(path / "projector.safetensors")))
        self.projector.eval()
        self.marker = self.config["prosody_token"]
        self.marker_id = self.tokenizer.convert_tokens_to_ids(self.marker)
        if self.marker_id != self.config["prosody_token_id"]:
            raise ValueError("Tokenizer does not match the released prosody marker")
        if self.config["fusion"] not in {"global_prepending", "interleaving"}:
            raise ValueError("Unsupported fusion policy")

    @torch.inference_mode()
    def prepare_inputs(self, encoded, instruction="", task="semantic"):
        """Combine ASR text, optional text instruction, and the trained prosody input."""
        interleave = self.config["fusion"] == "interleaving"
        transcript = encoded["transcript"]
        if self.marker in transcript or self.marker in instruction:
            raise ValueError("Transcript/instruction must not contain the reserved prosody marker")
        content = transcript
        if interleave:
            # Interleave prosody with transcript tokens only, in groups of five.
            ids = self.tokenizer(transcript, add_special_tokens=False)["input_ids"]
            group = self.config["transcript_tokens_per_prosody"]
            content = "".join(
                self.marker + self.tokenizer.decode(ids[i:i+group], clean_up_tokenization_spaces=False)
                for i in range(0, len(ids), group)
            )
        # AIR-Bench adds a text question/choices here. Spoken QA leaves this empty.
        content += "\n\n" + instruction if instruction else ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[task]},
            {"role": "user", "content": content},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if not interleave:
            prompt = prompt.replace("<|im_start|>user\n", "<|im_start|>user\n" + self.marker, 1)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        positions = (inputs.input_ids == self.marker_id).nonzero(as_tuple=True)[1]
        if not len(positions) or (not interleave and len(positions) != 1):
            raise ValueError("No usable prosody marker; the transcript may be empty")
        if inputs.input_ids[0, -3:].tolist() != [151644, 77091, 198]:
            raise ValueError("Malformed assistant generation prefix")
        vector = torch.as_tensor(encoded["prosody"], device=self.device, dtype=self.dtype)
        if interleave:
            if len(positions) > len(vector):
                raise ValueError("Too few WhisperPro states for the interleaving groups")
            # Preserve the original BF16 pooling order on CUDA.
            vector = torch.stack([chunk.mean(0) for chunk in torch.tensor_split(vector, len(positions), dim=0)])
        else:
            vector = vector.unsqueeze(0)
        if not torch.isfinite(vector).all():
            raise ValueError("Non-finite prosody")
        embeddings = self.llm.get_input_embeddings()(inputs.input_ids)
        projected = self.projector(vector)
        # Replace marker embeddings with acoustic features; keep text embeddings intact.
        for i, position in enumerate(positions):
            embeddings[0, position, :] = projected[i, :]
        return dict(inputs_embeds=embeddings, attention_mask=inputs.attention_mask)

    @torch.inference_mode()
    def respond(self, audio, instruction="", task="semantic", max_new_tokens=512,
                do_sample=True, temperature=.7, top_p=.9):
        """Answer spoken QA (empty instruction) or audio + text QA.

        Defaults match the paper's sampling policy. max_new_tokens caps the
        response only; an end-of-sequence token can stop generation earlier.
        """
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        encoded = self.encoder.encode(audio, token_level=self.config["fusion"] == "interleaving")
        inputs = self.prepare_inputs(encoded, instruction, task)
        if inputs["inputs_embeds"].shape[1] + max_new_tokens > self.llm.config.max_position_embeddings:
            raise ValueError("Transcript and requested response exceed the LLM context window")
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                      pad_token_id=self.tokenizer.pad_token_id)
        # Sampling parameters do not apply when greedy decoding is requested.
        if do_sample:
            kwargs.update(temperature=temperature, top_p=top_p)
        ids = self.llm.generate(**inputs, **kwargs)
        return {
            "transcript": encoded["transcript"],
            "response": self.tokenizer.decode(ids[0], skip_special_tokens=True),
            "duration": encoded["duration"],
            "num_segments": len(encoded["segment_spans"]),
        }
