"""Inference / text-to-speech generation for KoeTTS.

Given text (and optionally a voice prompt), autoregressively generates
audio tokens, then decodes them to a waveform.

Usage as library:
    from koe.generate import KoeTTSGenerator

    gen = KoeTTSGenerator.from_checkpoints(
        model_path="checkpoints/ar_model/final.pt",
        codec_path="checkpoints/tokenizer/tokenizer_final.pt",
        device="cuda",
    )
    wav = gen.generate("Hello world", prompt_path="prompt.wav")
    gen.save_wav(wav, "output.wav")
"""

from typing import Optional, Tuple

import torch

from koe.config import ModelConfig, TokenConfig, CodecConfig
from koe.model import KoeTTS
from koe.codec import KoeCodec
from koe.text import CharTokenizer


class KoeTTSGenerator:
    """End-to-end TTS generator: text → audio tokens → waveform."""

    def __init__(
        self,
        model: KoeTTS,
        codec: KoeCodec,
        tokenizer: CharTokenizer,
        token_config: TokenConfig,
        device: torch.device,
    ):
        self.model = model.eval()
        self.codec = codec
        self.tokenizer = tokenizer
        self.token_config = token_config
        self.device = device
        self.groups_per_frame = model.config.groups_per_frame

    @classmethod
    def from_checkpoints(
        cls,
        model_path: str,
        codec_path: str,
        device: str = "cuda",
    ) -> "KoeTTSGenerator":
        """Load generator from saved checkpoints."""
        dev = torch.device(device)

        # Load AR model
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        model_cfg = ModelConfig(**ckpt["config"]["model"]) if "config" in ckpt else ModelConfig()
        token_cfg = TokenConfig(**ckpt["config"]["token"]) if "config" in ckpt else TokenConfig()
        model = KoeTTS(model_cfg, token_cfg)
        model.load_state_dict(ckpt["model"])
        model = model.to(dev)

        # Load codec
        codec = KoeCodec.from_pretrained(codec_path, device=device)

        # Text tokenizer
        text_tokenizer = CharTokenizer(token_cfg)

        return cls(model, codec, text_tokenizer, token_cfg, dev)

    def _build_prefix(
        self,
        text: str,
        prompt_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the prefix sequence before autoregressive generation.

        Returns:
            token_ids:    [1, T_prefix]
            segment_ids:  [1, T_prefix]
            group_pos_ids:[1, T_prefix]
        """
        tc = self.token_config
        gpf = self.groups_per_frame

        # Encode text
        text_ids = self.tokenizer.encode(text)

        # Build sequence: [bos, text, text_sep, prompt?, audio_sep]
        tokens = [tc.bos_id]
        segments = [0]
        group_pos = [0]

        for tid in text_ids:
            tokens.append(tid)
            segments.append(0)
            group_pos.append(0)

        tokens.append(tc.text_sep_id)
        segments.append(0)
        group_pos.append(0)

        # Prompt audio tokens (if provided)
        if prompt_tokens is not None:
            # prompt_tokens: [1, T_p, 19] packed → flatten
            flat = prompt_tokens[0].reshape(-1)  # [T_p * 19]
            for i, tok in enumerate(flat.tolist()):
                tokens.append(int(tok) + tc.audio_offset)
                segments.append(1)
                group_pos.append(i % gpf)

        tokens.append(tc.audio_sep_id)
        segments.append(1 if prompt_tokens is not None else 0)
        group_pos.append(0)

        return (
            torch.tensor([tokens], dtype=torch.long, device=self.device),
            torch.tensor([segments], dtype=torch.long, device=self.device),
            torch.tensor([group_pos], dtype=torch.long, device=self.device),
        )

    @torch.no_grad()
    def generate(
        self,
        text: str,
        prompt_path: Optional[str] = None,
        max_audio_seconds: float = 30.0,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate speech from text.

        Args:
            text: input text to synthesize
            prompt_path: optional path to voice prompt audio
            max_audio_seconds: maximum output duration
            temperature: sampling temperature (lower = more deterministic)
            top_k: top-k filtering
            top_p: nucleus sampling threshold

        Returns:
            waveform tensor [1, 1, T_wav] at codec sample rate
        """
        tc = self.token_config
        gpf = self.groups_per_frame

        # Encode prompt if provided
        prompt_tokens = None
        if prompt_path:
            prompt_tokens = self.codec.encode_file(prompt_path, device=self.device)

        # Build prefix
        token_ids, segment_ids, group_pos_ids = self._build_prefix(text, prompt_tokens)

        # Prefill: run full prefix through model to populate KV-cache
        result = self.model(token_ids, segment_ids, group_pos_ids)
        kv_cache = result["kv_cache"]

        # Max tokens to generate
        max_audio_tokens = int(max_audio_seconds * self.codec.config.tokens_per_second)
        # Ensure we generate complete frames (multiples of 19)
        max_audio_tokens = (max_audio_tokens // gpf) * gpf

        # Autoregressive generation
        generated_audio_tokens = []
        group_idx = 0  # cycles 0-18 within each frame

        # Start with the last token of the prefix
        last_logits = result["logits"][:, -1:, :]  # [1, 1, vocab]
        # Sample first audio token from the last prefix position
        probs = torch.softmax(last_logits[:, 0, :] / temperature, dim=-1)
        if top_k:
            topk_vals, _ = torch.topk(probs, top_k)
            probs[probs < topk_vals[:, -1:]] = 0
            probs = probs / probs.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(probs, 1)  # [1, 1]

        for i in range(max_audio_tokens):
            tok_val = next_token.item()

            # Check for EOS — only valid at frame boundaries
            if tok_val == tc.eos_id and group_idx == 0 and i > 0:
                break

            generated_audio_tokens.append(tok_val)
            group_idx = (group_idx + 1) % gpf

            # Prepare next input
            cur_segment = torch.tensor([[2]], dtype=torch.long, device=self.device)
            cur_group = torch.tensor([[group_idx]], dtype=torch.long, device=self.device)

            # Generate next token
            next_token, kv_cache = self.model.generate_next_token(
                next_token, cur_segment, cur_group,
                kv_cache=kv_cache,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        if not generated_audio_tokens:
            raise RuntimeError("Model generated no audio tokens")

        # Trim to complete frames
        n_complete = (len(generated_audio_tokens) // gpf) * gpf
        generated_audio_tokens = generated_audio_tokens[:n_complete]

        # Convert to packed token tensor [1, T_z, 19]
        audio_ids = torch.tensor(generated_audio_tokens, dtype=torch.long, device=self.device)
        audio_ids = audio_ids - tc.audio_offset  # back to 0-16383 range
        audio_ids = audio_ids.clamp(0, tc.num_audio_tokens - 1)
        audio_ids = audio_ids.view(1, -1, gpf)  # [1, T_z, 19]

        # Decode to waveform
        wav = self.codec.decode(audio_ids)
        return wav

    def save_wav(self, wav: torch.Tensor, path: str, sample_rate: Optional[int] = None):
        """Save waveform tensor to file."""
        sr = sample_rate or self.codec.config.sample_rate
        if wav.dim() == 3:
            wav = wav.squeeze(0)  # [1, T]
        import torchaudio
        torchaudio.save(path, wav.cpu(), sr)

    def generate_to_file(
        self,
        text: str,
        output_path: str,
        prompt_path: Optional[str] = None,
        **kwargs,
    ):
        """Generate speech and save directly to file."""
        wav = self.generate(text, prompt_path=prompt_path, **kwargs)
        self.save_wav(wav, output_path)
        return output_path
