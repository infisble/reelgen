"""Generate an original background track with MusicGen on the local GPU (no keys, no licensing questions).

    python scripts/make_music.py "playful cartoon comedy, pizzicato strings, light percussion, upbeat" \
        --seconds 28 --out assets/music/zucchini_theme.wav

One track per series = a recognizable theme; the pipeline loops and ducks it under the voices (--music).
"""

import argparse
import os
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL = os.environ.get(
    "REELGEN_MUSIC_MODEL", "models/musicgen-stereo-small"
)  # or the HF id facebook/musicgen-stereo-small


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--seconds", type=float, default=28)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL)
    model = MusicgenForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    inputs = processor(text=[a.prompt], padding=True, return_tensors="pt").to(device)
    tokens = int(a.seconds * model.config.audio_encoder.frame_rate)  # ~50 tokens per second
    audio = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=tokens)
    sr = model.config.audio_encoder.sampling_rate
    data = audio[0].float().cpu().numpy()  # (channels, samples)
    data = data / max(1e-6, float(np.abs(data).max())) * 0.9
    pcm = (data.T * 32767).astype("<i2")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(a.out), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    print(f"saved {a.out} ({pcm.shape[0] / sr:.1f}s, {pcm.shape[1]}ch, {sr} Hz)")


if __name__ == "__main__":
    main()
