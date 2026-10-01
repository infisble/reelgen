"""Frames from a local open model (SDXL family) on the user's GPU: no API keys, no per-image cost.

Default model: DreamShaper XL v2 Turbo (good stylized 3D look in 6-8 steps). Fits an 8 GB card with
model CPU offload. Character consistency comes from a fixed character description + a per-character seed;
reference images (`refs`) are not used by this backend yet.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

MODEL = os.environ.get("REELGEN_LOCAL_IMAGE_MODEL", "Lykon/dreamshaper-xl-v2-turbo")
# short on purpose: CLIP reads only 77 tokens, the scene description must fit after the style
STYLE = "3D Pixar style animated movie still, cute cartoon character, big glossy eyes"
UNHAPPY = ("shock", "surpris", "sad", "cry", "tear", "angry", "furious", "scared", "fear", "worr", "ashamed")
NEGATIVE = (
    "text, letters, watermark, logo, caption, signature, blurry, lowres, deformed, extra limbs, "
    "extra fingers, ugly face, photo of a real vegetable, flat 2D drawing"
)


class LocalSDXLImages:
    def __init__(self, counters: dict, steps: int = 8, size: tuple[int, int] = (832, 1216)):
        self.k, self.steps, self.size = counters, steps, size
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Local image generation needs a CUDA GPU (torch.cuda.is_available() is False)"
                )
            pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=torch.float16, variant="fp16")
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True
            )
            pipe.enable_model_cpu_offload()  # 8 GB VRAM: keep only the active sub-model on the GPU
            pipe.vae.enable_tiling()  # diffusers >= 0.40: tiling lives on the VAE
            self._pipe = pipe
        return self._pipe

    def generate(self, prompt: str, out_png: Path, attempt: int, refs: list[Path] | None = None) -> None:
        import torch

        pipe = self._load()
        seed = int(hashlib.sha256(f"{prompt}|{attempt}".encode()).hexdigest()[:8], 16)
        w, h = self.size
        negative = NEGATIVE
        if "mascot" in prompt.lower():  # object characters: stop SDXL from drawing a person or an alien
            negative += ", human, person, human body, alien"
        guidance = 2.0
        if any(w in prompt.lower()[:80] for w in UNHAPPY):  # cute mascots default to a grin
            negative += ", smile, smiling, grin, happy, laughing"
            guidance = 3.0  # turbo models barely read the negative prompt at 2.0
        image = pipe(
            prompt=f"{STYLE}, {prompt}",
            negative_prompt=negative,
            num_inference_steps=self.steps,
            guidance_scale=guidance,
            width=w,
            height=h,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
        image.save(out_png)
        self.k["image_calls"] = self.k.get("image_calls", 0) + 1
