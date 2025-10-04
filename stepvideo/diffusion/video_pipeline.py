# stepvideo/diffusion/video_pipeline.py
"""
StepVideoPipeline — lightweight wrapper that loads the full Step-Video pipeline
from Hugging Face and exposes a small, stable interface for local inference.

This wrapper delegates to a HF DiffusionPipeline (downloaded via `from_pretrained`),
so you don't need separate VAE/caption microservices.
Environment variables:
 - HF_TOKEN      : (optional) Hugging Face token used for private repos
 - MODEL_REPO    : (optional) Hugging Face repo id (e.g. "stepfun-ai/Step-Video-TI2V")
"""

from typing import Any, List, Optional, Union
from dataclasses import dataclass
import os
import torch
from diffusers.utils import BaseOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import numpy as np

@dataclass
class StepVideoPipelineOutput(BaseOutput):
    video: Union[torch.Tensor, np.ndarray]

class StepVideoPipeline:
    """
    Wrapper around a Hugging Face DiffusionPipeline for Step-Video style generation.

    Usage:
        pipe = StepVideoPipeline(model_id="Wan-AI/Wan2.2-I2V-A8B")
        pipe.setup_device("cuda")           # optional, defaults to cuda if available
        out = pipe(prompt="...", first_image="img.png", num_frames=80, ...)
    """

    def __init__(self, model_id: Optional[str] = None, dtype: Optional[torch.dtype] = None):
        model_id = model_id or os.environ.get("MODEL_REPO", "stepfun-ai/Step-Video-TI2V")
        hf_token = os.environ.get("HF_TOKEN", None)

        # Choose dtype (bfloat16 is recommended when supported)
        self.dtype = dtype or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32)

        print(f"[StepVideoPipeline] Loading model from: {model_id} (dtype={self.dtype})")
        # Use `from_pretrained` to download/load the whole pipeline (transformer, vae, tokenizer, etc.)
        # `use_auth_token` or `token` arg usage differs between versions; HF accepts token= as well.
        kwargs = {}
        if hf_token:
            kwargs["token"] = hf_token  # huggingface_hub newer versions
        # Some diffusers versions accept use_auth_token; provide it as fallback
        try:
            self.pipe: DiffusionPipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                **kwargs
            )
        except TypeError:
            # fallback for older diffusers versions
            self.pipe: DiffusionPipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                use_auth_token=hf_token if hf_token else None
            )

        # Default device assignment (pipe.to(...) below or call setup_device)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.pipe.to(self.device)
        except Exception as e:
            print(f"[StepVideoPipeline] Warning moving pipeline to device failed: {e}")

        # Expose commonly-used components if present
        self.transformer = getattr(self.pipe, "transformer", None)
        self.vae = getattr(self.pipe, "vae", None)
        self.scheduler = getattr(self.pipe, "scheduler", None)
        self.tokenizer = getattr(self.pipe, "tokenizer", None)
        self.text_encoder = getattr(self.pipe, "text_encoder", None)

        # video processor / saving
        self.save_path = "./results"
        self.name_suffix = ""

    def setup_pipeline(self, args):
        """Optional compatibility with existing code that calls setup_pipeline(args)."""
        # map a couple common args to wrapper state
        self.save_path = getattr(args, "save_path", self.save_path)
        self.name_suffix = getattr(args, "name_suffix", self.name_suffix)
        return self

    def setup_device(self, device: Union[str, torch.device]):
        """Move internal pipeline to a given device (e.g. 'cuda:0')."""
        self.device = torch.device(device)
        self.pipe.to(self.device)
        return self

    def encode_prompt(self, prompt: str, neg_magic: str = "", pos_magic: str = ""):
        """
        Encode prompt into embeddings. Tries to use the pipeline's tokenizer & text_encoder
        if present; otherwise falls back to pipe._encode_prompt if available.
        Returns (prompt_embeds, clip_embedding_or_none, attention_mask_or_none)
        """
        if hasattr(self.pipe, "_encode_prompt"):
            # some pipelines provide internal helper
            out = self.pipe._encode_prompt(prompt, device=self.device)
            # _encode_prompt implementations vary; try to standardize
            prompt_embeds = out if isinstance(out, torch.Tensor) else out.get("prompt_embeds", out)
            return prompt_embeds, None, None

        if self.tokenizer is not None and self.text_encoder is not None:
            tok = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                enc = self.text_encoder(**tok)
                # many encoders return last_hidden_state
                prompt_embeds = getattr(enc, "last_hidden_state", enc)
            return prompt_embeds, None, tok.get("attention_mask", None)

        raise RuntimeError("No method to encode prompt found in pipeline (tokenizer/text_encoder/_encode_prompt missing).")

    def decode_vae(self, latents: torch.Tensor):
        """
        Decode latents into pixel/video tensor. Tries common VAE APIs with safe fallbacks.
        """
        if self.vae is None:
            # If the pipeline provides a direct helper, use it
            if hasattr(self.pipe, "decode_latents"):
                return self.pipe.decode_latents(latents)
            return latents

        # common methods: vae.decode, vae.decode_latents, vae.forward
        if hasattr(self.vae, "decode"):
            try:
                return self.vae.decode(latents)
            except Exception:
                pass
        if hasattr(self.vae, "decode_latents"):
            return self.vae.decode_latents(latents)
        # fallback: return latents
        return latents

    def encode_vae(self, image: Union[str, torch.Tensor, Any]):
        """
        Encode image/frame(s) into VAE latents. Tries common VAE APIs.
        Accepts file path or tensor.
        """
        if self.vae is None:
            raise RuntimeError("VAE not available on loaded pipeline.")

        if isinstance(image, str):
            from PIL import Image
            img = Image.open(image).convert("RGB")
            preprocess = getattr(self.pipe, "preprocess", None)
            if preprocess:
                img = preprocess(img)
            else:
                import torchvision.transforms as T
                img = T.ToTensor()(img).unsqueeze(0) * 2 - 1  # example normalization
        else:
            img = image

        if hasattr(self.vae, "encode"):
            return self.vae.encode(img)
        if hasattr(self.vae, "encode_latents"):
            return self.vae.encode_latents(img)

        raise RuntimeError("VAE does not expose a known encode method.")

    def __call__(self,
                 prompt: Optional[str] = None,
                 first_image: Optional[str] = None,
                 num_frames: int = 81,
                 height: int = 720,
                 width: int = 1280,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 time_shift: float = 13.0,
                 pos_magic: str = "",
                 neg_magic: str = "",
                 motion_score: float = 2.0,
                 output_type: str = "mp4",
                 output_file_name: Optional[str] = None,
                 return_dict: bool = True,
                 **kwargs):
        """
        High-level generation call — delegates to the underlying pipeline if possible.
        Keeps argument names compatible with earlier StepVideo usage.
        """
        # If underlying pipeline exposes a call() that accepts these args, call it.
        pipe_call = getattr(self.pipe, "__call__", None)
        if pipe_call is None:
            raise RuntimeError("Underlying HF pipeline has no __call__ method.")

        # Helpful mapping — many HF pipelines accept similar arguments but not identical names.
        call_kwargs = dict(
            prompt=prompt,
            first_image=first_image,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            time_shift=time_shift,
            pos_magic=pos_magic,
            neg_magic=neg_magic,
            motion_score=motion_score,
            output_type=output_type,
            output_file_name=output_file_name,
            return_dict=return_dict,
        )
        # pass other kwargs through
        call_kwargs.update(kwargs)

        # Call underlying pipeline (it may handle batching, VAE decode, etc.)
        out = pipe_call(**{k: v for k, v in call_kwargs.items() if v is not None})
        # Normalise returns: if pipeline returned a dataclass like StepVideoPipelineOutput, return .video
        if hasattr(out, "video"):
            return out.video
        return out

    # small convenience
    def save_video(self, video_tensor: Union[torch.Tensor, np.ndarray], filename: str):
        """If you need to save locally — use pipeline's utilities if available"""
        # try to use pipe's utility if available
        if hasattr(self.pipe, "save_video") and callable(self.pipe.save_video):
            return self.pipe.save_video(video_tensor, filename)
        # fallback naive saving using torchvision + imageio (simple 1-frame or sequence)
        try:
            import imageio
            if isinstance(video_tensor, torch.Tensor):
                v = video_tensor.detach().cpu().numpy()
            else:
                v = np.array(video_tensor)
            # v expected shape [f, c, h, w] or [c, h, w]
            if v.ndim == 4:
                frames = [(np.clip(((frame + 1) * 127.5), 0, 255)).astype(np.uint8).transpose(1, 2, 0) for frame in v]
            elif v.ndim == 3:
                frames = [(np.clip(((v + 1) * 127.5), 0, 255)).astype(np.uint8).transpose(1, 2, 0)]
            else:
                raise RuntimeError("Unexpected video tensor shape for saving.")
            imageio.mimwrite(filename, frames, fps=16)
            return filename
        except Exception as e:
            raise RuntimeError(f"Saving video failed: {e}")
