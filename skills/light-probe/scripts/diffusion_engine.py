"""Offline DiffusionLight-Turbo inference with a camera-space light estimate.

Adapted from DiffusionLight-Turbo-diffusers/inpaint.py and
DiffusionLight-Turbo/exposure2hdr.py (MIT; see THIRD_PARTY_LICENSE.txt).
No input image or inference request is sent to an external service.

``infer`` keeps the loaded pipeline in this Python process. The caller owns the
process lifetime. ``progress(fraction, message)`` may raise to cancel inference.
Fast is a 512px, EV0-only approximation, NOT an HDR measurement. Full is 1024px,
30 denoising steps per exposure, with EV 0/-2.5/-5 and the official HDR merge.
Neither mode recovers calibrated illuminance, camera pose, or scene geometry.
"""
from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Callable

import numpy as np
from PIL import Image, ImageOps

from model_registry import (COMPONENTS, acquire_lock, gpu_lock_path, hub_root,
                            model_status as registry_model_status, snapshot_path)

EVS = (0.0, -2.5, -5.0)
PROMPT = "a perfect mirrored reflective chrome ball sphere"
PROMPT_BLACK = "a perfect black dark mirrored reflective chrome ball sphere"
NEGATIVE = "matte, diffuse, flat, dull"
SEED = 200
SWITCH_LORA_TIMESTEP = 800
LUMA = np.array([0.212671, 0.715160, 0.072169], dtype=np.float32)

# The shared registry pins all inference/download components to immutable revisions.
_BUNDLE: dict | None = None
_INFERENCE_LOCK = threading.Lock()
_PROCESS_GPU_LOCK = None


class ModelUnavailableError(RuntimeError):
    """An explicitly downloaded local inference component is absent."""


class InferenceError(RuntimeError):
    """The generative pipeline did not produce a usable light probe."""


def _snapshot(key: str, root: Path | None = None) -> Path:
    return snapshot_path(key, root)


def model_status() -> dict:
    """Read-only local size check; --check --verify performs full hash validation."""
    return {**registry_model_status(), "loaded": _BUNDLE is not None}


def _notify(progress, fraction: float, message: str) -> None:
    if progress:
        progress(float(np.clip(fraction, 0, 1)), message)


def _acquire_process_gpu_lock() -> None:
    """Retain the advisory lock for as long as persistent model weights live."""
    global _PROCESS_GPU_LOCK
    if _PROCESS_GPU_LOCK is not None:
        return
    _PROCESS_GPU_LOCK = acquire_lock(
        gpu_lock_path(), "Another Light Probe process owns the GPU model lock; wait for it to exit")


def _release_unloaded_process_lock() -> None:
    global _PROCESS_GPU_LOCK
    if _BUNDLE is None and _PROCESS_GPU_LOCK is not None:
        _PROCESS_GPU_LOCK.release()
        _PROCESS_GPU_LOCK = None


def _load_local_lora(pipe, key: str, adapter: str) -> None:
    from safetensors.torch import load_file

    weights = load_file(str(_snapshot(key) / "pytorch_lora_weights.safetensors"), device="cpu")
    # These pinned official checkpoints contain an extra serialization wrapper:
    # unet.unet.<module>.lora.{down,up}.weight. Diffusers removes only one unet
    # prefix, so normalize that observed duplicate without modifying the cache.
    normalized = {}
    for name, tensor in weights.items():
        target = name[5:] if name.startswith("unet.unet.") else name
        if target in normalized:
            raise InferenceError(f"Duplicate normalized LoRA parameter: {target}")
        normalized[target] = tensor
    pipe.load_lora_weights(normalized, adapter_name=adapter)


def _load_pipeline(progress=None) -> dict:
    global _BUNDLE
    if _BUNDLE is not None:
        if _BUNDLE["root"] != str(hub_root()):
            raise RuntimeError("Model cache changed after pipeline load; start a new Python process")
        return _BUNDLE
    status = model_status()
    if not status["ready"]:
        raise ModelUnavailableError("Offline model files missing: " + "; ".join(status["missing"]) +
                                    ". Run download_models.py --download explicitly; inference never downloads models.")
    _acquire_process_gpu_lock()
    # Local paths plus local_files_only enforce offline operation even when
    # huggingface_hub was imported before these environment settings.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetInpaintPipeline, UniPCMultistepScheduler
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    threads = max(1, min(4, int(os.environ.get("LIGHT_PROBE_THREADS", "4"))))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(min(2, threads))
    except RuntimeError:
        pass  # Another caller already initialized the process-wide thread pool.
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16
    _notify(progress, 0.03, f"Loading local DiffusionLight models ({device}); first load may take time")
    vae = AutoencoderKL.from_pretrained(str(_snapshot("vae")), torch_dtype=dtype,
                                        local_files_only=True, use_safetensors=True)
    controlnet = ControlNetModel.from_pretrained(str(_snapshot("controlnet")), torch_dtype=dtype,
                                                variant="fp16", local_files_only=True, use_safetensors=True)
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        str(_snapshot("base")), controlnet=controlnet, vae=vae, torch_dtype=dtype,
        variant="fp16", add_watermarker=False, local_files_only=True, use_safetensors=True,
    )
    _load_local_lora(pipe, "turbo", "turbo")
    _load_local_lora(pipe, "exposure", "exposure")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing("auto")
    # Optional tiling bounds VAE memory at 1024px. It does not change image size.
    if os.environ.get("LIGHT_PROBE_VAE_TILING", "1") != "0":
        pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)

    original_prepare = pipe.prepare_mask_latents

    def bilinear_mask(self, mask, masked_image, batch_size, height, width, *args, **kwargs):
        mask = torch.nn.functional.interpolate(mask,
            size=(height // self.vae_scale_factor, width // self.vae_scale_factor),
            mode="bilinear", align_corners=False)
        return original_prepare(mask, masked_image, batch_size, height, width, *args, **kwargs)

    pipe.prepare_mask_latents = bilinear_mask.__get__(pipe, pipe.__class__)
    _notify(progress, 0.10, "Loading local depth estimator")
    # DPT stays on CPU to avoid a second GPU workload and fp16 depth instability.
    depth_processor = AutoImageProcessor.from_pretrained(str(_snapshot("depth")), local_files_only=True)
    depth_model = AutoModelForDepthEstimation.from_pretrained(str(_snapshot("depth")),
        local_files_only=True, torch_dtype=torch.float32, use_safetensors=True).eval()
    pipe.set_adapters(["turbo"], adapter_weights=[1.0])
    with torch.inference_mode():
        emb, neg_emb, pooled, neg_pooled = pipe.encode_prompt(PROMPT, negative_prompt=NEGATIVE)
        emb_black, _, pooled_black, _ = pipe.encode_prompt(PROMPT_BLACK, negative_prompt=NEGATIVE)
    _BUNDLE = {"pipe": pipe, "depth_processor": depth_processor, "depth_model": depth_model,
               "device": device, "root": str(hub_root()), "torch": torch,
               "emb": emb, "emb_black": emb_black, "pooled": pooled, "pooled_black": pooled_black,
               "neg_emb": neg_emb, "neg_pooled": neg_pooled}
    return _BUNDLE


def _circle(size: int) -> np.ndarray:
    y, x = np.mgrid[1:-1:complex(size), -1:1:complex(size)]
    return (x * x + y * y) <= 1


def _prepare_image(path: Path, anchor: tuple[float, float], size: int) -> tuple[Image.Image, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("Input image exceeds the 20 MiB limit")
    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source)
        if source.width * source.height > 80_000_000:
            raise ValueError("Input image exceeds 80 million pixels")
        # Composite transparent images on white before estimating scene lighting.
        if source.mode in ("RGBA", "LA") or (source.mode == "P" and "transparency" in source.info):
            rgba = source.convert("RGBA")
            image = Image.new("RGB", rgba.size, "white")
            image.paste(rgba, mask=rgba.getchannel("A"))
        else:
            image = source.convert("RGB")
        width, height = image.size
        if min(width, height) < 16:
            raise ValueError("Input image must be at least 16 pixels on each side")
        # Keep the full image scale and center the selected anchor. Image edges
        # are black padded, as recommended by the official inpainting pipeline.
        scale = size / max(width, height)
        resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
        offset = (round(size / 2 - anchor[0] * resized.width), round(size / 2 - anchor[1] * resized.height))
        canvas = Image.new("RGB", (size, size), "black")
        canvas.paste(resized, offset)
    return canvas, {"width": width, "height": height, "anchor": list(anchor), "canvas_size": size,
                    "scale": scale, "offset": list(offset), "ball_diameter": size // 4}


def environment_directions(height: int) -> np.ndarray:
    """Lat-long: center=+z viewer, right quarter=+x, top=+y; pixel centers."""
    row, column = np.mgrid[0:height, 0:2 * height]
    longitude = ((column + 0.5) / (2 * height) - 0.5) * (2 * np.pi)
    latitude = (0.5 - (row + 0.5) / height) * np.pi
    return np.stack((np.cos(latitude) * np.sin(longitude), np.sin(latitude),
                     np.cos(latitude) * np.cos(longitude)), axis=-1).astype(np.float32)


def mirror_ball_to_environment(ball: np.ndarray, height: int = 256) -> np.ndarray:
    """Map environment direction L to mirror normal normalize(L + V), V=+z.

    The visible mirror normal is (nx, ny, sqrt(1-nx²-ny²)); image y is downward.
    This implements the mirror reflection geometry, not a polar-coordinate crop.
    The exactly opposite direction L=-V is singular and is never a pixel center.
    """
    import cv2
    directions = environment_directions(height)
    normals = directions.copy()
    normals[..., 2] += 1
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
    map_x = ((normals[..., 0] + 1) * 0.5 * (ball.shape[1] - 1)).astype(np.float32)
    map_y = ((1 - normals[..., 1]) * 0.5 * (ball.shape[0] - 1)).astype(np.float32)
    return cv2.remap(ball, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def merge_exposures(environments: list[np.ndarray], exposures=EVS, gamma: float = 2.4) -> np.ndarray:
    """Official DiffusionLight luminance merge, in relative radiometric units."""
    if len(environments) != len(exposures) or len(environments) < 2:
        raise ValueError("HDR merge requires a matching bracket of at least two exposures")
    linear = [np.power(np.clip(image, 0, 1), gamma) / (2 ** ev) for image, ev in zip(environments, exposures)]
    luminances = [image @ LUMA for image in linear]
    merged_luminance = luminances[-1]
    for index in range(len(exposures) - 1, 0, -1):
        maximum = 1 / (2 ** exposures[index - 1])
        saturated = np.clip((luminances[index - 1] - 0.9 * maximum) / (0.1 * maximum), 0, 1)
        mask = saturated * (merged_luminance > luminances[index - 1])
        merged_luminance = luminances[index - 1] * (1 - mask) + merged_luminance * mask
    return (linear[0] * (merged_luminance / (luminances[0] + 1e-10))[..., None]).astype(np.float32)


def _validate_radiance(array: np.ndarray, name: str) -> None:
    if not np.isfinite(array).all():
        raise InferenceError(f"{name} contains non-finite values; no valid probe was produced")
    if array.size == 0 or float(np.max(array)) <= 1e-6 or float(np.mean(array)) <= 1e-7:
        raise InferenceError(f"{name} is black/empty; no valid probe was produced")


def _direction_marker(direction: np.ndarray, anchor, width: int, height: int) -> tuple[float, float]:
    # Screen markers depict directions only. No scene depth/focal length is known.
    dx, dy = float(direction[0]), -float(direction[1])
    x, y = anchor[0] * width, anchor[1] * height
    radius = 0.38 * min(width, height)
    for position, delta, extent in ((x, dx, width), (y, dy, height)):
        if abs(delta) > 1e-8:
            radius = min(radius, ((extent - position) if delta > 0 else position) / abs(delta))
    return float(np.clip((x + radius * dx) / width, 0, 1)), float(np.clip((y + radius * dy) / height, 0, 1))


def extract_lights(environment: np.ndarray, anchor=(0.5, 0.5), width=1, height=1, limit=3) -> list[dict]:
    """Dominant spherical lobes, weighted by luminance and pixel solid angle."""
    import cv2
    _validate_radiance(environment, "Environment")
    directions = environment_directions(environment.shape[0])
    luminance = environment @ LUMA
    # Horizontal wrap avoids inventing a seam at the back of the sphere.
    wrapped = np.pad(luminance, ((0, 0), (4, 4)), mode="wrap")
    smooth = cv2.GaussianBlur(wrapped, (9, 9), 0)[:, 4:-4]
    # Uniform probes do not support a dominant direction.
    median = float(np.median(smooth))
    if float(smooth.max()) <= max(1e-7, median * 1.08):
        return []
    latitude_weight = np.sqrt(np.maximum(0, 1 - directions[..., 1] ** 2))
    remaining = np.ones(luminance.shape, dtype=bool)
    candidates = []
    for _ in range(limit):
        scores = np.where(remaining, smooth, -1)
        peak_row, peak_column = np.unravel_index(int(np.argmax(scores)), scores.shape)
        peak = float(scores[peak_row, peak_column])
        if peak <= max(median * 1.12, float(smooth.max()) * 0.20, 1e-7):
            break
        peak_direction = directions[peak_row, peak_column]
        dots = np.einsum("ijk,k->ij", directions, peak_direction)
        neighborhood = (dots >= math.cos(math.radians(18))) & remaining
        weights = np.maximum(luminance - median, 0) * latitude_weight * neighborhood
        energy = float(weights.sum())
        if energy <= 1e-9:
            break
        direction = (directions * weights[..., None]).sum(axis=(0, 1)) / energy
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        rgb = (environment * weights[..., None]).sum(axis=(0, 1)) / energy
        candidates.append((energy, direction, rgb))
        remaining &= np.einsum("ijk,k->ij", directions, direction) < math.cos(math.radians(32))
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    largest = candidates[0][0] if candidates else 1
    result = []
    for index, (energy, direction, rgb) in enumerate(candidates):
        x, y = _direction_marker(direction, anchor, width, height)
        display_rgb = np.power(np.clip(rgb / max(float(rgb.max()), 1e-8), 0, 1), 1 / 2.4)
        color = "#" + "".join(f"{int(round(channel * 255)):02x}" for channel in display_rgb)
        result.append({"x": x, "y": y, "strength": round(float(energy / largest), 5),
                       "color": color, "label": f"추정 광원 {index + 1}",
                       "direction": [round(float(value), 7) for value in direction],
                       "projection": "orthographic_direction_marker"})
    return result


def infer(image_path: Path, output_dir: Path, quality="fast", anchor=(0.5, 0.5),
          progress: Callable[[float, str], None] | None = None) -> dict:
    """Generate a mirror-ball probe and estimate dominant camera-space lights.

    ``anchor`` uses normalized original-image coordinates, x right/y down.
    Direction vectors point from the anchor toward the source, +z to the viewer.
    Returned x/y are 2D direction markers, NOT recovered source pixel locations.
    The callback runs synchronously; raising an exception stops between steps.
    """
    if quality not in ("fast", "full"):
        raise ValueError("quality must be 'fast' or 'full'")
    try:
        anchor = tuple(float(value) for value in anchor)
    except (TypeError, ValueError) as exc:
        raise ValueError("anchor must contain two finite numbers in [0, 1]") from exc
    if len(anchor) != 2 or not all(math.isfinite(value) and 0 <= value <= 1 for value in anchor):
        raise ValueError("anchor must contain two finite numbers in [0, 1]")
    if not _INFERENCE_LOCK.acquire(blocking=False):
        raise RuntimeError("This process already has an active DiffusionLight inference")
    started = time.perf_counter()
    bundle = None
    try:
        size, steps = (512, 12) if quality == "fast" else (1024, 30)
        exposures = (0.0,) if quality == "fast" else EVS
        canvas, info = _prepare_image(Path(image_path), anchor, size)
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        canvas.save(out / "canvas.png")
        bundle = _load_pipeline(progress)
        torch, pipe = bundle["torch"], bundle["pipe"]
        _notify(progress, 0.15, "Estimating depth locally and placing the mirror ball")
        with torch.inference_mode():
            inputs = bundle["depth_processor"](images=canvas, return_tensors="pt")
            depth_output = bundle["depth_model"](**inputs).predicted_depth
            depth_output = torch.nn.functional.interpolate(depth_output.unsqueeze(1), (size, size),
                                                           mode="bicubic", align_corners=False)
            depth = depth_output.squeeze().cpu().numpy().astype(np.float32)
            if not np.isfinite(depth).all():
                raise InferenceError("Depth model produced non-finite values")
            depth = (depth - depth.min()) / max(float(depth.max() - depth.min()), 1e-8) * 255
            ball_size = size // 4
            edge = (size - ball_size) // 2
            ball_mask = _circle(ball_size)
            depth[edge:edge + ball_size, edge:edge + ball_size][ball_mask] = 255
            control_image = Image.fromarray(depth.astype(np.uint8)).convert("RGB")
            control_image.save(out / "depth.png")
            dilation = 10 if quality == "full" else 5
            mask = np.zeros((size, size), dtype=np.uint8)
            mask_size = ball_size + 2 * dilation
            mask[edge - dilation:edge + ball_size + dilation, edge - dilation:edge + ball_size + dilation] = _circle(mask_size) * 255
            mask_image = Image.fromarray(mask)
            mask_image.save(out / "mask.png")
            environments = []
            for exposure_index, exposure in enumerate(exposures):
                ratio = (exposure - EVS[-1]) / (0 - EVS[-1])
                switched = False
                # Unfused adapters avoid repeated fp16 fuse/unfuse rounding and
                # temporary full-weight copies; scale/switch match the reference.
                pipe.set_adapters(["turbo"], adapter_weights=[1.0])

                def on_step(pipeline, step_index, timestep, callback_kwargs):
                    nonlocal switched
                    if not switched and float(timestep) <= SWITCH_LORA_TIMESTEP:
                        pipeline.set_adapters(["exposure"], adapter_weights=[0.75])
                        switched = True
                    completed = (exposure_index + (step_index + 1) / steps) / len(exposures)
                    _notify(progress, 0.20 + completed * 0.70,
                            f"EV {exposure:g}: denoising {step_index + 1}/{steps}")
                    latents = callback_kwargs.get("latents")
                    if latents is not None and not bool(torch.isfinite(latents).all()):
                        raise InferenceError("Denoising produced non-finite values; inference stopped")
                    return callback_kwargs

                _notify(progress, 0.20 + exposure_index / len(exposures) * 0.70,
                        f"Generating EV {exposure:g} mirror ball ({size}px, {steps} steps)")
                output = pipe(
                    prompt_embeds=bundle["emb"] * ratio + bundle["emb_black"] * (1 - ratio),
                    pooled_prompt_embeds=bundle["pooled"] * ratio + bundle["pooled_black"] * (1 - ratio),
                    negative_prompt_embeds=bundle["neg_emb"], negative_pooled_prompt_embeds=bundle["neg_pooled"],
                    num_inference_steps=steps, strength=1.0, height=size, width=size,
                    image=canvas, mask_image=mask_image, control_image=control_image,
                    controlnet_conditioning_scale=0.5, callback_on_step_end=on_step,
                    generator=torch.Generator(device="cpu").manual_seed(SEED), output_type="np",
                ).images[0]
                _validate_radiance(output, f"EV {exposure:g} output")
                ball = output[edge:edge + ball_size, edge:edge + ball_size].astype(np.float32)
                _validate_radiance(ball[ball_mask], f"EV {exposure:g} mirror ball")
                tag = f"{int(abs(exposure * 10)):02d}"
                Image.fromarray(np.uint8(np.clip(output, 0, 1) * 255)).save(out / f"inpainted_ev{tag}.png")
                Image.fromarray(np.uint8(np.clip(ball, 0, 1) * 255)).save(out / f"ball_ev{tag}.png")
                environments.append(mirror_ball_to_environment(ball, 128 if quality == "fast" else 256))
                del output
        _notify(progress, 0.93, "Unwrapping the mirror ball and extracting dominant light directions")
        if quality == "full":
            import cv2
            radiance = merge_exposures(environments)
            _validate_radiance(radiance, "Merged HDR")
            hdr_path = out / "probe.hdr"
            if not cv2.imwrite(str(hdr_path), radiance[..., ::-1]):
                raise InferenceError("Unable to save the HDR probe")
            percentile = max(float(np.percentile(radiance @ LUMA, 95)), 1e-6)
            scaled = radiance / percentile
            preview = np.power(scaled / (1 + scaled), 1 / 2.4)
        else:
            # A single LDR exposure contains no recovered highlight dynamic range.
            radiance = np.power(np.clip(environments[0], 0, 1), 2.4)
            preview = environments[0]
            hdr_path = None
        _validate_radiance(radiance, "Final environment")
        preview_path = out / "probe_preview.png"
        Image.fromarray(np.uint8(np.clip(preview, 0, 1) * 255)).save(preview_path)
        lights = extract_lights(radiance, anchor, info["width"], info["height"])
        luminance = radiance @ LUMA
        note = ("DiffusionLight 생성 기반 조명 추정입니다. " +
                ("빠른 모드는 512px·12단계·EV0 단일 LDR 근사이며 HDR 복원이 아닙니다. " if quality == "fast" else
                 "1024px·노출별 30단계·EV 0/−2.5/−5를 합성한 상대 HDR입니다. ") +
                "3D 방향은 카메라 기준 +x 오른쪽, +y 위쪽, +z 관찰자 쪽입니다. "
                "사진 위 x/y는 기준점에서 방향을 정사영한 표시 좌표이며 실제 광원 위치가 아닙니다. "
                "광도는 상대값이며 lux, 재질 굴절률, 실제 장면 기하를 측정하지 않습니다.")
        if not lights:
            note += " 뚜렷한 지배 광원을 구분하지 못했습니다."
        result = {"mode": "diffusion", "quality": quality, "lights": lights,
                  "preview_path": str(preview_path), "elapsed_ms": round((time.perf_counter() - started) * 1000),
                  "note": note, "reliability": "low", "device": bundle["device"],
                  "width": info["width"], "height": info["height"], "anchor": list(anchor),
                  "environment_layout": "latitude-longitude, center +z, right quarter +x, top +y",
                  "relative_luminance_mean": float(luminance.mean()),
                  "contrast": float((luminance.max() - luminance.min()) / max(float(luminance.max()), 1e-8)),
                  "exposures": list(exposures), "steps_per_exposure": steps, "seed": SEED, "offline": True}
        if hdr_path:
            result["hdr_path"] = str(hdr_path)
        (out / "probe.json").write_text(json.dumps({**result, "preparation": info}, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
        _notify(progress, 1, "Local DiffusionLight inference complete")
        return result
    finally:
        if bundle is not None:
            # Keep weights for the next call, release unused inference buffers.
            gc.collect()
            if bundle["device"] == "mps":
                bundle["torch"].mps.empty_cache()
            elif bundle["device"] == "cuda":
                bundle["torch"].cuda.empty_cache()
        _release_unloaded_process_lock()
        _INFERENCE_LOCK.release()
