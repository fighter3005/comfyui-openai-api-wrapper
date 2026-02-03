import copy

MODEL_ID = "flux-2-klein-4b"

# --------------------------
# Text-to-Image (Generation)
# --------------------------
# Updated to match the new workflow:
# - Prompt node is PrimitiveStringMultiline (76.value)
# - Steps are 50
# - Width/Height primitives are 1232/1232 by default
# - Scheduler/LatentImage take width/height from those primitives
FLUX_KLEIN_4B_GEN = {
  "76": {
    "inputs": {
      "value": "Latina female with thick wavy hair, harbor boats and pastel houses behind. Breezy seaside light, warm tones, cinematic close-up. "
    },
    "class_type": "PrimitiveStringMultiline",
    "_meta": {"title": "Prompt"}
  },
  "80": {
    "inputs": {"filename_prefix": "Flux2-Klein", "images": ["75:65", 0]},
    "class_type": "SaveImage",
    "_meta": {"title": "Save Image"}
  },
  "75:61": {
    "inputs": {"sampler_name": "euler"},
    "class_type": "KSamplerSelect",
    "_meta": {"title": "KSamplerSelect"}
  },
  "75:63": {
    "inputs": {"cfg": 5, "model": ["75:70", 0], "positive": ["75:74", 0], "negative": ["75:67", 0]},
    "class_type": "CFGGuider",
    "_meta": {"title": "CFGGuider"}
  },
  "75:64": {
    "inputs": {"noise": ["75:73", 0], "guider": ["75:63", 0], "sampler": ["75:61", 0], "sigmas": ["75:62", 0], "latent_image": ["75:66", 0]},
    "class_type": "SamplerCustomAdvanced",
    "_meta": {"title": "SamplerCustomAdvanced"}
  },
  "75:65": {
    "inputs": {"samples": ["75:64", 0], "vae": ["75:72", 0]},
    "class_type": "VAEDecode",
    "_meta": {"title": "VAE Decode"}
  },
  "75:66": {
    "inputs": {"width": ["75:68", 0], "height": ["75:69", 0], "batch_size": 1},
    "class_type": "EmptyFlux2LatentImage",
    "_meta": {"title": "Empty Flux 2 Latent"}
  },
  "75:67": {
    "inputs": {"text": "", "clip": ["75:71", 0]},
    "class_type": "CLIPTextEncode",
    "_meta": {"title": "CLIP Text Encode (Negative Prompt)"}
  },
  "75:68": {
    "inputs": {"value": 1232},
    "class_type": "PrimitiveInt",
    "_meta": {"title": "Width"}
  },
  "75:69": {
    "inputs": {"value": 1232},
    "class_type": "PrimitiveInt",
    "_meta": {"title": "Height"}
  },
  "75:73": {
    "inputs": {"noise_seed": 457566357656},
    "class_type": "RandomNoise",
    "_meta": {"title": "RandomNoise"}
  },
  "75:72": {
    "inputs": {"vae_name": "flux2-vae.safetensors"},
    "class_type": "VAELoader",
    "_meta": {"title": "Load VAE"}
  },
  "75:74": {
    "inputs": {"text": ["76", 0], "clip": ["75:71", 0]},
    "class_type": "CLIPTextEncode",
    "_meta": {"title": "CLIP Text Encode (Positive Prompt)"}
  },
  "75:71": {
    "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2", "device": "default"},
    "class_type": "CLIPLoader",
    "_meta": {"title": "Load CLIP"}
  },
  "75:70": {
    "inputs": {"unet_name": "flux-2-klein-base-4b-fp8.safetensors", "weight_dtype": "default"},
    "class_type": "UNETLoader",
    "_meta": {"title": "Load Diffusion Model"}
  },
  "75:62": {
    "inputs": {"steps": 50, "width": ["75:68", 0], "height": ["75:69", 0]},
    "class_type": "Flux2Scheduler",
    "_meta": {"title": "Flux2Scheduler"}
  }
}

# ============================================================
# EDIT WORKFLOWS (NORMAL vs INFER-SIZE)
# ============================================================
# Rule requested (same as with dev turbo):
# - If no size is given / size not supported / size is 0x0 -> infer-size workflow.
# In your app.py you now return (0,0) for edit when size is auto/None, which triggers infer-size.

def _should_use_infer_size(width, height) -> bool:
    try:
        if width is None or height is None:
            return True
        w = int(width)
        h = int(height)
        return (w <= 0) or (h <= 0)
    except Exception:
        return True

# --------------------------
# Image Edit (1 image) - NORMAL (explicit 1232x1232 in workflow) [6]
# --------------------------
FLUX_KLEIN_4B_EDIT_1IMG = {
  "76": {"inputs": {"image": "pasted/image (5).png"}, "class_type": "LoadImage", "_meta": {"title": "Load Image"}},
  "94": {"inputs": {"filename_prefix": "Flux2-Klein-4b-base", "images": ["92:65", 0]}, "class_type": "SaveImage", "_meta": {"title": "Save Image"}},
  "92:61": {"inputs": {"sampler_name": "euler"}, "class_type": "KSamplerSelect", "_meta": {"title": "KSamplerSelect"}},
  "92:64": {"inputs": {"noise": ["92:73", 0], "guider": ["92:63", 0], "sampler": ["92:61", 0], "sigmas": ["92:62", 0], "latent_image": ["92:66", 0]}, "class_type": "SamplerCustomAdvanced", "_meta": {"title": "SamplerCustomAdvanced"}},
  "92:65": {"inputs": {"samples": ["92:64", 0], "vae": ["92:72", 0]}, "class_type": "VAEDecode", "_meta": {"title": "VAE Decode"}},
  "92:73": {"inputs": {"noise_seed": 432262096973497}, "class_type": "RandomNoise", "_meta": {"title": "RandomNoise"}},
  "92:70": {"inputs": {"unet_name": "flux-2-klein-base-4b-fp8.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader", "_meta": {"title": "Load Diffusion Model"}},
  "92:71": {"inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2", "device": "default"}, "class_type": "CLIPLoader", "_meta": {"title": "Load CLIP"}},
  "92:63": {"inputs": {"cfg": 5, "model": ["92:70", 0], "positive": ["92:79:77", 0], "negative": ["92:79:76", 0]}, "class_type": "CFGGuider", "_meta": {"title": "CFGGuider"}},
  "92:72": {"inputs": {"vae_name": "flux2-vae.safetensors"}, "class_type": "VAELoader", "_meta": {"title": "Load VAE"}},
  "92:74": {"inputs": {"text": "Make the Woman wear the BOSS t-shirt and cap.", "clip": ["92:71", 0]}, "class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text Encode (Positive Prompt)"}},
  "92:87": {"inputs": {"text": "", "clip": ["92:71", 0]}, "class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text Encode ( Negative Prompt)"}},
  "92:79:76": {"inputs": {"conditioning": ["92:87", 0], "latent": ["92:79:78", 0]}, "class_type": "ReferenceLatent", "_meta": {"title": "ReferenceLatent"}},
  "92:79:78": {"inputs": {"pixels": ["92:80", 0], "vae": ["92:72", 0]}, "class_type": "VAEEncode", "_meta": {"title": "VAE Encode"}},
  "92:79:77": {"inputs": {"conditioning": ["92:74", 0], "latent": ["92:79:78", 0]}, "class_type": "ReferenceLatent", "_meta": {"title": "ReferenceLatent"}},
  "92:66": {"inputs": {"width": 1232, "height": 1232, "batch_size": 1}, "class_type": "EmptyFlux2LatentImage", "_meta": {"title": "Empty Flux 2 Latent"}},
  "92:80": {"inputs": {"upscale_method": "nearest-exact", "megapixels": 1.25, "resolution_steps": 1, "image": ["76", 0]}, "class_type": "ImageScaleToTotalPixels", "_meta": {"title": "ImageScaleToTotalPixels"}},
  "92:62": {"inputs": {"steps": 40, "width": 1232, "height": 1232}, "class_type": "Flux2Scheduler", "_meta": {"title": "Flux2Scheduler"}}
}

# --------------------------
# Image Edit (1 image) - INFER SIZE [7]
# --------------------------
FLUX_KLEIN_4B_EDIT_1IMG_INFER_SIZE = {
  **copy.deepcopy(FLUX_KLEIN_4B_EDIT_1IMG),
  "92:66": {"inputs": {"width": ["92:81", 0], "height": ["92:81", 1], "batch_size": 1}, "class_type": "EmptyFlux2LatentImage", "_meta": {"title": "Empty Flux 2 Latent"}},
  "92:81": {"inputs": {"image": ["92:80", 0]}, "class_type": "GetImageSize", "_meta": {"title": "Get Image Size"}},
  "92:62": {"inputs": {"steps": 40, "width": ["92:81", 0], "height": ["92:81", 1]}, "class_type": "Flux2Scheduler", "_meta": {"title": "Flux2Scheduler"}}
}

# --------------------------
# Image Edit (2 images) - NORMAL [4]
# --------------------------
FLUX_KLEIN_4B_EDIT_2IMG = {
  **copy.deepcopy(FLUX_KLEIN_4B_EDIT_1IMG),
  "81": {"inputs": {"image": "pasted/image (6).png"}, "class_type": "LoadImage", "_meta": {"title": "Load Image"}},
  "92:88": {"inputs": {"upscale_method": "nearest-exact", "megapixels": 1.25, "resolution_steps": 1, "image": ["81", 0]}, "class_type": "ImageScaleToTotalPixels", "_meta": {"title": "ImageScaleToTotalPixels"}},
  "92:89:78": {"inputs": {"pixels": ["92:88", 0], "vae": ["92:72", 0]}, "class_type": "VAEEncode", "_meta": {"title": "VAE Encode"}},
  "92:89:76": {"inputs": {"conditioning": ["92:79:76", 0], "latent": ["92:89:78", 0]}, "class_type": "ReferenceLatent", "_meta": {"title": "ReferenceLatent"}},
  "92:89:77": {"inputs": {"conditioning": ["92:79:77", 0], "latent": ["92:89:78", 0]}, "class_type": "ReferenceLatent", "_meta": {"title": "ReferenceLatent"}},
  "92:63": {"inputs": {"cfg": 5, "model": ["92:70", 0], "positive": ["92:89:77", 0], "negative": ["92:89:76", 0]}, "class_type": "CFGGuider", "_meta": {"title": "CFGGuider"}}
}

# --------------------------
# Image Edit (2 images) - INFER SIZE [5]
# --------------------------
FLUX_KLEIN_4B_EDIT_2IMG_INFER_SIZE = {
  **copy.deepcopy(FLUX_KLEIN_4B_EDIT_2IMG),
  "92:66": {"inputs": {"width": ["92:81", 0], "height": ["92:81", 1], "batch_size": 1}, "class_type": "EmptyFlux2LatentImage", "_meta": {"title": "Empty Flux 2 Latent"}},
  "92:81": {"inputs": {"image": ["92:80", 0]}, "class_type": "GetImageSize", "_meta": {"title": "Get Image Size"}},
  "92:62": {"inputs": {"steps": 40, "width": ["92:81", 0], "height": ["92:81", 1]}, "class_type": "Flux2Scheduler", "_meta": {"title": "Flux2Scheduler"}}
}

# --------------------------
# Image Edit (3 images) - NORMAL [2]
# --------------------------
FLUX_KLEIN_4B_EDIT_3IMG = {
  **copy.deepcopy(FLUX_KLEIN_4B_EDIT_2IMG),
  "100": {"inputs": {"image": "pasted/image (8).png"}, "class_type": "LoadImage", "_meta": {"title": "Load Image"}},
  "92:85": {"inputs": {"upscale_method": "nearest-exact", "megapixels": 1.25, "resolution_steps": 1, "image": ["100", 0]}, "class_type": "ImageScaleToTotalPixels", "_meta": {"title": "ImageScaleToTotalPixels"}},
  "92:84:78": {"inputs": {"pixels": ["92:85", 0], "vae": ["92:72", 0]}, "class_type": "VAEEncode", "_meta": {"title": "VAE Encode"}},
  "92:84:76": {"inputs": {"conditioning": ["92:89:76", 0], "latent": ["92:84:78", 0]}, "class_type": "ReferenceLatent", "_meta": {"title": "ReferenceLatent"}},
  "92:84:77": {"inputs": {"conditioning": ["92:89:77", 0], "latent": ["92:84:78", 0]}, "class_type": "ReferenceLatent", "_meta": {"title": "ReferenceLatent"}},
  "92:63": {"inputs": {"cfg": 5, "model": ["92:70", 0], "positive": ["92:84:77", 0], "negative": ["92:84:76", 0]}, "class_type": "CFGGuider", "_meta": {"title": "CFGGuider"}}
}

# --------------------------
# Image Edit (3 images) - INFER SIZE [3]
# --------------------------
FLUX_KLEIN_4B_EDIT_3IMG_INFER_SIZE = {
  **copy.deepcopy(FLUX_KLEIN_4B_EDIT_3IMG),
  "92:66": {"inputs": {"width": ["92:81", 0], "height": ["92:81", 1], "batch_size": 1}, "class_type": "EmptyFlux2LatentImage", "_meta": {"title": "Empty Flux 2 Latent"}},
  "92:81": {"inputs": {"image": ["92:80", 0]}, "class_type": "GetImageSize", "_meta": {"title": "Get Image Size"}},
  "92:62": {"inputs": {"steps": 40, "width": ["92:81", 0], "height": ["92:81", 1]}, "class_type": "Flux2Scheduler", "_meta": {"title": "Flux2Scheduler"}}
}


def get_workflow(
    mode: str = "gen",
    prompt: str = "",
    width: int = 0,
    height: int = 0,
    seed: int = 0,
    images=None,
    **kwargs
):
    if images is None:
        images = []

    ignored = {k: v for k, v in kwargs.items() if v not in (None, "", [], {})}
    if ignored:
        print(f"[{MODEL_ID}] Ignoring unsupported parameters: {list(ignored.keys())}")

    if mode == "gen":
        wf = copy.deepcopy(FLUX_KLEIN_4B_GEN)
        wf["76"]["inputs"]["value"] = (prompt or "").rstrip() + "\n"
        wf["75:73"]["inputs"]["noise_seed"] = int(seed or 0)
        wf["75:68"]["inputs"]["value"] = int(width)
        wf["75:69"]["inputs"]["value"] = int(height)
        return wf

    # edit mode
    if len(images) < 1:
        raise ValueError(f"{MODEL_ID} edit requires at least 1 image")
    if len(images) > 3:
        print(f"[{MODEL_ID}] Received {len(images)} images; only 3 supported. Extra images will be ignored.")
        images = images[:3]

    use_infer = _should_use_infer_size(width, height)

    if len(images) == 1:
        wf = copy.deepcopy(FLUX_KLEIN_4B_EDIT_1IMG_INFER_SIZE if use_infer else FLUX_KLEIN_4B_EDIT_1IMG)
        wf["76"]["inputs"]["image"] = images[0]
    elif len(images) == 2:
        wf = copy.deepcopy(FLUX_KLEIN_4B_EDIT_2IMG_INFER_SIZE if use_infer else FLUX_KLEIN_4B_EDIT_2IMG)
        wf["76"]["inputs"]["image"] = images[0]
        wf["81"]["inputs"]["image"] = images[1]
    else:
        wf = copy.deepcopy(FLUX_KLEIN_4B_EDIT_3IMG_INFER_SIZE if use_infer else FLUX_KLEIN_4B_EDIT_3IMG)
        wf["76"]["inputs"]["image"] = images[0]
        wf["81"]["inputs"]["image"] = images[1]
        wf["100"]["inputs"]["image"] = images[2]

    # prompt + seed (node ids per new workflows)
    wf["92:74"]["inputs"]["text"] = prompt or ""
    wf["92:73"]["inputs"]["noise_seed"] = int(seed or 0)

    # For NORMAL (non-infer) edit workflows, width/height are hardcoded (1232x1232) in nodes 92:66 and 92:62.
    # Your requirement says: if size is supported and non-0x0, use the normal (not infer-size) version.
    # So we *do* set those explicit dimensions from the API if provided and non-zero, to honor the request.
    if not use_infer:
        wf["92:66"]["inputs"]["width"] = int(width)
        wf["92:66"]["inputs"]["height"] = int(height)
        wf["92:62"]["inputs"]["width"] = int(width)
        wf["92:62"]["inputs"]["height"] = int(height)

    return wf