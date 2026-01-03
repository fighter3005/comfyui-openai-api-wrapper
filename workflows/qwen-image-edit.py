import copy

MODEL_ID = "qwen-image-edit"

# Qwen Image Edit – 1 image
QWEN_IMAGE_EDIT = {
  "3": { "inputs": { "seed": 0, "steps": 4, "cfg": 2.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1, "model": ["75", 0], "positive": ["111", 0], "negative": ["110", 0], "latent_image": ["88", 0] }, "class_type": "KSampler" },
  "8": { "inputs": { "samples": ["3", 0], "vae": ["39", 0] }, "class_type": "VAEDecode" },
  "38": { "inputs": { "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default" }, "class_type": "CLIPLoader" },
  "39": { "inputs": { "vae_name": "qwen_image_vae.safetensors" }, "class_type": "VAELoader" },
  "60": { "inputs": { "filename_prefix": "ComfyUI", "images": ["8", 0] }, "class_type": "SaveImage" },
  "66": { "inputs": { "shift": 3, "model": ["89", 0] }, "class_type": "ModelSamplingAuraFlow" },
  "75": { "inputs": { "strength": 1, "model": ["66", 0] }, "class_type": "CFGNorm" },
  "78": { "inputs": { "image": "" }, "class_type": "LoadImage" },
  "88": { "inputs": { "pixels": ["93", 0], "vae": ["39", 0] }, "class_type": "VAEEncode" },
  "89": { "inputs": { "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "strength_model": 1, "model": ["390", 0] }, "class_type": "LoraLoaderModelOnly" },
  "93": { "inputs": { "upscale_method": "lanczos", "megapixels": 1, "resolution_steps": 1, "image": ["78", 0] }, "class_type": "ImageScaleToTotalPixels" },
  "110": { "inputs": { "prompt": "", "clip": ["38", 0], "vae": ["39", 0], "image1": ["93", 0] }, "class_type": "TextEncodeQwenImageEditPlus" },
  "111": { "inputs": { "prompt": "", "clip": ["38", 0], "vae": ["39", 0], "image1": ["93", 0] }, "class_type": "TextEncodeQwenImageEditPlus" },
  "112": { "inputs": { "width": 1024, "height": 1024, "batch_size": 1 }, "class_type": "EmptySD3LatentImage" },
  "390": { "inputs": { "unet_name": "qwen-image-edit-2511-Q4_K_M.gguf" }, "class_type": "UnetLoaderGGUF" }
}

# Qwen Image Edit – 2 images
QWEN_IMAGE_EDIT_2IMG = {
  **copy.deepcopy(QWEN_IMAGE_EDIT),
  "106": { "inputs": { "image": "" }, "class_type": "LoadImage" },
  "110": { "inputs": { "prompt": "", "clip": ["38", 0], "vae": ["39", 0], "image1": ["93", 0], "image2": ["393", 0] }, "class_type": "TextEncodeQwenImageEditPlus" },
  "111": { "inputs": { "prompt": "", "clip": ["38", 0], "vae": ["39", 0], "image1": ["93", 0], "image2": ["393", 0] }, "class_type": "TextEncodeQwenImageEditPlus" },
  "393": { "inputs": { "upscale_method": "lanczos", "megapixels": 1, "resolution_steps": 1, "image": ["106", 0] }, "class_type": "ImageScaleToTotalPixels" }
}

# Qwen Image Edit – 3 images
QWEN_IMAGE_EDIT_3IMG = {
  **copy.deepcopy(QWEN_IMAGE_EDIT_2IMG),
  "108": { "inputs": { "image": "" }, "class_type": "LoadImage" },
  "110": { "inputs": { "prompt": "", "clip": ["38", 0], "vae": ["39", 0], "image1": ["93", 0], "image2": ["393", 0], "image3": ["394", 0] }, "class_type": "TextEncodeQwenImageEditPlus" },
  "111": { "inputs": { "prompt": "", "clip": ["38", 0], "vae": ["39", 0], "image1": ["93", 0], "image2": ["393", 0], "image3": ["394", 0] }, "class_type": "TextEncodeQwenImageEditPlus" },
  "394": { "inputs": { "upscale_method": "lanczos", "megapixels": 1, "resolution_steps": 1, "image": ["108", 0] }, "class_type": "ImageScaleToTotalPixels" }
}

def get_workflow(
    prompt: str = "",
    images=None,              # list of uploaded ComfyUI image names
    seed: int = 0,
    width: int = 1024,        # not used by this workflow (logged/ignored if passed)
    height: int = 1024,       # not used by this workflow (logged/ignored if passed)
    **kwargs
):
    if images is None:
        images = []

    # log/ignore unsupported params
    ignored = {k: v for k, v in kwargs.items() if v not in (None, "", [], {})}
    if width or height:
        # This workflow uses megapixels scaling, not explicit w/h nodes.
        ignored.update({"width": width, "height": height})
    if ignored:
        print(f"[{MODEL_ID}] Ignoring unsupported parameters: {list(ignored.keys())}")

    if len(images) < 1:
        raise ValueError("qwen-image-edit requires at least 1 image")

    if len(images) > 3:
        print(f"[{MODEL_ID}] Received {len(images)} images; only 3 supported. Extra images will be ignored.")
        images = images[:3]

    if len(images) == 1:
        wf = copy.deepcopy(QWEN_IMAGE_EDIT)
    elif len(images) == 2:
        wf = copy.deepcopy(QWEN_IMAGE_EDIT_2IMG)
    else:
        wf = copy.deepcopy(QWEN_IMAGE_EDIT_3IMG)

    wf["111"]["inputs"]["prompt"] = prompt or ""
    wf["3"]["inputs"]["seed"] = int(seed or 0)
    wf["93"]["inputs"]["megapixels"] = 1.0

    # Primary image
    wf["78"]["inputs"]["image"] = images[0]

    # Optional refs
    if len(images) >= 2:
        wf["106"]["inputs"]["image"] = images[1]
    if len(images) >= 3:
        wf["108"]["inputs"]["image"] = images[2]

    return wf