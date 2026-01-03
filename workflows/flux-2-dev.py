import copy

MODEL_ID = "flux-2-dev"

FLUX_2_GEN = {
  "6": { "inputs": { "text": "", "clip": ["38", 0] }, "class_type": "CLIPTextEncode" },
  "8": { "inputs": { "samples": ["13", 0], "vae": ["10", 0] }, "class_type": "VAEDecode" },
  "9": { "inputs": { "filename_prefix": "Flux2", "images": ["8", 0] }, "class_type": "SaveImage" },
  "10": { "inputs": { "vae_name": "flux2-vae.safetensors" }, "class_type": "VAELoader" },
  "13": { "inputs": { "noise": ["25", 0], "guider": ["22", 0], "sampler": ["16", 0], "sigmas": ["48", 0], "latent_image": ["47", 0] }, "class_type": "SamplerCustomAdvanced" },
  "16": { "inputs": { "sampler_name": "euler" }, "class_type": "KSamplerSelect" },
  "22": { "inputs": { "model": ["67", 0], "conditioning": ["26", 0] }, "class_type": "BasicGuider" },
  "25": { "inputs": { "noise_seed": 0 }, "class_type": "RandomNoise" },
  "26": { "inputs": { "guidance": 4, "conditioning": ["6", 0] }, "class_type": "FluxGuidance" },
  "38": { "inputs": { "clip_name": "mistral_3_small_flux2_fp8.safetensors", "type": "flux2", "device": "default" }, "class_type": "CLIPLoader" },
  "47": { "inputs": { "width": 512, "height": 512, "batch_size": 1 }, "class_type": "EmptyFlux2LatentImage" },
  "48": { "inputs": { "steps": 20, "width": 512, "height": 512 }, "class_type": "Flux2Scheduler" },
  "67": { "inputs": { "unet_name": "flux2_dev_Q4_K_M.gguf" }, "class_type": "UnetLoaderGGUF" }
}

FLUX_2_EDIT_1IMG = {
  "6": { "inputs": { "text": "", "clip": ["38", 0] }, "class_type": "CLIPTextEncode" },
  "8": { "inputs": { "samples": ["13", 0], "vae": ["10", 0] }, "class_type": "VAEDecode" },
  "9": { "inputs": { "filename_prefix": "Flux2", "images": ["8", 0] }, "class_type": "SaveImage" },
  "10": { "inputs": { "vae_name": "flux2-vae.safetensors" }, "class_type": "VAELoader" },
  "13": { "inputs": { "noise": ["25", 0], "guider": ["22", 0], "sampler": ["16", 0], "sigmas": ["48", 0], "latent_image": ["47", 0] }, "class_type": "SamplerCustomAdvanced" },
  "16": { "inputs": { "sampler_name": "euler" }, "class_type": "KSamplerSelect" },
  "22": { "inputs": { "model": ["67", 0], "conditioning": ["43", 0] }, "class_type": "BasicGuider" },
  "25": { "inputs": { "noise_seed": 0 }, "class_type": "RandomNoise" },
  "26": { "inputs": { "guidance": 4, "conditioning": ["6", 0] }, "class_type": "FluxGuidance" },
  "38": { "inputs": { "clip_name": "mistral_3_small_flux2_fp8.safetensors", "type": "flux2", "device": "default" }, "class_type": "CLIPLoader" },
  "43": { "inputs": { "conditioning": ["26", 0], "latent": ["44", 0] }, "class_type": "ReferenceLatent" },
  "44": { "inputs": { "pixels": ["45", 0], "vae": ["10", 0] }, "class_type": "VAEEncode" },
  "45": { "inputs": { "upscale_method": "lanczos", "megapixels": 2, "resolution_steps": 1, "image": ["46", 0] }, "class_type": "ImageScaleToTotalPixels" },
  "46": { "inputs": { "image": "" }, "class_type": "LoadImage" },
  "47": { "inputs": { "width": 1088, "height": 1920, "batch_size": 1 }, "class_type": "EmptyFlux2LatentImage" },
  "48": { "inputs": { "steps": 20, "width": 1088, "height": 1920 }, "class_type": "Flux2Scheduler" },
  "67": { "inputs": { "unet_name": "flux2_dev_Q4_K_M.gguf" }, "class_type": "UnetLoaderGGUF" }
}

FLUX_2_EDIT_2IMG = {
  "6": { "inputs": { "text": "", "clip": ["38", 0] }, "class_type": "CLIPTextEncode" },
  "8": { "inputs": { "samples": ["13", 0], "vae": ["10", 0] }, "class_type": "VAEDecode" },
  "9": { "inputs": { "filename_prefix": "Flux2", "images": ["8", 0] }, "class_type": "SaveImage" },
  "10": { "inputs": { "vae_name": "flux2-vae.safetensors" }, "class_type": "VAELoader" },
  "13": { "inputs": { "noise": ["25", 0], "guider": ["22", 0], "sampler": ["16", 0], "sigmas": ["48", 0], "latent_image": ["47", 0] }, "class_type": "SamplerCustomAdvanced" },
  "16": { "inputs": { "sampler_name": "euler" }, "class_type": "KSamplerSelect" },
  "22": { "inputs": { "model": ["67", 0], "conditioning": ["39", 0] }, "class_type": "BasicGuider" },
  "25": { "inputs": { "noise_seed": 0 }, "class_type": "RandomNoise" },
  "26": { "inputs": { "guidance": 4, "conditioning": ["6", 0] }, "class_type": "FluxGuidance" },
  "38": { "inputs": { "clip_name": "mistral_3_small_flux2_fp8.safetensors", "type": "flux2", "device": "default" }, "class_type": "CLIPLoader" },
  "39": { "inputs": { "conditioning": ["43", 0], "latent": ["40", 0] }, "class_type": "ReferenceLatent" },
  "40": { "inputs": { "pixels": ["41", 0], "vae": ["10", 0] }, "class_type": "VAEEncode" },
  "41": { "inputs": { "upscale_method": "lanczos", "megapixels": 2, "resolution_steps": 1, "image": ["42", 0] }, "class_type": "ImageScaleToTotalPixels" },
  "42": { "inputs": { "image": "" }, "class_type": "LoadImage" },
  "43": { "inputs": { "conditioning": ["26", 0], "latent": ["44", 0] }, "class_type": "ReferenceLatent" },
  "44": { "inputs": { "pixels": ["45", 0], "vae": ["10", 0] }, "class_type": "VAEEncode" },
  "45": { "inputs": { "upscale_method": "lanczos", "megapixels": 2, "resolution_steps": 1, "image": ["46", 0] }, "class_type": "ImageScaleToTotalPixels" },
  "46": { "inputs": { "image": "" }, "class_type": "LoadImage" },
  "47": { "inputs": { "width": 512, "height": 512, "batch_size": 1 }, "class_type": "EmptyFlux2LatentImage" },
  "48": { "inputs": { "steps": 20, "width": 512, "height": 512 }, "class_type": "Flux2Scheduler" },
  "67": { "inputs": { "unet_name": "flux2_dev_Q4_K_M.gguf" }, "class_type": "UnetLoaderGGUF" }
}

FLUX_2_EDIT_3IMG = {
  **copy.deepcopy(FLUX_2_EDIT_2IMG),
  "68": { "inputs": { "image": "" }, "class_type": "LoadImage" },
  "69": { "inputs": { "upscale_method": "lanczos", "megapixels": 2, "resolution_steps": 1, "image": ["68", 0] }, "class_type": "ImageScaleToTotalPixels" },
  "70": { "inputs": { "pixels": ["69", 0], "vae": ["10", 0] }, "class_type": "VAEEncode" },
  "71": { "inputs": { "conditioning": ["39", 0], "latent": ["70", 0] }, "class_type": "ReferenceLatent" },
  "22": { "inputs": { "model": ["67", 0], "conditioning": ["71", 0] }, "class_type": "BasicGuider" }
}

def get_workflow(
    mode: str = "gen",
    prompt: str = "",
    width: int = 1024,
    height: int = 1024,
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
        wf = copy.deepcopy(FLUX_2_GEN)
        wf["6"]["inputs"]["text"] = prompt or ""
        wf["25"]["inputs"]["noise_seed"] = int(seed or 0)
        wf["47"]["inputs"]["width"] = int(width)
        wf["47"]["inputs"]["height"] = int(height)
        wf["48"]["inputs"]["width"] = int(width)
        wf["48"]["inputs"]["height"] = int(height)
        return wf

    # edit mode
    if len(images) < 1:
        raise ValueError("flux-2-dev edit requires at least 1 image")

    if len(images) > 3:
        print(f"[{MODEL_ID}] Received {len(images)} images; only 3 supported. Extra images will be ignored.")
        images = images[:3]

    if len(images) == 1:
        wf = copy.deepcopy(FLUX_2_EDIT_1IMG)
        wf["46"]["inputs"]["image"] = images[0]
    elif len(images) == 2:
        wf = copy.deepcopy(FLUX_2_EDIT_2IMG)
        wf["42"]["inputs"]["image"] = images[0]
        wf["46"]["inputs"]["image"] = images[1]
    else:
        wf = copy.deepcopy(FLUX_2_EDIT_3IMG)
        wf["42"]["inputs"]["image"] = images[0]
        wf["46"]["inputs"]["image"] = images[1]
        wf["68"]["inputs"]["image"] = images[2]

    wf["6"]["inputs"]["text"] = prompt or ""
    wf["25"]["inputs"]["noise_seed"] = int(seed or 0)

    # These workflows include explicit latent size nodes too
    wf["47"]["inputs"]["width"] = int(width)
    wf["47"]["inputs"]["height"] = int(height)
    wf["48"]["inputs"]["width"] = int(width)
    wf["48"]["inputs"]["height"] = int(height)

    return wf