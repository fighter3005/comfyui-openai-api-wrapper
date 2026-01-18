import copy

MODEL_ID = "flux-2-dev-turbo"

# --------------------------
# Text-to-Image (Generation)
# --------------------------
# Derived from your provided workflow by removing the image-reference chain
# and setting BasicGuider conditioning to the prompt guidance (node 26).
FLUX2_TURBO_GEN = {
  "6": {
    "inputs": {"text": "", "clip": ["38", 0]},
    "class_type": "CLIPTextEncode",
    "_meta": {"title": "CLIP Text Encode (Positive Prompt)"}
  },
  "8": {
    "inputs": {"samples": ["13", 0], "vae": ["10", 0]},
    "class_type": "VAEDecode",
    "_meta": {"title": "VAE Decode"}
  },
  "9": {
    "inputs": {"filename_prefix": "Flux2", "images": ["8", 0]},
    "class_type": "SaveImage",
    "_meta": {"title": "Save Image"}
  },
  "10": {
    "inputs": {"vae_name": "flux2-vae.safetensors"},
    "class_type": "VAELoader",
    "_meta": {"title": "Load VAE"}
  },
  "12": {
    "inputs": {"unet_name": "flux2_dev_fp8mixed.safetensors", "weight_dtype": "default"},
    "class_type": "UNETLoader",
    "_meta": {"title": "Load Diffusion Model"}
  },
  "13": {
    "inputs": {
      "noise": ["25", 0],
      "guider": ["22", 0],
      "sampler": ["16", 0],
      "sigmas": ["48", 0],
      "latent_image": ["47", 0]
    },
    "class_type": "SamplerCustomAdvanced",
    "_meta": {"title": "SamplerCustomAdvanced"}
  },
  "16": {
    "inputs": {"sampler_name": "euler"},
    "class_type": "KSamplerSelect",
    "_meta": {"title": "KSamplerSelect"}
  },
  "22": {
    "inputs": {
      "model": ["68", 0],
      "conditioning": ["26", 0]
    },
    "class_type": "BasicGuider",
    "_meta": {"title": "BasicGuider"}
  },
  "25": {
    "inputs": {"noise_seed": 0},
    "class_type": "RandomNoise",
    "_meta": {"title": "RandomNoise"}
  },
  "26": {
    "inputs": {"guidance": 4, "conditioning": ["6", 0]},
    "class_type": "FluxGuidance",
    "_meta": {"title": "FluxGuidance"}
  },
  "38": {
    "inputs": {"clip_name": "mistral_3_small_flux2_fp8.safetensors", "type": "flux2", "device": "default"},
    "class_type": "CLIPLoader",
    "_meta": {"title": "Load CLIP"}
  },
  "47": {
    "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    "class_type": "EmptyFlux2LatentImage",
    "_meta": {"title": "Empty Flux 2 Latent"}
  },
  "48": {
    "inputs": {"steps": 8, "width": 1024, "height": 1024},
    "class_type": "Flux2Scheduler",
    "_meta": {"title": "Flux2Scheduler"}
  },
  "68": {
    "inputs": {
      "lora_name": "Flux_2-Turbo-LoRA_comfyui.safetensors",
      "strength_model": 1,
      "model": ["12", 0]
    },
    "class_type": "LoraLoaderModelOnly",
    "_meta": {"title": "LoraLoaderModelOnly"}
  }
}

# --------------------------
# Image Edit (1 image)
# --------------------------
# Uses the "primary" image chain (node 46 -> 45 -> 44) and one ReferenceLatent (node 43).
FLUX2_TURBO_EDIT_1IMG = {
  "6": {
    "inputs": {"text": "", "clip": ["38", 0]},
    "class_type": "CLIPTextEncode",
    "_meta": {"title": "CLIP Text Encode (Positive Prompt)"}
  },
  "8": {
    "inputs": {"samples": ["13", 0], "vae": ["10", 0]},
    "class_type": "VAEDecode",
    "_meta": {"title": "VAE Decode"}
  },
  "9": {
    "inputs": {"filename_prefix": "Flux2", "images": ["8", 0]},
    "class_type": "SaveImage",
    "_meta": {"title": "Save Image"}
  },
  "10": {
    "inputs": {"vae_name": "flux2-vae.safetensors"},
    "class_type": "VAELoader",
    "_meta": {"title": "Load VAE"}
  },
  "12": {
    "inputs": {"unet_name": "flux2_dev_fp8mixed.safetensors", "weight_dtype": "default"},
    "class_type": "UNETLoader",
    "_meta": {"title": "Load Diffusion Model"}
  },
  "13": {
    "inputs": {
      "noise": ["25", 0],
      "guider": ["22", 0],
      "sampler": ["16", 0],
      "sigmas": ["48", 0],
      "latent_image": ["47", 0]
    },
    "class_type": "SamplerCustomAdvanced",
    "_meta": {"title": "SamplerCustomAdvanced"}
  },
  "16": {
    "inputs": {"sampler_name": "euler"},
    "class_type": "KSamplerSelect",
    "_meta": {"title": "KSamplerSelect"}
  },
  "22": {
    "inputs": {"model": ["68", 0], "conditioning": ["43", 0]},
    "class_type": "BasicGuider",
    "_meta": {"title": "BasicGuider"}
  },
  "25": {
    "inputs": {"noise_seed": 0},
    "class_type": "RandomNoise",
    "_meta": {"title": "RandomNoise"}
  },
  "26": {
    "inputs": {"guidance": 4, "conditioning": ["6", 0]},
    "class_type": "FluxGuidance",
    "_meta": {"title": "FluxGuidance"}
  },
  "38": {
    "inputs": {"clip_name": "mistral_3_small_flux2_fp8.safetensors", "type": "flux2", "device": "default"},
    "class_type": "CLIPLoader",
    "_meta": {"title": "Load CLIP"}
  },
  "43": {
    "inputs": {"conditioning": ["26", 0], "latent": ["44", 0]},
    "class_type": "ReferenceLatent",
    "_meta": {"title": "ReferenceLatent"}
  },
  "44": {
    "inputs": {"pixels": ["45", 0], "vae": ["10", 0]},
    "class_type": "VAEEncode",
    "_meta": {"title": "VAE Encode"}
  },
  "45": {
    "inputs": {
      "upscale_method": "lanczos",
      "megapixels": 2,
      "resolution_steps": 1,
      "image": ["46", 0]
    },
    "class_type": "ImageScaleToTotalPixels",
    "_meta": {"title": "ImageScaleToTotalPixels"}
  },
  "46": {
    "inputs": {"image": ""},
    "class_type": "LoadImage",
    "_meta": {"title": "Load Image"}
  },
  "47": {
    "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    "class_type": "EmptyFlux2LatentImage",
    "_meta": {"title": "Empty Flux 2 Latent"}
  },
  "48": {
    "inputs": {"steps": 8, "width": 1024, "height": 1024},
    "class_type": "Flux2Scheduler",
    "_meta": {"title": "Flux2Scheduler"}
  },
  "68": {
    "inputs": {
      "lora_name": "Flux_2-Turbo-LoRA_comfyui.safetensors",
      "strength_model": 1,
      "model": ["12", 0]
    },
    "class_type": "LoraLoaderModelOnly",
    "_meta": {"title": "LoraLoaderModelOnly"}
  }
}

# --------------------------
# Image Edit (2 images)
# --------------------------
# This is your provided workflow (as-is).
FLUX2_TURBO_EDIT_2IMG = {
  "6": {
    "inputs": {"text": "", "clip": ["38", 0]},
    "class_type": "CLIPTextEncode",
    "_meta": {"title": "CLIP Text Encode (Positive Prompt)"}
  },
  "8": {
    "inputs": {"samples": ["13", 0], "vae": ["10", 0]},
    "class_type": "VAEDecode",
    "_meta": {"title": "VAE Decode"}
  },
  "9": {
    "inputs": {"filename_prefix": "Flux2", "images": ["8", 0]},
    "class_type": "SaveImage",
    "_meta": {"title": "Save Image"}
  },
  "10": {
    "inputs": {"vae_name": "flux2-vae.safetensors"},
    "class_type": "VAELoader",
    "_meta": {"title": "Load VAE"}
  },
  "12": {
    "inputs": {"unet_name": "flux2_dev_fp8mixed.safetensors", "weight_dtype": "default"},
    "class_type": "UNETLoader",
    "_meta": {"title": "Load Diffusion Model"}
  },
  "13": {
    "inputs": {
      "noise": ["25", 0],
      "guider": ["22", 0],
      "sampler": ["16", 0],
      "sigmas": ["48", 0],
      "latent_image": ["47", 0]
    },
    "class_type": "SamplerCustomAdvanced",
    "_meta": {"title": "SamplerCustomAdvanced"}
  },
  "16": {
    "inputs": {"sampler_name": "euler"},
    "class_type": "KSamplerSelect",
    "_meta": {"title": "KSamplerSelect"}
  },
  "22": {
    "inputs": {"model": ["68", 0], "conditioning": ["39", 0]},
    "class_type": "BasicGuider",
    "_meta": {"title": "BasicGuider"}
  },
  "25": {
    "inputs": {"noise_seed": 0},
    "class_type": "RandomNoise",
    "_meta": {"title": "RandomNoise"}
  },
  "26": {
    "inputs": {"guidance": 4, "conditioning": ["6", 0]},
    "class_type": "FluxGuidance",
    "_meta": {"title": "FluxGuidance"}
  },
  "38": {
    "inputs": {"clip_name": "mistral_3_small_flux2_fp8.safetensors", "type": "flux2", "device": "default"},
    "class_type": "CLIPLoader",
    "_meta": {"title": "Load CLIP"}
  },
  "39": {
    "inputs": {"conditioning": ["43", 0], "latent": ["40", 0]},
    "class_type": "ReferenceLatent",
    "_meta": {"title": "ReferenceLatent"}
  },
  "40": {
    "inputs": {"pixels": ["41", 0], "vae": ["10", 0]},
    "class_type": "VAEEncode",
    "_meta": {"title": "VAE Encode"}
  },
  "41": {
    "inputs": {
      "upscale_method": "lanczos",
      "megapixels": 1,
      "resolution_steps": 1,
      "image": ["42", 0]
    },
    "class_type": "ImageScaleToTotalPixels",
    "_meta": {"title": "ImageScaleToTotalPixels"}
  },
  "42": {
    "inputs": {"image": ""},
    "class_type": "LoadImage",
    "_meta": {"title": "Load Image"}
  },
  "43": {
    "inputs": {"conditioning": ["26", 0], "latent": ["44", 0]},
    "class_type": "ReferenceLatent",
    "_meta": {"title": "ReferenceLatent"}
  },
  "44": {
    "inputs": {"pixels": ["45", 0], "vae": ["10", 0]},
    "class_type": "VAEEncode",
    "_meta": {"title": "VAE Encode"}
  },
  "45": {
    "inputs": {
      "upscale_method": "lanczos",
      "megapixels": 2,
      "resolution_steps": 1,
      "image": ["46", 0]
    },
    "class_type": "ImageScaleToTotalPixels",
    "_meta": {"title": "ImageScaleToTotalPixels"}
  },
  "46": {
    "inputs": {"image": ""},
    "class_type": "LoadImage",
    "_meta": {"title": "Load Image"}
  },
  "47": {
    "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    "class_type": "EmptyFlux2LatentImage",
    "_meta": {"title": "Empty Flux 2 Latent"}
  },
  "48": {
    "inputs": {"steps": 8, "width": 1024, "height": 1024},
    "class_type": "Flux2Scheduler",
    "_meta": {"title": "Flux2Scheduler"}
  },
  "68": {
    "inputs": {
      "lora_name": "Flux_2-Turbo-LoRA_comfyui.safetensors",
      "strength_model": 1,
      "model": ["12", 0]
    },
    "class_type": "LoraLoaderModelOnly",
    "_meta": {"title": "LoraLoaderModelOnly"}
  }
}

# --------------------------
# Image Edit (3 images)
# --------------------------
# Adds a third reference image chained on top of the 2-image conditioning result.
FLUX2_TURBO_EDIT_3IMG = {
  **copy.deepcopy(FLUX2_TURBO_EDIT_2IMG),

  # third image load + scale + encode
  "69": {
    "inputs": {"image": ""},
    "class_type": "LoadImage",
    "_meta": {"title": "Load Image (3rd)"}
  },
  "70": {
    "inputs": {
      "upscale_method": "lanczos",
      "megapixels": 1,
      "resolution_steps": 1,
      "image": ["69", 0]
    },
    "class_type": "ImageScaleToTotalPixels",
    "_meta": {"title": "ImageScaleToTotalPixels (3rd)"}
  },
  "71": {
    "inputs": {"pixels": ["70", 0], "vae": ["10", 0]},
    "class_type": "VAEEncode",
    "_meta": {"title": "VAE Encode (3rd)"}
  },

  # chain conditioning from node 39 (2-image result) with latent from third image
  "72": {
    "inputs": {"conditioning": ["39", 0], "latent": ["71", 0]},
    "class_type": "ReferenceLatent",
    "_meta": {"title": "ReferenceLatent (3rd)"}
  },

  # override BasicGuider to use the final (3-image) conditioning
  "22": {
    "inputs": {"model": ["68", 0], "conditioning": ["72", 0]},
    "class_type": "BasicGuider",
    "_meta": {"title": "BasicGuider"}
  },
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
        wf = copy.deepcopy(FLUX2_TURBO_GEN)
        wf["6"]["inputs"]["text"] = prompt or ""
        wf["25"]["inputs"]["noise_seed"] = int(seed or 0)
        wf["47"]["inputs"]["width"] = int(width)
        wf["47"]["inputs"]["height"] = int(height)
        wf["48"]["inputs"]["width"] = int(width)
        wf["48"]["inputs"]["height"] = int(height)
        return wf

    # edit mode
    if len(images) < 1:
        raise ValueError(f"{MODEL_ID} edit requires at least 1 image")
    if len(images) > 3:
        print(f"[{MODEL_ID}] Received {len(images)} images; only 3 supported. Extra images will be ignored.")
        images = images[:3]

    if len(images) == 1:
        wf = copy.deepcopy(FLUX2_TURBO_EDIT_1IMG)
        # primary image
        wf["46"]["inputs"]["image"] = images[0]
    elif len(images) == 2:
        wf = copy.deepcopy(FLUX2_TURBO_EDIT_2IMG)
        # mapping chosen to match your node intent:
        # - node 46 is the "main" image (megapixels=2)
        # - node 42 is the "reference" image (megapixels=1)
        wf["46"]["inputs"]["image"] = images[0]
        wf["42"]["inputs"]["image"] = images[1]
    else:
        wf = copy.deepcopy(FLUX2_TURBO_EDIT_3IMG)
        wf["46"]["inputs"]["image"] = images[0]
        wf["42"]["inputs"]["image"] = images[1]
        wf["69"]["inputs"]["image"] = images[2]

    # prompt + seed + output size
    wf["6"]["inputs"]["text"] = prompt or ""
    wf["25"]["inputs"]["noise_seed"] = int(seed or 0)

    wf["47"]["inputs"]["width"] = int(width)
    wf["47"]["inputs"]["height"] = int(height)
    wf["48"]["inputs"]["width"] = int(width)
    wf["48"]["inputs"]["height"] = int(height)

    return wf