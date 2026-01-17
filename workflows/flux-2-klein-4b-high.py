import copy

MODEL_ID = "flux-2-klein-4b"

# --------------------------
# Text-to-Image (Generation)
# --------------------------
FLUX_KLEIN_4B_GEN = {
  "76": {
    "inputs": {
      "value": "Upper body portrait of woman walking down a street at sunset, wearing a dark clean jeans on her thin waist, high black boots, a black leather jacket and a white t-shirt. High fashion, cinematic, professional photograph."
    },
    "class_type": "PrimitiveStringMultiline",
    "_meta": {"title": "Prompt"}
  },
  "80": {
    "inputs": {
      "filename_prefix": "Flux2-Klein",
      "images": ["75:65", 0]
    },
    "class_type": "SaveImage",
    "_meta": {"title": "Save Image"}
  },
  "75:61": {
    "inputs": {"sampler_name": "euler"},
    "class_type": "KSamplerSelect",
    "_meta": {"title": "KSamplerSelect"}
  },
  "75:62": {
    "inputs": {
      "steps": 50,
      "width": ["75:68", 0],
      "height": ["75:69", 0]
    },
    "class_type": "Flux2Scheduler",
    "_meta": {"title": "Flux2Scheduler"}
  },
  "75:63": {
    "inputs": {
      "cfg": 5,
      "model": ["75:70", 0],
      "positive": ["75:74", 0],
      "negative": ["75:67", 0]
    },
    "class_type": "CFGGuider",
    "_meta": {"title": "CFGGuider"}
  },
  "75:64": {
    "inputs": {
      "noise": ["75:73", 0],
      "guider": ["75:63", 0],
      "sampler": ["75:61", 0],
      "sigmas": ["75:62", 0],
      "latent_image": ["75:66", 0]
    },
    "class_type": "SamplerCustomAdvanced",
    "_meta": {"title": "SamplerCustomAdvanced"}
  },
  "75:65": {
    "inputs": {
      "samples": ["75:64", 0],
      "vae": ["75:72", 0]
    },
    "class_type": "VAEDecode",
    "_meta": {"title": "VAE Decode"}
  },
  "75:66": {
    "inputs": {
      "width": ["75:68", 0],
      "height": ["75:69", 0],
      "batch_size": 1
    },
    "class_type": "EmptyFlux2LatentImage",
    "_meta": {"title": "Empty Flux 2 Latent"}
  },
  "75:67": {
    "inputs": {
      "text": "",
      "clip": ["75:71", 0]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {"title": "CLIP Text Encode (Negative Prompt)"}
  },
  "75:68": {
    "inputs": {"value": 1360},
    "class_type": "PrimitiveInt",
    "_meta": {"title": "Width"}
  },
  "75:69": {
    "inputs": {"value": 1360},
    "class_type": "PrimitiveInt",
    "_meta": {"title": "Height"}
  },
  "75:73": {
    "inputs": {"noise_seed": 69},
    "class_type": "RandomNoise",
    "_meta": {"title": "RandomNoise"}
  },
  "75:72": {
    "inputs": {"vae_name": "flux2-vae.safetensors"},
    "class_type": "VAELoader",
    "_meta": {"title": "Load VAE"}
  },
  "75:74": {
    "inputs": {
      "text": ["76", 0],
      "clip": ["75:71", 0]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {"title": "CLIP Text Encode (Positive Prompt)"}
  },
  "75:71": {
    "inputs": {
      "clip_name": "qwen_3_4b.safetensors",
      "type": "flux2",
      "device": "default"
    },
    "class_type": "CLIPLoader",
    "_meta": {"title": "Load CLIP"}
  },
  "75:70": {
    "inputs": {
      "unet_name": "flux-2-klein-base-4b-fp8.safetensors",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader",
    "_meta": {"title": "Load Diffusion Model"}
  }
}

# --------------------------
# Image Edit (1 image)
# --------------------------
FLUX_KLEIN_4B_EDIT_1IMG = {
  "76": {
    "inputs": {"image": "pasted/image.png"},
    "class_type": "LoadImage",
    "_meta": {"title": "Load Image"}
  },
  "94": {
    "inputs": {
      "filename_prefix": "Flux2-Klein-4b-base",
      "images": ["92:65", 0]
    },
    "class_type": "SaveImage",
    "_meta": {"title": "Save Image"}
  },
  "92:61": {
    "inputs": {"sampler_name": "euler"},
    "class_type": "KSamplerSelect",
    "_meta": {"title": "KSamplerSelect"}
  },
  "92:64": {
    "inputs": {
      "noise": ["92:73", 0],
      "guider": ["92:63", 0],
      "sampler": ["92:61", 0],
      "sigmas": ["92:62", 0],
      "latent_image": ["92:66", 0]
    },
    "class_type": "SamplerCustomAdvanced",
    "_meta": {"title": "SamplerCustomAdvanced"}
  },
  "92:65": {
    "inputs": {
      "samples": ["92:64", 0],
      "vae": ["92:72", 0]
    },
    "class_type": "VAEDecode",
    "_meta": {"title": "VAE Decode"}
  },
  "92:73": {
    "inputs": {"noise_seed": 432262096973497},
    "class_type": "RandomNoise",
    "_meta": {"title": "RandomNoise"}
  },
  "92:70": {
    "inputs": {
      "unet_name": "flux-2-klein-base-4b-fp8.safetensors",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader",
    "_meta": {"title": "Load Diffusion Model"}
  },
  "92:71": {
    "inputs": {
      "clip_name": "qwen_3_4b.safetensors",
      "type": "flux2",
      "device": "default"
    },
    "class_type": "CLIPLoader",
    "_meta": {"title": "Load CLIP"}
  },
  "92:63": {
    "inputs": {
      "cfg": 5,
      "model": ["92:70", 0],
      "positive": ["92:79:77", 0],
      "negative": ["92:79:76", 0]
    },
    "class_type": "CFGGuider",
    "_meta": {"title": "CFGGuider"}
  },
  "92:62": {
    "inputs": {
      "steps": 50,
      "width": ["92:81", 0],
      "height": ["92:81", 1]
    },
    "class_type": "Flux2Scheduler",
    "_meta": {"title": "Flux2Scheduler"}
  },
  "92:72": {
    "inputs": {"vae_name": "flux2-vae.safetensors"},
    "class_type": "VAELoader",
    "_meta": {"title": "Load VAE"}
  },
  "92:74": {
    "inputs": {
      "text": "Make the Woman wear the BOSS t-shirt.",
      "clip": ["92:71", 0]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {"title": "CLIP Text Encode (Positive Prompt)"}
  },
  "92:87": {
    "inputs": {"text": "", "clip": ["92:71", 0]},
    "class_type": "CLIPTextEncode",
    "_meta": {"title": "CLIP Text Encode ( Negative Prompt)"}
  },
  "92:79:76": {
    "inputs": {
      "conditioning": ["92:87", 0],
      "latent": ["92:79:78", 0]
    },
    "class_type": "ReferenceLatent",
    "_meta": {"title": "ReferenceLatent"}
  },
  "92:79:78": {
    "inputs": {
      "pixels": ["92:80", 0],
      "vae": ["92:72", 0]
    },
    "class_type": "VAEEncode",
    "_meta": {"title": "VAE Encode"}
  },
  "92:79:77": {
    "inputs": {
      "conditioning": ["92:74", 0],
      "latent": ["92:79:78", 0]
    },
    "class_type": "ReferenceLatent",
    "_meta": {"title": "ReferenceLatent"}
  },
  "92:66": {
    "inputs": {
      "width": ["92:81", 0],
      "height": ["92:81", 1],
      "batch_size": 1
    },
    "class_type": "EmptyFlux2LatentImage",
    "_meta": {"title": "Empty Flux 2 Latent"}
  },
  "92:81": {
    "inputs": {"image": ["92:80", 0]},
    "class_type": "GetImageSize",
    "_meta": {"title": "Get Image Size"}
  },
  "92:80": {
    "inputs": {
      "upscale_method": "nearest-exact",
      "megapixels": 1,
      "resolution_steps": 1,
      "image": ["76", 0]
    },
    "class_type": "ImageScaleToTotalPixels",
    "_meta": {"title": "ImageScaleToTotalPixels"}
  }
}

# --------------------------
# Image Edit (2 images)
# Chain: primary (92:79) -> ref1 (92:89)
# CFGGuider uses 92:89 outputs
# --------------------------
FLUX_KLEIN_4B_EDIT_2IMG = {
  **copy.deepcopy(FLUX_KLEIN_4B_EDIT_1IMG),

  # second image
  "81": {
    "inputs": {"image": "pasted/image_ref.png"},
    "class_type": "LoadImage",
    "_meta": {"title": "Load Image"}
  },

  "92:88": {
    "inputs": {
      "upscale_method": "nearest-exact",
      "megapixels": 1,
      "resolution_steps": 1,
      "image": ["81", 0]
    },
    "class_type": "ImageScaleToTotalPixels",
    "_meta": {"title": "ImageScaleToTotalPixels"}
  },

  "92:89:78": {
    "inputs": {
      "pixels": ["92:88", 0],
      "vae": ["92:72", 0]
    },
    "class_type": "VAEEncode",
    "_meta": {"title": "VAE Encode"}
  },

  "92:89:76": {
    "inputs": {
      "conditioning": ["92:79:76", 0],
      "latent": ["92:89:78", 0]
    },
    "class_type": "ReferenceLatent",
    "_meta": {"title": "ReferenceLatent"}
  },

  "92:89:77": {
    "inputs": {
      "conditioning": ["92:79:77", 0],
      "latent": ["92:89:78", 0]
    },
    "class_type": "ReferenceLatent",
    "_meta": {"title": "ReferenceLatent"}
  },

  # override CFGGuider to use final conditioning from 2-image chain
  "92:63": {
    "inputs": {
      "cfg": 5,
      "model": ["92:70", 0],
      "positive": ["92:89:77", 0],
      "negative": ["92:89:76", 0]
    },
    "class_type": "CFGGuider",
    "_meta": {"title": "CFGGuider"}
  },
}

# --------------------------
# Image Edit (3 images)
# Chain: primary (92:79) -> ref1 (92:89) -> ref2 (92:84)
# CFGGuider uses 92:84 outputs
# --------------------------
FLUX_KLEIN_4B_EDIT_3IMG = {
  **copy.deepcopy(FLUX_KLEIN_4B_EDIT_2IMG),

  # third image
  "100": {
    "inputs": {"image": "pasted/image_ref2.png"},
    "class_type": "LoadImage",
    "_meta": {"title": "Load Image"}
  },

  "92:85": {
    "inputs": {
      "upscale_method": "nearest-exact",
      "megapixels": 1,
      "resolution_steps": 1,
      "image": ["100", 0]
    },
    "class_type": "ImageScaleToTotalPixels",
    "_meta": {"title": "ImageScaleToTotalPixels"}
  },

  "92:84:78": {
    "inputs": {
      "pixels": ["92:85", 0],
      "vae": ["92:72", 0]
    },
    "class_type": "VAEEncode",
    "_meta": {"title": "VAE Encode"}
  },

  "92:84:76": {
    "inputs": {
      "conditioning": ["92:89:76", 0],
      "latent": ["92:84:78", 0]
    },
    "class_type": "ReferenceLatent",
    "_meta": {"title": "ReferenceLatent"}
  },

  "92:84:77": {
    "inputs": {
      "conditioning": ["92:89:77", 0],
      "latent": ["92:84:78", 0]
    },
    "class_type": "ReferenceLatent",
    "_meta": {"title": "ReferenceLatent"}
  },

  # override CFGGuider to use final conditioning from 3-image chain
  "92:63": {
    "inputs": {
      "cfg": 5,
      "model": ["92:70", 0],
      "positive": ["92:84:77", 0],
      "negative": ["92:84:76", 0]
    },
    "class_type": "CFGGuider",
    "_meta": {"title": "CFGGuider"}
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
        wf = copy.deepcopy(FLUX_KLEIN_4B_GEN)
        wf["76"]["inputs"]["value"] = (prompt or "").rstrip() + "\n"
        wf["75:73"]["inputs"]["noise_seed"] = int(seed or 0)
        wf["75:68"]["inputs"]["value"] = int(width)
        wf["75:69"]["inputs"]["value"] = int(height)
        return wf

    # edit mode
    if len(images) < 1:
        raise ValueError("flux-2-klein-4b edit requires at least 1 image")
    if len(images) > 3:
        print(f"[{MODEL_ID}] Received {len(images)} images; only 3 supported. Extra images will be ignored.")
        images = images[:3]

    if len(images) == 1:
        wf = copy.deepcopy(FLUX_KLEIN_4B_EDIT_1IMG)
        wf["76"]["inputs"]["image"] = images[0]
    elif len(images) == 2:
        wf = copy.deepcopy(FLUX_KLEIN_4B_EDIT_2IMG)
        wf["76"]["inputs"]["image"] = images[0]
        wf["81"]["inputs"]["image"] = images[1]
    else:
        wf = copy.deepcopy(FLUX_KLEIN_4B_EDIT_3IMG)
        wf["76"]["inputs"]["image"] = images[0]
        wf["81"]["inputs"]["image"] = images[1]
        wf["100"]["inputs"]["image"] = images[2]

    # prompt + seed
    wf["92:74"]["inputs"]["text"] = prompt or ""
    wf["92:73"]["inputs"]["noise_seed"] = int(seed or 0)

    # Note: edit workflows derive width/height from image via GetImageSize.
    # The wrapper passes width/height anyway; we intentionally ignore them here.

    return wf