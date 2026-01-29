import copy

MODEL_ID = "qwen image 2025"

# ============================================================
# TEXT-TO-IMAGE (GENERATION)
# - Uses qwen_image_2512_fp8_e4m3fn.safetensors (+ lightning LoRA)
# ============================================================
QWEN_IMAGE_2025_GEN = {
  "90": {
    "inputs": {
      "filename_prefix": "Qwen-Image-2512",
      "images": [
        "92:8",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "91": {
    "inputs": {
      "value": ""
    },
    "class_type": "PrimitiveStringMultiline",
    "_meta": {
      "title": "Prompt"
    }
  },
  "92:39": {
    "inputs": {
      "vae_name": "qwen_image_vae.safetensors"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "92:38": {
    "inputs": {
      "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
      "type": "qwen_image",
      "device": "default"
    },
    "class_type": "CLIPLoader",
    "_meta": {
      "title": "Load CLIP"
    }
  },
  "92:66": {
    "inputs": {
      "shift": 3.1000000000000005,
      "model": [
        "92:73",
        0
      ]
    },
    "class_type": "ModelSamplingAuraFlow",
    "_meta": {
      "title": "ModelSamplingAuraFlow"
    }
  },
  "92:6": {
    "inputs": {
      "text": [
        "91",
        0
      ],
      "clip": [
        "92:38",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Positive Prompt)"
    }
  },
  "92:8": {
    "inputs": {
      "samples": [
        "92:3",
        0
      ],
      "vae": [
        "92:39",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "92:58": {
    "inputs": {
      "width": 1328,
      "height": 1328,
      "batch_size": 1
    },
    "class_type": "EmptySD3LatentImage",
    "_meta": {
      "title": "EmptySD3LatentImage"
    }
  },
  "92:7": {
    "inputs": {
      "text": "低分辨率,低画质,肢体畸形,手指畸形,画面过饱和,蜡像感,人脸无细节,过度光滑,画面具有AI感。构图混乱。文字模糊,扭曲",
      "clip": [
        "92:38",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Negative Prompt)"
    }
  },
  "92:73": {
    "inputs": {
      "lora_name": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
      "strength_model": 1,
      "model": [
        "92:37",
        0
      ]
    },
    "class_type": "LoraLoaderModelOnly",
    "_meta": {
      "title": "LoraLoaderModelOnly"
    }
  },
  "92:37": {
    "inputs": {
      "unet_name": "qwen_image_2512_fp8_e4m3fn.safetensors",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader",
    "_meta": {
      "title": "Load Diffusion Model"
    }
  },
  "92:3": {
    "inputs": {
      "seed": 0,
      "steps": 4,
      "cfg": 1,
      "sampler_name": "euler",
      "scheduler": "simple",
      "denoise": 1,
      "model": [
        "92:66",
        0
      ],
      "positive": [
        "92:6",
        0
      ],
      "negative": [
        "92:7",
        0
      ],
      "latent_image": [
        "92:58",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  }
}

# ============================================================
# EDIT WORKFLOWS (1/2/3 IMAGES) + INFER-SIZE VARIANTS
# - Uses qwen-image-edit-2511-Q4_K_M.gguf
# - Infer-size: uses VAEEncode(latent_image) instead of EmptySD3LatentImage
# ============================================================

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
# Edit: 1 image (NORMAL)
# --------------------------
QWEN_IMAGE_2025_EDIT_1IMG = {
  "3": {
    "inputs": {
      "seed": 40709252805689,
      "steps": 20,
      "cfg": 2.5,
      "sampler_name": "euler",
      "scheduler": "simple",
      "denoise": 1,
      "model": [
        "75",
        0
      ],
      "positive": [
        "111",
        0
      ],
      "negative": [
        "110",
        0
      ],
      "latent_image": [
        "112",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "8": {
    "inputs": {
      "samples": [
        "3",
        0
      ],
      "vae": [
        "39",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "38": {
    "inputs": {
      "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
      "type": "qwen_image",
      "device": "default"
    },
    "class_type": "CLIPLoader",
    "_meta": {
      "title": "Load CLIP"
    }
  },
  "39": {
    "inputs": {
      "vae_name": "qwen_image_vae.safetensors"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "60": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "66": {
    "inputs": {
      "shift": 3,
      "model": [
        "390",
        0
      ]
    },
    "class_type": "ModelSamplingAuraFlow",
    "_meta": {
      "title": "ModelSamplingAuraFlow"
    }
  },
  "75": {
    "inputs": {
      "strength": 1,
      "model": [
        "66",
        0
      ]
    },
    "class_type": "CFGNorm",
    "_meta": {
      "title": "CFGNorm"
    }
  },
  "78": {
    "inputs": {
      "image": ""
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "93": {
    "inputs": {
      "upscale_method": "lanczos",
      "megapixels": 1.5,
      "resolution_steps": 1,
      "image": [
        "78",
        0
      ]
    },
    "class_type": "ImageScaleToTotalPixels",
    "_meta": {
      "title": "ImageScaleToTotalPixels"
    }
  },
  "110": {
    "inputs": {
      "prompt": "",
      "clip": [
        "38",
        0
      ],
      "vae": [
        "39",
        0
      ],
      "image1": [
        "93",
        0
      ]
    },
    "class_type": "TextEncodeQwenImageEditPlus",
    "_meta": {
      "title": "TextEncodeQwenImageEditPlus"
    }
  },
  "111": {
    "inputs": {
      "prompt": "",
      "clip": [
        "38",
        0
      ],
      "vae": [
        "39",
        0
      ],
      "image1": [
        "93",
        0
      ]
    },
    "class_type": "TextEncodeQwenImageEditPlus",
    "_meta": {
      "title": "TextEncodeQwenImageEditPlus"
    }
  },
  "112": {
    "inputs": {
      "width": 1232,
      "height": 1232,
      "batch_size": 1
    },
    "class_type": "EmptySD3LatentImage",
    "_meta": {
      "title": "EmptySD3LatentImage"
    }
  },
  "390": {
    "inputs": {
      "unet_name": "qwen-image-edit-2511-Q4_K_M.gguf"
    },
    "class_type": "UnetLoaderGGUF",
    "_meta": {
      "title": "Unet Loader (GGUF)"
    }
  }
}

# --------------------------
# Edit: 1 image (INFER SIZE)
# --------------------------
QWEN_IMAGE_2025_EDIT_1IMG_INFER_SIZE = {
  **copy.deepcopy(QWEN_IMAGE_2025_EDIT_1IMG),
  "88": {
    "inputs": {
      "pixels": [
        "93",
        0
      ],
      "vae": [
        "39",
        0
      ]
    },
    "class_type": "VAEEncode",
    "_meta": {
      "title": "VAE Encode"
    }
  }
}
QWEN_IMAGE_2025_EDIT_1IMG_INFER_SIZE["3"]["inputs"]["latent_image"] = ["88", 0]

# --------------------------
# Edit: 2 images (NORMAL)
# --------------------------
QWEN_IMAGE_2025_EDIT_2IMG = {
  **copy.deepcopy(QWEN_IMAGE_2025_EDIT_1IMG),
  "106": {
    "inputs": {
      "image": ""
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "393": {
    "inputs": {
      "upscale_method": "lanczos",
      "megapixels": 1.5,
      "resolution_steps": 1,
      "image": [
        "106",
        0
      ]
    },
    "class_type": "ImageScaleToTotalPixels",
    "_meta": {
      "title": "ImageScaleToTotalPixels"
    }
  }
}
QWEN_IMAGE_2025_EDIT_2IMG["111"]["inputs"]["image2"] = ["393", 0]

# --------------------------
# Edit: 2 images (INFER SIZE)
# --------------------------
QWEN_IMAGE_2025_EDIT_2IMG_INFER_SIZE = copy.deepcopy(QWEN_IMAGE_2025_EDIT_2IMG)
QWEN_IMAGE_2025_EDIT_2IMG_INFER_SIZE["88"] = {
  "inputs": {
    "pixels": [
      "93",
      0
    ],
    "vae": [
      "39",
      0
    ]
  },
  "class_type": "VAEEncode",
  "_meta": {
    "title": "VAE Encode"
  }
}
QWEN_IMAGE_2025_EDIT_2IMG_INFER_SIZE["3"]["inputs"]["latent_image"] = ["88", 0]

# --------------------------
# Edit: 3 images (NORMAL)
# --------------------------
QWEN_IMAGE_2025_EDIT_3IMG = {
  **copy.deepcopy(QWEN_IMAGE_2025_EDIT_2IMG),
  "108": {
    "inputs": {
      "image": ""
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "394": {
    "inputs": {
      "upscale_method": "lanczos",
      "megapixels": 1.5,
      "resolution_steps": 1,
      "image": [
        "108",
        0
      ]
    },
    "class_type": "ImageScaleToTotalPixels",
    "_meta": {
      "title": "ImageScaleToTotalPixels"
    }
  }
}
# For 3 images, the Qwen edit node supports image3 as well
QWEN_IMAGE_2025_EDIT_3IMG["111"]["inputs"]["image3"] = ["394", 0]
QWEN_IMAGE_2025_EDIT_3IMG["110"]["inputs"]["image3"] = ["394", 0]

# --------------------------
# Edit: 3 images (INFER SIZE)
# --------------------------
QWEN_IMAGE_2025_EDIT_3IMG_INFER_SIZE = copy.deepcopy(QWEN_IMAGE_2025_EDIT_3IMG)
QWEN_IMAGE_2025_EDIT_3IMG_INFER_SIZE["88"] = {
  "inputs": {
    "pixels": [
      "93",
      0
    ],
    "vae": [
      "39",
      0
    ]
  },
  "class_type": "VAEEncode",
  "_meta": {
    "title": "VAE Encode"
  }
}
QWEN_IMAGE_2025_EDIT_3IMG_INFER_SIZE["3"]["inputs"]["latent_image"] = ["88", 0]


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
        wf = copy.deepcopy(QWEN_IMAGE_2025_GEN)
        wf["91"]["inputs"]["value"] = (prompt or "").rstrip() + "\n"
        wf["92:3"]["inputs"]["seed"] = int(seed or 0)
        wf["92:58"]["inputs"]["width"] = int(width)
        wf["92:58"]["inputs"]["height"] = int(height)
        return wf

    # edit mode
    if len(images) < 1:
        raise ValueError(f"{MODEL_ID} edit requires at least 1 image")
    if len(images) > 3:
        print(f"[{MODEL_ID}] Received {len(images)} images; only 3 supported. Extra images will be ignored.")
        images = images[:3]

    use_infer = _should_use_infer_size(width, height)

    if len(images) == 1:
        wf = copy.deepcopy(QWEN_IMAGE_2025_EDIT_1IMG_INFER_SIZE if use_infer else QWEN_IMAGE_2025_EDIT_1IMG)
        wf["78"]["inputs"]["image"] = images[0]
    elif len(images) == 2:
        wf = copy.deepcopy(QWEN_IMAGE_2025_EDIT_2IMG_INFER_SIZE if use_infer else QWEN_IMAGE_2025_EDIT_2IMG)
        wf["78"]["inputs"]["image"] = images[0]
        wf["106"]["inputs"]["image"] = images[1]
    else:
        wf = copy.deepcopy(QWEN_IMAGE_2025_EDIT_3IMG_INFER_SIZE if use_infer else QWEN_IMAGE_2025_EDIT_3IMG)
        wf["78"]["inputs"]["image"] = images[0]
        wf["106"]["inputs"]["image"] = images[1]
        wf["108"]["inputs"]["image"] = images[2]

    # prompt + seed
    wf["111"]["inputs"]["prompt"] = prompt or ""
    wf["3"]["inputs"]["seed"] = int(seed or 0)

    # If NOT infer-size, honor explicit width/height
    if not use_infer:
        wf["112"]["inputs"]["width"] = int(width)
        wf["112"]["inputs"]["height"] = int(height)

    return wf