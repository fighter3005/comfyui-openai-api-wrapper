import copy

MODEL_ID = "z-image-turbo"

Z_IMAGE_TURBO_GEN = {
  "9": {
    "inputs": {"filename_prefix": "z-image", "images": ["57:8", 0]},
    "class_type": "SaveImage"
  },
  "58": {
    "inputs": {"value": ""},
    "class_type": "PrimitiveStringMultiline"
  },
  "57:30": {
    "inputs": {
      "clip_name": "qwen_3_4b.safetensors",
      "type": "lumina2",
      "device": "default"
    },
    "class_type": "CLIPLoader"
  },
  "57:29": {
    "inputs": {"vae_name": "ae.safetensors"},
    "class_type": "VAELoader"
  },
  "57:33": {
    "inputs": {"conditioning": ["57:27", 0]},
    "class_type": "ConditioningZeroOut"
  },
  "57:8": {
    "inputs": {"samples": ["57:3", 0], "vae": ["57:29", 0]},
    "class_type": "VAEDecode"
  },
  "57:28": {
    "inputs": {
      "unet_name": "z_image_turbo_bf16.safetensors",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader"
  },
  "57:27": {
    "inputs": {"text": ["58", 0], "clip": ["57:30", 0]},
    "class_type": "CLIPTextEncode"
  },
  "57:13": {
    "inputs": {"width": 1232, "height": 1232, "batch_size": 1},
    "class_type": "EmptySD3LatentImage"
  },
  "57:3": {
    "inputs": {
      "seed": 0,
      "steps": 4,
      "cfg": 1,
      "sampler_name": "res_multistep",
      "scheduler": "simple",
      "denoise": 1,
      "model": ["57:11", 0],
      "positive": ["57:27", 0],
      "negative": ["57:33", 0],
      "latent_image": ["57:13", 0]
    },
    "class_type": "KSampler"
  },
  "57:11": {
    "inputs": {"shift": 3, "model": ["57:28", 0]},
    "class_type": "ModelSamplingAuraFlow"
  }
}


def get_workflow(
    mode: str = "gen",
    prompt: str = "",
    width: int = 1232,
    height: int = 1232,
    seed: int = 0,
    **kwargs
):
    if mode != "gen":
        raise ValueError(f"{MODEL_ID} does not support edit mode")

    ignored = {k: v for k, v in kwargs.items() if v not in (None, "", [], {})}
    if ignored:
        print(f"[{MODEL_ID}] Ignoring unsupported parameters: {list(ignored.keys())}")

    wf = copy.deepcopy(Z_IMAGE_TURBO_GEN)

    wf["58"]["inputs"]["value"] = (prompt or "").rstrip() + "\n"
    wf["57:3"]["inputs"]["seed"] = int(seed or 0)
    wf["57:13"]["inputs"]["width"] = int(width)
    wf["57:13"]["inputs"]["height"] = int(height)

    return wf