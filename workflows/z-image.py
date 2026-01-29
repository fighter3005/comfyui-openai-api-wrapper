import copy

MODEL_ID = "z-image"

Z_IMAGE_GEN = {
  "9": {
    "inputs": {"filename_prefix": "z-image", "images": ["65", 0]},
    "class_type": "SaveImage"
  },
  "62": {
    "inputs": {
      "clip_name": "qwen_3_4b.safetensors",
      "type": "lumina2",
      "device": "default"
    },
    "class_type": "CLIPLoader"
  },
  "63": {
    "inputs": {"vae_name": "ae.safetensors"},
    "class_type": "VAELoader"
  },
  "65": {
    "inputs": {"samples": ["69", 0], "vae": ["63", 0]},
    "class_type": "VAEDecode"
  },
  "66": {
    "inputs": {
      "unet_name": "z_image_bf16.safetensors",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader"
  },
  "67": {
    "inputs": {"text": "", "clip": ["62", 0]},
    "class_type": "CLIPTextEncode"
  },
  "68": {
    "inputs": {"width": 1232, "height": 1232, "batch_size": 1},
    "class_type": "EmptySD3LatentImage"
  },
  "69": {
    "inputs": {
      "seed": 0,
      "steps": 30,
      "cfg": 4,
      "sampler_name": "res_multistep",
      "scheduler": "simple",
      "denoise": 1,
      "model": ["70", 0],
      "positive": ["67", 0],
      "negative": ["71", 0],
      "latent_image": ["68", 0]
    },
    "class_type": "KSampler"
  },
  "70": {
    "inputs": {"shift": 3, "model": ["66", 0]},
    "class_type": "ModelSamplingAuraFlow"
  },
  "71": {
    "inputs": {"text": "", "clip": ["62", 0]},
    "class_type": "CLIPTextEncode"
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

    wf = copy.deepcopy(Z_IMAGE_GEN)

    wf["67"]["inputs"]["text"] = prompt or ""
    wf["69"]["inputs"]["seed"] = int(seed or 0)
    wf["68"]["inputs"]["width"] = int(width)
    wf["68"]["inputs"]["height"] = int(height)

    return wf