import copy

MODEL_ID = "z-image-turbo"

Z_IMAGE_TURBO_GEN = {
  "6":  { "inputs": { "text": "", "clip": ["30", 1] }, "class_type": "CLIPTextEncode" },
  "33": { "inputs": { "text": "", "clip": ["30", 1] }, "class_type": "CLIPTextEncode" },

  "35": { "inputs": { "guidance": 3.0, "conditioning": ["6", 0] }, "class_type": "FluxGuidance" },

  "27": { "inputs": { "width": 1024, "height": 1024, "batch_size": 1 }, "class_type": "EmptySD3LatentImage" },

  "30": { "inputs": { "ckpt_name": "z-image-turbo-fp8.safetensors" }, "class_type": "CheckpointLoaderSimple" },

  "31": {
    "inputs": {
      "seed": 0,
      "steps": 8,
      "cfg": 1,
      "sampler_name": "euler",
      "scheduler": "simple",
      "denoise": 1,
      "model": ["30", 0],
      "positive": ["35", 0],
      "negative": ["33", 0],
      "latent_image": ["27", 0]
    },
    "class_type": "KSampler"
  },

  "8": { "inputs": { "samples": ["31", 0], "vae": ["30", 2] }, "class_type": "VAEDecode" },
  "9": { "inputs": { "filename_prefix": "z-image-turbo", "images": ["8", 0] }, "class_type": "SaveImage" }
}


def get_workflow(prompt: str = "", width: int = 1024, height: int = 1024, seed: int = 0, **kwargs):
    ignored = {k: v for k, v in kwargs.items() if v not in (None, "", [], {})}
    if ignored:
        print(f"[{MODEL_ID}] Ignoring unsupported parameters: {list(ignored.keys())}")

    wf = copy.deepcopy(Z_IMAGE_TURBO_GEN)
    wf["6"]["inputs"]["text"] = prompt or ""
    wf["31"]["inputs"]["seed"] = int(seed or 0)
    wf["27"]["inputs"]["width"] = int(width)
    wf["27"]["inputs"]["height"] = int(height)
    return wf