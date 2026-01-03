import copy

MODEL_ID = "flux-dev-checkpoint"

FLUX_DEV_CHECKPOINT = {
  "6": { "inputs": { "text": "", "clip": ["30", 1] }, "class_type": "CLIPTextEncode" },
  "8": { "inputs": { "samples": ["31", 0], "vae": ["30", 2] }, "class_type": "VAEDecode" },
  "9": { "inputs": { "filename_prefix": "ComfyUI", "images": ["8", 0] }, "class_type": "SaveImage" },
  "27": { "inputs": { "width": 1024, "height": 1024, "batch_size": 1 }, "class_type": "EmptySD3LatentImage" },
  "30": { "inputs": { "ckpt_name": "flux1-dev-fp8.safetensors" }, "class_type": "CheckpointLoaderSimple" },
  "31": { "inputs": { "seed": 0, "steps": 20, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1, "model": ["30", 0], "positive": ["35", 0], "negative": ["33", 0], "latent_image": ["27", 0] }, "class_type": "KSampler" },
  "33": { "inputs": { "text": "", "clip": ["30", 1] }, "class_type": "CLIPTextEncode" },
  "35": { "inputs": { "guidance": 3.5, "conditioning": ["6", 0] }, "class_type": "FluxGuidance" }
}

def get_workflow(prompt: str = "", width: int = 1024, height: int = 1024, seed: int = 0, **kwargs):
    ignored = {k: v for k, v in kwargs.items() if v not in (None, "", [], {})}
    if ignored:
        print(f"[{MODEL_ID}] Ignoring unsupported parameters: {list(ignored.keys())}")

    wf = copy.deepcopy(FLUX_DEV_CHECKPOINT)
    wf["6"]["inputs"]["text"] = prompt or ""
    wf["31"]["inputs"]["seed"] = int(seed or 0)
    wf["27"]["inputs"]["width"] = int(width)
    wf["27"]["inputs"]["height"] = int(height)
    return wf