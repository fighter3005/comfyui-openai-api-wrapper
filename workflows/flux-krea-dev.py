import copy

MODEL_ID = "flux-krea-dev"

FLUX_KREA_DEV = {
  "8": { "inputs": { "samples": ["31", 0], "vae": ["39", 0] }, "class_type": "VAEDecode" },
  "9": { "inputs": { "filename_prefix": "flux_krea/flux_krea", "images": ["8", 0] }, "class_type": "SaveImage" },
  "27": { "inputs": { "width": 1024, "height": 1024, "batch_size": 1 }, "class_type": "EmptySD3LatentImage" },
  "31": { "inputs": { "seed": 0, "steps": 20, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1, "model": ["53", 0], "positive": ["45", 0], "negative": ["42", 0], "latent_image": ["27", 0] }, "class_type": "KSampler" },
  "39": { "inputs": { "vae_name": "ae.safetensors" }, "class_type": "VAELoader" },
  "42": { "inputs": { "conditioning": ["45", 0] }, "class_type": "ConditioningZeroOut" },
  "45": { "inputs": { "text": "", "clip": ["52", 0] }, "class_type": "CLIPTextEncode" },
  "52": { "inputs": { "clip_name1": "clip_l.safetensors", "clip_name2": "t5-v1_1-xxl-encoder-Q4_K_M.gguf", "type": "flux" }, "class_type": "DualCLIPLoaderGGUF" },
  "53": { "inputs": { "unet_name": "flux1-krea-dev-Q4_K_M.gguf" }, "class_type": "UnetLoaderGGUF" }
}

def get_workflow(prompt: str = "", width: int = 1024, height: int = 1024, seed: int = 0, **kwargs):
    ignored = {k: v for k, v in kwargs.items() if v not in (None, "", [], {})}
    if ignored:
        print(f"[{MODEL_ID}] Ignoring unsupported parameters: {list(ignored.keys())}")

    wf = copy.deepcopy(FLUX_KREA_DEV)
    wf["45"]["inputs"]["text"] = prompt or ""
    wf["31"]["inputs"]["seed"] = int(seed or 0)
    wf["27"]["inputs"]["width"] = int(width)
    wf["27"]["inputs"]["height"] = int(height)
    return wf