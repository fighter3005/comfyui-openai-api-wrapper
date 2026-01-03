import copy

MODEL_ID = "sd3-5-simple"

SD3_5_SIMPLE = {
  "3": { "inputs": { "seed": 0, "steps": 20, "cfg": 4.01, "sampler_name": "euler", "scheduler": "sgm_uniform", "denoise": 1, "model": ["4", 0], "positive": ["16", 0], "negative": ["40", 0], "latent_image": ["53", 0] }, "class_type": "KSampler" },
  "4": { "inputs": { "ckpt_name": "sd3.5_large_fp8_scaled.safetensors" }, "class_type": "CheckpointLoaderSimple" },
  "8": { "inputs": { "samples": ["3", 0], "vae": ["4", 2] }, "class_type": "VAEDecode" },
  "9": { "inputs": { "filename_prefix": "ComfyUI", "images": ["8", 0] }, "class_type": "SaveImage" },
  "16": { "inputs": { "text": "", "clip": ["4", 1] }, "class_type": "CLIPTextEncode" },
  "40": { "inputs": { "text": "", "clip": ["4", 1] }, "class_type": "CLIPTextEncode" },
  "53": { "inputs": { "width": 1024, "height": 1024, "batch_size": 1 }, "class_type": "EmptySD3LatentImage" }
}

def get_workflow(prompt: str = "", width: int = 1024, height: int = 1024, seed: int = 0, **kwargs):
    ignored = {k: v for k, v in kwargs.items() if v not in (None, "", [], {})}
    if ignored:
        print(f"[{MODEL_ID}] Ignoring unsupported parameters: {list(ignored.keys())}")

    wf = copy.deepcopy(SD3_5_SIMPLE)
    wf["16"]["inputs"]["text"] = prompt or ""
    wf["3"]["inputs"]["seed"] = int(seed or 0)
    wf["53"]["inputs"]["width"] = int(width)
    wf["53"]["inputs"]["height"] = int(height)
    return wf