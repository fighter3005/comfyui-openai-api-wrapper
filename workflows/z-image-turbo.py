import copy

MODEL_ID = "z-image-turbo"

Z_IMAGE_TURBO_GEN = {
  "3": { "inputs": { "seed": 0, "steps": 10, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1, "model": ["30", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["13", 0] }, "class_type": "KSampler" },
  "6": { "inputs": { "text": "", "clip": ["29", 0] }, "class_type": "CLIPTextEncode" },
  "7": { "inputs": { "text": "blurry ugly bad", "clip": ["29", 0] }, "class_type": "CLIPTextEncode" },
  "8": { "inputs": { "samples": ["3", 0], "vae": ["17", 0] }, "class_type": "VAEDecode" },
  "9": { "inputs": { "filename_prefix": "ComfyUI", "images": ["8", 0] }, "class_type": "SaveImage" },
  "13": { "inputs": { "width": 1024, "height": 1024, "batch_size": 1 }, "class_type": "EmptySD3LatentImage" },
  "17": { "inputs": { "vae_name": "ae.safetensors" }, "class_type": "VAELoader" },
  "29": { "inputs": { "clip_name": "Qwen_3_4b-Q8_0.gguf", "type": "lumina2" }, "class_type": "CLIPLoaderGGUF" },
  "30": { "inputs": { "unet_name": "Z_Image_Q6_K.gguf" }, "class_type": "UnetLoaderGGUF" }
}

def get_workflow(prompt: str = "", width: int = 1024, height: int = 1024, seed: int = 0, **kwargs):
    ignored = {k: v for k, v in kwargs.items() if v not in (None, "", [], {})}
    if ignored:
        print(f"[{MODEL_ID}] Ignoring unsupported parameters: {list(ignored.keys())}")

    wf = copy.deepcopy(Z_IMAGE_TURBO_GEN)
    wf["6"]["inputs"]["text"] = prompt or ""
    wf["3"]["inputs"]["seed"] = int(seed or 0)
    wf["13"]["inputs"]["width"] = int(width)
    wf["13"]["inputs"]["height"] = int(height)
    return wf