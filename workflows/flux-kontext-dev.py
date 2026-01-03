import copy

MODEL_ID = "flux-kontext-dev"

FLUX_KONTEXT_DEV = {
  "6": { "inputs": { "text": "", "clip": ["194", 0] }, "class_type": "CLIPTextEncode" },
  "8": { "inputs": { "samples": ["31", 0], "vae": ["39", 0] }, "class_type": "VAEDecode" },
  "31": { "inputs": { "seed": 0, "steps": 20, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1, "model": ["192", 0], "positive": ["35", 0], "negative": ["135", 0], "latent_image": ["124", 0] }, "class_type": "KSampler" },
  "35": { "inputs": { "guidance": 2.5, "conditioning": ["177", 0] }, "class_type": "FluxGuidance" },
  "39": { "inputs": { "vae_name": "ae.safetensors" }, "class_type": "VAELoader" },
  "42": { "inputs": { "image": ["146", 0] }, "class_type": "FluxKontextImageScale" },
  "124": { "inputs": { "pixels": ["42", 0], "vae": ["39", 0] }, "class_type": "VAEEncode" },
  "135": { "inputs": { "conditioning": ["6", 0] }, "class_type": "ConditioningZeroOut" },
  "136": { "inputs": { "filename_prefix": "ComfyUI", "images": ["8", 0] }, "class_type": "SaveImage" },
  "142": { "inputs": { "image": "", "refresh": "refresh" }, "class_type": "LoadImage" },
  "146": { "inputs": { "direction": "right", "match_image_size": True, "spacing_width": 0, "spacing_color": "white", "image1": ["142", 0] }, "class_type": "ImageStitch" },
  "173": { "inputs": { "images": ["42", 0] }, "class_type": "PreviewImage" },
  "177": { "inputs": { "conditioning": ["6", 0], "latent": ["124", 0] }, "class_type": "ReferenceLatent" },
  "192": { "inputs": { "unet_name": "flux1-kontext-dev-Q4_K_M.gguf" }, "class_type": "UnetLoaderGGUF" },
  "194": { "inputs": { "clip_name1": "clip_l.safetensors", "clip_name2": "t5-v1_1-xxl-encoder-Q4_K_M.gguf", "type": "flux" }, "class_type": "DualCLIPLoaderGGUF" }
}

def get_workflow(
    prompt: str = "",
    images=None,   # list of uploaded names
    seed: int = 0,
    width: int = 1024,   # unused
    height: int = 1024,  # unused
    **kwargs
):
    if images is None:
        images = []

    ignored = {k: v for k, v in kwargs.items() if v not in (None, "", [], {})}
    if width or height:
        ignored.update({"width": width, "height": height})
    if ignored:
        print(f"[{MODEL_ID}] Ignoring unsupported parameters: {list(ignored.keys())}")

    if len(images) < 1:
        raise ValueError("flux-kontext-dev requires 1 image")

    if len(images) > 1:
        print(f"[{MODEL_ID}] Received {len(images)} images; only 1 supported. Extra images will be ignored.")
        images = images[:1]

    wf = copy.deepcopy(FLUX_KONTEXT_DEV)
    wf["6"]["inputs"]["text"] = prompt or ""
    wf["142"]["inputs"]["image"] = images[0]
    wf["31"]["inputs"]["seed"] = int(seed or 0)
    return wf