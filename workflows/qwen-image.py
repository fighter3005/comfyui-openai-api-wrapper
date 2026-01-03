import copy

MODEL_ID = "qwen-image"

# --- Qwen Image 2512 (Text-to-Image) ---
QWEN_IMAGE_2512_GEN = {
  "90": {
    "inputs": {"filename_prefix": "Qwen-Image-2512", "images": ["92:8", 0]},
    "class_type": "SaveImage",
    "_meta": {"title": "Save Image"}
  },
  "91": {
    "inputs": {"value": "Urban alleyway at dusk.\n"},
    "class_type": "PrimitiveStringMultiline",
    "_meta": {"title": "Prompt"}
  },
  "92:8": {
    "inputs": {"samples": ["92:3", 0], "vae": ["92:39", 0]},
    "class_type": "VAEDecode",
    "_meta": {"title": "VAE Decode"}
  },
  "92:66": {
    "inputs": {"shift": 3.1000000000000005, "model": ["92:73", 0]},
    "class_type": "ModelSamplingAuraFlow",
    "_meta": {"title": "ModelSamplingAuraFlow"}
  },
  "92:73": {
    "inputs": {"lora_name": "Qwen-Image-Lightning-4steps-V1.0.safetensors", "strength_model": 1, "model": ["92:76", 0]},
    "class_type": "LoraLoaderModelOnly",
    "_meta": {"title": "LoraLoaderModelOnly"}
  },
  "92:6": {
    "inputs": {"text": ["91", 0], "clip": ["92:38", 0]},
    "class_type": "CLIPTextEncode",
    "_meta": {"title": "CLIP Text Encode (Positive Prompt)"}
  },
  "92:7": {
    "inputs": {"text": "低分辨率,低画质,肢体畸形,手指畸形,画面过饱和,蜡像感,人脸无细节,过度光滑,画面具有AI感。构图混乱。文字模糊,扭曲", "clip": ["92:38", 0]},
    "class_type": "CLIPTextEncode",
    "_meta": {"title": "CLIP Text Encode (Negative Prompt)"}
  },
  "92:38": {
    "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"},
    "class_type": "CLIPLoader",
    "_meta": {"title": "Load CLIP"}
  },
  "92:39": {
    "inputs": {"vae_name": "qwen_image_vae.safetensors"},
    "class_type": "VAELoader",
    "_meta": {"title": "Load VAE"}
  },
  "92:58": {
    "inputs": {"width": 1328, "height": 1328, "batch_size": 1},
    "class_type": "EmptySD3LatentImage",
    "_meta": {"title": "EmptySD3LatentImage"}
  },
  "92:3": {
    "inputs": {
      "seed": 43565364,
      "steps": 4,
      "cfg": 1,
      "sampler_name": "euler",
      "scheduler": "simple",
      "denoise": 1,
      "model": ["92:66", 0],
      "positive": ["92:6", 0],
      "negative": ["92:7", 0],
      "latent_image": ["92:58", 0]
    },
    "class_type": "KSampler",
    "_meta": {"title": "KSampler"}
  },
  "92:76": {
    "inputs": {"unet_name": "qwen-image-2512-Q4_K_M.gguf"},
    "class_type": "UnetLoaderGGUF",
    "_meta": {"title": "Unet Loader (GGUF)"}
  }
}

def get_workflow(
    prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    seed: int = 0,
    **kwargs
):
    ignored = {k: v for k, v in kwargs.items() if v not in (None, "", [], {})}
    if ignored:
        print(f"[{MODEL_ID}] Ignoring unsupported parameters: {list(ignored.keys())}")

    wf = copy.deepcopy(QWEN_IMAGE_2512_GEN)

    # Prompt is fed via PrimitiveStringMultiline node "91"
    wf["91"]["inputs"]["value"] = (prompt or "").rstrip() + "\n"

    # Seed & size
    wf["92:3"]["inputs"]["seed"] = int(seed or 0)
    wf["92:58"]["inputs"]["width"] = int(width)
    wf["92:58"]["inputs"]["height"] = int(height)

    return wf