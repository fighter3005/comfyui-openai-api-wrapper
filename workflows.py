import copy

# Workflow Definitions

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

QWEN_IMAGE_EDIT = {
  "3": { "inputs": { "seed": 0, "steps": 4, "cfg": 2.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1, "model": ["75", 0], "positive": ["111", 0], "negative": ["110", 0], "latent_image": ["88", 0] }, "class_type": "KSampler" },
  "8": { "inputs": { "samples": ["3", 0], "vae": ["39", 0] }, "class_type": "VAEDecode" },
  "37": { "inputs": { "unet_name": "qwen_image_edit_2509_fp8_e4m3fn.safetensors", "weight_dtype": "default" }, "class_type": "UNETLoader" },
  "38": { "inputs": { "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default" }, "class_type": "CLIPLoader" },
  "39": { "inputs": { "vae_name": "qwen_image_vae.safetensors" }, "class_type": "VAELoader" },
  "60": { "inputs": { "filename_prefix": "ComfyUI", "images": ["8", 0] }, "class_type": "SaveImage" },
  "66": { "inputs": { "shift": 3, "model": ["89", 0] }, "class_type": "ModelSamplingAuraFlow" },
  "75": { "inputs": { "strength": 1, "model": ["66", 0] }, "class_type": "CFGNorm" },
  "78": { "inputs": { "image": "" }, "class_type": "LoadImage" },
  "88": { "inputs": { "pixels": ["93", 0], "vae": ["39", 0] }, "class_type": "VAEEncode" },
  "89": { "inputs": { "lora_name": "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors", "strength_model": 1, "model": ["37", 0] }, "class_type": "LoraLoaderModelOnly" },
  "93": { "inputs": { "upscale_method": "lanczos", "megapixels": 1, "image": ["78", 0] }, "class_type": "ImageScaleToTotalPixels" },
  "110": { "inputs": { "prompt": "", "clip": ["38", 0], "vae": ["39", 0], "image1": ["93", 0] }, "class_type": "TextEncodeQwenImageEditPlus" },
  "111": { "inputs": { "prompt": "", "clip": ["38", 0], "vae": ["39", 0], "image1": ["93", 0] }, "class_type": "TextEncodeQwenImageEditPlus" },
  "112": { "inputs": { "width": 1024, "height": 1024, "batch_size": 1 }, "class_type": "EmptySD3LatentImage" }
}

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

# Flux 2 Workflows
FLUX_2_GEN = {
  "6": { "inputs": { "text": "", "clip": ["38", 0] }, "class_type": "CLIPTextEncode" },
  "8": { "inputs": { "samples": ["13", 0], "vae": ["10", 0] }, "class_type": "VAEDecode" },
  "9": { "inputs": { "filename_prefix": "Flux2", "images": ["8", 0] }, "class_type": "SaveImage" },
  "10": { "inputs": { "vae_name": "flux2-vae.safetensors" }, "class_type": "VAELoader" },
  "13": { "inputs": { "noise": ["25", 0], "guider": ["22", 0], "sampler": ["16", 0], "sigmas": ["48", 0], "latent_image": ["47", 0] }, "class_type": "SamplerCustomAdvanced" },
  "16": { "inputs": { "sampler_name": "euler" }, "class_type": "KSamplerSelect" },
  "22": { "inputs": { "model": ["67", 0], "conditioning": ["26", 0] }, "class_type": "BasicGuider" },
  "25": { "inputs": { "noise_seed": 0 }, "class_type": "RandomNoise" },
  "26": { "inputs": { "guidance": 4, "conditioning": ["6", 0] }, "class_type": "FluxGuidance" },
  "38": { "inputs": { "clip_name": "mistral_3_small_flux2_fp8.safetensors", "type": "flux2", "device": "default" }, "class_type": "CLIPLoader" },
  "47": { "inputs": { "width": 512, "height": 512, "batch_size": 1 }, "class_type": "EmptyFlux2LatentImage" },
  "48": { "inputs": { "steps": 20, "width": 512, "height": 512 }, "class_type": "Flux2Scheduler" },
  "67": { "inputs": { "unet_name": "flux2_dev_Q4_K_M.gguf" }, "class_type": "UnetLoaderGGUF" }
}

FLUX_2_EDIT_1IMG = {
  "6": { "inputs": { "text": "", "clip": ["38", 0] }, "class_type": "CLIPTextEncode" },
  "8": { "inputs": { "samples": ["13", 0], "vae": ["10", 0] }, "class_type": "VAEDecode" },
  "9": { "inputs": { "filename_prefix": "Flux2", "images": ["8", 0] }, "class_type": "SaveImage" },
  "10": { "inputs": { "vae_name": "flux2-vae.safetensors" }, "class_type": "VAELoader" },
  "13": { "inputs": { "noise": ["25", 0], "guider": ["22", 0], "sampler": ["16", 0], "sigmas": ["48", 0], "latent_image": ["47", 0] }, "class_type": "SamplerCustomAdvanced" },
  "16": { "inputs": { "sampler_name": "euler" }, "class_type": "KSamplerSelect" },
  "22": { "inputs": { "model": ["67", 0], "conditioning": ["43", 0] }, "class_type": "BasicGuider" },
  "25": { "inputs": { "noise_seed": 0 }, "class_type": "RandomNoise" },
  "26": { "inputs": { "guidance": 4, "conditioning": ["6", 0] }, "class_type": "FluxGuidance" },
  "38": { "inputs": { "clip_name": "mistral_3_small_flux2_fp8.safetensors", "type": "flux2", "device": "default" }, "class_type": "CLIPLoader" },
  "43": { "inputs": { "conditioning": ["26", 0], "latent": ["44", 0] }, "class_type": "ReferenceLatent" },
  "44": { "inputs": { "pixels": ["45", 0], "vae": ["10", 0] }, "class_type": "VAEEncode" },
  "45": { "inputs": { "upscale_method": "lanczos", "megapixels": 2, "image": ["46", 0] }, "class_type": "ImageScaleToTotalPixels" },
  "46": { "inputs": { "image": "" }, "class_type": "LoadImage" },
  "47": { "inputs": { "width": 1088, "height": 1920, "batch_size": 1 }, "class_type": "EmptyFlux2LatentImage" },
  "48": { "inputs": { "steps": 20, "width": 1088, "height": 1920 }, "class_type": "Flux2Scheduler" },
  "67": { "inputs": { "unet_name": "flux2_dev_Q4_K_M.gguf" }, "class_type": "UnetLoaderGGUF" }
}

FLUX_2_EDIT_2IMG = {
  "6": { "inputs": { "text": "", "clip": ["38", 0] }, "class_type": "CLIPTextEncode" },
  "8": { "inputs": { "samples": ["13", 0], "vae": ["10", 0] }, "class_type": "VAEDecode" },
  "9": { "inputs": { "filename_prefix": "Flux2", "images": ["8", 0] }, "class_type": "SaveImage" },
  "10": { "inputs": { "vae_name": "flux2-vae.safetensors" }, "class_type": "VAELoader" },
  "13": { "inputs": { "noise": ["25", 0], "guider": ["22", 0], "sampler": ["16", 0], "sigmas": ["48", 0], "latent_image": ["47", 0] }, "class_type": "SamplerCustomAdvanced" },
  "16": { "inputs": { "sampler_name": "euler" }, "class_type": "KSamplerSelect" },
  "22": { "inputs": { "model": ["67", 0], "conditioning": ["39", 0] }, "class_type": "BasicGuider" },
  "25": { "inputs": { "noise_seed": 0 }, "class_type": "RandomNoise" },
  "26": { "inputs": { "guidance": 4, "conditioning": ["6", 0] }, "class_type": "FluxGuidance" },
  "38": { "inputs": { "clip_name": "mistral_3_small_flux2_fp8.safetensors", "type": "flux2", "device": "default" }, "class_type": "CLIPLoader" },
  "39": { "inputs": { "conditioning": ["43", 0], "latent": ["40", 0] }, "class_type": "ReferenceLatent" },
  "40": { "inputs": { "pixels": ["41", 0], "vae": ["10", 0] }, "class_type": "VAEEncode" },
  "41": { "inputs": { "upscale_method": "lanczos", "megapixels": 1, "image": ["42", 0] }, "class_type": "ImageScaleToTotalPixels" },
  "42": { "inputs": { "image": "" }, "class_type": "LoadImage" },
  "43": { "inputs": { "conditioning": ["26", 0], "latent": ["44", 0] }, "class_type": "ReferenceLatent" },
  "44": { "inputs": { "pixels": ["45", 0], "vae": ["10", 0] }, "class_type": "VAEEncode" },
  "45": { "inputs": { "upscale_method": "lanczos", "megapixels": 2, "image": ["46", 0] }, "class_type": "ImageScaleToTotalPixels" },
  "46": { "inputs": { "image": "" }, "class_type": "LoadImage" },
  "47": { "inputs": { "width": 512, "height": 512, "batch_size": 1 }, "class_type": "EmptyFlux2LatentImage" },
  "48": { "inputs": { "steps": 20, "width": 512, "height": 512 }, "class_type": "Flux2Scheduler" },
  "67": { "inputs": { "unet_name": "flux2_dev_Q4_K_M.gguf" }, "class_type": "UnetLoaderGGUF" }
}

FLUX_2_EDIT_3IMG = {
  "6": { "inputs": { "text": "", "clip": ["38", 0] }, "class_type": "CLIPTextEncode" },
  "8": { "inputs": { "samples": ["13", 0], "vae": ["10", 0] }, "class_type": "VAEDecode" },
  "9": { "inputs": { "filename_prefix": "Flux2", "images": ["8", 0] }, "class_type": "SaveImage" },
  "10": { "inputs": { "vae_name": "flux2-vae.safetensors" }, "class_type": "VAELoader" },
  "13": { "inputs": { "noise": ["25", 0], "guider": ["22", 0], "sampler": ["16", 0], "sigmas": ["48", 0], "latent_image": ["47", 0] }, "class_type": "SamplerCustomAdvanced" },
  "16": { "inputs": { "sampler_name": "euler" }, "class_type": "KSamplerSelect" },
  "22": { "inputs": { "model": ["67", 0], "conditioning": ["71", 0] }, "class_type": "BasicGuider" },
  "25": { "inputs": { "noise_seed": 0 }, "class_type": "RandomNoise" },
  "26": { "inputs": { "guidance": 4, "conditioning": ["6", 0] }, "class_type": "FluxGuidance" },
  "38": { "inputs": { "clip_name": "mistral_3_small_flux2_fp8.safetensors", "type": "flux2", "device": "default" }, "class_type": "CLIPLoader" },
  "39": { "inputs": { "conditioning": ["43", 0], "latent": ["40", 0] }, "class_type": "ReferenceLatent" },
  "40": { "inputs": { "pixels": ["41", 0], "vae": ["10", 0] }, "class_type": "VAEEncode" },
  "41": { "inputs": { "upscale_method": "lanczos", "megapixels": 1, "image": ["42", 0] }, "class_type": "ImageScaleToTotalPixels" },
  "42": { "inputs": { "image": "" }, "class_type": "LoadImage" },
  "43": { "inputs": { "conditioning": ["26", 0], "latent": ["44", 0] }, "class_type": "ReferenceLatent" },
  "44": { "inputs": { "pixels": ["45", 0], "vae": ["10", 0] }, "class_type": "VAEEncode" },
  "45": { "inputs": { "upscale_method": "lanczos", "megapixels": 2, "image": ["46", 0] }, "class_type": "ImageScaleToTotalPixels" },
  "46": { "inputs": { "image": "" }, "class_type": "LoadImage" },
  "47": { "inputs": { "width": 1088, "height": 1920, "batch_size": 1 }, "class_type": "EmptyFlux2LatentImage" },
  "48": { "inputs": { "steps": 20, "width": 1088, "height": 1920 }, "class_type": "Flux2Scheduler" },
  "67": { "inputs": { "unet_name": "flux2_dev_Q4_K_M.gguf" }, "class_type": "UnetLoaderGGUF" },
  "68": { "inputs": { "image": "" }, "class_type": "LoadImage" },
  "69": { "inputs": { "upscale_method": "lanczos", "megapixels": 2, "image": ["68", 0] }, "class_type": "ImageScaleToTotalPixels" },
  "70": { "inputs": { "pixels": ["69", 0], "vae": ["10", 0] }, "class_type": "VAEEncode" },
  "71": { "inputs": { "conditioning": ["39", 0], "latent": ["70", 0] }, "class_type": "ReferenceLatent" }
}

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

SD3_5_SIMPLE = {
  "3": { "inputs": { "seed": 0, "steps": 20, "cfg": 4.01, "sampler_name": "euler", "scheduler": "sgm_uniform", "denoise": 1, "model": ["4", 0], "positive": ["16", 0], "negative": ["40", 0], "latent_image": ["53", 0] }, "class_type": "KSampler" },
  "4": { "inputs": { "ckpt_name": "sd3.5_large_fp8_scaled.safetensors" }, "class_type": "CheckpointLoaderSimple" },
  "8": { "inputs": { "samples": ["3", 0], "vae": ["4", 2] }, "class_type": "VAEDecode" },
  "9": { "inputs": { "filename_prefix": "ComfyUI", "images": ["8", 0] }, "class_type": "SaveImage" },
  "16": { "inputs": { "text": "", "clip": ["4", 1] }, "class_type": "CLIPTextEncode" },
  "40": { "inputs": { "text": "", "clip": ["4", 1] }, "class_type": "CLIPTextEncode" },
  "53": { "inputs": { "width": 1024, "height": 1024, "batch_size": 1 }, "class_type": "EmptySD3LatentImage" }
}

FLUX_SCHNELL = {
  "6": { "inputs": { "text": "", "clip": ["30", 1] }, "class_type": "CLIPTextEncode" },
  "8": { "inputs": { "samples": ["31", 0], "vae": ["30", 2] }, "class_type": "VAEDecode" },
  "9": { "inputs": { "filename_prefix": "ComfyUI", "images": ["8", 0] }, "class_type": "SaveImage" },
  "27": { "inputs": { "width": 1024, "height": 1024, "batch_size": 1 }, "class_type": "EmptySD3LatentImage" },
  "30": { "inputs": { "ckpt_name": "flux1-schnell-fp8.safetensors" }, "class_type": "CheckpointLoaderSimple" },
  "31": { "inputs": { "seed": 0, "steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1, "model": ["30", 0], "positive": ["6", 0], "negative": ["33", 0], "latent_image": ["27", 0] }, "class_type": "KSampler" },
  "33": { "inputs": { "text": "", "clip": ["30", 1] }, "class_type": "CLIPTextEncode" }
}

def get_workflow(name, mode="gen", img_count=0):
    if name == "flux-kontext-dev":
        return copy.deepcopy(FLUX_KONTEXT_DEV)
    elif name == "qwen-image-edit-2509":
        return copy.deepcopy(QWEN_IMAGE_EDIT)
    elif name == "flux-krea-dev":
        return copy.deepcopy(FLUX_KREA_DEV)
    elif name == "z-image-turbo":
        return copy.deepcopy(Z_IMAGE_TURBO_GEN)
    elif name == "flux-dev-checkpoint":
        return copy.deepcopy(FLUX_DEV_CHECKPOINT)
    elif name == "sd3-5-simple":
        return copy.deepcopy(SD3_5_SIMPLE)
    elif name == "flux-schnell":
        return copy.deepcopy(FLUX_SCHNELL)
    # Flux 2 Logic
    elif name == "flux-2-dev":
        if mode == "gen":
            return copy.deepcopy(FLUX_2_GEN)
        elif mode == "edit":
            if img_count == 1:
                return copy.deepcopy(FLUX_2_EDIT_1IMG)
            elif img_count == 2:
                return copy.deepcopy(FLUX_2_EDIT_2IMG)
            elif img_count == 3:
                return copy.deepcopy(FLUX_2_EDIT_3IMG)
    return None