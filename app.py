import os
import io
import json
import uuid
import random
import time
import base64
import threading
import requests
import websocket

from flask import Flask, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename
from PIL import Image

import workflows  # workflows package (workflows/__init__.py)

app = Flask(__name__)

# ============================================================
# Configuration
# ============================================================
COMFY_HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1")
COMFY_PORT = os.environ.get("COMFYUI_PORT", "8188")
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"
WS_URL = f"ws://{COMFY_HOST}:{COMFY_PORT}/ws"

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL")  # optional, e.g. https://api.example.com

# In-memory temporary URL store (for response_format="url")
_TEMP_IMAGES = {}
_TEMP_LOCK = threading.Lock()
_TEMP_TTL_SECONDS = int(os.environ.get("TEMP_IMAGE_TTL_SECONDS", "3600"))

# ============================================================
# ComfyUI Client Logic
# ============================================================
def upload_image(file_storage, image_type="input"):
    """
    Uploads an image file to ComfyUI.
    Adds a unique prefix to filename to prevent overwriting.
    """
    original_name = secure_filename(file_storage.filename)
    if not original_name:
        original_name = "image.png"

    unique_prefix = str(uuid.uuid4())[:8]
    filename = f"{unique_prefix}_{original_name}"

    files = {"image": (filename, file_storage.read(), file_storage.content_type)}
    data = {"type": image_type, "overwrite": "true"}

    response = requests.post(f"{COMFY_URL}/upload/image", files=files, data=data)
    response.raise_for_status()

    # reset
    try:
        file_storage.seek(0)
    except Exception:
        pass

    result = response.json()
    return result.get("name", filename)

def upload_image_bytes(image_bytes: bytes, filename="image.png", content_type="image/png", image_type="input"):
    """
    Uploads raw bytes to ComfyUI as an image file.
    """
    safe_name = secure_filename(filename) or "image.png"
    unique_prefix = str(uuid.uuid4())[:8]
    final_name = f"{unique_prefix}_{safe_name}"

    files = {"image": (final_name, image_bytes, content_type)}
    data = {"type": image_type, "overwrite": "true"}

    response = requests.post(f"{COMFY_URL}/upload/image", files=files, data=data)
    response.raise_for_status()
    result = response.json()
    return result.get("name", final_name)

def queue_prompt(prompt_workflow):
    client_id = str(uuid.uuid4())
    payload = {"prompt": prompt_workflow, "client_id": client_id}
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{COMFY_URL}/prompt", json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["prompt_id"], client_id

def get_history(prompt_id):
    response = requests.get(f"{COMFY_URL}/history/{prompt_id}")
    response.raise_for_status()
    return response.json()

def get_image_raw(filename, subfolder, folder_type):
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    response = requests.get(f"{COMFY_URL}/view", params=params)
    response.raise_for_status()
    return response.content

def execute_workflow(workflow_dict):
    """
    Full execution: Queue -> WS wait -> Download.
    Returns a list of raw image bytes.
    """
    prompt_id, client_id = queue_prompt(workflow_dict)

    ws = websocket.WebSocket()
    ws.connect(f"{WS_URL}?clientId={client_id}")

    try:
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message.get("type") == "executing":
                    data = message.get("data", {})
                    if data.get("node") is None and data.get("prompt_id") == prompt_id:
                        break
    finally:
        try:
            ws.close()
        except Exception:
            pass

    history = get_history(prompt_id).get(prompt_id, {})
    outputs = history.get("outputs", {})

    images = []
    for node_id in outputs:
        node_output = outputs[node_id]
        if "images" in node_output:
            for img in node_output["images"]:
                raw = get_image_raw(img["filename"], img["subfolder"], img["type"])
                images.append(raw)

    return images

# ============================================================
# Helpers: OpenAI-ish parsing and formatting
# ============================================================
def _now():
    return int(time.time())

def parse_size(size_str):
    """
    Parses OpenAI 'size' param into (width, height).
    Accepts:
      - "auto" / None => (1024, 1024)
      - "1024x1024" etc => parsed
    """
    if not size_str or size_str == "auto":
        return 1024, 1024
    if isinstance(size_str, str) and "x" in size_str:
        try:
            w, h = map(int, size_str.split("x"))
            return w, h
        except Exception:
            return 1024, 1024
    return 1024, 1024

def normalize_model_id(model_id):
    if not model_id:
        return "flux-kontext-dev"
    if model_id.startswith("openai/"):
        return model_id[7:]
    if model_id.startswith("comfy-"):
        return model_id[6:]
    return model_id

def clamp_int(val, default, lo, hi, name):
    if val is None:
        return default
    try:
        iv = int(val)
        if iv < lo:
            print(f"[api] {name}={iv} below min {lo}; clamping.")
            return lo
        if iv > hi:
            print(f"[api] {name}={iv} above max {hi}; clamping.")
            return hi
        return iv
    except Exception:
        return default

def convert_image_bytes(raw_bytes: bytes, output_format: str = "png", output_compression: int = 100, background: str = None) -> bytes:
    """
    Convert raw image bytes (from ComfyUI) to desired output_format: png/jpeg/webp.
    output_compression: 0-100 for jpeg/webp.
    background: transparent/opaque/auto (we only use it to decide alpha flattening for jpeg).
    """
    output_format = (output_format or "png").lower()
    if output_format not in ("png", "jpeg", "webp"):
        raise ValueError(f"Unsupported output_format: {output_format}")

    img = Image.open(io.BytesIO(raw_bytes))

    # For JPEG: must be RGB (no alpha). For webp/png: can keep alpha.
    if output_format == "jpeg":
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            # Flatten alpha (background param may request opaque; default white)
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
    else:
        # png/webp: keep alpha if present
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")

    out = io.BytesIO()
    if output_format == "png":
        img.save(out, format="PNG")
    elif output_format == "webp":
        q = clamp_int(output_compression, 100, 0, 100, "output_compression")
        img.save(out, format="WEBP", quality=q, method=6)
    elif output_format == "jpeg":
        q = clamp_int(output_compression, 100, 0, 100, "output_compression")
        img.save(out, format="JPEG", quality=q, optimize=True)
    return out.getvalue()

def mime_for_output_format(fmt: str) -> str:
    fmt = (fmt or "png").lower()
    if fmt == "png":
        return "image/png"
    if fmt == "jpeg":
        return "image/jpeg"
    if fmt == "webp":
        return "image/webp"
    return "application/octet-stream"

def store_temp_image(image_bytes: bytes, output_format: str) -> str:
    token = str(uuid.uuid4())
    expires_at = time.time() + _TEMP_TTL_SECONDS
    with _TEMP_LOCK:
        _TEMP_IMAGES[token] = (image_bytes, output_format, expires_at)
    return token

def cleanup_temp_images():
    now = time.time()
    with _TEMP_LOCK:
        dead = [k for k, (_, __, exp) in _TEMP_IMAGES.items() if exp < now]
        for k in dead:
            _TEMP_IMAGES.pop(k, None)

def make_public_url(path: str) -> str:
    # prefer configured public base URL, else infer from request
    base = PUBLIC_BASE_URL or request.host_url.rstrip("/")
    return f"{base}{path}"

def openai_error(message: str, status: int = 400, param: str = None, code: str = None, err_type: str = "invalid_request_error"):
    payload = {
        "error": {
            "message": message,
            "type": err_type,
            "param": param,
            "code": code
        }
    }
    return jsonify(payload), status

def build_images_response(created: int, data_list, response_format: str, output_format: str = None, size: str = None, quality: str = None, background: str = None):
    """
    OpenAI Images API response shape:
      { created, data: [ {b64_json|url} ... ] }
    Additional top-level fields exist in OpenAI for GPT image models; we include some when known.
    """
    resp = {"created": created, "data": data_list}
    # optional (best-effort)
    if output_format:
        resp["output_format"] = output_format
    if size:
        resp["size"] = size
    if quality:
        resp["quality"] = quality
    if background:
        resp["background"] = background
    return jsonify(resp)

def sse_format(event_name: str, data_obj: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data_obj, separators=(',', ':'))}\n\n"

# ============================================================
# Routes: temp image serving (for response_format="url")
# ============================================================
@app.route("/v1/images/tmp/<token>", methods=["GET"])
def get_temp_image(token):
    cleanup_temp_images()
    with _TEMP_LOCK:
        item = _TEMP_IMAGES.get(token)
    if not item:
        return openai_error("Image URL expired or not found.", status=404)

    image_bytes, output_format, _exp = item
    return send_file(
        io.BytesIO(image_bytes),
        mimetype=mime_for_output_format(output_format),
        as_attachment=False,
        download_name=f"image.{output_format}"
    )

# ============================================================
# OpenAI Compatible Endpoints
# ============================================================
@app.route("/v1/models", methods=["GET"])
def list_models():
    """
    Dynamically lists available models based on workflows/*.py files
    via workflows.get_supported_models().
    """
    if not hasattr(workflows, "get_supported_models"):
        # Safety fallback (should not happen in the refactored layout)
        return jsonify({"object": "list", "data": []})

    supported = workflows.get_supported_models()
    created = _now()

    return jsonify({
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "created": created, "owned_by": "comfyui"}
            for model_id in supported
        ]
    })

# ------------------------------------------------------------
# POST /v1/images/generations
# ------------------------------------------------------------
@app.route("/v1/images/generations", methods=["POST"])
def images_generations():
    if not request.is_json:
        return openai_error("Request must be application/json", param="Content-Type")

    data = request.get_json(silent=True) or {}

    raw_model_id = data.get("model", "flux-krea-dev")
    model_id = normalize_model_id(raw_model_id)

    prompt_text = data.get("prompt")
    if not prompt_text:
        return openai_error("Missing required parameter: prompt", param="prompt")

    n = clamp_int(data.get("n"), 1, 1, 10, "n")
    size_str = data.get("size", "auto")
    width, height = parse_size(size_str)

    # OpenAI Image API params we accept (best-effort)
    response_format = (data.get("response_format") or "b64_json").lower()
    if response_format not in ("b64_json", "url"):
        return openai_error("response_format must be 'b64_json' or 'url'", param="response_format")

    output_format = (data.get("output_format") or "png").lower()
    output_compression = clamp_int(data.get("output_compression"), 100, 0, 100, "output_compression")
    background = data.get("background")  # transparent/opaque/auto (best-effort)
    quality = data.get("quality")        # ignored by most comfy workflows
    partial_images = clamp_int(data.get("partial_images"), 0, 0, 3, "partial_images")
    stream = bool(data.get("stream") or False)

    # Warn about unsupported but ignore
    for k in ("moderation", "style", "user"):
        if data.get(k) not in (None, "", [], {}):
            print(f"[{model_id}] Ignoring unsupported parameter: {k}")

    def run_generation_once():
        seed = random.randint(1, 10**15)
        wf = workflows.get_workflow(
            model_id,
            mode="gen",
            prompt=prompt_text,
            width=width,
            height=height,
            seed=seed,
            quality=quality,
            background=background,
        )
        if not wf:
            raise ValueError(f"Model {model_id} (raw: {raw_model_id}) not supported for generations.")

        raw_images = execute_workflow(wf)
        if not raw_images:
            raise ValueError("No images returned from workflow execution.")

        # Convert each returned image into requested output_format
        converted = [convert_image_bytes(b, output_format=output_format, output_compression=output_compression, background=background) for b in raw_images]
        return converted

    def build_data_items(images_bytes):
        items = []
        if response_format == "b64_json":
            for b in images_bytes:
                items.append({"b64_json": base64.b64encode(b).decode("utf-8")})
        else:
            for b in images_bytes:
                token = store_temp_image(b, output_format)
                items.append({"url": make_public_url(f"/v1/images/tmp/{token}")})
        return items

    # Streaming (SSE) per OpenAI Image Streaming: partial + completed events
    if stream:
        if n != 1:
            print("[api] stream=true with n!=1: only the first image will be streamed; others ignored.")
            n = 1

        def gen():
            created_at = _now()
            try:
                imgs = run_generation_once()
                # Only one "image" conceptually; we may still have multiple outputs from Comfy.
                final_img = imgs[0]

                # Emit a "partial_image" if requested (best-effort; we send the same bytes)
                if partial_images > 0:
                    yield sse_format("image_generation.partial_image", {
                        "type": "image_generation.partial_image",
                        "b64_json": base64.b64encode(final_img).decode("utf-8"),
                        "created_at": created_at,
                        "size": f"{width}x{height}",
                        "quality": quality or "auto",
                        "background": background or "auto",
                        "output_format": output_format,
                        "partial_image_index": 0
                    })

                yield sse_format("image_generation.completed", {
                    "type": "image_generation.completed",
                    "b64_json": base64.b64encode(final_img).decode("utf-8"),
                    "created_at": created_at,
                    "size": f"{width}x{height}",
                    "quality": quality or "auto",
                    "background": background or "auto",
                    "output_format": output_format,
                })
            except Exception as e:
                err = {"error": {"message": str(e), "type": "server_error"}}
                yield sse_format("error", err)

        return Response(gen(), mimetype="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })

    # Non-streaming
    created = _now()
    out_images = []
    try:
        while len(out_images) < n:
            imgs = run_generation_once()
            out_images.extend(imgs)
        out_images = out_images[:n]
    except Exception as e:
        return openai_error(str(e), status=400)

    return build_images_response(
        created=created,
        data_list=build_data_items(out_images),
        response_format=response_format,
        output_format=output_format,
        size=f"{width}x{height}",
        quality=quality or "auto",
        background=background or "auto",
    )

# ------------------------------------------------------------
# POST /v1/images/edits
# ------------------------------------------------------------
@app.route("/v1/images/edits", methods=["POST"])
def images_edits():
    # OpenAI expects multipart/form-data for edits, but we also allow JSON fallback.
    prompt_text = None
    raw_model_id = None
    response_format = None
    output_format = None
    output_compression = None
    size_str = None
    n = 1
    stream = False
    partial_images = 0
    background = None
    quality = None
    input_fidelity = None

    uploaded_names = []

    if request.is_json:
        data = request.get_json(silent=True) or {}
        prompt_text = data.get("prompt")
        raw_model_id = data.get("model", "flux-kontext-dev")
        response_format = (data.get("response_format") or "b64_json").lower()
        output_format = (data.get("output_format") or "png").lower()
        output_compression = clamp_int(data.get("output_compression"), 100, 0, 100, "output_compression")
        size_str = data.get("size", "auto")
        n = clamp_int(data.get("n"), 1, 1, 10, "n")
        stream = bool(data.get("stream") or False)
        partial_images = clamp_int(data.get("partial_images"), 0, 0, 3, "partial_images")
        background = data.get("background")
        quality = data.get("quality")
        input_fidelity = data.get("input_fidelity")

        images_field = data.get("image") or data.get("images")
        if not images_field:
            return openai_error("No image provided. Provide 'image' (string or array) or use multipart/form-data.", param="image")

        if not isinstance(images_field, list):
            images_field = [images_field]

        # Accept data URLs ("data:image/png;base64,...") or raw base64
        for idx, item in enumerate(images_field):
            if not isinstance(item, str):
                print("[api] JSON image item is not a string; ignoring.")
                continue
            b64_part = item
            content_type = "image/png"
            filename = f"image_{idx}.png"
            if item.startswith("data:"):
                try:
                    header, b64_part = item.split(",", 1)
                    # data:image/png;base64
                    if header.startswith("data:") and ";base64" in header:
                        content_type = header[5:].split(";")[0]
                        ext = content_type.split("/")[-1]
                        filename = f"image_{idx}.{ext}"
                except Exception:
                    pass
            try:
                image_bytes = base64.b64decode(b64_part)
            except Exception:
                print("[api] Could not base64-decode a JSON image; ignoring.")
                continue

            uploaded_names.append(upload_image_bytes(image_bytes, filename=filename, content_type=content_type))

    else:
        # multipart/form-data
        prompt_text = request.form.get("prompt")
        raw_model_id = request.form.get("model", "flux-kontext-dev")
        response_format = (request.form.get("response_format") or "b64_json").lower()
        output_format = (request.form.get("output_format") or "png").lower()
        output_compression = clamp_int(request.form.get("output_compression"), 100, 0, 100, "output_compression")
        size_str = request.form.get("size", "auto")
        n = clamp_int(request.form.get("n"), 1, 1, 10, "n")
        stream = (request.form.get("stream", "false").lower() == "true")
        partial_images = clamp_int(request.form.get("partial_images"), 0, 0, 3, "partial_images")
        background = request.form.get("background")
        quality = request.form.get("quality")
        input_fidelity = request.form.get("input_fidelity")

        # Collect images from common OpenAI styles:
        files = []
        for key in ("image", "image[]", "file", "file[]"):
            files.extend(request.files.getlist(key))

        # Fallback: collect all file keys
        if not files and request.files:
            print(f"[api] Warning: 'image' key not found. Found keys: {list(request.files.keys())}")
            for key in request.files:
                files.extend(request.files.getlist(key))

        if not files:
            return openai_error("No image provided. Ensure multipart/form-data includes an image file.", param="image")

        # mask is supported by OpenAI edits API, but may not be supported by your Comfy workflows
        if "mask" in request.files:
            print("[api] mask provided but not supported by current workflows; ignoring.")

        for f in files:
            uploaded_names.append(upload_image(f))

    if not prompt_text:
        return openai_error("Missing required parameter: prompt", param="prompt")

    if response_format not in ("b64_json", "url"):
        return openai_error("response_format must be 'b64_json' or 'url'", param="response_format")

    if output_format not in ("png", "jpeg", "webp"):
        return openai_error("output_format must be 'png', 'jpeg', or 'webp'", param="output_format")

    width, height = parse_size(size_str)
    model_id = normalize_model_id(raw_model_id)

    # Log/ignore unsupported OpenAI params
    if input_fidelity not in (None, "", [], {}):
        print(f"[{model_id}] Ignoring unsupported parameter: input_fidelity={input_fidelity}")
    for k in ("user", "moderation"):
        if (request.form.get(k) if not request.is_json else (request.get_json(silent=True) or {}).get(k)) not in (None, "", [], {}):
            print(f"[{model_id}] Ignoring unsupported parameter: {k}")

    def run_edit_once():
        seed = random.randint(1, 10**15)
        wf = workflows.get_workflow(
            model_id,
            mode="edit",
            prompt=prompt_text,
            width=width,
            height=height,
            seed=seed,
            images=uploaded_names,
            quality=quality,
            background=background
        )
        if not wf:
            raise ValueError(f"Model {model_id} (raw: {raw_model_id}) not supported for edits.")

        raw_images = execute_workflow(wf)
        if not raw_images:
            raise ValueError("No images returned from workflow execution.")

        converted = [convert_image_bytes(b, output_format=output_format, output_compression=output_compression, background=background) for b in raw_images]
        return converted

    def build_data_items(images_bytes):
        items = []
        if response_format == "b64_json":
            for b in images_bytes:
                items.append({"b64_json": base64.b64encode(b).decode("utf-8")})
        else:
            for b in images_bytes:
                token = store_temp_image(b, output_format)
                items.append({"url": make_public_url(f"/v1/images/tmp/{token}")})
        return items

    # Streaming SSE for edits
    if stream:
        if n != 1:
            print("[api] stream=true with n!=1: only the first image will be streamed; others ignored.")
            n = 1

        def gen():
            created_at = _now()
            try:
                imgs = run_edit_once()
                final_img = imgs[0]

                if partial_images > 0:
                    yield sse_format("image_edit.partial_image", {
                        "type": "image_edit.partial_image",
                        "b64_json": base64.b64encode(final_img).decode("utf-8"),
                        "created_at": created_at,
                        "size": f"{width}x{height}",
                        "quality": quality or "auto",
                        "background": background or "auto",
                        "output_format": output_format,
                        "partial_image_index": 0
                    })

                yield sse_format("image_edit.completed", {
                    "type": "image_edit.completed",
                    "b64_json": base64.b64encode(final_img).decode("utf-8"),
                    "created_at": created_at,
                    "size": f"{width}x{height}",
                    "quality": quality or "auto",
                    "background": background or "auto",
                    "output_format": output_format,
                })
            except Exception as e:
                err = {"error": {"message": str(e), "type": "server_error"}}
                yield sse_format("error", err)

        return Response(gen(), mimetype="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })

    created = _now()
    out_images = []
    try:
        while len(out_images) < n:
            imgs = run_edit_once()
            out_images.extend(imgs)
        out_images = out_images[:n]
    except Exception as e:
        return openai_error(str(e), status=400)

    return build_images_response(
        created=created,
        data_list=build_data_items(out_images),
        response_format=response_format,
        output_format=output_format,
        size=f"{width}x{height}",
        quality=quality or "auto",
        background=background or "auto",
    )

# ------------------------------------------------------------
# POST /v1/images/variations (not implemented in this wrapper)
# ------------------------------------------------------------
@app.route("/v1/images/variations", methods=["POST"])
def images_variations():
    return openai_error("images/variations is not supported by this ComfyUI wrapper.", status=501)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)