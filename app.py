import os
import json
import uuid
import random
import time
import base64
import requests
import websocket
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import workflows

app = Flask(__name__)

# Configuration
COMFY_HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1")
COMFY_PORT = os.environ.get("COMFYUI_PORT", "8188")
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"
WS_URL = f"ws://{COMFY_HOST}:{COMFY_PORT}/ws"

# --- Helpers ---

def parse_size(size_str):
    """Parses OpenAI 'size' param (e.g., '1024x1024') into (width, height)."""
    try:
        if not size_str or "x" not in size_str:
            return 1024, 1024
        w, h = map(int, size_str.split("x"))
        return w, h
    except:
        return 1024, 1024

# --- ComfyUI Client Logic ---

def upload_image(file_storage, image_type="input"):
    """Uploads an image file to ComfyUI [2]."""
    filename = secure_filename(file_storage.filename)
    files = {"image": (filename, file_storage.read(), file_storage.content_type)}
    data = {"type": image_type, "overwrite": "true"}
    
    try:
        response = requests.post(f"{COMFY_URL}/upload/image", files=files, data=data)
        response.raise_for_status()
        # ComfyUI returns the name and subfolder
        result = response.json()
        return result.get("name", filename)
    except Exception as e:
        print(f"Error uploading image: {e}")
        raise

def queue_prompt(prompt_workflow):
    """Queues a workflow via the /prompt endpoint [1]."""
    client_id = str(uuid.uuid4())
    p = {"prompt": prompt_workflow, "client_id": client_id}
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(f"{COMFY_URL}/prompt", json=p, headers=headers)
        response.raise_for_status()
        return response.json()['prompt_id'], client_id
    except Exception as e:
        print(f"Error queuing prompt: {e}")
        raise

def get_history(prompt_id):
    """Retrieves history for a specific prompt [2]."""
    response = requests.get(f"{COMFY_URL}/history/{prompt_id}")
    return response.json()

def get_image_raw(filename, subfolder, folder_type):
    """Downloads the raw image data [2][4]."""
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    response = requests.get(f"{COMFY_URL}/view", params=params)
    return response.content

def execute_workflow(workflow):
    """
    Full execution cycle: Connect WS -> Queue -> Wait -> Download.
    Based on WebSocket logic from source [4].
    """
    prompt_id, client_id = queue_prompt(workflow)
    
    ws = websocket.WebSocket()
    ws.connect(f"{WS_URL}?clientId={client_id}")
    
    # Wait for execution completion
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            # Check if our prompt is done
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break # Execution is done
    ws.close()

    # Fetch results from history
    history = get_history(prompt_id).get(prompt_id, {})
    outputs = history.get('outputs', {})
    
    generated_images = []
    
    for node_id in outputs:
        node_output = outputs[node_id]
        if 'images' in node_output:
            for image in node_output['images']:
                raw_data = get_image_raw(image['filename'], image['subfolder'], image['type'])
                b64_img = base64.b64encode(raw_data).decode('utf-8')
                generated_images.append({"b64_json": b64_img})
                
    return generated_images

# --- OpenAI Compatible Endpoints ---

@app.route('/v1/models', methods=['GET'])
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {"id": "flux-kontext-dev", "object": "model", "created": int(time.time()), "owned_by": "comfyui"},
            {"id": "qwen-image-edit-2509", "object": "model", "created": int(time.time()), "owned_by": "comfyui"},
            {"id": "flux-krea-dev", "object": "model", "created": int(time.time()), "owned_by": "comfyui"},
            {"id": "flux-2-dev", "object": "model", "created": int(time.time()), "owned_by": "comfyui"}
        ]
    })

@app.route('/v1/images/generations', methods=['POST'])
def generate_image():
    """Handles image generation (Text-to-Image)."""
    data = request.json
    model_id = data.get("model", "flux-krea-dev")
    prompt_text = data.get("prompt")
    size_str = data.get("size", "512x512")
    width, height = parse_size(size_str)
    
    # Select workflow
    if model_id == "flux-2-dev":
        workflow = workflows.get_workflow(model_id, mode="gen")
    elif model_id == "flux-krea-dev":
        workflow = workflows.get_workflow(model_id)
    else:
        return jsonify({"error": {"message": f"Model {model_id} not supported for generations."}}), 400

    seed = random.randint(1, 10**15)

    try:
        if model_id == "flux-krea-dev":
            # 1. Update Prompt (Node 45)
            workflow["45"]["inputs"]["text"] = prompt_text
            # 2. Randomize Seed (Node 31)
            workflow["31"]["inputs"]["seed"] = seed
            # 3. Update Size (Node 27)
            workflow["27"]["inputs"]["width"] = width
            workflow["27"]["inputs"]["height"] = height

        elif model_id == "flux-2-dev":
            # 1. Update Prompt (Node 6)
            workflow["6"]["inputs"]["text"] = prompt_text
            # 2. Randomize Seed (Node 25 uses 'noise_seed')
            workflow["25"]["inputs"]["noise_seed"] = seed
            # 3. Update Size (EmptyFlux2LatentImage: 47, Flux2Scheduler: 48)
            workflow["47"]["inputs"]["width"] = width
            workflow["47"]["inputs"]["height"] = height
            workflow["48"]["inputs"]["width"] = width
            workflow["48"]["inputs"]["height"] = height

        images = execute_workflow(workflow)
        return jsonify({
            "created": int(time.time()),
            "data": images
        })
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500


@app.route('/v1/images/edits', methods=['POST'])
def edit_image():
    """Handles image editing (Image+Text-to-Image)."""
    # OpenAI usually allows only 1 image 'image', but we support multiple for flux-2
    # We use getlist to capture multiple files if sent as 'image' or 'image[]'
    files = request.files.getlist('image')
    if not files:
        # Fallback to check specific keys if client sends distinct keys
        if 'image' in request.files:
            files = [request.files['image']]
    
    if not files:
         return jsonify({"error": {"message": "No image provided."}}), 400

    prompt_text = request.form.get("prompt")
    model_id = request.form.get("model", "flux-kontext-dev")
    size_str = request.form.get("size", "512x512")
    width, height = parse_size(size_str)
    seed = random.randint(1, 10**15)

    # Determine workflow based on model and image count
    workflow = None
    
    if model_id == "flux-kontext-dev":
        workflow = workflows.get_workflow(model_id)
    elif model_id == "qwen-image-edit-2509":
        workflow = workflows.get_workflow(model_id)
    elif model_id == "flux-2-dev":
        img_count = len(files)
        if img_count < 1 or img_count > 2:
            return jsonify({"error": {"message": f"Flux 2 edit requires 1 or 2 images, got {img_count}."}}), 400
        workflow = workflows.get_workflow(model_id, mode="edit", img_count=img_count)
    
    if not workflow:
        return jsonify({"error": {"message": f"Model {model_id} not found."}}), 400

    try:
        # Upload images and get filenames
        uploaded_names = []
        for f in files:
            uploaded_names.append(upload_image(f))

        # --- Apply Updates to Workflow ---

        if model_id == "flux-kontext-dev":
            workflow["6"]["inputs"]["text"] = prompt_text
            workflow["142"]["inputs"]["image"] = uploaded_names[0]
            workflow["31"]["inputs"]["seed"] = seed
            # Note: Flux Kontext workflow provided is complex and relies on ImageStitch/Scale inputs.
            # Changing latent size might break Stitching logic, so strictly ignoring size here unless mapped manually.

        elif model_id == "qwen-image-edit-2509":
            workflow["111"]["inputs"]["prompt"] = prompt_text
            workflow["78"]["inputs"]["image"] = uploaded_names[0]
            workflow["3"]["inputs"]["seed"] = seed
            # 1MP Scale logic as requested
            workflow["93"]["inputs"]["megapixels"] = 1.0 

        elif model_id == "flux-2-dev":
            # Common updates
            workflow["6"]["inputs"]["text"] = prompt_text
            workflow["25"]["inputs"]["noise_seed"] = seed
            # Update Size (Node 47 & 48)
            workflow["47"]["inputs"]["width"] = width
            workflow["47"]["inputs"]["height"] = height
            workflow["48"]["inputs"]["width"] = width
            workflow["48"]["inputs"]["height"] = height

            if len(files) == 1:
                # Node 46 is input image
                workflow["46"]["inputs"]["image"] = uploaded_names[0]
            elif len(files) == 2:
                # Node 42 is image 1, Node 46 is image 2
                workflow["42"]["inputs"]["image"] = uploaded_names[0]
                workflow["46"]["inputs"]["image"] = uploaded_names[1]

        images = execute_workflow(workflow)
        
        return jsonify({
            "created": int(time.time()),
            "data": images
        })
        
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)