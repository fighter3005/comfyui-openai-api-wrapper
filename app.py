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

# --- ComfyUI Client Logic ---
def upload_image(file_storage, image_type="input"):
    """
    Uploads an image file to ComfyUI [1][2].
    Adds a unique prefix to the filename to prevent overwriting when multiple
    files share the same name (common with proxies/APIs).
    """
    original_name = secure_filename(file_storage.filename)
    if not original_name:
        original_name = "image.png"
    
    # Generate a unique filename using UUID
    unique_prefix = str(uuid.uuid4())[:8]
    filename = f"{unique_prefix}_{original_name}"
    
    # ComfyUI expects the file field to be named 'image'
    files = {"image": (filename, file_storage.read(), file_storage.content_type)}
    data = {"type": image_type, "overwrite": "true"}
    
    try:
        response = requests.post(f"{COMFY_URL}/upload/image", files=files, data=data)
        response.raise_for_status()
        
        # Reset file pointer
        file_storage.seek(0)
        
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
    """Retrieves history for a specific prompt [1][2]."""
    response = requests.get(f"{COMFY_URL}/history/{prompt_id}")
    return response.json()

def get_image_raw(filename, subfolder, folder_type):
    """Downloads the raw image data [1]."""
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    response = requests.get(f"{COMFY_URL}/view", params=params)
    return response.content

def execute_workflow(workflow):
    """
    Full execution cycle: Connect WS -> Queue -> Wait -> Download.
    Based on WebSocket logic from source [2][4].
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

def normalize_model_id(model_id):
    """
    Handles prefixes from LiteLLM.
    Your config maps to 'openai/flux-2-dev', so we strip 'openai/' or 'comfy-'.
    """
    if not model_id:
        return "flux-kontext-dev" # Default
    if model_id.startswith("openai/"):
        return model_id[7:]
    if model_id.startswith("comfy-"):
        return model_id[6:]
    return model_id

# --- OpenAI Compatible Endpoints ---
@app.route('/v1/models', methods=['GET'])
def list_models():
    """Lists available models compatible with this wrapper."""
    return jsonify({
        "object": "list",
        "data": [
            {"id": "flux-kontext-dev", "object": "model", "created": int(time.time()), "owned_by": "comfyui"},
            {"id": "qwen-image-edit", "object": "model", "created": int(time.time()), "owned_by": "comfyui"},
            {"id": "flux-krea-dev", "object": "model", "created": int(time.time()), "owned_by": "comfyui"},
            {"id": "flux-2-dev", "object": "model", "created": int(time.time()), "owned_by": "comfyui"},
            {"id": "z-image-turbo", "object": "model", "created": int(time.time()), "owned_by": "comfyui"},
            {"id": "flux-dev-checkpoint", "object": "model", "created": int(time.time()), "owned_by": "comfyui"},
            {"id": "sd3-5-simple", "object": "model", "created": int(time.time()), "owned_by": "comfyui"},
            {"id": "flux-schnell", "object": "model", "created": int(time.time()), "owned_by": "comfyui"}
        ]
    })

@app.route('/v1/images/generations', methods=['POST'])
def generate_image():
    """Handles image generation (Text-to-Image)."""
    data = request.json
    raw_model_id = data.get("model", "flux-krea-dev")
    model_id = normalize_model_id(raw_model_id)
    prompt_text = data.get("prompt")
    size_str = data.get("size", "1024x1024")
    width, height = parse_size(size_str)
    
    # Select workflow
    workflow = workflows.get_workflow(model_id, mode="gen")
    
    if not workflow:
        return jsonify({"error": {"message": f"Model {model_id} (raw: {raw_model_id}) not supported for generations."}}), 400
        
    seed = random.randint(1, 10**15)
    
    try:
        # Map parameters to specific nodes based on the model/workflow structure
        if model_id == "flux-krea-dev":
            workflow["45"]["inputs"]["text"] = prompt_text
            workflow["31"]["inputs"]["seed"] = seed
            workflow["27"]["inputs"]["width"] = width
            workflow["27"]["inputs"]["height"] = height
            
        elif model_id == "flux-2-dev":
            workflow["6"]["inputs"]["text"] = prompt_text
            workflow["25"]["inputs"]["noise_seed"] = seed
            workflow["47"]["inputs"]["width"] = width
            workflow["47"]["inputs"]["height"] = height
            workflow["48"]["inputs"]["width"] = width
            workflow["48"]["inputs"]["height"] = height
            
        elif model_id == "z-image-turbo":
            workflow["6"]["inputs"]["text"] = prompt_text
            workflow["3"]["inputs"]["seed"] = seed
            workflow["13"]["inputs"]["width"] = width
            workflow["13"]["inputs"]["height"] = height
            
        # New models added from context [1][2][3]
        elif model_id == "flux-dev-checkpoint":
            workflow["6"]["inputs"]["text"] = prompt_text
            workflow["31"]["inputs"]["seed"] = seed
            workflow["27"]["inputs"]["width"] = width
            workflow["27"]["inputs"]["height"] = height
            
        elif model_id == "sd3-5-simple":
            workflow["16"]["inputs"]["text"] = prompt_text
            workflow["3"]["inputs"]["seed"] = seed
            workflow["53"]["inputs"]["width"] = width
            workflow["53"]["inputs"]["height"] = height
            
        elif model_id == "flux-schnell":
            workflow["6"]["inputs"]["text"] = prompt_text
            workflow["31"]["inputs"]["seed"] = seed
            workflow["27"]["inputs"]["width"] = width
            workflow["27"]["inputs"]["height"] = height
            
        images = execute_workflow(workflow)
        
        return jsonify({
            "created": int(time.time()),
            "data": images
        })
    except Exception as e:
        print(e)
        return jsonify({"error": {"message": str(e)}}), 500

@app.route('/v1/images/edits', methods=['POST'])
def edit_image():
    """Handles image editing (Image+Text-to-Image)."""
    # 1. Robust File Retrieval
    # Check 'image', 'file', or fallback to iterating all files
    files = request.files.getlist('image')
    if not files:
        files = request.files.getlist('file')
    if not files and request.files:
        print(f"Warning: Specific key not found. Found keys: {list(request.files.keys())}")
        for key in request.files:
            files.extend(request.files.getlist(key))
            
    if not files:
         return jsonify({"error": {"message": "No image provided. Ensure multipart/form-data includes an image file."}}), 400
         
    # 2. Extract Parameters
    prompt_text = request.form.get("prompt")
    raw_model_id = request.form.get("model", "flux-kontext-dev")
    model_id = normalize_model_id(raw_model_id)
    size_str = request.form.get("size", "1024x1024")
    width, height = parse_size(size_str)
    seed = random.randint(1, 10**15)
    
    # 3. Determine workflow
    workflow = None
    if model_id == "flux-kontext-dev":
        workflow = workflows.get_workflow(model_id)
    elif model_id == "qwen-image-edit":
        workflow = workflows.get_workflow(model_id)
    elif model_id == "flux-2-dev":
        img_count = len(files)
        if img_count < 1 or img_count > 3:
            return jsonify({"error": {"message": f"Flux 2 edit requires 1, 2, or 3 images, got {img_count}."}}), 400
        workflow = workflows.get_workflow(model_id, mode="edit", img_count=img_count)
        
    if not workflow:
        return jsonify({"error": {"message": f"Model {model_id} (raw: {raw_model_id}) not found."}}), 400

    try:
        # Upload images
        uploaded_names = []
        for f in files:
            uploaded_names.append(upload_image(f))
            
        # --- Apply Updates to Workflow ---
        if model_id == "flux-kontext-dev":
            workflow["6"]["inputs"]["text"] = prompt_text
            workflow["142"]["inputs"]["image"] = uploaded_names[0]
            workflow["31"]["inputs"]["seed"] = seed
            
        elif model_id == "qwen-image-edit":
            workflow["111"]["inputs"]["prompt"] = prompt_text
            workflow["78"]["inputs"]["image"] = uploaded_names[0]
            workflow["3"]["inputs"]["seed"] = seed
            workflow["93"]["inputs"]["megapixels"] = 1.0
            
        elif model_id == "flux-2-dev":
            workflow["6"]["inputs"]["text"] = prompt_text
            workflow["25"]["inputs"]["noise_seed"] = seed
            workflow["47"]["inputs"]["width"] = width
            workflow["47"]["inputs"]["height"] = height
            workflow["48"]["inputs"]["width"] = width
            workflow["48"]["inputs"]["height"] = height
            
            if len(files) == 1:
                workflow["46"]["inputs"]["image"] = uploaded_names[0]
            elif len(files) == 2:
                workflow["42"]["inputs"]["image"] = uploaded_names[0]
                workflow["46"]["inputs"]["image"] = uploaded_names[1]
            elif len(files) == 3:
                workflow["42"]["inputs"]["image"] = uploaded_names[0]
                workflow["46"]["inputs"]["image"] = uploaded_names[1]
                workflow["68"]["inputs"]["image"] = uploaded_names[2]
                
        images = execute_workflow(workflow)
        
        return jsonify({
            "created": int(time.time()),
            "data": images
        })
    except Exception as e:
        print(f"Workflow execution failed: {e}")
        return jsonify({"error": {"message": str(e)}}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)