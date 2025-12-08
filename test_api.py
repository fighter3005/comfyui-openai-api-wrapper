import base64
import os
import requests
from openai import OpenAI
from PIL import Image

# Configuration
API_BASE = "https://openai.kemme.xyz/v1" 
API_KEY = "sk-Sqap8QLVpLUrea8H3ZjS0g"

# SET TIMEOUT: 3600 seconds = 1 Hour (covers the >30 min requirement)
TIMEOUT_SECONDS = 3600 

# Initialize OpenAI Client with the custom timeout
# This applies to all calls made via 'client.images.generate', etc.
client = OpenAI(
    base_url=API_BASE, 
    api_key=API_KEY,
    timeout=TIMEOUT_SECONDS
)

def save_b64_image(b64_data, filename):
    """Decodes base64 string and saves to file."""
    try:
        image_data = base64.b64decode(b64_data)
        with open(filename, "wb") as f:
            f.write(image_data)
        print(f"✅ Saved: {filename}")
    except Exception as e:
        print(f"❌ Error saving {filename}: {e}")

# --- Test 1: Image Generation (Flux 2) ---
def test_generation():
    print("\n--- Testing Generation (Flux 2) ---")
    try:
        # Uses the timeout defined in the client initialization
        response = client.images.generate(
            model="comfy-flux-2-dev",
            prompt="A cinematic photo of a robot painting a canvas",
            size="512x512",
            response_format="b64_json" 
        )
        save_b64_image(response.data[0].b64_json, "output_gen_flux2.png")
    except Exception as e:
        print(f"❌ Generation failed: {e}")

# --- Test 2: Standard Image Edit (Qwen) ---
def test_edit_standard():
    print("\n--- Testing Edit (Qwen - 1 Image) ---")
    try:
        # Uses the timeout defined in the client initialization
        with open("test_input.png", "rb") as img:
            response = client.images.edit(
                model="comfy-qwen-image-edit-2509",
                image=img,
                prompt="Turn the background into a blue ocean",
                size="768x1280", 
                response_format="b64_json"
            )
        save_b64_image(response.data[0].b64_json, "output_edit_qwen.png")
    except Exception as e:
        print(f"❌ Edit failed: {e}")

# --- Test 3: Multi-Image Edit (Flux 2) ---
def test_edit_multi_image():
    print("\n--- Testing Multi-Image Edit (Flux 2 - Custom) ---")
    
    url = f"{API_BASE}/images/edits"
    
    # Ensure these files exist before running
    files = [
        ('image', ('img1.png', open("test_input.png", 'rb'), 'image/png')),
        ('image', ('img2.png', open("test_input2.jpg", 'rb'), 'image/jpg'))
    ]
    data = {
        "model": "comfy-flux-2-dev",
        "prompt": "Make the woman wear the Boss t-shirt.",
        "size": "768x1280"
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    try:
        # Added explicit 'timeout' argument here for requests
        response = requests.post(
            url, 
            files=files, 
            data=data, 
            headers=headers,
            timeout=TIMEOUT_SECONDS 
        )
        
        if response.status_code == 200:
            result = response.json()
            b64_str = result['data'][0]['b64_json']
            save_b64_image(b64_str, "output_edit_flux2_multi.png")
        else:
            print(f"❌ Multi-image edit failed: {response.text}")
    except Exception as e:
        print(f"❌ Request error: {e}")
    finally:
        # Close file handles
        for _, file_tuple in files:
            file_tuple[1].close()

if __name__ == "__main__":
    # create_dummy_image() # Uncomment if you need to generate input files
    # test_generation()
    # test_edit_standard()
    test_edit_multi_image()