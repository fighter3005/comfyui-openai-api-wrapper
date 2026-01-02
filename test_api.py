import base64
import requests
from openai import OpenAI

# =========================
# Configuration
# =========================
API_BASE = "http://localhost:5000/v1"
API_KEY = "sk-local-test"  # Wrapper does not validate keys
TIMEOUT_SECONDS = 3600

client = OpenAI(
    base_url=API_BASE,
    api_key=API_KEY,
    timeout=TIMEOUT_SECONDS
)

# =========================
# Helpers
# =========================
def save_b64_image(b64_data, filename):
    try:
        image_data = base64.b64decode(b64_data)
        with open(filename, "wb") as f:
            f.write(image_data)
        print(f"✅ Saved: {filename}")
    except Exception as e:
        print(f"❌ Error saving {filename}: {e}")

# =========================
# Test 1: z-image Generation
# =========================
def test_generation_flux2():
    print("\n--- z-image Generation ---")
    response = client.images.generate(
        model="z-image-turbo",
        prompt="A cinematic photo of a robot painting a canvas",
        size="512x512",
        response_format="b64_json"
    )
    save_b64_image(response.data[0].b64_json, "z-image-turbo_gen.png")

# =========================
# Test 2: Qwen Image Edit (1 image)
# =========================
def test_qwen_edit():
    print("\n--- Qwen Image Edit (1 image) ---")

    with open("test_input.png", "rb") as img:
        response = client.images.edit(
            model="qwen-image-edit",
            image=img,
            prompt="Turn the background into a blue ocean",
            size="768x1280",
            response_format="b64_json"
        )

    save_b64_image(response.data[0].b64_json, "qwen_edit.png")

# =========================
# Test 3: Flux 2 Multi-Image Edit
# =========================
def test_flux2_multi_edit():
    print("\n--- Flux 2 Multi-Image Edit (2 images) ---")

    url = f"{API_BASE}/images/edits"

    files = [
        ("image", ("img1.png", open("test_input.png", "rb"), "image/png")),
        ("image", ("img2.jpg", open("test_input2.jpg", "rb"), "image/jpeg")),
    ]

    data = {
        "model": "flux-2-dev",
        "prompt": "Make the woman wear a Boss t-shirt",
        "size": "768x1280",
    }

    try:
        response = requests.post(
            url,
            files=files,
            data=data,
            timeout=TIMEOUT_SECONDS,
        )

        if response.status_code != 200:
            print("❌ Edit failed:", response.text)
            return

        result = response.json()
        save_b64_image(result["data"][0]["b64_json"], "flux2_multi_edit.png")

    finally:
        for _, file_tuple in files:
            file_tuple[1].close()

# =========================
# Main
# =========================
if __name__ == "__main__":
    test_generation_flux2()
    test_qwen_edit()
    test_flux2_multi_edit()