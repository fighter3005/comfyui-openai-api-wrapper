import os
import re
import json
import base64
import requests

API_BASE = os.environ.get("API_BASE", "http://localhost:5000/v1")
TIMEOUT_SECONDS = 3600

TEST_IMG1 = os.environ.get("TEST_IMG1", "test_input.png")
TEST_IMG2 = os.environ.get("TEST_IMG2", "test_input2.jpg")


def save_bytes(path, b):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b)
    print("saved:", path)


def save_b64_image(b64_data, filename):
    save_bytes(filename, base64.b64decode(b64_data))


def get_models():
    r = requests.get(f"{API_BASE}/models", timeout=60)
    r.raise_for_status()
    data = r.json()
    ids = [m["id"] for m in data.get("data", [])]
    print("models:", ids)
    return ids


def test_generation_b64_png():
    payload = {
        "model": "z-image-turbo",
        "prompt": "A cinematic photo of a robot painting a canvas",
        "size": "512x512",
        "response_format": "b64_json",
        "output_format": "png",
    }
    r = requests.post(f"{API_BASE}/images/generations", json=payload, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    out = r.json()
    save_b64_image(out["data"][0]["b64_json"], "out/gen_b64_png.png")


def test_generation_url_jpeg():
    payload = {
        "model": "z-image-turbo",
        "prompt": "A studio portrait of a cat wearing sunglasses",
        "size": "512x512",
        "response_format": "url",
        "output_format": "jpeg",
        "output_compression": 80,
    }
    r = requests.post(f"{API_BASE}/images/generations", json=payload, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    out = r.json()
    url = out["data"][0]["url"]
    img = requests.get(url, timeout=TIMEOUT_SECONDS).content
    save_bytes("out/gen_url.jpeg", img)


def test_generation_n2_webp_b64():
    payload = {
        "model": "z-image-turbo",
        "prompt": "A watercolor painting of a lighthouse on a cliff",
        "size": "512x512",
        "n": 2,
        "response_format": "b64_json",
        "output_format": "webp",
        "output_compression": 90,
    }
    r = requests.post(f"{API_BASE}/images/generations", json=payload, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    out = r.json()
    for i, item in enumerate(out["data"]):
        save_b64_image(item["b64_json"], f"out/gen_n2_{i}.webp")


def _parse_sse(stream_iter_lines):
    """
    Minimal SSE parser: yields (event_name, data_dict).
    """
    event_name = None
    data_lines = []

    for raw in stream_iter_lines:
        if raw is None:
            continue

        line = raw.decode("utf-8", errors="replace").rstrip("\n")

        # IMPORTANT: blank line terminates one SSE event
        if line == "":
            if data_lines:
                data_str = "\n".join(data_lines)
                try:
                    yield event_name, json.loads(data_str)
                except Exception:
                    yield event_name, {"_raw": data_str}
            event_name = None
            data_lines = []
            continue

        # comment/keepalive lines (optional)
        if line.startswith(":"):
            continue

        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())


def test_generation_stream_partial():
    payload = {
        "model": "z-image-turbo",
        "prompt": "A tiny dragon sitting on a book, ultra detailed",
        "size": "512x512",
        "stream": True,
        "partial_images": 1,
        "output_format": "png",
    }
    with requests.post(f"{API_BASE}/images/generations", json=payload, stream=True, timeout=TIMEOUT_SECONDS) as r:
        r.raise_for_status()
        last_b64 = None
        for event, data in _parse_sse(r.iter_lines()):
            print("SSE event:", event, "type:", data.get("type"))
            if "b64_json" in data:
                last_b64 = data["b64_json"]
            if data.get("type") == "image_generation.completed":
                break
        if last_b64:
            save_b64_image(last_b64, "out/gen_stream_final.png")


def test_edit_multipart_image_key_b64():
    # standard single image upload using key 'image'
    with open(TEST_IMG1, "rb") as img:
        files = {"image": ("input.png", img, "image/png")}
        data = {
            "model": "qwen-image-edit",
            "prompt": "Turn the background into a blue ocean",
            "size": "768x1280",
            "response_format": "b64_json",
            "output_format": "png",
        }
        r = requests.post(f"{API_BASE}/images/edits", files=files, data=data, timeout=TIMEOUT_SECONDS)
        r.raise_for_status()
        out = r.json()
        save_b64_image(out["data"][0]["b64_json"], "out/edit_multipart_image.png")


def test_edit_multipart_image_array_multi():
    # OpenAI docs show image[] repeated [3]
    files = []
    f1 = open(TEST_IMG1, "rb")
    f2 = open(TEST_IMG2, "rb")
    try:
        files = [
            ("image[]", ("img1.png", f1, "image/png")),
            ("image[]", ("img2.jpg", f2, "image/jpeg")),
        ]
        data = {
            "model": "flux-2-dev",
            "prompt": "Make the person wear the boss t-shirt.",
            "size": "768x1280",
            "response_format": "b64_json",
            "output_format": "webp",
            "output_compression": "85",
        }
        r = requests.post(f"{API_BASE}/images/edits", files=files, data=data, timeout=TIMEOUT_SECONDS)
        r.raise_for_status()
        out = r.json()
        save_b64_image(out["data"][0]["b64_json"], "out/edit_multi_imagearray.webp")
    finally:
        f1.close()
        f2.close()


def test_edit_multipart_file_key_url():
    # some clients might send 'file' instead of 'image'
    with open(TEST_IMG1, "rb") as img:
        files = {"file": ("input.png", img, "image/png")}
        data = {
            "model": "flux-kontext-dev",
            "prompt": "Make the womans pullover green.",
            "size": "1024x1024",
            "response_format": "url",
            "output_format": "jpeg",
            "output_compression": "75",
        }
        r = requests.post(f"{API_BASE}/images/edits", files=files, data=data, timeout=TIMEOUT_SECONDS)
        r.raise_for_status()
        out = r.json()
        url = out["data"][0]["url"]
        img_bytes = requests.get(url, timeout=TIMEOUT_SECONDS).content
        save_bytes("out/edit_filekey_url.jpeg", img_bytes)


def test_edit_json_base64_data_url():
    # JSON edit mode: send image as data URL
    with open(TEST_IMG1, "rb") as f:
        raw = f.read()
    data_url = "data:image/png;base64," + base64.b64encode(raw).decode("utf-8")

    payload = {
        "model": "qwen-image-edit",
        "prompt": "Add a small red balloon in the sky",
        "size": "768x1280",
        "response_format": "b64_json",
        "output_format": "png",
        "image": data_url,
    }
    r = requests.post(f"{API_BASE}/images/edits", json=payload, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    out = r.json()
    save_b64_image(out["data"][0]["b64_json"], "out/edit_json_dataurl.png")


def test_edit_stream():
    # stream edit response (SSE) [2]
    with open(TEST_IMG1, "rb") as img:
        files = {"image": ("input.png", img, "image/png")}
        data = {
            "model": "qwen-image-edit",
            "prompt": "Make the sky purple",
            "size": "768x1280",
            "stream": "true",
            "partial_images": "1",
            "output_format": "png",
        }
        with requests.post(f"{API_BASE}/images/edits", files=files, data=data, stream=True, timeout=TIMEOUT_SECONDS) as r:
            r.raise_for_status()
            last_b64 = None
            for event, payload in _parse_sse(r.iter_lines()):
                print("SSE event:", event, "type:", payload.get("type"))
                if "b64_json" in payload:
                    last_b64 = payload["b64_json"]
                if payload.get("type") == "image_edit.completed":
                    break
            if last_b64:
                save_b64_image(last_b64, "out/edit_stream_final.png")


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)

    get_models()

    test_generation_b64_png()
    test_generation_url_jpeg()
    test_generation_n2_webp_b64()
    test_generation_stream_partial()

    test_edit_multipart_image_key_b64()
    test_edit_multipart_image_array_multi()
    test_edit_multipart_file_key_url()
    test_edit_json_base64_data_url()
    test_edit_stream()

    print("All tests finished.")