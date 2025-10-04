import runpod
import subprocess
import uuid
import sys
import os
import requests

def generate_video(input_params):
    """
    Step-Video-TI2V handler for RunPod.
    Supports both text-to-video (T2V) and image-to-video (I2V).
    """
    prompt = input_params.get("prompt", "A cinematic view of a futuristic city at night.")
    image_url = input_params.get("image", None)
    output_name = f"/workspace/output_{uuid.uuid4().hex}.mp4"

    # ✅ Download image if provided
    img_path = None
    if image_url:
        try:
            img_path = f"/workspace/{uuid.uuid4().hex}.png"
            r = requests.get(image_url, timeout=30)
            r.raise_for_status()
            with open(img_path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            return {"error": f"Failed to download image: {str(e)}"}

    # ✅ Build the command for Step-Video-TI2V
    cmd = [sys.executable, "run_single.py", "--prompt", prompt, "--output_file_name", output_name]

    # ⚙️ Add image if provided
    if img_path:
        cmd += ["--first_image_path", img_path]

    # ✅ Run the model
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, check=True)
        logs = result.stdout + "\n" + result.stderr
    except subprocess.CalledProcessError as e:
        return {"error": str(e), "logs": e.stderr}

    # ✅ Return generated video info
    return {
        "video_path": output_name,
        "logs": logs,
        "prompt": prompt,
        "input_image": image_url if image_url else None
    }

runpod.serverless.start({"handler": generate_video})
