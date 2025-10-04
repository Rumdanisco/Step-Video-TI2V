import torch
from diffusers import DiffusionPipeline
from stepvideo.utils import setup_seed
from stepvideo.config import parse_args
import os


def main():
    args = parse_args()
    setup_seed(args.seed)

    # 🧠 Load model from Hugging Face
    model_repo = os.getenv("MODEL_REPO", "stepfun-ai/Step-Video-TI2V")
    token = os.getenv("HF_TOKEN", None)

    print(f"🚀 Loading Step-Video-TI2V from Hugging Face: {model_repo}")

    # ✅ Use Hugging Face DiffusionPipeline
    pipeline = DiffusionPipeline.from_pretrained(
        model_repo,
        torch_dtype=torch.float16,
        use_auth_token=token
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    prompt = args.prompt
    print(f"🎬 Generating video for prompt: {prompt}")

    # 🧩 Run generation (for text-to-video or image-to-video)
    video = pipeline(prompt=prompt)

    output_file = args.output_file_name or "/workspace/output.mp4"
    video["videos"][0].save(output_file)

    print(f"✅ Generation complete: {output_file}")
    return output_file


if __name__ == "__main__":
    main()
