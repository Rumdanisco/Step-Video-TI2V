import torch
from stepvideo.diffusion.video_pipeline import StepVideoPipeline
from stepvideo.utils import setup_seed
from stepvideo.config import parse_args
import os


def main():
    args = parse_args()
    setup_seed(args.seed)

    # 🧠 Load model from Hugging Face
    model_repo = os.getenv("MODEL_REPO", "stepfun-ai/Step-Video-TI2V")
    token = os.getenv("HUGGINGFACE_TOKEN", None)

    print(f"🚀 Loading Step-Video-TI2V model from: {model_repo}")

    # ✅ Correct keyword: use_auth_token (not token)
    pipeline = StepVideoPipeline.from_pretrained(
        model_repo,
        torch_dtype=torch.float16,
        use_auth_token=token
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)

    prompt = args.prompt
    print(f"🎬 Generating video for prompt: {prompt}")

    videos = pipeline(
        prompt=prompt,
        first_image=args.first_image_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        time_shift=args.time_shift,
        pos_magic=args.pos_magic,
        neg_magic=args.neg_magic,
        output_file_name=args.output_file_name or args.prompt[:50],
        motion_score=args.motion_score,
    )

    print("✅ Generation complete.")
    return videos


if __name__ == "__main__":
    main()
