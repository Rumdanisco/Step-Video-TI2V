import torch
from stepvideo.diffusion.video_pipeline import StepVideoPipeline
from stepvideo.config import parse_args
from stepvideo.utils import setup_seed

if __name__ == "__main__":
    args = parse_args()

    print("✅ Step-Video running in single-GPU mode (no distributed init).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(args.seed)

    # Load model
    pipeline = StepVideoPipeline.from_pretrained(
        args.model_dir
    ).to(dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device=device)

    pipeline.setup_pipeline(args)

    prompt = args.prompt

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
        output_file_name=args.output_file_name or prompt[:50],
        motion_score=args.motion_score,
    )

    print(f"✅ Video generation complete: {args.output_file_name or prompt[:50]}")
