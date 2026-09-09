import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# 添加当前目录至 Python 模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import VideoGenerationConfig, OUTPUT_DIR
from src.pipeline import WanT2VLowVramPipeline

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("T2V.AvatarBatch")

# 数字人核心人设固定特征 Prompt 模板
# 来源于 media/数字人大图正面.png 与 数字人多角度.png 的像素级解析
AVATAR_BASE_PROMPT = (
    "A photorealistic mature Asian man with a very short silver buzz-cut hairstyle, "
    "groomed salt-and-pepper goatee, wearing a clean plain white crewneck t-shirt with a tiny red chest logo, "
    "standing in a modern high-tech digital studio with curved giant digital screens displaying glowing blue and purple data visualizations, "
    "professional studio softbox lighting, 8k resolution, highly detailed skin texture, cinematic quality."
)

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，残影，模糊，扭曲，变形，多余的肢体，多余的手指，融化的物体，低分辨率，卡通，粗糙"
)

# 对应截图口播文案与 5 秒卡点分镜规划 (5 个镜头各 1 秒)
SHOTS_CONFIG = [
    {
        "id": "shot_01",
        "time": "0-1s",
        "sub_line": "这一刻，",
        "action": "The digital avatar materializes from gentle holographic code particles and soft light flares, slowly opening his eyes with a calm and gentle expression, looking forward."
    },
    {
        "id": "shot_02",
        "time": "1-2s",
        "sub_line": "我从代码中醒来。",
        "action": "The digital avatar slightly raises his chin, eyes firmly and confidently focusing directly at the camera lens, natural head motion."
    },
    {
        "id": "shot_03",
        "time": "2-3s",
        "sub_line": "你好，",
        "action": "The digital avatar gently raises his right hand toward chest level in an elegant welcoming open-palm gesture, friendly smile."
    },
    {
        "id": "shot_04",
        "time": "3-4s",
        "sub_line": "我是你的",
        "action": "The digital avatar smiles warmly, nodding his head slightly and politely toward the viewer, welcoming atmosphere."
    },
    {
        "id": "shot_05",
        "time": "4-5s",
        "sub_line": "数字人伙伴。",
        "action": "The digital avatar smoothly lowers his right hand back to his side, looking steadily and professionally at the camera, stable posture."
    }
]


def build_shot_item(shot_info: dict, seed: int = 42) -> dict:
    """
    构造标准分镜头数据包
    依据 4n+1 数学约束：底层潜空间生成 9 帧，导出时截取前 8 帧对应严格 1 秒 @ 8fps
    """
    shot_id = shot_info["id"]
    full_prompt = f"{AVATAR_BASE_PROMPT} Action: {shot_info['action']}"
    return {
        "id": shot_id,
        "prompt": full_prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "num_frames": 9,
        "target_num_frames": 8,
        "seed": seed,
        "output_filename": f"avatar_{shot_id}.mp4",
        "time": shot_info["time"],
        "sub_line": shot_info["sub_line"]
    }


def main():
    parser = argparse.ArgumentParser(description="数字人 5 秒时序卡点视频生成流水线")
    parser.add_argument("--shot", choices=["all", "shot_01", "shot_02", "shot_03", "shot_04", "shot_05"], default="all", help="指定渲染的分镜镜头（默认全部）")
    parser.add_argument("--width", type=int, default=512, help="视频宽度（默认 512）")
    parser.add_argument("--height", type=int, default=288, help="视频高度（默认 288）")
    parser.add_argument("--steps", type=int, default=20, help="去噪采样步数（20~28 步，默认 20）")
    parser.add_argument("--seed", type=int, default=42, help="随机数种子（固定人物面貌一致性）")
    parser.add_argument("--output", type=str, default="avatar_speech_5s.mp4", help="最终 5 秒成品视频文件名")
    args = parser.parse_args()

    logger.info("==================================================================")
    logger.info("  🚀 数字人 5 秒动作卡点视频流水线 (Wan2.1 4GB Low-VRAM 引擎)")
    logger.info(f"  分辨率: {args.width}x{args.height} | 帧率: 8 fps | 总帧数: 40 帧 | 步数: {args.steps}")
    logger.info("==================================================================")

    # 准备基础配置
    base_config = VideoGenerationConfig(
        prompt=AVATAR_BASE_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        width=args.width,
        height=args.height,
        num_frames=9,
        target_num_frames=8,
        fps=8,
        num_inference_steps=args.steps,
        guidance_scale=5.0,
        seed=args.seed,
        vram_limit_gb=3.6
    )

    pipeline = WanT2VLowVramPipeline(base_config)

    targets_info = [s for s in SHOTS_CONFIG if args.shot == "all" or s["id"] == args.shot]
    shots_to_render = [build_shot_item(s, seed=args.seed) for s in targets_info]

    concat_name = args.output if args.shot == "all" else None
    audit = pipeline.generate_multi_shots(shots_to_render, concat_output_filename=concat_name)

    # 若完成了完整 5 秒合成，自动抽帧提取 5 个关键卡点预览图供审计验证
    if audit.get("final_video"):
        final_video_path = audit["final_video"]
        logger.info(f"正在对成品视频 {final_video_path} 提取 1fps 关键帧预览图...")
        thumb_pattern = str(OUTPUT_DIR / "avatar_5s_thumb_%02d.jpg")
        cmd = [
            "ffmpeg", "-y", "-i", str(final_video_path),
            "-vf", "fps=1",
            thumb_pattern
        ]
        subprocess.run(cmd, capture_output=True)
        logger.info("关键帧预览提取完成！")

    logger.info("==================================================================")
    logger.info(f" 🎉 任务执行完毕！总耗时: {audit['total_elapsed_sec']}s | 峰值显存: {audit['gpu_peak_vram_mb']}MB")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
