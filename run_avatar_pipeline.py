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

# 数字人核心人设固定特征基准 Prompt 模板
# 设计原因：
# 配合时序首帧潜空间条件锚定（I2V驱动模式），正向提示词无需再盲目描摹发型与五官细节，
# 全面聚焦于演播室柔光环境、真实皮肤质感与清晰画质，为后续帧动作展开提供纯净语义支撑。
AVATAR_BASE_PROMPT = (
    "A photorealistic mature Asian man with a natural very short shaved buzz-cut hairstyle, "
    "neat natural hairline, groomed salt-and-pepper goatee, wearing a clean plain white crewneck t-shirt with a tiny red chest logo, "
    "standing in a modern high-tech broadcast studio with ambient soft studio lighting, 8k resolution, highly detailed realistic skin texture, cinematic quality."
)

# 负向提示词清洗
# 设计原因：
# 彻底清除此前引入的所有侵入式发型排斥词（pompadour, quiff）与特效光晕词，
# 仅保留基础画质与变形防御，杜绝负向排斥向量场破坏面部骨相。
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，残影，模糊，扭曲，变形，多余的肢体，多余的手指，融化的物体，低分辨率，卡通，粗糙，光晕，白雾"
)

# 对应截图口播文案与 5 秒卡点分镜规划 (5 个镜头各 1 秒)
SHOTS_CONFIG = [
    {
        "id": "shot_01",
        "time": "0-1s",
        "sub_line": "这一刻，",
        "action": "The digital avatar gently and slowly opens his eyes with a calm and confident expression, looking steadily forward at the camera, natural subtle head movement."
    },
    {
        "id": "shot_02",
        "time": "1-2s",
        "sub_line": "我从代码中醒来。",
        "action": "The digital avatar slightly raises his chin, eyes firmly and confidently focusing directly at the camera lens, natural confident expression."
    },
    {
        "id": "shot_03",
        "time": "2-3s",
        "sub_line": "你好，",
        "action": "The digital avatar gently raises his right hand toward chest level in an elegant welcoming open-palm gesture, friendly subtle smile."
    },
    {
        "id": "shot_04",
        "time": "3-4s",
        "sub_line": "我是你的",
        "action": "The digital avatar smiles warmly, nodding his head slightly and politely toward the viewer, welcoming professional posture."
    },
    {
        "id": "shot_05",
        "time": "4-5s",
        "sub_line": "数字人伙伴。",
        "action": "The digital avatar smoothly lowers his right hand back to his side, looking steadily and professionally at the camera, stable posture."
    }
]


def build_shot_item(shot_info: dict, seed: int = 42, filename_prefix: str = "", first_frame_path: str = None, strength: float = 0.20) -> dict:
    """
    构造标准分镜头数据包
    设计原因：
    底层锁定 9 帧 (4n+1，n=2)，在首帧潜变量锚定与 strength=0.20 先验加噪下严格继承原图人物骨相与五官，
    保留 80% 真实五官潜变量，杜绝重绘导致的面容走样与大众脸漂移；
    配合 FP32 VAE 解码、时序对比度保真恢复与自适应保边去条纹滤波，截取前 8 帧输出，严格对齐 8 帧 @ 8fps 1.0 秒业务需求。
    """
    shot_id = shot_info["id"]
    full_prompt = f"{AVATAR_BASE_PROMPT} Action: {shot_info['action']}"
    return {
        "id": shot_id,
        "prompt": full_prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "first_frame_path": first_frame_path,
        "strength": strength,
        "num_frames": 9,
        "target_num_frames": 8,
        "seed": seed,
        "output_filename": f"{filename_prefix}avatar_{shot_id}.mp4",
        "time": shot_info["time"],
        "sub_line": shot_info["sub_line"]
    }


def main():
    parser = argparse.ArgumentParser(description="数字人 5 秒时序卡点视频生成流水线")
    parser.add_argument("--shot", choices=["all", "shot_01", "shot_02", "shot_03", "shot_04", "shot_05"], default="all", help="指定渲染的分镜镜头（默认全部）")
    parser.add_argument("--width", type=int, default=512, help="视频宽度（默认 512）")
    parser.add_argument("--height", type=int, default=288, help="视频高度（默认 288）")
    parser.add_argument("--steps", type=int, default=20, help="去噪采样步数（20~28 步，默认 20）")
    parser.add_argument("--strength", type=float, default=0.20, help="首帧结构先验去噪强度（微表情推荐 0.18~0.25，默认 0.20）")
    parser.add_argument("--seed", type=int, default=42, help="随机数种子（固定人物面貌一致性）")
    parser.add_argument("--prefix", type=str, default="", help="分镜输出文件名前缀（如 fixed_）")
    parser.add_argument("--force", action="store_true", help="强制重新渲染，不复用已有分镜缓存")
    parser.add_argument("--first-frame", type=str, default="media/数字人大图正面.png", help="首帧参考数字人图像路径（默认 media/数字人大图正面.png）")
    parser.add_argument("--output", type=str, default="avatar_speech_5s.mp4", help="最终成品视频文件名")
    args = parser.parse_args()

    logger.info("==================================================================")
    logger.info("  🚀 数字人时序卡点视频流水线 (Wan2.1 4GB Low-VRAM 引擎)")
    logger.info(f"  分辨率: {args.width}x{args.height} | 帧率: 8 fps | 目标: {args.shot} | 步数: {args.steps} | 强度: {args.strength}")
    if args.first_frame:
        logger.info(f"  首帧定义: {args.first_frame} (I2V 条件注入模式)")
    logger.info("==================================================================")

    # 准备基础配置
    base_config = VideoGenerationConfig(
        prompt=AVATAR_BASE_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        first_frame_path=args.first_frame,
        strength=args.strength,
        width=args.width,
        height=args.height,
        num_frames=9,
        target_num_frames=8,
        fps=8,
        num_inference_steps=args.steps,
        # 调优 guidance_scale = 2.0
        # 设计原因：
        # 在首帧潜变量注入模式下，CFG=2.0 专注于引导眼皮眨动与呼吸微表情，杜绝高 CFG (5.0) 的通用文本先验对抗原图骨相五官。
        guidance_scale=2.0,
        seed=args.seed,
        vram_limit_gb=3.6
    )

    pipeline = WanT2VLowVramPipeline(base_config)

    targets_info = [s for s in SHOTS_CONFIG if args.shot == "all" or s["id"] == args.shot]
    shots_to_render = [build_shot_item(s, seed=args.seed, filename_prefix=args.prefix, first_frame_path=args.first_frame, strength=args.strength) for s in targets_info]

    # 若指定了 --force，提前清理对应已有视频文件以确保强制重新渲染
    if args.force:
        for s in shots_to_render:
            target_p = OUTPUT_DIR / s["output_filename"]
            if target_p.exists():
                logger.info(f"检测到 --force 参数，删除已有缓存视频以强制重新渲染: {target_p.name}")
                target_p.unlink()

    concat_name = args.output if args.shot == "all" else None
    audit = pipeline.generate_multi_shots(shots_to_render, concat_output_filename=concat_name)

    # 关键帧预览抽取
    if audit.get("final_video"):
        final_video_path = audit["final_video"]
        logger.info(f"正在对成品视频 {final_video_path} 提取 1fps 关键帧预览图...")
        thumb_pattern = str(OUTPUT_DIR / f"{args.prefix}avatar_5s_thumb_%02d.jpg")
        cmd = [
            "ffmpeg", "-y", "-i", str(final_video_path),
            "-vf", "fps=1",
            thumb_pattern
        ]
        subprocess.run(cmd, capture_output=True)
        logger.info("关键帧预览提取完成！")
    elif args.shot != "all" and audit.get("shot_results"):
        single_video = audit["shot_results"][0]["output_file"]
        logger.info(f"正在为单个分镜头 {single_video} 生成全画幅对比图、面部微距对齐图与动图预览...")
        import imageio
        from PIL import Image, ImageDraw, ImageFont
        try:
            reader = imageio.get_reader(single_video)
            v_frames = [f for f in reader]
            reader.close()
            if len(v_frames) >= 2:
                # 1. 导出轻量级预览动图 GIF
                gif_path = OUTPUT_DIR / f"{args.prefix}{args.shot}_preview.gif"
                imageio.mimsave(str(gif_path), v_frames, fps=args.fps if hasattr(args, "fps") else 8, loop=0)
                logger.info(f"动态预览 GIF 已生成: {gif_path}")

                # 2. 抽取首帧（基准）与动作帧（第 4 帧或最后帧）拼合横向全画幅对比
                f_first = Image.fromarray(v_frames[0])
                act_idx = min(4, len(v_frames) - 1)
                f_action = Image.fromarray(v_frames[act_idx])
                
                w, h = f_first.size
                canvas = Image.new("RGB", (w * 2, h))
                canvas.paste(f_first, (0, 0))
                canvas.paste(f_action, (w, 0))
                cmp_path = OUTPUT_DIR / f"{args.prefix}{args.shot}_compare.png"
                canvas.save(str(cmp_path))
                logger.info(f"首帧与动作帧并排对比图已生成: {cmp_path}")

                # 3. 抽取面部微距特写对比（验证骨相五官一致性与横波纹消除）
                # 人脸居中区域大约 x: 180~320, y: 50~220 (140x170)
                face_box = (180, 50, 320, 220)
                crop_face_first = f_first.crop(face_box)
                crop_face_act = f_action.crop(face_box)
                cw, ch = crop_face_first.size
                face_canvas = Image.new("RGB", (cw * 2, ch))
                face_canvas.paste(crop_face_first, (0, 0))
                face_canvas.paste(crop_face_act, (cw, 0))
                face_cmp_path = OUTPUT_DIR / f"{args.prefix}{args.shot}_face_alignment.png"
                face_canvas.save(str(face_cmp_path))
                logger.info(f"面部微距对齐切片已生成: {face_cmp_path}")
        except Exception as e:
            logger.warning(f"生成微距检验图时遇到非致命异常: {e}")

    logger.info("==================================================================")
    logger.info(f" 🎉 任务执行完毕！总耗时: {audit['total_elapsed_sec']}s | 峰值显存: {audit['gpu_peak_vram_mb']}MB")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
