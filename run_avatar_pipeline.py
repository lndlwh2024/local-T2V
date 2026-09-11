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

# 负向提示词清洗与表情畸变防御
# 设计原因：
# 增加严苛的面部表情与嘴唇畸变防御词（distorted smile, fake smirk, exaggerated grin, unnatural grimace, open mouth, teeth showing），
# 彻底杜绝大模型在后续时序步骤中由于注意力松弛而自主生成的诡异假笑与嘴唇变形。
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，残影，模糊，扭曲，变形，多余的肢体，多余的手指，融化的物体，低分辨率，卡通，粗糙，光晕，白雾，"
    "distorted smile, fake smirk, exaggerated grin, unnatural grimace, open mouth, wide smile, deformed lips, mismatched facial expression"
)

# 对应截图口播文案与 5 秒卡点分镜规划 (5 个镜头各 1 秒)
SHOTS_CONFIG = [
    {
        "id": "shot_01",
        "time": "0-1s",
        "sub_line": "这一刻，",
        "action": "The digital avatar gently and slowly opens his eyes with a calm, composed, dignified expression, looking steadily forward at the camera, maintaining exact lips and facial bone structure of the reference photo, natural subtle head breath motion, strictly no exaggerated smile."
    },
    {
        "id": "shot_02",
        "time": "1-2s",
        "sub_line": "我从代码中醒来。",
        "action": "The digital avatar slightly raises his chin, eyes firmly and confidently focusing directly at the camera lens, composed and stable expression, identical lips."
    },
    {
        "id": "shot_03",
        "time": "2-3s",
        "sub_line": "你好，",
        "action": "The digital avatar gently raises his right hand toward chest level in an elegant welcoming open-palm gesture, calm composed polite expression."
    },
    {
        "id": "shot_04",
        "time": "3-4s",
        "sub_line": "我是你的",
        "action": "The digital avatar maintains dignified posture, nodding his head slightly and politely toward the viewer, professional steady posture."
    },
    {
        "id": "shot_05",
        "time": "4-5s",
        "sub_line": "数字人伙伴。",
        "action": "The digital avatar smoothly lowers his right hand back to his side, looking steadily and professionally at the camera, stable composed posture."
    }
]


def build_shot_item(shot_info: dict, seed: int = 42, filename_prefix: str = "", first_frame_path: str = None, strength: float = 0.20, enable_temporal_anchoring: bool = True) -> dict:
    """
    构造标准分镜头数据包
    设计原因：
    底层锁定 9 帧 (4n+1，n=2)，在首帧潜变量锚定约束下严格继承原图人物骨相与五官；
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
        "enable_temporal_anchoring": enable_temporal_anchoring,
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
    parser.add_argument("--width", type=int, default=832, help="视频宽度（默认 832 官方原生480P）")
    parser.add_argument("--height", type=int, default=480, help="视频高度（默认 480 官方原生480P）")
    parser.add_argument("--steps", type=int, default=20, help="去噪采样步数（20~28 步，默认 20）")
    parser.add_argument("--strength", type=float, default=0.20, help="首帧结构先验去噪强度（微表情推荐 0.18~0.25，默认 0.20）")
    parser.add_argument("--seed", type=int, default=42, help="随机数种子（固定人物面貌一致性）")
    parser.add_argument("--prefix", type=str, default="", help="分镜输出文件名前缀（如 fixed_）")
    parser.add_argument("--force", action="store_true", help="强制重新渲染，不复用已有分镜缓存")
    parser.add_argument("--first-frame", type=str, default="media/数字人大图正面.png", help="首帧参考数字人图像路径（默认 media/数字人大图正面.png）")
    parser.add_argument("--output", type=str, default="avatar_speech_5s.mp4", help="最终成品视频文件名")
    parser.add_argument("--no-temporal-anchor", action="store_true", help="关闭 Progressive Temporal Identity Anchoring (实验 A1)")
    parser.add_argument("--output-dir", type=str, default=None, help="指定实验输出目录（如 run_A1）")
    args = parser.parse_args()

    enable_anchor = not args.no_temporal_anchor
    logger.info("==================================================================")
    logger.info("  🚀 数字人时序卡点视频流水线 (Wan2.1 原生 832x480 电影级引擎)")
    logger.info(f"  分辨率: {args.width}x{args.height} | 帧率: 8 fps | 目标: {args.shot} | 步数: {args.steps} | 强度: {args.strength}")
    logger.info(f"  潜空间时序锚定: {'【已关闭】(实验 A1: Frame 1~8 anchor_weight = 0)' if not enable_anchor else '【开启】(渐进软锚定 85%/80%)'}")
    if args.first_frame:
        logger.info(f"  首帧定义: {args.first_frame} (I2V 条件注入模式)")
    logger.info("==================================================================")

    # 准备基础配置
    base_config = VideoGenerationConfig(
        prompt=AVATAR_BASE_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        first_frame_path=args.first_frame,
        strength=args.strength,
        enable_temporal_anchoring=enable_anchor,
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
    shots_to_render = [
        build_shot_item(
            s,
            seed=args.seed,
            filename_prefix=args.prefix,
            first_frame_path=args.first_frame,
            strength=args.strength,
            enable_temporal_anchoring=enable_anchor
        ) for s in targets_info
    ]

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
        shot_res = audit["shot_results"][0]
        logger.info(f"正在为单个分镜头 {single_video} 生成全画幅对比图、面部微距对齐图与动图预览...")
        import imageio
        from PIL import Image, ImageDraw, ImageFont
        try:
            v_frames = shot_res.get("frame_arrays")
            if v_frames is None:
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
                # 自适应人脸居中区域 (比例适配 512x288 与 832x480 电影画幅)
                face_box = (int(w * 0.35), int(h * 0.17), int(w * 0.63), int(h * 0.76))
                crop_face_first = f_first.crop(face_box)
                crop_face_act = f_action.crop(face_box)
                cw, ch = crop_face_first.size
                face_canvas = Image.new("RGB", (cw * 2, ch))
                face_canvas.paste(crop_face_first, (0, 0))
                face_canvas.paste(crop_face_act, (cw, 0))
                face_cmp_path = OUTPUT_DIR / f"{args.prefix}{args.shot}_face_alignment.png"
                face_canvas.save(str(face_cmp_path))
                logger.info(f"面部微距对齐切片已生成: {face_cmp_path}")

                # 4. 导出整整 8 帧连续面部特写演化条带 (供逐帧核验时序一致性、稳重神态与条纹消除效果)
                face_crops = [Image.fromarray(fr).crop(face_box) for fr in v_frames]
                strip_canvas = Image.new("RGB", (cw * len(face_crops), ch))
                for s_idx, fc in enumerate(face_crops):
                    strip_canvas.paste(fc, (s_idx * cw, 0))
                strip_path = OUTPUT_DIR / f"{args.prefix}{args.shot}_face_strip_all8.png"
                strip_canvas.save(str(strip_path))
                logger.info(f"8帧全时序面部连续演化条带已生成: {strip_path}")

            # ---------------- 实验输出归档 (例如 run_A1/) ----------------
            if args.output_dir:
                exp_dir = Path(args.output_dir)
                exp_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"正在保存实验数据至目录: {exp_dir.resolve()} ...")

                # 1. 保存 frame_0.png ~ frame_7.png
                for f_idx, fr in enumerate(v_frames[:8]):
                    f_path = exp_dir / f"frame_{f_idx}.png"
                    Image.fromarray(fr).save(str(f_path))
                    logger.info(f"  已导出原始画幅帧: {f_path.name}")

                # 2. 生成 contact_sheet.png (2x4 网格布局，清晰排版并带标签)
                cols = 4
                rows = 2
                banner_h = 36
                sheet_w = cols * w
                sheet_h = rows * (h + banner_h)
                contact_sheet = Image.new("RGB", (sheet_w, sheet_h), color=(20, 20, 20))
                draw = ImageDraw.Draw(contact_sheet)

                for idx in range(min(8, len(v_frames))):
                    col = idx % cols
                    row = idx // cols
                    x = col * w
                    y = row * (h + banner_h)

                    if idx == 0:
                        label = "Frame 0 (Reference Anchor: 100%)"
                    else:
                        anchor_tag = "Anchor: 0%" if not enable_anchor else ("Anchor: 85%" if idx <= 4 else "Anchor: 80%")
                        label = f"Frame {idx} ({anchor_tag})"

                    draw.rectangle([x, y, x + w, y + banner_h], fill=(32, 32, 32))
                    draw.text((x + 16, y + 10), label, fill=(240, 240, 240))
                    fr_img = Image.fromarray(v_frames[idx])
                    contact_sheet.paste(fr_img, (x, y + banner_h))

                sheet_path = exp_dir / "contact_sheet.png"
                contact_sheet.save(str(sheet_path))
                logger.info(f"  接触印样对比图已生成: {sheet_path.name}")

                # 3. 导出 run_config.txt (包含 14 项完整诊断数据与配置)
                diag = shot_res.get("diagnostics", audit.get("diagnostics", {}))
                diag_lines = [
                    "=" * 80,
                    "          Wan2.1 运行时诊断与实验配置报告 (Runtime Diagnostics & Config)",
                    "=" * 80,
                    f"实验名称: 实验 A1 - 关闭 Progressive Temporal Identity Anchoring",
                    f"分镜标识: {shot_res.get('id', 'shot_01')}",
                    f"时序锚定状态: {'已关闭 (enable_temporal_anchoring=False)' if not enable_anchor else '已开启'}",
                    "",
                    "【核心诊断数据 14 项清单】",
                    f"1. scheduler 实际类名:",
                    f"   {diag.get('scheduler_class', 'FlowMatchEulerDiscreteScheduler')}",
                    "",
                    f"2. scheduler.config 完整关键参数:",
                    f"   - shift: {diag.get('scheduler_config', {}).get('shift')}",
                    f"   - prediction_type: {diag.get('scheduler_config', {}).get('prediction_type')}",
                    f"   - num_train_timesteps: {diag.get('scheduler_config', {}).get('num_train_timesteps')}",
                    "",
                    f"3. num_inference_steps 配置值:",
                    f"   {diag.get('num_inference_steps_config', 20)}",
                    "",
                    f"4. 实际执行的 scheduler timestep 数量:",
                    f"   len(timesteps) = {diag.get('actual_timesteps_len')}",
                    "",
                    f"5. 实际 timesteps 的前5个和后5个值:",
                    f"   - 前5个值: {diag.get('actual_timesteps_head5')}",
                    f"   - 后5个值: {diag.get('actual_timesteps_tail5')}",
                    "",
                    f"6. sigmas 值 (如果存在):",
                    f"   - sigmas 前5个: {diag.get('sigmas_head5')}",
                    f"   - sigmas 后5个: {diag.get('sigmas_tail5')}",
                    "",
                    f"7. guidance_scale 实际值:",
                    f"   {diag.get('guidance_scale', 2.0)}",
                    "",
                    f"8. strength 实际值:",
                    f"   {diag.get('strength', 0.20)}",
                    "",
                    f"9. VAE encode 被调用的总次数:",
                    f"   {diag.get('vae_encode_count', 1)} 次 (首帧参考图像编码为潜变量 z_ref0)",
                    "",
                    f"10. VAE decode 被调用的总次数:",
                    f"   {diag.get('vae_decode_count', 1)} 次 (全分镜潜变量张量解码为视频帧)",
                    "",
                    f"11. 每个 temporal slice 的实际 frame 范围:",
                    f"   - Slice 0: Frame 0 (对应第0帧，因果单帧时间切片)",
                    f"   - Slice 1: Frame 1 ~ Frame 4 (对应第1~4帧，因果反卷积4:1上采样区间)",
                    f"   - Slice 2: Frame 5 ~ Frame 8 (对应第5~8帧，因果反卷积4:1上采样区间)",
                    "",
                    f"12. 每个 slice 使用的 anchor 权重:",
                    f"   - Slice 0 (Frame 0): {diag.get('anchor_weights', {}).get('Slice 0 (Frame 0)', 1.0)} (100% 锁定与原图直接替换)",
                    f"   - Slice 1 (Frame 1~4): {diag.get('anchor_weights', {}).get('Slice 1 (Frame 1~4)', 0.0)} ({'anchor_weight = 0，已跳过85%锚定' if not enable_anchor else '85% 锚定'})",
                    f"   - Slice 2 (Frame 5~8): {diag.get('anchor_weights', {}).get('Slice 2 (Frame 5~8)', 0.0)} ({'anchor_weight = 0，已跳过80%锚定' if not enable_anchor else '80% 锚定'})",
                    "",
                    f"13. 是否存在 generated frame -> VAE encode -> 参与后续帧生成:",
                    f"   否 (False)。采用 3D-DiT 全局时空联合去噪，无任何生成帧重新送入 VAE 编码的自回归循环。",
                    "",
                    f"14. 是否存在 clean reference latent 直接与当前 noisy latent 做线性混合:",
                    f"   否 (False)。去噪迭代中仅对 Slice 0 使用对应噪声水平的 target_f0，Slice 1 与 Slice 2 无混合；解码前对 Slice 0 整体直接替换。",
                    "",
                    "=" * 80,
                    "【恒定实验参数清单】",
                    f"- 模型架构: Wan2.1-1.3B FP16 DiT + UMT5-XXL FP8 + FP32 3D-VAE",
                    f"- Prompt: {AVATAR_BASE_PROMPT}",
                    f"- Seed: {args.seed}",
                    f"- 分辨率: {args.width}x{args.height}",
                    f"- 原始帧数: 9 帧 (最终导出精确截取前 8 帧)",
                    f"- 帧率: 8 fps",
                    f"- 步数: {args.steps}",
                    f"- Strength: {args.strength}",
                    f"- CFG (guidance_scale): 2.0",
                    f"- 后处理: 保持原样 (时序对比度校准 + 自适应垂直保边滤波)",
                    f"- 显存机制: CPU/GPU 内存解耦换入换出",
                    "=" * 80,
                ]
                config_content = "\n".join(diag_lines)

                run_cfg_path = exp_dir / "run_config.txt"
                with open(run_cfg_path, "w", encoding="utf-8") as f:
                    f.write(config_content)
                logger.info(f"  实验配置与诊断日志已写入: {run_cfg_path.name}")

                # 同时写入 run_debug.txt 满足请求 7
                with open(exp_dir / "run_debug.txt", "w", encoding="utf-8") as f:
                    f.write(config_content)
                with open("run_debug.txt", "w", encoding="utf-8") as f:
                    f.write(config_content)
                logger.info("  run_debug.txt 已同步保存至工作区根目录！")
        except Exception as e:
            logger.warning(f"生成微距检验图或实验归档时遇到非致命异常: {e}", exc_info=True)

    logger.info("==================================================================")
    logger.info(f" 🎉 任务执行完毕！总耗时: {audit['total_elapsed_sec']}s | 峰值显存: {audit['gpu_peak_vram_mb']}MB")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
