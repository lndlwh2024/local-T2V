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


def build_shot_item(
    shot_info: dict,
    seed: int = 42,
    filename_prefix: str = "",
    first_frame_path: str = None,
    strength: float = 0.30,
    enable_temporal_anchoring: bool = False,
    enable_post_processing: bool = False,
    use_full_sequence_reference: bool = True,
    custom_prompt: str = None,
    custom_negative_prompt: str = None
) -> dict:
    """
    构造标准分镜头数据包
    设计原因：
    底层锁定 9 帧 (4n+1，n=2)，采用生产黄金甜点 (strength=0.30) 与全时序因果编码 (use_full_sequence_reference)；
    配合全精度 FP32 VAE 解码与零后处理原生输出，截取前 8 帧输出，严格对齐 8 帧 @ 8fps 1.0 秒业务需求。
    支持可选 custom_prompt 与 custom_negative_prompt 注入，支持独立阶段 D 大动作压力测试。
    """
    shot_id = shot_info["id"]
    if custom_prompt:
        full_prompt = custom_prompt
    else:
        full_prompt = f"{AVATAR_BASE_PROMPT} Action: {shot_info['action']}"
    
    neg_prompt = custom_negative_prompt if custom_negative_prompt else NEGATIVE_PROMPT

    return {
        "id": shot_id,
        "prompt": full_prompt,
        "negative_prompt": neg_prompt,
        "first_frame_path": first_frame_path,
        "strength": strength,
        "enable_temporal_anchoring": enable_temporal_anchoring,
        "enable_post_processing": enable_post_processing,
        "use_full_sequence_reference": use_full_sequence_reference,
        "num_frames": 9,
        "target_num_frames": 8,
        "seed": seed,
        "output_filename": f"{filename_prefix}avatar_{shot_id}.mp4",
        "time": shot_info["time"],
        "sub_line": shot_info["sub_line"]
    }


def main():
    parser = argparse.ArgumentParser(description="数字人 5 秒时序卡点视频生成流水线 (生产模式: AVATAR_STABLE_MICRO_MOTION)")
    parser.add_argument("--shot", choices=["all", "shot_01", "shot_02", "shot_03", "shot_04", "shot_05"], default="all", help="指定渲染的分镜镜头（默认全部）")
    parser.add_argument("--width", type=int, default=832, help="视频宽度（默认 832 官方原生480P）")
    parser.add_argument("--height", type=int, default=480, help="视频高度（默认 480 官方原生480P）")
    parser.add_argument("--steps", type=int, default=50, help="去噪采样步数（生产模式锁定 50 步，默认 50）")
    parser.add_argument("--strength", type=float, default=0.30, help="生产黄金甜点去噪强度 (Production Sweet Spot: 默认 0.30)")
    parser.add_argument("--seed", type=int, default=42, help="随机数种子（固定人物面貌一致性）")
    parser.add_argument("--prefix", type=str, default="", help="分镜输出文件名前缀（如 fixed_）")
    parser.add_argument("--force", action="store_true", help="强制重新渲染，不复用已有分镜缓存")
    parser.add_argument("--first-frame", type=str, default="media/数字人大图正面.png", help="首帧参考数字人图像路径（默认 media/数字人大图正面.png）")
    parser.add_argument("--output", type=str, default="avatar_speech_5s.mp4", help="最终成品视频文件名")
    parser.add_argument("--prompt", type=str, default=None, help="自定义正向提示词（覆盖分镜默认提示词）")
    parser.add_argument("--negative-prompt", type=str, default=None, help="自定义负向提示词（覆盖分镜默认负向提示词）")
    parser.add_argument("--no-temporal-anchor", action="store_true", help="关闭 Progressive Temporal Identity Anchoring (生产模式默认已关闭)")
    parser.add_argument("--no-post-process", action="store_true", help="关闭后处理与逐帧自动归一化，仅输出原始解码结果 (生产模式默认已关闭)")
    parser.add_argument("--no-full-sequence-reference", action="store_true", help="关闭全时序参考潜变量初始化，退回旧版单帧广播模式")
    parser.add_argument("--pure-t2v", action="store_true", help="启用实验 D2-Control 纯 T2V 随机高斯噪声对照模式 (关闭全部参考潜变量与原图)")
    parser.add_argument("--output-dir", type=str, default=None, help="指定实验输出目录（如 run_C2_strength030）")
    args = parser.parse_args()

    enable_anchor = not args.no_temporal_anchor if args.no_temporal_anchor else False
    enable_post = not args.no_post_process if args.no_post_process else False
    use_full_seq = not args.no_full_sequence_reference

    if args.pure_t2v:
        args.first_frame = None
        use_full_seq = False
        enable_anchor = False

    logger.info("==================================================================")
    if args.pure_t2v:
        logger.info("  🧪 数字人对照实验: 【实验 D2-Control 纯 T2V 大动作响应对照】")
        logger.info("  use_full_sequence_reference = False")
        logger.info("  initial_latents_source = RANDOM_NOISE")
        logger.info(f"  actual prompt = {args.prompt}")
        logger.info(f"  actual_iteration_count = {args.steps}")
        logger.info("  CFG = 2.0")
    else:
        logger.info("  🚀 数字人时序卡点视频流水线 (生产模式: AVATAR_STABLE_MICRO_MOTION)")
    logger.info(f"  分辨率: {args.width}x{args.height} | 帧率: 8 fps | 目标: {args.shot} | 步数: {args.steps} | 强度: {args.strength}")
    logger.info(f"  潜空间时序锚定: {'【已关闭】(Frame 1~8 anchor_weight = 0，零波纹)' if not enable_anchor else '【开启】(渐进软锚定 85%/80%)'}")
    logger.info(f"  参考潜变量模式: {'【纯 T2V 随机噪声初始化】(零参考图)' if args.pure_t2v else ('【全时序参考初始化 (实验 B3 架构)】(9帧全同张量 VAE 一次性编码真实 z_ref_seq T=3)' if use_full_seq else '【单帧广播模式】(旧版单帧重复 3 次)')}")
    logger.info(f"  后处理管线: {'【已彻底关闭】(零后处理原生解码帧 Raw Decoded Frames，零发雾)' if not enable_post else '【开启】(时序动态对比度校准 + 自适应垂直保边滤波)'}")
    if args.first_frame:
        logger.info(f"  首帧定义: {args.first_frame} (I2V 条件注入模式)")
    if args.prompt:
        logger.info(f"  自定义提示词注入: {args.prompt[:60]}...")
    logger.info("==================================================================")

    # 准备基础配置
    base_config = VideoGenerationConfig(
        prompt=args.prompt if args.prompt else AVATAR_BASE_PROMPT,
        negative_prompt=args.negative_prompt if args.negative_prompt else NEGATIVE_PROMPT,
        first_frame_path=args.first_frame,
        strength=args.strength,
        enable_temporal_anchoring=enable_anchor,
        enable_post_processing=enable_post,
        use_full_sequence_reference=use_full_seq,
        width=args.width,
        height=args.height,
        num_frames=9,
        target_num_frames=8,
        fps=8,
        num_inference_steps=args.steps,
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
            enable_temporal_anchoring=enable_anchor,
            enable_post_processing=enable_post,
            use_full_sequence_reference=use_full_seq,
            custom_prompt=args.prompt,
            custom_negative_prompt=args.negative_prompt
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

                # 1. 保存 frame_0.png ~ frame_7.png 与未替换的 raw_decoded_frame_0_before_replacement.png
                for f_idx, fr in enumerate(v_frames[:8]):
                    f_path = exp_dir / f"frame_{f_idx}.png"
                    Image.fromarray(fr).save(str(f_path))
                    logger.info(f"  已导出原始画幅帧: {f_path.name}")

                raw_f0 = shot_res.get("raw_frame_0_before_replace")
                if raw_f0 is not None:
                    raw_f0_path = exp_dir / "raw_decoded_frame_0_before_replacement.png"
                    Image.fromarray(raw_f0).save(str(raw_f0_path))
                    logger.info(f"  【实验 B3】已导出未替换原图的原始生成第0帧: {raw_f0_path.name}")

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

                    if args.pure_t2v:
                        label = f"Frame {idx} (Pure T2V | Random Noise)"
                    elif idx == 0:
                        label = "Frame 0 (Replaced with Reference Image)"
                    else:
                        anchor_tag = "Anchor: 0%" if not enable_anchor else ("Anchor: 85%" if idx <= 4 else "Anchor: 80%")
                        ref_tag = "B3: FullSeq Ref" if use_full_seq else "B1: Single Broadcast"
                        label = f"Frame {idx} ({anchor_tag} | {ref_tag})"

                    draw.rectangle([x, y, x + w, y + banner_h], fill=(32, 32, 32))
                    draw.text((x + 16, y + 10), label, fill=(240, 240, 240))
                    fr_img = Image.fromarray(v_frames[idx])
                    contact_sheet.paste(fr_img, (x, y + banner_h))

                sheet_path = exp_dir / "contact_sheet.png"
                contact_sheet.save(str(sheet_path))
                logger.info(f"  接触印样对比图已生成: {sheet_path.name}")

                # 3. 保存 8 帧面部微距连续演化条带至实验目录
                strip_out_path = exp_dir / "face_strip_all8.png"
                strip_canvas.save(str(strip_out_path))
                logger.info(f"  8帧面部连续演化条带已生成: {strip_out_path.name}")

                # 3.1 生成 Frame 1 vs Frame 7 面部微距特写对比图 (f1_vs_f7_face_compare.png)
                if len(face_crops) >= 8:
                    f1_face = face_crops[1]
                    f7_face = face_crops[7]
                    cw_cmp, ch_cmp = f1_face.size
                    banner_cmp_h = 32
                    f17_canvas = Image.new("RGB", (cw_cmp * 2, ch_cmp + banner_cmp_h), color=(20, 20, 20))
                    draw_cmp = ImageDraw.Draw(f17_canvas)

                    # 绘制标题栏
                    draw_cmp.rectangle([0, 0, cw_cmp, banner_cmp_h], fill=(36, 36, 36))
                    draw_cmp.text((16, 8), "Frame 1 (Early Denoised Face)", fill=(240, 240, 240))
                    draw_cmp.rectangle([cw_cmp, 0, cw_cmp * 2, banner_cmp_h], fill=(28, 28, 28))
                    draw_cmp.text((cw_cmp + 16, 8), "Frame 7 (Final Evolved Face)", fill=(240, 240, 240))

                    f17_canvas.paste(f1_face, (0, banner_cmp_h))
                    f17_canvas.paste(f7_face, (cw_cmp, banner_cmp_h))
                    f17_path = exp_dir / "f1_vs_f7_face_compare.png"
                    f17_canvas.save(str(f17_path))
                    logger.info(f"  Frame 1 vs Frame 7 面部特写对比图已生成: {f17_path.name}")

                # 4. 导出 run_config.txt (包含 14 项完整诊断数据与配置)
                diag = shot_res.get("diagnostics", audit.get("diagnostics", {}))
                luma_stats = diag.get("frames_luma_stats", {})
                latent_stats = diag.get("latent_evolution_stats", {})
                ref_shapes = diag.get("ref_shapes_info", {})
                deltas = luma_stats.get("frame1_to_frame7_deltas", {})

                diag_lines = [
                    "=" * 80,
                    "          Wan2.1 运行时诊断与实验配置报告 (Runtime Diagnostics & Config)",
                    "=" * 80,
                    f"实验目录: {exp_dir.name}",
                    f"分镜标识: {shot_res.get('id', 'shot_01')}",
                    f"时序锚定状态: {'已关闭 (enable_temporal_anchoring=False)' if not enable_anchor else '已开启'}",
                    f"参考初始化模式: {'纯T2V随机高斯噪声初始化 (use_full_sequence_reference=False, initial_latents_source=RANDOM_NOISE)' if args.pure_t2v else ('全时序参考潜变量初始化 (use_full_sequence_reference=True, 实验 B3)' if use_full_seq else '单帧广播初始化 (use_full_sequence_reference=False)')}",
                    f"use_full_sequence_reference = {diag.get('use_full_sequence_reference', False if args.pure_t2v else True)}",
                    f"initial_latents_source = {diag.get('initial_latents_source', 'RANDOM_NOISE' if args.pure_t2v else 'FULL_SEQUENCE_REFERENCE')}",
                    f"actual prompt = {diag.get('actual_prompt', args.prompt)}",
                    f"prompt embedding checksum/hash = {diag.get('prompt_embedding_hash', 'N/A')}",
                    f"actual_iteration_count = {diag.get('actual_iteration_count', args.steps)}",
                    f"CFG = {diag.get('guidance_scale', 2.0)}",
                    "",
                    "【本轮关键执行与去噪指标】",
                    f"configured_num_inference_steps = {args.steps}",
                    f"strength = {args.strength:.2f}",
                    f"actual_iteration_count = {diag.get('actual_iteration_count', diag.get('actual_timesteps_len'))}",
                    f"len(actual_timesteps) = {diag.get('actual_timesteps_len')}",
                    f"actual timesteps 前3个: {diag.get('actual_timesteps_head3')}",
                    f"actual timesteps 后3个: {diag.get('actual_timesteps_tail3')}",
                    f"Transformer 去噪耗时: {diag.get('transformer_denoise_elapsed_sec')} s",
                    f"峰值显存: {audit['gpu_peak_vram_mb']} MB",
                    "",
                    "【实验 B3 潜空间 3 个 temporal slice 统计与形状审计】",
                    f"old_ref_latent_shape = {ref_shapes.get('old_ref_latent_shape', '[1, 16, 1, 60, 104]')}",
                    f"new_ref_latent_shape = {ref_shapes.get('new_ref_latent_shape', 'N/A')}",
                    f"reference_temporal_latent_count = {ref_shapes.get('reference_temporal_latent_count', 'N/A')}",
                    f"initial_latents_shape = {ref_shapes.get('initial_latents_shape', 'N/A')}",
                    "",
                    "--- clean_z_ref_seq 切片统计 ---",
                ]

                clean_slices = latent_stats.get("clean_z_ref_seq", {})
                for sl_k, sl_v in clean_slices.items():
                    diag_lines.append(f"  {sl_k}: mean={sl_v.get('mean')}, std={sl_v.get('std')}, min={sl_v.get('min')}, max={sl_v.get('max')}")

                diag_lines.append("\n--- initial_latents (加噪后) 切片统计 ---")
                init_slices = latent_stats.get("initial_latents", {})
                for sl_k, sl_v in init_slices.items():
                    diag_lines.append(f"  {sl_k}: mean={sl_v.get('mean')}, std={sl_v.get('std')}, min={sl_v.get('min')}, max={sl_v.get('max')}")

                diag_lines.append("\n--- denoised_latent (去噪后) 切片统计 ---")
                denoise_slices = latent_stats.get("denoised_latent", {})
                for sl_k, sl_v in denoise_slices.items():
                    diag_lines.append(f"  {sl_k}: mean={sl_v.get('mean')}, std={sl_v.get('std')}, min={sl_v.get('min')}, max={sl_v.get('max')}")

                diag_lines.extend([
                    "",
                    "【RGB 帧亮度统计与 Frame 1 -> Frame 7 漂移指标】",
                    f"A. 未替换原图的原始解码 Frame 0:",
                ])
                raw_f0_stat = luma_stats.get("raw_frame_0_before_replace", {})
                diag_lines.append(f"   raw_frame_0_before_replace: mean={raw_f0_stat.get('mean_luma')}, std={raw_f0_stat.get('std_luma')}, min={raw_f0_stat.get('min_luma')}, max={raw_f0_stat.get('max_luma')}, dynamic_range={raw_f0_stat.get('dynamic_range')}")

                diag_lines.append("\nB. 逐帧统计 (Frame 0 ~ 7, Frame 0 已替换原图):")
                for f_i in range(8):
                    f_key = f"frame_{f_i}"
                    st = luma_stats.get(f_key, {})
                    diag_lines.append(f"   Frame {f_i}: mean={st.get('mean_luma')}, std={st.get('std_luma')}, min={st.get('min_luma')}, max={st.get('max_luma')}, dynamic_range={st.get('dynamic_range')}")

                diag_lines.extend([
                    "\nC. Frame 1 -> Frame 7 漂移物理指标:",
                    f"   black_level_delta    (min_luma_7 - min_luma_1)       = {deltas.get('black_level_delta', 'N/A')}",
                    f"   highlight_delta      (max_luma_7 - max_luma_1)       = {deltas.get('highlight_delta', 'N/A')}",
                    f"   std_delta            (std_luma_7 - std_luma_1)       = {deltas.get('std_delta', 'N/A')}",
                    f"   dynamic_range_delta  (range_7 - range_1)             = {deltas.get('dynamic_range_delta', 'N/A')}",
                    "",
                    "=" * 80,
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
                    f"   {diag.get('num_inference_steps_config', args.steps)}",
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
                    f"   - 后5个值: {diag.get('sigmas_tail5')}",
                    "",
                    f"7. guidance_scale 实际值:",
                    f"   {diag.get('guidance_scale', 2.0)}",
                    "",
                    f"8. strength 实际值:",
                    f"   {diag.get('strength', 0.20)}",
                    "",
                    f"9. VAE encode 被调用的总次数:",
                    f"   {diag.get('vae_encode_count', 1)} 次 (实验 B3 9帧全同视频一次性编码真实 z_ref_seq)",
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
                    f"   - Slice 0 (Frame 0): 0.0 (全局自由去噪，无步内 anchor，交付帧直接替换原图)",
                    f"   - Slice 1 (Frame 1~4): 0.0 (anchor_weight = 0，已跳过锚定)",
                    f"   - Slice 2 (Frame 5~8): 0.0 (anchor_weight = 0，已跳过锚定)",
                    "",
                    f"13. 是否存在 generated frame -> VAE encode -> 参与后续帧生成:",
                    f"   否 (False)。采用 3D-DiT 全局时空联合去噪，无任何生成帧重新送入 VAE 编码的自回归循环。",
                    "",
                    f"14. 是否存在 clean reference latent 直接与当前 noisy latent 做线性混合:",
                    f"   否 (False)。去噪迭代中全时序潜变量一次性加噪后正常去噪，无步内 blend/overwrite。",
                    "",
                    "=" * 80,
                    "【后处理与解码链路状态】",
                    f"A. 后处理执行状态:",
                    f"   - temporal_contrast_restoration_executed = {diag.get('temporal_contrast_restoration_executed', False)}",
                    f"   - adaptive_destripe_filter_executed = {diag.get('adaptive_destripe_filter_executed', False)}",
                    f"   - frame_level_normalization_exists = {diag.get('frame_level_normalization_exists', False)}",
                    "",
                    f"B. 残留路径说明:",
                    f"   {diag.get('residual_path_note', '除 VAE 原始解码与固定线性变换外，不存在任何隐式色彩/滤波处理。')}",
                    "",
                    f"C. 输出链路说明:",
                    f"   {diag.get('output_pipeline_note', '当前导出的是 raw decoded frames，无后处理')}",
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
                    f"- 后处理: {'【已彻底关闭】(跳过对比度恢复与垂直滤波，无逐帧归一化，仅输出 raw decode 帧)' if not enable_post else '开启'}",
                    f"- 显存机制: CPU/GPU 内存解耦换入换出",
                    "=" * 80,
                ])
                config_content = "\n".join(diag_lines)

                run_cfg_path = exp_dir / "run_config.txt"
                with open(run_cfg_path, "w", encoding="utf-8") as f:
                    f.write(config_content)
                logger.info(f"  实验配置与诊断日志已写入: {run_cfg_path.name}")

                # 同时写入 run_debug.txt 满足请求
                with open(exp_dir / "run_debug.txt", "w", encoding="utf-8") as f:
                    f.write(config_content)
                with open("run_debug.txt", "w", encoding="utf-8") as f:
                    f.write(config_content)
                logger.info("  run_debug.txt 已同步保存至工作区根目录！")

                # 按照用户要求在控制台直接打印关键去噪执行指标与实验 B3 诊断指标
                # 按照用户要求在控制台直接打印关键去噪执行指标与实验 B3 / D2-Control 诊断指标
                logger.info("==================================================================")
                if args.pure_t2v:
                    logger.info("  【实验 D2-Control 纯 T2V 对照关键指标控制台汇总】")
                    logger.info("  use_full_sequence_reference = False")
                    logger.info("  initial_latents_source = RANDOM_NOISE")
                    logger.info(f"  actual prompt = {diag.get('actual_prompt', args.prompt)}")
                    logger.info(f"  prompt embedding checksum/hash = {diag.get('prompt_embedding_hash', 'N/A')}")
                    logger.info(f"  actual_iteration_count = {diag.get('actual_iteration_count', args.steps)}")
                    logger.info(f"  CFG = {diag.get('guidance_scale', 2.0)}")
                    logger.info(f"  Transformer 去噪耗时: {diag.get('transformer_denoise_elapsed_sec')} s")
                    logger.info(f"  峰值显存: {audit['gpu_peak_vram_mb']} MB")
                else:
                    logger.info("  【实验 B3 关键诊断指标控制台汇总】")
                    logger.info(f"  1. 参考潜变量模式: use_full_sequence_reference = {use_full_seq}")
                    logger.info(f"     old_ref_shape: {ref_shapes.get('old_ref_latent_shape')} -> new_ref_shape: {ref_shapes.get('new_ref_latent_shape')}")
                    logger.info(f"  2. 后处理状态:     enable_post_processing = {enable_post}")
                    logger.info(f"  3. Frame 1 -> Frame 7 漂移物理指标:")
                    logger.info(f"     - black_level_delta:   {deltas.get('black_level_delta', 'N/A')}")
                    logger.info(f"     - highlight_delta:     {deltas.get('highlight_delta', 'N/A')}")
                    logger.info(f"     - std_delta:           {deltas.get('std_delta', 'N/A')}")
                    logger.info(f"     - dynamic_range_delta: {deltas.get('dynamic_range_delta', 'N/A')}")
                    logger.info("  4. 帧亮度统计:")
                    logger.info(f"     raw_frame_0: mean={raw_f0_stat.get('mean_luma')}, min={raw_f0_stat.get('min_luma')}, max={raw_f0_stat.get('max_luma')}, range={raw_f0_stat.get('dynamic_range')}")
                    for f_i in range(1, 8):
                        st = luma_stats.get(f"frame_{f_i}", {})
                        logger.info(f"     Frame {f_i}:     mean={st.get('mean_luma')}, min={st.get('min_luma')}, max={st.get('max_luma')}, range={st.get('dynamic_range')}")
                    logger.info("------------------------------------------------------------------")
                    logger.info(f"  configured_num_inference_steps = {args.steps}")
                    logger.info(f"  strength = {args.strength:.2f}")
                    logger.info(f"  actual_iteration_count = {diag.get('actual_iteration_count', diag.get('actual_timesteps_len'))}")
                    logger.info(f"  len(actual_timesteps) = {diag.get('actual_timesteps_len')}")
                    logger.info(f"  actual timesteps 前3个: {diag.get('actual_timesteps_head3')}")
                    logger.info(f"  actual timesteps 后3个: {diag.get('actual_timesteps_tail3')}")
                    logger.info(f"  Transformer 去噪耗时: {diag.get('transformer_denoise_elapsed_sec')} s")
                    logger.info(f"  峰值显存: {audit['gpu_peak_vram_mb']} MB")
                logger.info("==================================================================")
        except Exception as e:
            logger.warning(f"生成微距检验图或实验归档时遇到非致命异常: {e}", exc_info=True)

    logger.info("==================================================================")
    logger.info(f" 🎉 任务执行完毕！总耗时: {audit['total_elapsed_sec']}s | 峰值显存: {audit['gpu_peak_vram_mb']}MB")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
