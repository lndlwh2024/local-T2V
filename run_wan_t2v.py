import argparse
import logging
import sys
import time
from pathlib import Path

# 添加当前目录至 Python 模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import VideoGenerationConfig, OUTPUT_DIR
from src.pipeline import WanT2VLowVramPipeline


def setup_logging(log_level: str = "INFO") -> None:
    """
    配置规范的分级日志系统
    同时输出到控制台与 output/wan_t2v_execution.log 文件
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    log_file = OUTPUT_DIR / "wan_t2v_execution.log"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
    ]

    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )


def main():
    parser = argparse.ArgumentParser(
        description="Wan 2.1 4GB 显存端到端本地文生视频一键运行脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A futuristic cyberpunk city at night, rain-slicked streets reflecting neon lights, a creator standing in front of high-rise window looking at a glowing blue holographic system blueprint, cinematic forward motion, realistic physics.",
        help="文生视频自然语言提示词（默认采用赛博朋克都市基准镜头）"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="色调艳丽，过曝，静态，残影，模糊，扭曲，变形，多余的肢体，融化的物体",
        help="负向提示词"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="视频宽度（必须是 16 的倍数，4G 显存推荐 480）"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=272,
        help="视频高度（必须是 16 的倍数，4G 显存推荐 272）"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=33,
        help="视频帧数（必须满足 4n+1，如 33 帧、49 帧）"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="视频采样帧率（原生 8~12 fps）"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="DiT 去噪步数（推荐 20 步兼顾质量与速度）"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,
        help="分类器自由引导尺度 (CFG Scale)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机数种子，用于结果复现"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 MP4 视频文件名（若不指定则按时间戳自动命名）"
    )
    parser.add_argument(
        "--vram-limit",
        type=float,
        default=3.6,
        help="显存安全阈值上限（GB，4G 显存推荐 3.6GB）"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志记录级别"
    )

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("T2V.Main")

    logger.info("=========================================================")
    logger.info("  Wan 2.1 4GB 显存端到端文生视频推理引擎（Quadro T1000 适配版）")
    logger.info("=========================================================")

    try:
        config = VideoGenerationConfig(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            fps=args.fps,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            output_filename=args.output,
            vram_limit_gb=args.vram_limit
        )
        config.validate()
    except ValueError as e:
        logger.error(f"输入参数校验未通过: {e}")
        sys.exit(1)

    try:
        pipeline = WanT2VLowVramPipeline(config)
        audit_result = pipeline.generate()

        logger.info("================== 任务审计清单 ==================")
        logger.info(f"生成视频路径: {audit_result['output_file']}")
        logger.info(f"画面规格: {audit_result['width']}x{audit_result['height']} ({audit_result['num_frames']} 帧)")
        logger.info(f"去噪步数: {audit_result['num_inference_steps']} 步")
        logger.info(f"总计算耗时: {audit_result['total_elapsed_sec']} 秒")
        logger.info(f"GPU 峰值显存: {audit_result['gpu_peak_vram_mb']} MB (警戒线: {args.vram_limit * 1024} MB)")
        logger.info("==================================================")
    except Exception as e:
        logger.error(f"视频生成运行发生异常: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
