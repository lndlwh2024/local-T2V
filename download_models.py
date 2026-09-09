import argparse
import logging
import os
import sys
from pathlib import Path

# 配置标准日志记录器
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("T2V.Downloader")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DIFFUSION_DIR = MODELS_DIR / "diffusion_models"
TEXT_ENCODER_DIR = MODELS_DIR / "text_encoders"
VAE_DIR = MODELS_DIR / "vae"

# 模型资产常量配置
MODELS_CATALOG = {
    "diffusion_gguf": {
        "filename": "Wan2.1-T2V-1.3B-Q4_K_M.gguf",
        "target_dir": DIFFUSION_DIR,
        "modelscope_repo": "city96/Wan2.1-T2V-1.3B-GGUF",
        "huggingface_repo": "city96/Wan2.1-T2V-1.3B-GGUF",
        "description": "Wan 2.1 1.3B Q4_K_M GGUF 量化主干模型 (~1.2GB)"
    },
    "vae": {
        "filename": "Wan2.1_VAE.pth",
        "target_dir": VAE_DIR,
        "modelscope_repo": "Wan-AI/Wan2.1-T2V-1.3B",
        "huggingface_repo": "Wan-AI/Wan2.1-T2V-1.3B",
        "subfolder": "Wan2.1_VAE.pth",
        "description": "Wan 2.1 官方 3D Causal VAE 解码器权重"
    },
    "text_encoder": {
        "target_dir": TEXT_ENCODER_DIR,
        "modelscope_repo": "Wan-AI/Wan2.1-T2V-1.3B",
        "huggingface_repo": "Wan-AI/Wan2.1-T2V-1.3B",
        "subfolder": "google/umt5-xxl",
        "description": "UMT5-XXL 文本编码器与分词器（驻留 40GB CPU 内存）"
    }
}


def download_from_modelscope(repo_id: str, target_dir: Path, filename: str = None, subfolder: str = None) -> None:
    """
    使用 ModelScope 魔搭社区 API 下载资产（国内网络环境下速度快且稳定）
    """
    try:
        from modelscope.hub.file_download import model_file_download
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        logger.error("未检测到 modelscope 依赖，请在虚拟环境中执行: uv pip install modelscope")
        sys.exit(1)

    logger.info(f"正在从 ModelScope 拉取仓库 [{repo_id}] 到 {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)

    if filename:
        model_file_download(
            model_id=repo_id,
            file_path=filename,
            local_dir=str(target_dir)
        )
    elif subfolder:
        snapshot_download(
            model_id=repo_id,
            allow_patterns=[f"{subfolder}/*"],
            local_dir=str(target_dir)
        )
    else:
        snapshot_download(
            model_id=repo_id,
            local_dir=str(target_dir)
        )
    logger.info(f"ModelScope 资源 [{repo_id}] 下载完成！")


def download_from_huggingface(repo_id: str, target_dir: Path, filename: str = None, subfolder: str = None) -> None:
    """
    使用 Hugging Face Hub（支持 HF-Mirror 镜像加速）下载资产
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        logger.error("未检测到 huggingface_hub 依赖，请在虚拟环境中执行: uv pip install huggingface_hub")
        sys.exit(1)

    # 优先检测或配置国内镜像源以防止连接超时
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logger.info("已自动激活 HF_ENDPOINT=https://hf-mirror.com 镜像加速")

    logger.info(f"正在从 HuggingFace 镜像拉取仓库 [{repo_id}] 到 {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)

    if filename:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(target_dir)
        )
    elif subfolder:
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{subfolder}/*"],
            local_dir=str(target_dir)
        )
    else:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir)
        )
    logger.info(f"HuggingFace 资源 [{repo_id}] 下载完成！")


def main():
    parser = argparse.ArgumentParser(description="Wan 2.1 文生视频本地模型资产下载器")
    parser.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default="modelscope",
        help="下载源：优先推荐 modelscope（国内极速），亦可选 huggingface（镜像）"
    )
    parser.add_argument(
        "--asset",
        choices=["all", "diffusion_gguf", "vae", "text_encoder"],
        default="all",
        help="指定下载的模型资产类型，默认为全部"
    )
    args = parser.parse_args()

    logger.info(f"启动模型下载流水线 | 下载源: {args.source} | 目标资产: {args.asset}")

    assets_to_download = (
        MODELS_CATALOG.keys() if args.asset == "all" else [args.asset]
    )

    for asset_key in assets_to_download:
        item = MODELS_CATALOG[asset_key]
        logger.info(f"--> 开始处理: {item['description']}")
        
        # 边界检查：若文件已存在且大小正常，则跳过下载，避免重复消耗带宽与等待
        target_dir = item["target_dir"]
        filename = item.get("filename")
        if filename and (target_dir / filename).exists():
            file_size_mb = (target_dir / filename).stat().st_size / (1024 * 1024)
            if file_size_mb > 10:  # 排除空文件或损坏残存文件
                logger.info(f"文件已存在 ({file_size_mb:.2f} MB)，跳过下载: {filename}")
                continue

        if args.source == "modelscope":
            download_from_modelscope(
                repo_id=item["modelscope_repo"],
                target_dir=target_dir,
                filename=filename,
                subfolder=item.get("subfolder")
            )
        else:
            download_from_huggingface(
                repo_id=item["huggingface_repo"],
                target_dir=target_dir,
                filename=filename,
                subfolder=item.get("subfolder")
            )

    logger.info("所有请求的模型资产已就绪！")


if __name__ == "__main__":
    main()
