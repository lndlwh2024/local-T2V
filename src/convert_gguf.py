import gc
import logging
import time
from pathlib import Path
import gguf
import torch
from safetensors.torch import save_file
from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers
from diffusers import AutoencoderKLWan, WanTransformer3DModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("T2V.ConvertGGUF")

BASE_DIR = Path(__file__).resolve().parent.parent
GGUF_PATH = BASE_DIR / "models" / "diffusion_models" / "Wan2.1-T2V-1.3B-Q4_K_M.gguf"
OUTPUT_SAFETENSORS = BASE_DIR / "models" / "diffusion_models" / "diffusion_pytorch_model.safetensors"
CONFIG_PATH = BASE_DIR / "models" / "diffusion_models" / "config.json"
VAE_PTH_PATH = BASE_DIR / "models" / "vae" / "Wan2.1_VAE.pth"
VAE_DIR = BASE_DIR / "models" / "vae" / "diffusers_vae"


def convert_gguf_to_diffusers_safetensors():
    """
    将 GGUF Q4_K_M 量化文件精确反量化并无损映射为 Diffusers 标准 safetensors 格式
    """
    if OUTPUT_SAFETENSORS.exists() and OUTPUT_SAFETENSORS.stat().st_size > 2 * 1024**3:
        logger.info(f"目标 Safetensors 文件已存在 ({OUTPUT_SAFETENSORS.stat().st_size / 1024**2:.2f} MB)，跳过转换。")
        return

    logger.info(f"正在打开 GGUF 文件: {GGUF_PATH}...")
    reader = gguf.GGUFReader(str(GGUF_PATH))
    total_tensors = len(reader.tensors)
    logger.info(f"GGUF 总张量数: {total_tensors}")

    raw_sd = {}
    start_time = time.time()
    
    for i, t in enumerate(reader.tensors):
        from gguf.quants import dequantize
        d = dequantize(t.data, t.tensor_type)
        raw_sd[t.name] = torch.from_numpy(d.copy()).to(torch.float16)
        if (i + 1) % 100 == 0 or (i + 1) == total_tensors:
            logger.info(f"反量化进度: {i + 1}/{total_tensors} ({(i + 1) / total_tensors * 100:.1f}%)")

    logger.info(f"所有张量反量化完成，耗时: {time.time() - start_time:.2f}s。正在执行 Diffusers 键名标准化映射...")
    converted_sd = convert_wan_transformer_to_diffusers(raw_sd)
    logger.info(f"标准化键名映射完成，张量数: {len(converted_sd)}")

    del raw_sd
    gc.collect()

    logger.info(f"正在落盘保存至: {OUTPUT_SAFETENSORS}...")
    save_file(converted_sd, str(OUTPUT_SAFETENSORS))
    logger.info(f"落盘完成！大小: {OUTPUT_SAFETENSORS.stat().st_size / 1024**2:.2f} MB")

    # 保存 1.3B 官方 Transformer 配置文件
    logger.info("保存 Transformer 架构配置文件 config.json...")
    config = WanTransformer3DModel.load_config("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="transformer")
    import json
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(dict(config), f, indent=2)
    logger.info("Transformer 配置已成功保存！")


def convert_vae_to_diffusers():
    """
    将 Wan2.1_VAE.pth 转换为标准 Diffusers VAE 目录，加快后续毫秒级加载
    """
    if (VAE_DIR / "diffusion_pytorch_model.safetensors").exists():
        logger.info("标准 VAE 已就绪，跳过转换。")
        return

    logger.info(f"正在从 {VAE_PTH_PATH} 加载原始 VAE 并导出为标准 Diffusers 格式...")
    vae = AutoencoderKLWan.from_single_file(str(VAE_PTH_PATH))
    vae.save_pretrained(str(VAE_DIR))
    logger.info(f"标准 VAE 已保存至 {VAE_DIR}！")


if __name__ == "__main__":
    convert_gguf_to_diffusers_safetensors()
    convert_vae_to_diffusers()
