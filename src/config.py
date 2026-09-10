import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 项目基础目录结构定义
# 使用绝对路径以避免不同工作目录下执行脚本导致的相对路径寻址失败
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DIFFUSION_DIR = MODELS_DIR / "diffusion_models"
TEXT_ENCODER_DIR = MODELS_DIR / "text_encoders"
VAE_DIR = MODELS_DIR / "vae"
OUTPUT_DIR = BASE_DIR / "output"

# 确保关键目录存在，避免因输出目录缺失引发 IO 异常
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIFFUSION_DIR.mkdir(parents=True, exist_ok=True)
TEXT_ENCODER_DIR.mkdir(parents=True, exist_ok=True)
VAE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class VideoGenerationConfig:
    """
    视频生成参数配置类
    负责承载生成参数，并实施严格的合法性校验与边界限制，防止恶意输入或非法尺寸导致显存溢出
    """
    prompt: str
    negative_prompt: str = "色调艳丽，过曝，静态，残影，模糊，扭曲，变形，多余的肢体，融化的物体"
    # 分辨率设为 480x272 是针对 4GB Quadro T1000 的甜点参数
    # 尺寸需为 16 的倍数以适配 Wan2.1 3D-VAE 的 8x 下采样与 Patchify 卷积切分
    width: int = 480
    height: int = 272
    # 帧数设定必须满足 4n + 1 公式（如 9, 33, 41, 49），因 3D Causal VAE 时间轴压缩比为 4:1
    num_frames: int = 33
    # 目标裁切帧数：当设定时，底层按 num_frames (4n+1) 生成，导出时自动精确截断为 target_num_frames
    target_num_frames: Optional[int] = None
    fps: int = 8
    num_inference_steps: int = 20
    # 引导强度设为 3.2，消除高 CFG (5.0) 引起的边缘吉布斯振铃、双重发际线重影与高频干涉
    guidance_scale: float = 3.2
    seed: Optional[int] = 42
    # 物理显存安全警戒线：4096MB 显存需扣除系统合成器及驱动保留的 ~400MB，故硬锁定在 3.6GB
    vram_limit_gb: float = 3.6
    output_filename: Optional[str] = None

    def validate(self) -> None:
        """
        参数安全与边界校验
        根据安全规范，永远不信任客户端输入，严防越界参数击穿 4GB 显存
        """
        if not self.prompt or not self.prompt.strip():
            raise ValueError("Prompt 提示词不能为空")
        
        # 限制 prompt 最大长度，防止恶意长文本注入耗尽 CPU 内存与 Tokenizer 资源
        if len(self.prompt) > 1000:
            raise ValueError("Prompt 长度超过限制（最大支持 1000 字符）")
        
        # 必须满足 16 像素对齐边界条件，否则 VAE 解码与 Patch 投影会出现尺寸错位崩溃
        if self.width % 16 != 0 or self.height % 16 != 0:
            raise ValueError(f"分辨率 ({self.width}x{self.height}) 必须是 16 的整倍数")
            
        # 4GB 显存硬性分辨率上限约束：严禁超过 640x360，否则无论如何量化都会发生 CUDA OOM
        if self.width * self.height > 640 * 360:
            raise ValueError(f"当前硬件为 4GB 显存，总像素数量不得超过 640x360（当前请求: {self.width}x{self.height}）")
            
        # 若指定了 target_num_frames，自动校准底层 num_frames 为满足 4n+1 的最小合法整数
        if self.target_num_frames is not None:
            if self.target_num_frames <= 0:
                raise ValueError("目标帧数 target_num_frames 必须为正整数")
            # 计算 >= target_num_frames 且满足 (N - 1) % 4 == 0 的值
            remainder = (self.target_num_frames - 1) % 4
            if remainder == 0:
                self.num_frames = self.target_num_frames
            else:
                self.num_frames = self.target_num_frames + (4 - remainder)

        # 验证底层帧数边界：Wan 3D VAE 结构要求 (num_frames - 1) % 4 == 0
        if (self.num_frames - 1) % 4 != 0:
            raise ValueError(f"底层计算帧数 ({self.num_frames}) 必须满足 (N - 1) % 4 == 0，推荐 33 或 41 帧")
            
        if self.num_frames > 81:
            raise ValueError("4GB 显存下帧数不能超过 81 帧，否则时空注意力张量显存将超限")

        if self.num_inference_steps < 5 or self.num_inference_steps > 100:
            raise ValueError("推理步数需处于 [5, 100] 合理区间")
