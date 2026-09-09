import gc
import logging
import time
from contextlib import contextmanager
from typing import Generator, Optional

logger = logging.getLogger("T2V.MemorySentinel")


class MemorySentinel:
    """
    显存与内存哨兵
    针对 NVIDIA Quadro T1000（4GB 显存）极端受限硬件设计的内存管线保护机制。
    
    设计原因：
    PyTorch 默认的 Caching Allocator 为加速显存分配，在张量生命周期结束后并不会立即归还系统显存，
    在视频潜空间去噪多步迭代中，易导致显存碎片化累积并最终引发 CUDA OOM。
    因此本哨兵提供主动显存追踪、硬性警戒线监测与强制碎片回收功能。
    """

    def __init__(self, vram_limit_gb: float = 3.6):
        self.vram_limit_bytes = int(vram_limit_gb * 1024 * 1024 * 1024)
        self.vram_limit_gb = vram_limit_gb

    @staticmethod
    def force_clean_memory() -> None:
        """
        深度显存与内存垃圾回收
        必须执行两步：
        1. gc.collect() 强制 Python 回收无引用的中间激活张量（如注意力权重、局部变量）；
        2. torch.cuda.empty_cache() 将 PyTorch 缓存池中的空闲显存块真正释放给显卡驱动。
        """
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass

    @staticmethod
    def get_memory_info() -> dict:
        """
        获取当前实时的 GPU 显存与 CPU 内存使用情况
        单位转换为 MB，便于日志记录与性能审计
        """
        info = {
            "gpu_allocated_mb": 0.0,
            "gpu_reserved_mb": 0.0,
            "gpu_max_allocated_mb": 0.0,
        }
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                max_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
                info["gpu_allocated_mb"] = round(allocated, 2)
                info["gpu_reserved_mb"] = round(reserved, 2)
                info["gpu_max_allocated_mb"] = round(max_alloc, 2)
        except ImportError:
            pass
        return info

    def log_status(self, stage_prefix: str = "") -> None:
        """
        打印结构化内存状态日志，使用合理的日志级别（DEBUG/INFO/WARNING）
        """
        info = self.get_memory_info()
        prefix = f"[{stage_prefix}] " if stage_prefix else ""
        logger.info(
            f"{prefix}显存现状: 已分配 {info['gpu_allocated_mb']}MB | "
            f"保留池 {info['gpu_reserved_mb']}MB | 峰值 {info['gpu_max_allocated_mb']}MB"
        )

    def check_safety(self, current_stage: str = "") -> bool:
        """
        边界安全检测：当已分配显存逼近安全阈值（如 3.6GB）时发出告警并触发紧急清理
        """
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                if allocated >= self.vram_limit_bytes:
                    logger.warning(
                        f"警告: 阶段 [{current_stage}] 显存已分配 {allocated / 1024**3:.2f}GB，"
                        f"超出安全警戒线 {self.vram_limit_gb}GB！正在启动紧急回收..."
                    )
                    self.force_clean_memory()
                    return False
        except ImportError:
            pass
        return True

    @contextmanager
    def guard(self, stage_name: str) -> Generator[None, None, None]:
        """
        上下文管理器：包裹关键计算阶段（如文本编码、去噪循环、VAE 解码）
        进入前清理缓存，退出后强制释放并输出该阶段显存消耗与耗时
        """
        logger.info(f">>> 进入阶段: [{stage_name}]")
        self.force_clean_memory()
        start_time = time.time()
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.force_clean_memory()
            info = self.get_memory_info()
            logger.info(
                f"<<< 完成阶段: [{stage_name}] 耗时: {elapsed:.2f}s | "
                f"当前显存占用: {info['gpu_allocated_mb']}MB | 历史峰值: {info['gpu_max_allocated_mb']}MB"
            )
