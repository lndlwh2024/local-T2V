import sys
import unittest
from pathlib import Path

# 将项目根目录加入搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import VideoGenerationConfig
from src.memory_sentinel import MemorySentinel


class TestWanT2VComponents(unittest.TestCase):
    """
    Wan2.1 4GB 显存文生视频核心组件单元测试
    验证边界安全防护、输入合法性校验与显存哨兵逻辑
    """

    def test_valid_config(self):
        """测试正常参数配置下的合法性"""
        config = VideoGenerationConfig(
            prompt="A beautiful sunrise over the ocean",
            width=480,
            height=272,
            num_frames=33,
            num_inference_steps=20
        )
        # 应该正常通过，不抛出异常
        config.validate()
        self.assertEqual(config.width, 480)
        self.assertEqual(config.height, 272)
        self.assertEqual(config.num_frames, 33)

    def test_empty_prompt_raises_error(self):
        """测试空 Prompt 防御机制"""
        config = VideoGenerationConfig(prompt="   ")
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("不能为空", str(ctx.exception))

    def test_resolution_alignment_defense(self):
        """测试非 16 像素整除边界拦截"""
        config = VideoGenerationConfig(prompt="Test", width=481, height=272)
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("16 的整倍数", str(ctx.exception))

    def test_oversized_resolution_defense(self):
        """测试 4GB 显存超限分辨率硬性拦截"""
        config = VideoGenerationConfig(prompt="Test", width=1280, height=720)
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("不得超过 640x360", str(ctx.exception))

    def test_frame_rule_defense(self):
        """测试 3D-VAE (4n+1) 帧数法则边界拦截"""
        config = VideoGenerationConfig(prompt="Test", num_frames=30)
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("(N - 1) % 4 == 0", str(ctx.exception))

    def test_memory_sentinel_metrics(self):
        """测试显存哨兵状态提取与回收机制"""
        sentinel = MemorySentinel(vram_limit_gb=3.6)
        info = sentinel.get_memory_info()
        self.assertIn("gpu_allocated_mb", info)
        self.assertIn("gpu_reserved_mb", info)
        self.assertIn("gpu_max_allocated_mb", info)
        # 测试强制垃圾清理无异常抛出
        sentinel.force_clean_memory()


if __name__ == "__main__":
    unittest.main()
