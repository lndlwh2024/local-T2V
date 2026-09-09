import logging
from pathlib import Path
from typing import Optional, Tuple
import torch

logger = logging.getLogger("T2V.TextEncoder")


class CpuTextEncoderManager:
    """
    CPU 文本编码器管理器
    
    设计关键（为什么必须绑定 CPU）：
    Wan2.1 依赖 UMT5-XXL 作为文本条件特征提取器，其权重达 11GB+。
    在 4GB 显存显卡（NVIDIA Quadro T1000）上，若将文本编码器载入显存会瞬间导致 CUDA OOM；
    而本机实测配备了高达 40GB 的物理内存，因此将 UMT5-XXL 100% 绑定于 CPU（device='cpu'）执行推理，
    仅将最终得到的微小特征张量（Prompt Embedding，约几兆字节）传输至 GPU，
    从而达成在文本特征抽取阶段【对 4G 显卡显存 0 占用】的极限工程优化。
    """

    def __init__(self, model_dir: Path, torch_dtype: torch.dtype = torch.bfloat16):
        self.model_dir = model_dir
        self.torch_dtype = torch_dtype
        self.tokenizer = None
        self.text_encoder = None
        self.device = torch.device("cpu")

    def load(self) -> None:
        """
        在 CPU 物理内存中延迟加载 Tokenizer 与 Text Encoder
        """
        if self.text_encoder is not None:
            logger.info("CPU 文本编码器已就绪，复用现有模型缓存。")
            return

        logger.info(f"正在从 {self.model_dir} 加载 UMT5 文本编码器到 CPU 内存（利用本机 40GB RAM）...")
        
        try:
            from transformers import AutoTokenizer, UMT5EncoderModel
            
            # 使用 local_files_only 确保网络抖动或断网时不发生隐式下载阻塞
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_dir),
                local_files_only=True
            )
            # 严格强制加载至 CPU，防止误分配到 CUDA
            self.text_encoder = UMT5EncoderModel.from_pretrained(
                str(self.model_dir),
                torch_dtype=self.torch_dtype,
                device_map="cpu",
                local_files_only=True
            )
            self.text_encoder.eval()
            logger.info("UMT5 文本编码器在 CPU 内存加载完成，未占用任何物理显存。")
        except Exception as e:
            logger.error(f"加载 CPU 文本编码器失败: {e}", exc_info=True)
            raise

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        max_sequence_length: int = 512
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        使用 CPU 对 Prompt 与 Negative Prompt 进行特征编码
        
        返回值：
        - prompt_embeds: 形状为 [1, seq_len, hidden_dim] 的嵌入张量（位于 CPU）
        - negative_prompt_embeds: 负向嵌入张量（若未提供则为 None）
        """
        if self.text_encoder is None:
            self.load()

        logger.info(f"正在 CPU 上对提示词进行编码，提示词长度: {len(prompt)} 字符")
        
        with torch.no_grad():
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(self.device)
            attention_mask = text_inputs.attention_mask.to(self.device)

            # 在 CPU 上执行前向传播
            prompt_embeds = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )[0]

            neg_embeds = None
            if negative_prompt:
                neg_inputs = self.tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_sequence_length,
                    truncation=True,
                    return_tensors="pt",
                )
                neg_input_ids = neg_inputs.input_ids.to(self.device)
                neg_attention_mask = neg_inputs.attention_mask.to(self.device)
                neg_embeds = self.text_encoder(
                    input_ids=neg_input_ids,
                    attention_mask=neg_attention_mask
                )[0]

        logger.info(f"提示词 CPU 编码完成，嵌入张量维度: {prompt_embeds.shape}")
        return prompt_embeds, neg_embeds

    def unload(self) -> None:
        """
        手动释放 CPU 内存中的模型对象（当需要极端释放物理内存时调用）
        """
        if self.text_encoder is not None:
            logger.info("正在释放 CPU 文本编码器以清理系统物理内存...")
            del self.text_encoder
            del self.tokenizer
            self.text_encoder = None
            self.tokenizer = None
            import gc
            gc.collect()
