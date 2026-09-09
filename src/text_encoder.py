import gc
import logging
from pathlib import Path
from typing import Optional, Tuple
import torch

logger = logging.getLogger("T2V.TextEncoder")


class CpuTextEncoderManager:
    """
    CPU 物理内存文本编码器管理器
    
    设计核心机制：
    1. 零显存占用原则：
       将 5.4GB 的 UMT5-XXL 文本编码器严格限定在 CPU 物理内存中加载与执行（利用本机实测 40GB RAM）。
       杜绝文本编码模型进入 GPU 物理显存，为 4GB Quadro T1000 节省超过 5GB 的宝贵显存空间。
    2. 标准嵌入对齐：
       输出 Wan2.1 要求的 [batch_size, max_seq_len, 4096] 维度嵌入张量，
       计算完成后仅将此微小张量（约几兆字节）转移至 GPU。
    """

    def __init__(
        self,
        tokenizer_dir: Path,
        model_dir: Path,
        max_sequence_length: int = 226,
        torch_dtype: torch.dtype = torch.float16
    ):
        self.tokenizer_dir = tokenizer_dir
        self.model_dir = model_dir
        self.max_sequence_length = max_sequence_length
        self.torch_dtype = torch_dtype
        self.tokenizer = None
        self.text_encoder = None
        self.device = torch.device("cpu")

    def load(self) -> None:
        """
        在 CPU 物理内存中按需加载 Tokenizer 与 Text Encoder
        """
        if self.text_encoder is not None:
            return

        logger.info(f"正在从 {self.tokenizer_dir} 加载分词器...")
        from transformers import AutoTokenizer, UMT5EncoderModel

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_dir))

        logger.info(f"正在从 {self.model_dir} 加载 UMT5 文本编码器至 CPU 物理内存（利用 40GB RAM，不挤占 GPU 显存）...")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            str(self.model_dir),
            torch_dtype=self.torch_dtype,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        self.text_encoder.eval()
        logger.info("UMT5 文本编码器在 CPU 内存加载完成，未占用任何显卡物理显存。")

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        target_device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在 CPU 上对 Prompt 和 Negative Prompt 进行特征抽取，并按需对齐序列长度
        """
        if self.text_encoder is None:
            self.load()

        logger.info(f"正在 CPU 上对提示词进行语义编码: '{prompt[:60]}...'")
        target_device = target_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            # 编码正向提示词
            text_inputs = self.tokenizer(
                [prompt],
                padding="max_length",
                max_length=self.max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(self.device)
            mask = text_inputs.attention_mask.to(self.device)
            seq_lens = mask.gt(0).sum(dim=1).long()

            prompt_embeds = self.text_encoder(input_ids, mask).last_hidden_state
            prompt_embeds = prompt_embeds.to(dtype=self.torch_dtype)
            prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
            prompt_embeds = torch.stack(
                [torch.cat([u, u.new_zeros(self.max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds],
                dim=0
            )

            # 编码负向提示词
            neg_prompt = negative_prompt or ""
            neg_inputs = self.tokenizer(
                [neg_prompt],
                padding="max_length",
                max_length=self.max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            neg_input_ids = neg_inputs.input_ids.to(self.device)
            neg_mask = neg_inputs.attention_mask.to(self.device)
            neg_seq_lens = neg_mask.gt(0).sum(dim=1).long()

            neg_embeds = self.text_encoder(neg_input_ids, neg_mask).last_hidden_state
            neg_embeds = neg_embeds.to(dtype=self.torch_dtype)
            neg_embeds = [u[:v] for u, v in zip(neg_embeds, neg_seq_lens)]
            neg_embeds = torch.stack(
                [torch.cat([u, u.new_zeros(self.max_sequence_length - u.size(0), u.size(1))]) for u in neg_embeds],
                dim=0
            )

        # 仅将轻量结果嵌入张量转移至 GPU
        prompt_embeds = prompt_embeds.to(target_device)
        neg_embeds = neg_embeds.to(target_device)

        logger.info(f"文本编码完成，输出张量形状: {prompt_embeds.shape}，已传入计算设备: {target_device}")
        return prompt_embeds, neg_embeds
