import torch
from typing import Optional, List, Dict, Any, Generator
import logging
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline,
    TextStreamer
)
from huggingface_hub import login
import os

from ..core.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.LLM_MODEL
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_model()
    
    def _init_model(self):
        """初始化LLM模型"""
        try:
            # 检查是否有HF Token（如果需要）
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                login(token=hf_token)
            
            # 量化配置
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 设置padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # 创建pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto"
            )
            
            logger.info(f"模型 {self.model_name} 加载成功，设备: {self.device}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None,
                max_tokens: int = None,
                temperature: float = None,
                top_p: float = None,
                stream: bool = False) -> Generator[str, None, None]:
        """生成文本"""
        try:
            # 构建完整prompt
            if system_prompt:
                full_prompt = f"<|system|>\n{system_prompt}\n<|end|>\n<|user|>\n{prompt}\n<|end|>\n<|assistant|>\n"
            else:
                full_prompt = f"<|user|>\n{prompt}\n<|end|>\n<|assistant|>\n"
            
            # 生成参数
            generation_config = {
                "max_new_tokens": max_tokens or settings.LLM_MAX_TOKENS,
                "temperature": temperature or settings.LLM_TEMPERATURE,
                "top_p": top_p or settings.LLM_TOP_P,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            if stream:
                # 流式生成
                inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    for _ in self.model.generate(
                        **inputs,
                        **generation_config,
                        streamer=TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True),
                        max_new_tokens=generation_config["max_new_tokens"]
                    ):
                        # 这里简化处理，实际应该从streamer获取
                        pass
                yield ""  # 简化实现
            else:
                # 批量生成
                result = self.pipeline(
                    full_prompt,
                    max_new_tokens=generation_config["max_new_tokens"],
                    temperature=generation_config["temperature"],
                    top_p=generation_config["top_p"],
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False
                )
                
                yield result[0]['generated_text']
                
        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            yield f"生成错误: {str(e)}"
    
    def get_completion(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      **kwargs) -> str:
        """获取完整回复（非流式）"""
        for response in self.generate(prompt, system_prompt, stream=False, **kwargs):
            return response
        return ""