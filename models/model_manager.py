"""Model manager for language models"""

import torch
from typing import Dict, Any, Optional
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
                          AutoModelForTokenClassification,
                          TokenClassificationPipeline, 
                          pipeline,
                          BitsAndBytesConfig)
from sentence_transformers import SentenceTransformer
from ..config import get_config

class OpenSourceModelManager:

    def __init__(self, device: str = "auto"):
        self.config = get_config()
        self.device = device
        self._setup_device()

        self._load_models = {}

        print(f"using device: {self.device}")

    def _setup_device(self):
        """setup the best available device"""
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"CUDA available: {torch.cuda.get_device_name()}")
            # elif hasattr(torch.backends, 'mps') -> if using apple silicon, code this up
            else:
                self.device = "cpu"
                print("using CPU")

    def get_llm(self) ->  Dict[str, Any]:
        """get model for text generation"""
        if "llm" not in self._load_models:
            print("Loading language model...")

            try:
                model_name = self.config.MODELS["llm"]["name"]

                if self.device == "cuda":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                else:
                    quantization_config = None
                
                tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side = "left")

                if tokenizer.pad_token is None:
                    tokenizer.pad_token =  tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map ="auto" if self.device == "cuda" else None,
                    torch_dtype = torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage = True
                )

                if self.device != "cuda":
                    model = model.to(self.device)

                self._loaded_models["llm"] = {
                    "model": model,
                    "tokenizer": tokenizer
                }

                print(f"✅ Language model loaded: {model_name}")
                
            except Exception as e:
                print(f"Error loading main model, trying alternative: {e}")
                self._load_fallback_llm()
        
        return self._loaded_models["llm"]