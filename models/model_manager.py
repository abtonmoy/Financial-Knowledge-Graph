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

                print(f"Language model loaded: {model_name}")
                
            except Exception as e:
                print(f"Error loading main model, trying alternative: {e}")
                self._load_fallback_llm()
        
        return self._loaded_models["llm"]
    
    def _load_fallback_llm(self):
        """Load fallback language model"""
        model_name = self.config.MODELS["llm"]["alternative"]
        tokenizer =  AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model =  model.to(self.device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self._loaded_models["llm"] = {
            "model": model,
            "tokenizer": tokenizer
        }
        print(f"Fallback language model loaded: {model_name}")
    
    def get_ner_pipeline(self):
        """Get NER pipeline for entity extraction."""
        if "ner" not in self._load_models:
            print("Loading NER model...")
            try:
                model_name = self.config.MODELS["ner"]["name"]
                ner_pipeline = pipeline(
                    model =  model_name,
                    tokenizer = model_name,
                    aggregation_strategy = "simple",
                    device=0 if self.device=="cuda" else -1 
                )
                self._load_models["ner"] = ner_pipeline
                print(f"NER model loaded: {model_name}")
            except Exception as e:
                print(f"Error loading main NER model, trying alternative: {e}")
                self._load_fallback_ner()
        
        return self._loaded_models["ner"]
    
    def _load_fallback_ner(self):
        """Load fallback NER model"""
        model_name = self.config.MODELS["ner"]["alternative"]
        ner_pipeline = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple",
            device=0 if self.device == "cuda" else -1
        )
        self._loaded_models["ner"] = ner_pipeline
        print(f"Fallback NER model loaded: {model_name}")

    def get_embeddings_model(self):
        if "embeddings" not in self._load_models:
            print("Loading embeddings model...")
            try:
                model_name = self.config.MODELS["embeddings"]["name"]
                embedder = SentenceTransformer(model_name, device=self.device)
                self._loaded_models["embeddings"] = embedder
                print(f"Embeddings model loaded: {model_name}")
            except Exception as e:
                print(f"Error loading embeddings model: {e}")
                self._load_fallback_embeddings()
        
        return self._loaded_models["embeddings"]
    
    def _load_fallback_embeddings(self):
        """Load fallback embeddings model."""
        model_name = self.config.MODELS["embeddings"]["alternative"]
        embedder = SentenceTransformer(model_name, device=self.device)
        self._loaded_models["embeddings"] = embedder
        print(f"Fallback embeddings model loaded: {model_name}")

    def generate_text(self, prompt: str, max_length: Optional[int] = None, temperature: Optional[float]=None) -> str:
        """Generate text using the local LLM"""
        max_length = max_length or self.config.MAX_GENERATION_LENGTH
        temperature = temperature or self.config.GENERATION_TEMPERATURE

        llm = self.get_llm()
        model = llm["model"]
        tokenizer = llm["tokenizer"]

        try:
            # encode the prompt
            inputs =  tokenizer.encode(prompt, return_tensor="pt", truncation=True, max_length=400)
            inputs = inputs.to(self.device)

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=min(max_length, inputs.shape[1]+200),
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )

            # Decode the response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the new part -> remove the prompt()
            response = generated_text[len(prompt):].strip()

            return response if response else "I need more context to provide a specific answer"
        
        except Exception as e:
            print(f"Error generating text: {e}")
            return f"Error generating response: {str(e)}"
        
    def get_loaded_models(self) -> list:
        """Get list of currently loaded models"""
        return list(self._loaded_models.key())
    
    def unload_model(self, model_type: str):
        """Unload a specific model to free memory"""
        if model_type in self._load_models:
            del self._load_models[model_type]
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print(f" Unloaded {model_type} model")
    
    def unload_all_models(self):
        """Unload all models to free memory."""
        self._loaded_models.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(" All models unloaded")