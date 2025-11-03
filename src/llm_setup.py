"""
LLM setup for MPT-7B-Instruct model.
Handles loading and configuring the free LLM for text generation.
"""

import warnings
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.language_models import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import ConfigDict

# Suppress common transformers warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def get_best_device():
    """Auto-detect best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


class MPTInstructLLM(LLM):
    """Wrapper for MPT-7B-Instruct model to work with LangChain."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model_name: str = "mosaicml/mpt-7b-instruct"
    device: str = "auto"  # "auto" will detect MPS/CUDA/CPU, or specify "mps", "cuda", "cpu"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Auto-detect device if "auto" is specified
        if self.device == "auto":
            detected_device = get_best_device()
            self.device = detected_device
            print(f"🤖 Auto-detected device: {detected_device}")
            if detected_device == "mps":
                print("  ✅ Using Apple Silicon Neural Engine (MPS) - Faster inference!")
        object.__setattr__(self, '_pipeline', None)
        object.__setattr__(self, '_tokenizer', None)
    
    @property
    def _llm_type(self) -> str:
        return "mpt-instruct"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[list] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """Generate text using MPT-7B-Instruct."""
        if self._pipeline is None:
            self._initialize_model()
        
        # Format prompt for instruction-following
        formatted_prompt = f"### Instruction:\n{prompt}\n### Response:\n"
        
        # Generate response
        outputs = self._pipeline(
            formatted_prompt,
            max_new_tokens=self.max_length,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        
        response = outputs[0]["generated_text"]
        # Extract just the response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        # Stop generation at stop sequences
        if stop:
            for stop_seq in stop:
                if stop_seq in response:
                    response = response.split(stop_seq)[0].strip()
        
        return response
    
    def _initialize_model(self):
        """Initialize the model and tokenizer."""
        device_name = self.device.upper() if self.device != "mps" else "Apple Neural Engine (MPS)"
        print(f"Loading {self.model_name}... Using {device_name}")
        print("Progress: ", end="", flush=True)
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        print("✓ Tokenizer loaded")
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        object.__setattr__(self, '_tokenizer', tokenizer)
        
        if self.device == "mps":
            print("Loading model weights to Apple Neural Engine... (faster than CPU!)")
        elif self.device == "cuda":
            print("Loading model weights to GPU...")
        else:
            print("Loading model weights to CPU... (this may take 30-60 seconds)")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use device_map for CUDA, manual move for MPS/CPU
            if self.device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype="auto",
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype="auto",
                    device_map=None,
                    low_cpu_mem_usage=True,
                )
                # Move to appropriate device
                if self.device == "mps":
                    model = model.to("mps")
                elif self.device == "cpu":
                    model = model.to("cpu")
        
        print("Setting up generation pipeline...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Set device for pipeline (transformers uses torch.device objects)
            if self.device == "mps":
                pipeline_device = torch.device("mps")
            elif self.device == "cuda":
                pipeline_device = 0  # CUDA device index
            else:
                pipeline_device = -1  # CPU
            
            pipeline_obj = pipeline(
                "text-generation",
                model=model,
                tokenizer=self._tokenizer,
                device=pipeline_device,
            )
        
        object.__setattr__(self, '_pipeline', pipeline_obj)
        
        device_display = "Apple Neural Engine (MPS)" if self.device == "mps" else self.device.upper()
        print(f"✓ Model loaded successfully on {device_display}")
        print("  (Model is cached - subsequent queries will be instant!)")

