import torch
import torch.nn as nn
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Dict, List, Optional
from dataclasses import dataclass
import os
from pathlib import Path
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


@dataclass
class ModelConfig:
    name: str
    hf_id: str


SCRIPT_DIR = Path(__file__).parent
BENCH_DIR = SCRIPT_DIR / "bench_results"
BENCH_DIR.mkdir(exist_ok=True)
MODELS = [    
    # LLAMA
    # ModelConfig("Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct"),
    # ModelConfig("Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"),
    # ModelConfig("Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"),

    # QWEN
    ModelConfig("Qwen3-0.6B-Base", "Qwen/Qwen3-0.6B-Base"),
    # ModelConfig("Qwen3-4B-Base", "Qwen/Qwen3-4B-Base"),
    # ModelConfig("Qwen3-8B", "Qwen/Qwen3-8B"),
    # ModelConfig("Qwen3-14B", "Qwen/Qwen3-14B"),
    # ModelConfig("Qwen3-32B", "Qwen/Qwen3-32B"),
]

DEVICE = torch.device(os.getenv("DEVICE", "cpu"))

def get_bool_env(env_var, default=False):
    value = os.getenv(env_var, str(default))
    return value.lower() in ('true', '1', 'yes', 'y', 't')

COMPILE = get_bool_env("COMPILE", False)
RANDOM_WEIGHTS = get_bool_env("RANDOM_WEIGHTS", True)


class SimpleLLMBenchmark:
    def __init__(self, use_random_weights: bool = False, use_torch_compile: bool = False, model_config: ModelConfig = MODELS[0]):
        self.use_random_weights = use_random_weights
        self.use_torch_compile = use_torch_compile
        self.tokenizer = self.load_tokenizer(model_config)
        self.save_dir = Path(SCRIPT_DIR) / "weights" / model_config.name.lower()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def load_tokenizer(self, model_config: ModelConfig) -> nn.Module:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_model(self, model_config: ModelConfig) -> nn.Module:
        model_path = self.save_dir / "model"
        if self.use_random_weights:
            model_files = list(model_path.glob("*.bin")) + list(model_path.glob("*.safetensors"))
            has_weights = len(model_files) > 0
            if has_weights:
                print(f"Loading existing random weights for {model_config.name} from {model_path}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    device_map=DEVICE
                )
            else:
                print(f"Creating new random weights for {model_config.name} and saving to {model_path}")
                config = AutoConfig.from_pretrained(model_config.hf_id)
                model = AutoModelForCausalLM.from_config(config)
                model: Qwen3ForCausalLM
                model.save_pretrained(model_path)
                model = model.to(DEVICE)
        else:
            print(f"Loading {model_config.name} with PRETRAINED weights...")
            model = AutoModelForCausalLM.from_pretrained(
                model_config.hf_id,
                # torch_dtype=torch.bfloat16,
                device_map=DEVICE
            )
        
        if self.use_torch_compile:
            print("Compiling model with torch.compile...")
            model = torch.compile(model, backend='inductor')

        return model.to(DEVICE)
    
    def generate_test_inputs(self):
        test_texts = [
            "Apples are red",
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Python is a high-level programming language",
            "The capital of France is Paris"
        ]
        return [self.tokenizer(text, return_tensors="pt") for text in test_texts]
    
    def benchmark_inference(self, model: nn.Module, inputs: List[Dict], num_runs: int = 5):
        results = []
        
        for i, input_batch in enumerate(inputs):
            input_batch = {k: v.to(DEVICE) for k, v in input_batch.items()}
            
            with torch.no_grad(): # warmup
                _ = model(**input_batch)

            # torch.npu.synchronize()
            start_time = time.time()
            
            for _ in range(num_runs):
                with torch.no_grad():
                    outputs = model(**input_batch)
                    assert not torch.isnan(outputs.logits).any(), f"Output tensor contains nans."

            # torch.npu.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs
            results.append({
                "input_idx": i,
                "avg_time_ms": avg_time * 1000,
                "output_shape": outputs.logits.shape
            })
        
        return results
    
    def measure_memory(self, model: torch.nn.Module):
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
        
        dummy_input = self.tokenizer("test", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _ = model(**dummy_input)
        
        return {
            "max_memory_gb": torch.npu.max_memory_allocated() / 1024**3,
            "current_memory_gb": torch.npu.memory_allocated() / 1024**3
        }
    
    def run_model_analysis(self, model: torch.nn.Module, model_config: ModelConfig):
        try:
            total_params = sum(p.numel() for p in model.parameters())
            memory_stats = self.measure_memory(model)
            test_inputs = self.generate_test_inputs()
            inference_results = self.benchmark_inference(model, test_inputs)
            
            print(model)
            
            return {
                "model_name": model_config.name,
                "total_params": total_params,
                "memory_gb": memory_stats["max_memory_gb"],
                "inference_times": inference_results,
                # "summary": model_summary,
                "weight_type": "RANDOM" if self.use_random_weights else "PRETRAINED",
                "compiled": self.use_torch_compile
            }
            
        except Exception as e:
            return {
                "model_name": model_config.name,
                "error": str(e),
                "weight_type": "RANDOM" if self.use_random_weights else "PRETRAINED"
            }
    
    def run_model(self, model_config: ModelConfig):
        print(f"\n{'='*50}")
        print(f"Testing {model_config.name}")
        print(f"{'='*50}")
        try:
            model = self.load_model(model_config)
            model_result = self.run_model_analysis(model, model_config)
            del model
            torch.npu.empty_cache()
            
        except Exception as e:
            print(f"Error with {model_config.name}: {e}")
            model_result = {
                "model_name": model_config.name,
                "error": str(e)
            }
        return model_result
    
    def print_results(self, result: Dict):
        """Print formatted results"""
        print(f"\n{'='*60}")
        print("MODEL BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Weight type: {'RANDOM' if self.use_random_weights else 'PRETRAINED'}")
        print(f"Torch compile: {'ENABLED' if self.use_torch_compile else 'DISABLED'}")
        print(f"{'='*60}")
        
        if "error" in result:
            print(f"{result['model_name']}: ERROR - {result['error']}")
            return        
        print(f"\n{result['model_name']}:")
        print(f"  Parameters: {result['total_params']:,}")
        print(f"  Memory: {result['memory_gb']:.2f} GB")
        print(f"  Weight type: {result['weight_type']}")
        print(f"  Compiled: {result['compiled']}")
        
        print("  Inference times (ms):")
        for time_result in result['inference_times']:
            print(f"    Input {time_result['input_idx']}: {time_result['avg_time_ms']:.2f}ms")



if __name__ == "__main__":
    for model in MODELS:
        print(f"\n{'#'*70}")
        print(f"CONFIG: Random={RANDOM_WEIGHTS}, Compile={COMPILE}, Device={DEVICE}")
        print(f"{'#'*70}")
        
        benchmark = SimpleLLMBenchmark(
            use_random_weights=RANDOM_WEIGHTS,
            use_torch_compile=COMPILE,
            model_config=model
        )
        
        result = benchmark.run_model(model)
        benchmark.print_results(result)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{model.name}_benchmark_{timestamp}.json"
        import json
        with open(BENCH_DIR / filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Results saved to {filename}")
