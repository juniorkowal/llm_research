import torch
import torch.nn as nn
import time
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
import argparse

try:
    from third_party.qwen3.model.qwen3 import Qwen3Dense, Qwen3Config, Qwen3MoE
    from third_party.qwen3.model.processor import Processor
except ImportError:
    print("Warning: Qwen3 pure torch implementation not available")

try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
except ImportError:
    print("Warning: Transformers not available")

@dataclass
class ModelConfig:
    name: str
    hf_id: str
    qwen_config: Optional[Dict] = None
    implementation: Literal["torch", "transformers"] = "transformers"

SCRIPT_DIR = Path(__file__).parent
BENCH_DIR = SCRIPT_DIR / "bench_results"
BENCH_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR = SCRIPT_DIR / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

QWEN_CONFIGS = {
    "qwen3_06b": {
        "n_embed": 1024,
        "n_heads": 16,
        "n_kv_heads": 8,
        "n_layer": 28,
        "n_mlp": 3072,
        "rope_theta": 1000000,
        "rms_norm_eps": 1e-06,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
        "head_dim": 128
    },
    "qwen3_4b": {
        "n_embed": 2560,
        "n_heads": 32,
        "n_kv_heads": 8,
        "n_layer": 36,
        "n_mlp": 9728,
        "rope_theta": 1000000,
        "rms_norm_eps": 1e-06,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
        "head_dim": 128
    }
}

MODELS = [    
    ModelConfig("Qwen3-0.6B-Base", "Qwen/Qwen3-0.6B-Base", QWEN_CONFIGS["qwen3_06b"], "transformers"),
    ModelConfig("Qwen3-0.6B-Torch", "Qwen/Qwen3-0.6B-Base", QWEN_CONFIGS["qwen3_06b"], "torch"),
    ModelConfig("Qwen3-4B-Base", "Qwen/Qwen3-4B-Base", QWEN_CONFIGS["qwen3_4b"], "transformers"),
    ModelConfig("Qwen3-4B-Torch", "Qwen/Qwen3-4B-Base", QWEN_CONFIGS["qwen3_4b"], "torch"),
]

def get_device():
    device_str = os.getenv("DEVICE", "cpu")
    return torch.device(device_str)

def get_bool_env(env_var, default=False):
    value = os.getenv(env_var, str(default))
    return value.lower() in ('true', '1', 'yes', 'y', 't')

def get_rand_input(size: int = 32):
    random_input = torch.randint(low=1000, high=4000, size=[1, size], dtype=torch.int64)
    random_input[0, 0] = 101
    random_input[0, size-1] = 102
    return random_input

def save_random_weights(model: nn.Module, filename: str | Path):
    """Save model weights using SafeTensors"""
    weights = model.state_dict()
    save_file(weights, filename)
    print(f"Weights saved to {filename}")

def load_random_weights(model: nn.Module, filename: str | Path, strict: bool = True):
    """Load model weights from SafeTensors file"""
    state_dict = {}
    with safe_open(filename, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    print(f"Model weights loaded from {filename}")
    model.load_state_dict(state_dict, strict=strict)
    return model

def compile_model(model: nn.Module):
    """Compile model using torch.compile"""
    return torch.compile(model)

class LLMBenchmark:
    def __init__(self, use_random_weights: bool = False, use_torch_compile: bool = False):
        self.use_random_weights = use_random_weights
        self.use_torch_compile = use_torch_compile
        self.device = get_device()
        self.tokenizer = None#self._load_tokenizer()

    def _load_tokenizer(self):
        """Load a generic tokenizer for benchmarking"""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except ImportError:
            print("Warning: Transformers not available, using dummy tokenizer")
            return None

    def _create_torch_model(self, model_config: ModelConfig):
        """Create pure PyTorch implementation of the model"""
        if not hasattr(model_config, 'qwen_config') or not model_config.qwen_config:
            raise ValueError("Qwen config required for pure torch implementation")
        
        try:
            config = Qwen3Config(**model_config.qwen_config)
            model = Qwen3MoE(config)
            return model
        except NameError:
            raise ImportError("Pure torch Qwen implementation not available")

    def _create_transformers_model(self, model_config: ModelConfig):
        """Create transformers implementation of the model"""
        try:
            config = AutoConfig.from_pretrained(model_config.hf_id)
            model = AutoModelForCausalLM.from_config(config)
            return model
        except NameError:
            raise ImportError("Transformers implementation not available")

    def load_model(self, model_config: ModelConfig):
        """Load model based on implementation type"""
        model_path = WEIGHTS_DIR / f"{model_config.name.lower()}_{model_config.implementation}" / "model"
        model_path.mkdir(parents=True, exist_ok=True)
        
        if self.use_random_weights:
            return self._load_random_weights_model(model_config, model_path)
        else:
            return self._load_pretrained_model(model_config)

    def _load_random_weights_model(self, model_config: ModelConfig, model_path: Path):
        """Load or create model with random weights"""
        safetensors_file = model_path / "model.safetensors"
        
        if safetensors_file.exists():
            print(f"Loading existing random weights for {model_config.name} ({model_config.implementation})")
            
            if model_config.implementation == "torch":
                model = self._create_torch_model(model_config)
            else:
                model = self._create_transformers_model(model_config)
            
            load_random_weights(model, safetensors_file)
        else:
            print(f"Creating new random weights for {model_config.name} ({model_config.implementation})")
            
            if model_config.implementation == "torch":
                model = self._create_torch_model(model_config)
            else:
                model = self._create_transformers_model(model_config)
            
            save_random_weights(model, safetensors_file)
        
        model = model.to(self.device)
        if self.use_torch_compile:
            model = compile_model(model)
        
        return model

    def _load_pretrained_model(self, model_config: ModelConfig):
        """Load pretrained model - only works with transformers"""
        if model_config.implementation != "transformers":
            raise ValueError("Pretrained weights only available for transformers implementation")
        
        print(f"Loading {model_config.name} with pretrained weights...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_config.hf_id,
                device_map=self.device
            )
        except NameError:
            raise ImportError("Transformers not available for pretrained loading")
        
        if self.use_torch_compile:
            model = compile_model(model)
        
        return model

    def generate_test_inputs(self, num_samples: int = 5):
        """Generate test inputs for benchmarking"""
        test_texts = [
            "Apples are red",
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Python is a high-level programming language",
            "The capital of France is Paris"
        ][:num_samples]
        
        if self.tokenizer:
            return [self.tokenizer(text, return_tensors="pt") for text in test_texts]
        else:
            return [{"input_ids": get_rand_input()} for _ in range(num_samples)]

    def benchmark_inference(self, model: nn.Module, inputs: List[Dict], num_runs: int = 5):
        """Benchmark inference performance"""
        results = []
        
        for i, input_batch in enumerate(inputs):
            input_batch = {k: v.to(self.device) for k, v in input_batch.items()}
            
            # Warmup
            with torch.no_grad():
                _ = model(**input_batch)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_runs):
                with torch.no_grad():
                    outputs = model(**input_batch)
                    # Check for NaNs in output
                    if hasattr(outputs, 'logits'):
                        assert not torch.isnan(outputs.logits).any(), "Output tensor contains NaNs"
                    elif hasattr(outputs, 'last_hidden_state'):
                        assert not torch.isnan(outputs.last_hidden_state).any(), "Output tensor contains NaNs"
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs
            
            # Get output shape
            if hasattr(outputs, 'logits'):
                output_shape = outputs.logits.shape
            elif hasattr(outputs, 'last_hidden_state'):
                output_shape = outputs.last_hidden_state.shape
            else:
                output_shape = "unknown"
            
            results.append({
                "input_idx": i,
                "avg_time_ms": avg_time * 1000,
                "output_shape": output_shape
            })
        
        return results

    def measure_memory(self, model: nn.Module):
        """Measure memory usage"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            dummy_input = self.generate_test_inputs(1)[0]
            dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}
            
            with torch.no_grad():
                _ = model(**dummy_input)
            
            return {
                "max_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "current_memory_gb": torch.cuda.memory_allocated() / 1024**3
            }
        else:
            return {"max_memory_gb": 0, "current_memory_gb": 0}

    def run_benchmark(self, model_config: ModelConfig):
        """Run complete benchmark for a model"""
        print(f"\n{'='*50}")
        print(f"Testing {model_config.name} ({model_config.implementation} implementation)")
        print(f"{'='*50}")
        
        try:
            model = self.load_model(model_config)
            
            # Get model statistics
            total_params = sum(p.numel() for p in model.parameters())
            memory_stats = self.measure_memory(model)
            test_inputs = self.generate_test_inputs()
            inference_results = self.benchmark_inference(model, test_inputs)
            
            result = {
                "model_name": model_config.name,
                "implementation": model_config.implementation,
                "total_params": total_params,
                "memory_gb": memory_stats["max_memory_gb"],
                "inference_times": inference_results,
                "weight_type": "RANDOM" if self.use_random_weights else "PRETRAINED",
                "compiled": self.use_torch_compile,
                "device": str(self.device)
            }
            
            del model
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error with {model_config.name}: {e}")
            result = {
                "model_name": model_config.name,
                "implementation": model_config.implementation,
                "error": str(e),
                "weight_type": "RANDOM" if self.use_random_weights else "PRETRAINED"
            }
        
        return result

    def print_results(self, result: Dict):
        """Print formatted benchmark results"""
        print(f"\n{'='*60}")
        print("MODEL BENCHMARK RESULTS")
        print(f"{'='*60}")
        
        if "error" in result:
            print(f"{result['model_name']}: ERROR - {result['error']}")
            return
        
        print(f"Model: {result['model_name']}")
        print(f"Implementation: {result['implementation']}")
        print(f"Parameters: {result['total_params']:,}")
        print(f"Memory: {result['memory_gb']:.2f} GB")
        print(f"Weight type: {result['weight_type']}")
        print(f"Compiled: {result['compiled']}")
        print(f"Device: {result['device']}")
        
        print("\nInference times (ms):")
        for time_result in result['inference_times']:
            print(f"  Input {time_result['input_idx']}: {time_result['avg_time_ms']:.2f}ms")

def main():
    parser = argparse.ArgumentParser(description="LLM Benchmarking Tool")
    parser.add_argument("--random-weights", action="store_true", help="Use random weights instead of pretrained")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu, cuda, etc.)")
    parser.add_argument("--implementation", choices=["torch", "transformers", "both"], default="both", 
                       help="Which implementation to use (torch, transformers, or both)")
    parser.add_argument("--models", nargs="+", help="Specific models to benchmark")
    
    args = parser.parse_args()
    
    # Set environment variables from args
    os.environ["DEVICE"] = args.device
    os.environ["RANDOM_WEIGHTS"] = str(args.random_weights)
    os.environ["COMPILE"] = str(args.compile)
    
    # Filter models based on implementation
    models_to_test = MODELS
    if args.implementation != "both":
        models_to_test = [m for m in MODELS if m.implementation == args.implementation]
    
    # Filter models if specified
    if args.models:
        models_to_test = [m for m in models_to_test if m.name in args.models or m.hf_id in args.models]
    
    print(f"\n{'#'*70}")
    print(f"CONFIG: Random={args.random_weights}, Compile={args.compile}, Device={args.device}")
    print(f"Implementation: {args.implementation}")
    print(f"Models: {[m.name for m in models_to_test]}")
    print(f"{'#'*70}")
    
    benchmark = LLMBenchmark(
        use_random_weights=args.random_weights,
        use_torch_compile=args.compile
    )
    
    all_results = []
    for model_config in models_to_test:
        result = benchmark.run_benchmark(model_config)
        benchmark.print_results(result)
        all_results.append(result)
        
        # Save individual result
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{model_config.name}_{model_config.implementation}_benchmark_{timestamp}.json"
        with open(BENCH_DIR / filename, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {filename}")
    
    # Save combined results
    combined_filename = f"combined_benchmark_{timestamp}.json"
    with open(BENCH_DIR / combined_filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Combined results saved to {combined_filename}")

if __name__ == "__main__":
    main()