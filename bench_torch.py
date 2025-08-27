from third_party.qwen3.model.qwen3 import Qwen3Dense, Qwen3Config, Qwen3MoE
from third_party.qwen3.model.processor import Processor
from torchinfo import summary
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import torch.nn as nn
import os


SCRIPT_DIR = Path(__file__).parent
BENCH_DIR = SCRIPT_DIR / "bench_results"
BENCH_DIR.mkdir(exist_ok=True)
DEVICE = torch.device(os.getenv("DEVICE", "cpu"))
PREFIX = 'torch'

def get_bool_env(env_var, default=False):
    value = os.getenv(env_var, str(default))
    return value.lower() in ('true', '1', 'yes', 'y', 't')

COMPILE = get_bool_env("COMPILE", False)
RANDOM_WEIGHTS = os.getenv("RANDOM_WEIGHTS", True)

qwen_configs = {
    "qwen3_06b" : Qwen3Config(
        n_embed=1024,
        n_heads=16,
        n_kv_heads=8,
        n_layer=28,
        n_mlp=3072,
        rope_theta=1000000,
        rms_norm_eps=1e-06,
        vocab_size=151936,
        tie_word_embeddings=True,
        head_dim=128#,64
    ),
    "qwen3_4b" : Qwen3Config(
        n_embed=2560,
        n_heads=32,
        n_kv_heads=8,
        n_layer=36,
        n_mlp=9728,
        rope_theta=1000000,
        rms_norm_eps=1e-06,
        vocab_size=151936,
        tie_word_embeddings=True,
        head_dim=128
    ),
    "qwen3_8b" : Qwen3Config(
        n_embed=4096,
        n_heads=32,
        n_kv_heads=8,
        n_layer=36,
        n_mlp=12288,
        rope_theta=1000000,
        rms_norm_eps=1e-06,
        vocab_size=151936,
        tie_word_embeddings=False,
        head_dim=128
    ),
    "qwen3_14b" : Qwen3Config(
        n_embed=5120,
        n_heads=40,
        n_kv_heads=8,
        n_layer=40,
        n_mlp=17408,
        rope_theta=1000000,
        rms_norm_eps=1e-06,
        vocab_size=151936,
        tie_word_embeddings=False,
        head_dim=128
    ),
    "qwen3_32b" : Qwen3Config(
        n_embed=5120,
        n_heads=64,
        n_kv_heads=8,
        n_layer=64,
        n_mlp=25600,
        rope_theta=1000000,
        rms_norm_eps=1e-06,
        vocab_size=151936,
        tie_word_embeddings=False,
        head_dim=80
    )
}


def get_rand_input(size: int = 32):
    random_input = torch.randint(low=1000, high=4000, size=[1,size], dtype=torch.int64)
    random_input[0,0] = 101
    random_input[0,size-1] = 102
    return random_input


def save_random_weights(model: nn.Module, filename: str | Path):
    weights = model.state_dict()
    save_file(weights, filename)

def load_random_weights(model, filename="model_weights.safetensors", strict=True):
    state_dict = {}
    with safe_open(filename, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    print(f"Model weights loaded from {filename}")
    model.load_state_dict(state_dict, strict=strict)
    return model


def compile_model(model: nn.Module):
    return torch.compile(model)


if __name__ == "__main__":
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    random_input = get_rand_input(size=45)
    model = Qwen3MoE(qwen_configs["qwen3_06b"])
    # model.save_pretrained = Qwen3ForCausalLM.save_pretrained

    save_dir = Path(SCRIPT_DIR) / "weights" / PREFIX /"qwen3_06b" / 'model'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_random_weights(model, filename=save_dir)
    # model.save_pretrained(model, save_directory=save_dir)

    print(random_input)
    summary(model, input_data = random_input)

    model = model#.to('npu')
    random_input = random_input#.to('npu')
    # model = torch.compile(model)
    print(model(random_input))

