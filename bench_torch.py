from third_party.qwen3.model.qwen3 import Qwen3Dense, Qwen3Config, Qwen3MoE
from third_party.qwen3.model.processor import Processor
from torchinfo import summary
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import torch.nn as nn
import os
import torch_npu


SCRIPT_DIR = Path(__file__).parent
BENCH_DIR = SCRIPT_DIR / "bench_results"
BENCH_DIR.mkdir(exist_ok=True)
DEVICE = torch.device(os.getenv("DEVICE", "npu"))
PREFIX = 'torch'

def get_bool_env(env_var, default=False):
    value = os.getenv(env_var, str(default))
    return value.lower() in ('true', '1', 'yes', 'y', 't')

COMPILE = get_bool_env("COMPILE", False)
RANDOM_WEIGHTS = get_bool_env("RANDOM_WEIGHTS", True)

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


def cc():
    experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=[
                            torch_npu.profiler.ExportType.Text,
                                    torch_npu.profiler.ExportType.Db
                                        ],
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                            record_op_args=True
                            )
                            
    return experimental_config


def create_profiler(EAGER_MODE_PROF=True):

    experimental_config = torch_npu.profiler._ExperimentalConfig(aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                                                                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1, 
                                                                    record_op_args=EAGER_MODE_PROF )
    experimental_config=cc()

    prof = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,#False,
            profile_memory=True,#False,
            with_stack=True,#False,
            schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=0, skip_first=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir"),
            experimental_config=experimental_config)

    return prof


if __name__ == "__main__":
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    random_input = get_rand_input(size=45)
    model = Qwen3MoE(qwen_configs["qwen3_06b"])
    # model.save_pretrained = Qwen3ForCausalLM.save_pretrained

    save_dir = Path(SCRIPT_DIR) / "weights" / PREFIX /"qwen3_06b" / 'model'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_random_weights(model, filename=f"{save_dir}.safetensors")
    # model.save_pretrained(model, save_directory=save_dir)

    print(random_input)
    summary(model, input_data = random_input)

    model = model.to(DEVICE)
    random_input = random_input.to(DEVICE)

    model = torch.compile(model) if COMPILE else None

    for _ in range(2):
        model(random_input)

    with create_profiler() as prof:
        for _ in range(2):
            model(random_input)
            prof.step()
    # print(prof)
