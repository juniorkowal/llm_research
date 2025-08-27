from third_party.qwen3.model.qwen3 import Qwen3Dense, Qwen3Config, Qwen3MoE
from third_party.qwen3.model.processor import Processor
from torchinfo import summary
import torch


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

def get_rand_input():
    ...

if __name__ == "__main__":

    model = Qwen3MoE(qwen_configs["qwen3_06b"])
    random_input = torch.randint(low=1030, high=4000, size=[1,45], dtype=torch.int64)
    random_input[0,0] = 101
    random_input[0,44] = 102
    print(random_input)
    summary(model, input_data = random_input)

    model = model.to('npu')
    random_input = random_input.to('npu')
    # model = torch.compile(model)
    print(model(random_input))

