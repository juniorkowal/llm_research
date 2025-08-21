import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig#, TransformerLa
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from huggingface_hub import configure_http_backend
from torchinfo import summary
import os
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = 'true'

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kv_dim, i):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.num_heads_kv = num_heads / 4
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.q_proj = model_weights[f"model.layers.{i}.self_attn.q_proj.weight"] #nn.Linear(embed_dim, num_heads * head_dim)  # Projection for Q
        self.k_proj = model_weights[f"model.layers.{i}.self_attn.k_proj.weight"].T #nn.Linear(embed_dim, num_heads_kv * head_dim)  # Projection for K
        self.v_proj = model_weights[f"model.layers.{i}.self_attn.v_proj.weight"].T #nn.Linear(embed_dim, num_heads_kv * head_dim)  # Projection for V
        self.o_proj = model_weights[f"model.layers.{i}.self_attn.o_proj.weight"] #nn.Linear(embed_dim, num_heads * head_dim)  # Output projection

    def forward(self, x):
        bsz, seq_len, embed_dim = x.size()


        # Project Q, K, and V
        q = torch.matmul(x, self.q_proj)  # (bsz, seq_len, num_heads * head_dim)
        k = torch.matmul(x, self.k_proj)  # (bsz, seq_len, num_heads_kv * head_dim)
        v = torch.matmul(x, self.v_proj)  # (bsz, seq_len, num_heads_kv * head_dim)

        # Transpose to get (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ v

        # Combine heads and pass through output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, embed_dim)
        return torch.matmul(attn_output, self.o_proj)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, i):
        super(FeedForward, self).__init__()
        self.gate_proj = model_weights[f"model.layers.{i}.mlp.gate_proj.weight"].T #nn.Linear(embed_dim, ff_dim)
        self.up_proj = model_weights[f"model.layers.{i}.mlp.up_proj.weight"].T #nn.Linear(embed_dim, ff_dim)
        self.down_proj = model_weights[f"model.layers.{i}.mlp.down_proj.weight"].T #nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        # Gated Linear Unit (GLU)
        gate_x = F.gelu(x @ self.gate_proj)
        up_x = x @ self.up_proj
        x = gate_x * up_x
        x = x @ self.down_proj
        return x


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads, kv_dim, i):
        super(TransformerLayer, self).__init__()
        self.self_attn = SelfAttention(embed_dim, num_heads, kv_dim, i)
        self.input_layernorm = model_weights[f"model.layers.{i}.input_layernorm.weight"]  # nn.LayerNorm(embed_dim)
        self.post_attention_layernorm = model_weights[f"model.layers.{i}.post_attention_layernorm.weight"] # nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, ff_dim, i)
        self.eps = 1e-5

    def forward(self, x):

        # Self-attention block
        residual = x

        ## input layer normalization
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        x = self.input_layernorm * x

        x = self.self_attn(x) + residual

        # Feed-forward block
        residual = x

        ## input layer normalization
        mean = x.mean(dim=-1, keepdim=True)  # this means last dimension layer normalization
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        x = self.input_layernorm * x

        x = self.mlp(x) + residual
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, ff_dim, num_heads, kv_dim):
        super(TransformerModel, self).__init__()
        self.embed_tokens = model_weights['model.embed_tokens.weight'] # here
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, ff_dim, num_heads, kv_dim, i)
            for i in range(num_layers)
        ])
        self.norm = model_weights['model.norm.weight'] # nn.LayerNorm(embed_dim)
        self.lm_head = model_weights['lm_head.weight'].T # nn.Linear(embed_dim, vocab_size, bias=False)
        self.eps = 1e-5

    def forward(self, sentence):
        # Embed tokens

        tokenized_output = tokenizer(sentence, return_tensors="pt")
        input_ids = tokenized_output["input_ids"]

        x = (self.embed_tokens)[input_ids]

        # Pass through each layer
        for layer in self.layers:
            x = layer(x)

        # Normalize output
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(variance + self.eps)  # Layer normalization
        x = x * self.norm  # Scale

        # Project to vocabulary size
        logits = x @ self.lm_head
        return logits


# if __name__ == "__main__":
    
#     model_name = "meta-llama/Llama-3.2-1B"   # llama 3.1 8B is a scaled up ( in terms of layers) version of this

#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     input_text = """ Apples are red"""
#     tokenized_inputs = tokenizer(input_text, return_tensors="pt")
#     tokenized_inputs

#     model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

#     # model.config

#     model_weights = model.state_dict()
#     # for k, v in model_weights.items():
#     #     print(k, v.shape)

#     # Define model parameters based on layer structure
#     vocab_size = 128256
#     embed_dim = 2048
#     num_layers = 16
#     ff_dim = 8192
#     num_heads = 8  # Based on 2048 / 8 = 256 per head
#     kv_dim = 512  # Separate projection sizes for k and v

#     # Initialize model
#     model = TransformerModel(vocab_size, embed_dim, num_layers, ff_dim, num_heads, kv_dim)

#     # Example
#     sentence = "apples are "
#     logits = model.forward(sentence)
#     print("Logits shape:", logits.shape)  # Expected: (batch_size, seq_len, vocab_size)

#     token_ids = torch.argmax(logits, dim=-1)  # Shape: [batch_size, seq_len]
#     decoded_tokens = [tokenizer.decode(ids, skip_special_tokens=True) for ids in token_ids]

#     print(f"Token IDs: {token_ids}")
#     print(f"Decoded Tokens: {decoded_tokens}")



if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    model_name = "meta-llama/Llama-3.2-1B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_text = """ Apples are red"""
    tokenized_inputs = tokenizer(input_text, return_tensors="pt")
    print("Tokenized inputs:", tokenized_inputs)

    config = AutoConfig.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    
    # model = model.half()
    
    # if torch.npu.is_available():
    #     model = model.npu()
    #     device_map = "auto"
    # else:
    #     device_map = None

    print("Model config:", model.config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Get model weights (now they're random)
    model_weights = model.state_dict()
    # for k, v in model_weights.items():
    #     print(k, v.shape, v.dtype)

    sentence = "apples are "
    
    inputs = tokenizer(sentence, return_tensors="pt")
    # if torch.npu.is_available():
    #     inputs = {k: v.npu() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    print("Logits shape:", logits.shape) 

    token_ids = torch.argmax(logits, dim=-1)
    decoded_tokens = tokenizer.decode(token_ids[0], skip_special_tokens=True)

    print(f"Token IDs: {token_ids}")
    print(f"Decoded Tokens: {decoded_tokens}")