# Megakernel Feasibility Study for LLMs on NPU

A research project investigating the feasibility of implementing megakernels for Large Language Models using Triton on Torch-NPU.

## Overview

This project tests whether megakernels (single large kernels for entire LLM forward passes) are viable on NPU hardware using Triton, contrary to conventional wisdom that Triton is too high-level for such implementations.

## Prerequisites

- **MindStudio Insight**: Required for profiling. Download from [Mindstudio Insight](https://www.hiascend.com/developer/download/community/result?module=sto%2Bcann)

## Quick Start

```bash
# Initialize submodules
git submodule update --init --recursive

# Basic test run
python bench_torch.py

# Run with torch.compile enabled
COMPILE=True python bench_torch.py

# Debug compilation with Triton - creates torch_compile_debug folder with triton code
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_FX_GRAPH_CACHE=0 COMPILE=True python bench_torch.py
```

## Configuration

Environment variables control execution:
- `DEVICE`: Device to use (default: "npu")
- `COMPILE`: Enable torch.compile (default: False) use it for triton compilation
- `RANDOM_WEIGHTS`: Use random weights instead of pretrained (default: True) after creation, random weights will be reused on subsequent runs; perfect for environments where you cannot download from huggingface

## Models

Testing uses Qwen3 family models (0.6B, 4B, 8B, 14B, 32B) from [tiny-qwen](https://github.com/Emericen/tiny-qwen) with modified attention operations and interleave for NPU torch.compile compatibility.

## Profiling

Use MindStudio Insight for performance analysis:

```bash
COMPILE=True DEVICE=npu TRITON_ALWAYS_COMPILE=1 \
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_FX_GRAPH_CACHE=0 \
msprof --ascendcl=on --runtime-api=on --task-time=on \
--aicpu=on --ai-core=on --aic-metrics=ResourceConflictRatio \
--output=msprof_out/ python bench_torch.py
```

Visualize results using MindStudio Insight after profiling.

## Fusion Control

Adjust fusion behavior with:
```python
import torch._inductor.config as cfg
cfg.max_fusion_size = 1  # Controls maximum fusion size
```
Tested in inductor_fusion_test.py

## TLDR
1. Run bench_torch.py

## Resources
https://dev-discuss.pytorch.org/t/how-to-turn-off-inlining-force-materialization-in-torchinductor-during-torch-compile/3198/3
https://github.com/pytorch/pytorch/blob/bf8431ba062efa9ff0cdd5032a3ddf2e007a3216/torch/_inductor/config.py#L584-L585
https://docs.pytorch.org/assets/pytorch2-2.pdf
https://gitee.com/ascend/triton-ascend/blob/master/docs/sources/mindstudio-guide/01-msProf_op.md
https://www.hiascend.com/developer/download/community/result?module=sto%2Bcann
https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
https://github.com/Emericen/tiny-qwen
