realize_reads_threshold = 0
realize_opcount_threshold = 0


TORCH_COMPILE_DEBUG=1
TORCHINDUCTOR_FX_GRAPH_CACHE=0
TORCHINDUCTOR_PROLOGUE_FUSION
TORCHINDUCTOR_GRAPH_PARTITION
TORCHINDUCTOR_MIN_NUM_SPLIT=1
TORCHINDUCTOR_PERMUTE_FUSION=1

https://dev-discuss.pytorch.org/t/how-to-turn-off-inlining-force-materialization-in-torchinductor-during-torch-compile/3198/3
https://github.com/pytorch/pytorch/blob/bf8431ba062efa9ff0cdd5032a3ddf2e007a3216/torch/_inductor/config.py#L584-L585
https://docs.pytorch.org/assets/pytorch2-2.pdf
