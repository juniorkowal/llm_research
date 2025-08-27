import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._inductor.config as cfg
# config.
# config.allow_fusion = False 

cfg.realize_reads_threshold = 0
cfg.realize_opcount_threshold = 0

cfg.pre_grad_fusion_options = {}
cfg.post_grad_fusion_options = {}
cfg._fuse_ddp_communication_passes={}
# cfg._ddp_optimization_mode=["no_optimization"]
# cfg.pattern_matcher=False
cfg.max_fusion_size=0
# cfg.split_cat_fx_passes=

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(100, 50)
        
    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x) # inductor fuses e.g. relu and sigmoid; can we disable that?
        x = torch.sigmoid(x)
        return x

model = SimpleModel().cuda()
x = torch.randn(32, 100).cuda()

compiled_model = torch.compile(model, mode="reduce-overhead")

with torch.no_grad():
    for _ in range(2):
        output = compiled_model(x)

print("Model compiled successfully!")