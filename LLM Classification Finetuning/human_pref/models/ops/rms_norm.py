import sys
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['XFORMERS_MORE_DETAILS'] = '0'

# Mock objects to satisfy hardware-specific libraries
class Mock:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return self
    def __getattr__(self, name): return self
    @staticmethod
    def from_seqlens(*args, **kwargs): return Mock()

mock_obj = Mock()
for m in ['triton', 'xformers', 'xformers.ops', 'xformers.ops.fmha', 'xformers.ops.fmha.attn_bias']:
    sys.modules[m] = mock_obj

import torch
def rms_norm(x, weight, eps=1e-6):
    x_f = x.float()
    v = x_f.pow(2).mean(-1, keepdim=True)
    return (x_f * torch.rsqrt(v+eps) * weight.float()).to(x.dtype)
