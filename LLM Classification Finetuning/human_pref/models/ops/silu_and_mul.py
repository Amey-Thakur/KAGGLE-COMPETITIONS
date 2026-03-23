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
def silu_and_mul_fwd(x):
    d = x.shape[-1]//2
    return torch.nn.functional.silu(x[...,:d]) * x[...,d:]
