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
def fused_rotary_emb(q, k, cos, sin, pos_ids=None, out_q=None, out_k=None):
    def rot(x): d = x.shape[-1]; return torch.cat([-x[..., d//2:], x[..., :d//2]], dim=-1)
    q_o, k_o = q*cos + rot(q)*sin, k*cos + rot(k)*sin
    if out_q is not None: out_q.copy_(q_o)
    if out_k is not None: out_k.copy_(k_o)
    return q_o, k_o
