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
import torch.nn.functional as F
def context_attention_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal):
    out = torch.zeros_like(q)
    for i in range(len(cu_seqlens_q)-1):
        sq, eq = cu_seqlens_q[i], cu_seqlens_q[i+1]
        sk, ek = cu_seqlens_k[i], cu_seqlens_k[i+1]
        qi, ki, vi = q[sq:eq].unsqueeze(0).transpose(1,2), k[sk:ek].unsqueeze(0).transpose(1,2), v[sk:ek].unsqueeze(0).transpose(1,2)
        res = F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal, scale=softmax_scale)
        out[sq:eq] = res.transpose(1,2).squeeze(0)
    return out
