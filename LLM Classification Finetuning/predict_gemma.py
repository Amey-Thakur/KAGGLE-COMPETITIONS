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
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# The following libraries are hardware-specific and often missing on local Windows machines.
# assurance_fix.py mocks them at runtime, and we use type: ignore to silence IDE warnings.
try:
    from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask # type: ignore
except ImportError:
    BlockDiagonalCausalMask = None # type: ignore

try:
    from human_pref.models.modeling_gemma2 import Gemma2ForSequenceClassification # type: ignore
    from human_pref.data.processors import ProcessorPAB # type: ignore
    from human_pref.data.dataset import LMSYSDataset # type: ignore
    from human_pref.data.collators import VarlenCollator, ShardedMaxTokensCollator # type: ignore
    from human_pref.utils import to_device # type: ignore
except ImportError as e:
    print(f"⚠️ Warning: human_pref module not found. Runtime will fail if models are not present locally.")
    # Initialize as None to allow script compilation without 17+ errors
    Gemma2ForSequenceClassification = None
    ProcessorPAB = None
    LMSYSDataset = None
    VarlenCollator = None
    ShardedMaxTokensCollator = None
    to_device = None

import os

possible_checkpoints = [
    "/kaggle/input/datasets/tascj0/lmsys-checkpoints-0-0805",
    "/kaggle/input/lmsys-checkpoints-0-0805",
    "checkpoints/lmsys-checkpoints-0-0805"
]
model_name_or_path = next((p for p in possible_checkpoints if os.path.exists(p)), possible_checkpoints[0])
csv_path = "test.parquet"

if not os.path.exists(csv_path):
    print("test.parquet not found. Please run prepare_datasets.py.")
    exit(1)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
processor = ProcessorPAB(
    tokenizer=tokenizer,
    max_length=4096,
    support_system_role=False,
)
dataset = LMSYSDataset(
    csv_file=csv_path,
    query=None,
    processor=processor,
    include_swap=False,
    is_parquet=True,
)
dataloader = DataLoader(
    dataset,
    batch_size=80,
    num_workers=4,
    collate_fn=ShardedMaxTokensCollator(
        max_tokens=8192, base_collator=VarlenCollator()
    ),
)

# Allocate hidden layers evenly across hardware to prevent OOM
num_hidden_layers = 42
device_map = {
    "model.embed_tokens": "cuda:0",
    "model.norm": "cuda:1",
    "score": "cuda:1",
}
for i in range(num_hidden_layers // 2):
    device_map[f"model.layers.{i}"] = "cuda:0"
for i in range(num_hidden_layers // 2, num_hidden_layers):
    device_map[f"model.layers.{i}"] = "cuda:1"

model = Gemma2ForSequenceClassification.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map=device_map,
)

# Generate positional parameters per device
config = model.config
dim = config.head_dim
inv_freq = 1.0 / (
    getattr(config, 'rope_theta', 10000.0) ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
)
inv_freq0 = inv_freq.to("cuda:0")
inv_freq1 = inv_freq.to("cuda:1")

is_first = True
hidden_states = None
outs = []

# Execute dual device synchronization loops
for batch in tqdm(dataloader, desc="Gemma Inference"):
    for micro_batch in batch:
        input_ids = to_device(micro_batch["input_ids"], "cuda:0")
        seq_info = {
            "cu_seqlens": micro_batch["cu_seqlens"],
            "position_ids": micro_batch["position_ids"],
            "max_seq_len": micro_batch["max_seq_len"],
            "attn_bias": BlockDiagonalCausalMask.from_seqlens(micro_batch["seq_lens"]),
        }
        seq_info = to_device(seq_info, "cuda:0")

        if is_first:
            with torch.no_grad(), torch.cuda.amp.autocast():
                prev_hidden_states = model.forward_part1(input_ids, seq_info, inv_freq0)
            is_first = False
            prev_seq_info, prev_hidden_states = to_device(
                [seq_info, prev_hidden_states], "cuda:1"
            )
            continue

        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = model.forward_part2(prev_hidden_states, prev_seq_info, inv_freq1)
            hidden_states = model.forward_part1(input_ids, seq_info, inv_freq0)

            prev_seq_info, prev_hidden_states = to_device(
                [seq_info, hidden_states], "cuda:1"
            )
            outs.append(logits.cpu())

# Resolve terminal micro batch computation
with torch.no_grad(), torch.cuda.amp.autocast():
    logits = model.forward_part2(prev_hidden_states, prev_seq_info, inv_freq1)
    outs.append(logits.cpu())

# Consolidate spatial predictions
pred = torch.cat(outs, dim=0)
prob = pred.softmax(-1)

np.save("prob_gemma.npy", prob.numpy())
print("Gemma pipeline successfully finalized.")
