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

import numpy as np
import pandas as pd
import os

# Verify artifacts exist prior to merging logic
if not os.path.exists("prob_gemma.npy") or not os.path.exists("prob_llama.npy"):
    print("Inference probability tensors are missing. Creating dummy data for demonstration.")
    # Assuming test.parquet has 2 rows from prepare_datasets.py
    prob_gemma = np.array([[0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    prob_llama = np.array([[0.2, 0.7, 0.1], [0.4, 0.2, 0.4]])
else:
    prob_gemma = np.load("prob_gemma.npy")
    prob_llama = np.load("prob_llama.npy")

if os.path.exists("test.parquet"):
    df = pd.read_parquet("test.parquet")
else:
    df = pd.DataFrame({"id": ["123", "456"]})

# Geometrically map output probabilities back corresponding to the parameter swap
# Index translation: 0 (A) -> 1 (B), 1 (B) -> 0 (A), 2 (Tie) -> 2 (Tie)
prob_llama_mapped = prob_llama[:, [1, 0, 2]]

# Execute ensemble
blended_preds = np.average(
    [prob_gemma, prob_llama_mapped],
    axis=0,
    weights=[0.57, 0.43]
)

# Structure final delivery schema
submission = pd.DataFrame({
    "id": df["id"],
    "winner_model_a": blended_preds[:, 0],
    "winner_model_b": blended_preds[:, 1],
    "winner_tie": blended_preds[:, 2],
})

submission.to_csv("submission.csv", index=False)
print("Ensemble successful. Submission initialized.")
print(submission.head())
