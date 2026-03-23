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

import pandas as pd
import os

# Create dummy test CSV if it's missing (for local testing)
test_csv_path = "/kaggle/input/competitions/llm-classification-finetuning/test.csv"
if not os.path.exists(test_csv_path):
    # Try local alternative
    test_csv_path = "test.csv"
    if not os.path.exists(test_csv_path):
        print("test.csv not found. Creating a dummy one for validation.")
        pd.DataFrame({
            "id": ["123", "456"],
            "response_a": ["Hello world", "Good morning"],
            "response_b": ["Hi there", "Hello sir"]
        }).to_csv("test.csv", index=False)
        test_csv_path = "test.csv"

# Read the testing sequence inputs
df = pd.read_csv(test_csv_path)

# Initialize target columns which the SequenceClassification processor requires
df["winner_model_a"] = 1
df["winner_model_b"] = 0
df["winner_tie"]     = 0
df.to_parquet("test.parquet", index=False)

# Transpose responses to create an inverse view for the second model
df["response_a"], df["response_b"] = df["response_b"], df["response_a"]
df.to_parquet("test_swap.parquet", index=False)
print("Dataset representations generated successfully.")
