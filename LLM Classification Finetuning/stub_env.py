import os

def create_stubs():
    print("Generating environment stubs to silence IDE warnings...")
    
    # Base directory
    base = "human_pref"
    
    # Subdirectories
    dirs = [
        f"{base}/models",
        f"{base}/data",
        f"{base}/data/processors",
        f"{base}/data/collators",
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/__init__.py", "w") as f: f.write("")

    # human_pref/utils.py
    with open(f"{base}/utils.py", "w") as f:
        f.write('def to_device(obj, device):\n    """Stub for moving objects to device."""\n    return obj\n')

    # human_pref/models/modeling_gemma2.py
    with open(f"{base}/models/modeling_gemma2.py", "w") as f:
        f.write("""
class Gemma2ForSequenceClassification:
    @classmethod
    def from_pretrained(cls, *args, **kwargs): return cls()
    def forward_part1(self, *args, **kwargs): return None
    def forward_part2(self, *args, **kwargs): return None
""")

    # human_pref/models/modeling_llama.py
    with open(f"{base}/models/modeling_llama.py", "w") as f:
        f.write("""
class LlamaForSequenceClassification:
    @classmethod
    def from_pretrained(cls, *args, **kwargs): return cls()
    def forward_part1(self, *args, **kwargs): return None
    def forward_part2(self, *args, **kwargs): return None
""")

    # human_pref/data/processors.py
    with open(f"{base}/data/processors.py", "w") as f:
        f.write("class ProcessorPAB:\n    def __init__(self, *args, **kwargs): pass\n")

    # human_pref/data/dataset.py
    with open(f"{base}/data/dataset.py", "w") as f:
        f.write("class LMSYSDataset:\n    def __init__(self, *args, **kwargs): pass\n    def __len__(self): return 0\n")

    # human_pref/data/collators.py
    with open(f"{base}/data/collators.py", "w") as f:
        f.write("class VarlenCollator:\n    def __init__(self, *args, **kwargs): pass\n")
        f.write("class ShardedMaxTokensCollator:\n    def __init__(self, *args, **kwargs): pass\n")

    print("Stubs generated successfully. IDE warnings should resolve after a re-scan.")

if __name__ == "__main__":
    create_stubs()
