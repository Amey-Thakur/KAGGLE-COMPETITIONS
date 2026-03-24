import os
import sys

# Diagnostic to find exactly where the Titanic utility script was mounted.
print("--- Searching for ANY notebook/utility script sources ---")
possible_dirs = ["/kaggle/usr/lib", "/kaggle/working", "/kaggle/input"]

for d in possible_dirs:
    if os.path.exists(d):
        print(f"\nScanning: {d}")
        for root, dirs, files in os.walk(d):
            for file in files:
                if "titanic" in file.lower():
                    print(f"FOUND MOUNTED PATH: {os.path.join(root, file)}")

print("\n--- Current sys.path ---")
for p in sys.path:
    print(p)
