import os
import sys
from pathlib import Path

def auto_mount_all_utilities(search_root="/kaggle/usr/lib", verbose=True):
    """
    Automatically finds all Kaggle Utility Scripts in your environment and 
    adds their parent directories to sys.path so they can be imported directly.
    """
    if not os.path.exists(search_root):
        if verbose: print(f"--- [Universal Loader] Search root {search_root} not found. Skipping. ---")
        return

    mounted_paths = []
    
    # Kaggle Utility Scripts are typically mounted in: /kaggle/usr/lib/<username>/<script-name>/<script>.py
    # We want to add the '<script-name>' folder to sys.path.
    for root, dirs, files in os.walk(search_root):
        for file in files:
            if file.endswith(".py") and "__init__" not in file:
                script_dir = os.path.abspath(root)
                if script_dir not in sys.path:
                    sys.path.append(script_dir)
                    mounted_paths.append(script_dir)
                    if verbose: print(f"--- [Universal Loader] MOUNTED: {file} at {script_dir} ---")

    if not mounted_paths and verbose:
        print("--- [Universal Loader] No modular utility scripts found in /kaggle/usr/lib. ---")
    
    return mounted_paths

def diagnostic_report():
    """Prints a full report of the current Python environment."""
    print("--- Python Version ---")
    print(sys.version)
    print("\n--- Current sys.path ---")
    for p in sys.path:
        print(p)
    print("\n--- Mounted /kaggle Directories ---")
    for d in ["/kaggle/input", "/kaggle/working", "/kaggle/usr/lib"]:
        if os.path.exists(d):
            print(f"EXISTS: {d}")
        else:
            print(f"MISSING: {d}")

# Optional: Run immediately if executed directly
if __name__ == "__main__":
    auto_mount_all_utilities()
    diagnostic_report()
