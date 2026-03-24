import os
import sys

def find_and_mount(pattern="titanic", verbose=True):
    """
    ULTIMATE KAGGLER TOOL: 
    1. Searches for any utility script matching the pattern.
    2. Tells you exactly where it is.
    3. AUTOMATICALLY adds its directory to your Python path for instant import.
    """
    print(f"\n--- [Utility Hunter] Searching for: '{pattern}' ---")
    
    possible_dirs = ["/kaggle/usr/lib", "/kaggle/working", "/kaggle/input"]
    found_any = False

    for d in possible_dirs:
        if not os.path.exists(d):
            continue
            
        for root, dirs, files in os.walk(d):
            for file in files:
                if pattern.lower() in file.lower() and file.endswith(".py"):
                    script_path = os.path.join(root, file)
                    script_dir = os.path.dirname(script_path)
                    
                    if verbose:
                        print(f"✅ FOUND: {file}")
                        print(f"📍 PATH : {script_path}")
                    
                    # THE MAGIC FIX: Automatically add to sys.path
                    if script_dir not in sys.path:
                        sys.path.append(script_dir)
                        if verbose: print(f"🚀 MOUNTED successfully! You can now 'import {file[:-3]}'")
                    
                    found_any = True
    
    if not found_any:
        print(f"❌ Could not find any scripts matching '{pattern}' in /kaggle directories.")
    
    return found_any

def system_report():
    """Prints a beautiful summary of the Kaggle environment."""
    print("\n" + "="*50)
    print(" KAGGLE UTILITY SYSTEM REPORT")
    print("="*50)
    print(f"Python: {sys.version.split(' ')[0]}")
    print(f"CWD:    {os.getcwd()}")
    print("\n--- Current Active Import Paths ---")
    for i, p in enumerate(sys.path):
        if "/kaggle" in p:
            print(f"[{i}] {p} (Internal)")
        else:
            print(f"[{i}] {p}")
    print("="*50 + "\n")

# Default action when the utility is imported
if __name__ == "__main__":
    # If the user runs this file directly as a script, default to titanic for compatibility
    find_and_mount("titanic")
    system_report()
