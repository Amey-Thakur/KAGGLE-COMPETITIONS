import os

# Simply verifying that a Kaggle model is attached to earn the badge.
model_path = "/kaggle/input/models/google/bert/tensorflow2/answer-equivalence-bem/1"
if os.path.exists(model_path):
    print(f"Found Kaggle Model Surface at {model_path}. Earning the Notebook Modeler badge!")
else:
    print(f"Model surface STILL not found locally at {model_path}. (Wait for processing).")
