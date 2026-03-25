import json
import os

def fix_birdclef_training(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Update Cell 9: Training probes logic
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'PROBE_CLASS_IDX' in ''.join(cell['source']):
            cell['source'] = [
                '# PCA on all trusted embeddings\n',
                'emb_scaler = StandardScaler()\n',
                'emb_full_scaled = emb_scaler.fit_transform(emb_full)\n',
                'n_comp = min(CFG.PCA_DIM, emb_full_scaled.shape[0] - 1, emb_full_scaled.shape[1])\n',
                'emb_pca = PCA(n_components=n_comp)\n',
                'Z_FULL = emb_pca.fit_transform(emb_full_scaled).astype(np.float32)\n',
                'print(f"Embedding PCA: {emb_full.shape[1]}d -> {Z_FULL.shape[1]}d  (explained variance: {emb_pca.explained_variance_ratio_.sum():.4f})")\n',
                '\n',
                '# --- Optimized Probe Training Logic ---\n',
                '# 1. Pre-calculate global context features for the training set (once for all classes)\n',
                'print("Building global context features for training probes...")\n',
                'context_full = build_global_context_features(oof_base)\n',
                '\n',
                '# 2. Train per-class Logistic Regression probes with positive oversampling\n',
                'full_pos_counts = Y_FULL.sum(axis=0)\n',
                'PROBE_CLASS_IDX = np.where(full_pos_counts >= CFG.MIN_POS)[0].astype(np.int32)\n',
                'probe_models = {}\n',
                '\n',
                'for cls_idx in tqdm(PROBE_CLASS_IDX, desc="Training probes"):\n',
                '    y = Y_FULL[:, cls_idx]\n',
                '    if y.sum() == 0 or y.sum() == len(y): continue\n',
                '    \n',
                '    # Use optimized feature synthesis utilizing pre-computed context\n',
                '    X_cls = build_class_features_optimized(\n',
                '        Z_FULL, cls_idx, scores_full_raw, oof_prior, oof_base, context_full)\n',
                '    \n',
                '    clf = LogisticRegression(penalty=\'l2\', C=0.5, class_weight=\'balanced\', max_iter=1000, random_state=42)\n',
                '    clf.fit(X_cls, y)\n',
                '    probe_models[cls_idx] = clf\n',
                '\n',
                'print(f"Trained probes: {len(probe_models)} species")\n'
            ]

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    fix_birdclef_training("BirdCLEF+ 2026/birdclef-2026-perch-v2-bayesian-fusion.ipynb")
