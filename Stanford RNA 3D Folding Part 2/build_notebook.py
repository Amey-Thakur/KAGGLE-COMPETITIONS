#!/usr/bin/env python3
"""
Generates the Stanford RNA 3D Folding Part 2 notebook.
Uses the proven TBM + Protenix-v1 hybrid pipeline from the top public solutions.
"""
import json, textwrap
from typing import Any, Dict, List

def _s(text: str) -> List[str]:
    """Convert a multiline string into a list of notebook source lines."""
    lines = textwrap.dedent(text).strip().split("\n")
    if not lines:
        return []
    result = []
    for i in range(len(lines) - 1):
        result.append(lines[i] + "\n")
    result.append(lines[-1])
    return result

def build():
    cells: List[Dict[str, Any]] = []

    def md(src: str):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": _s(src)})

    def code(src: str):
        cells.append({
            "cell_type": "code", 
            "execution_count": None,
            "metadata": {}, 
            "outputs": [], 
            "source": _s(src)
        })

    # Section 1: Header
    md("""
        # Stanford RNA 3D Folding Part 2: TBM & Protenix-v1 

        **Author:** [Amey Thakur](https://www.kaggle.com/ameythakur20)

        This notebook predicts 3D RNA coordinates using a highly optimized two-phase pipeline. Phase 1 searches the training set for structural templates via pairwise sequence alignment. Phase 2 fills remaining slots with Protenix-v1 diffusion inference. Long sequences are chunked and stitched back together with Kabsch alignment. The output populates five independent structural models per target to maximize the TM-score metric.

        **Outline:**

        1. [Install Dependencies](#1-install-dependencies)
        2. [Imports and Runtime Mode](#2-imports-and-runtime-mode)
        3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
        4. [Paths, Constants and Utilities](#4-paths-constants-and-utilities)
        5. [Template-Based Modeling (TBM)](#5-template-based-modeling-tbm)
        6. [Protenix-v1 Inference](#6-protenix-v1-inference)
        7. [Run Pipeline](#7-run-pipeline)
        8. [Verify Submission](#8-verify-submission)
        9. [Analysis Summary](#9-analysis-summary)

        ***
    """)

    # 1. Install Dependencies
    md("## 1. Install Dependencies\n\nWe install fixed local wheel versions to ensure reproducible runtime environments without internet access.")
    code("""
        !pip install --no-index --no-deps /kaggle/input/datasets/kami1976/biopython-cp312/biopython-1.86-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
        !pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/biotite/biotite-1.6.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
        !pip install --no-index --no-deps /kaggle/input/datasets/amirrezaaleyasin/rdkit-2025-9-5/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl
    """)

    # 2. Imports and Runtime Mode
    md("## 2. Imports and Runtime Mode\n\nWe define a flag to detect Kaggle versus local execution to facilitate testing.")
    code("""
        import os
        import sys
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        IS_KAGGLE = True
        LOCAL_N_SAMPLES = None

        if IS_KAGGLE:
            print("Running in KAGGLE COMPETITION mode.")
        else:
            print(f"Running in LOCAL mode (first {LOCAL_N_SAMPLES} targets).")
    """)

    # 3. Exploratory Data Analysis (EDA)
    md("""
        ## 3. Exploratory Data Analysis (EDA)

        Before setting up the inference constants, we construct basic visualizations to understand the sequence length distributions. This helps establish why MAX_SEQ_LEN chunking thresholds are selected.
    """)
    code(r"""
        DATA_BASE = "/kaggle/input/competitions/stanford-rna-3d-folding-2"
        DEFAULT_TEST_CSV = f"{DATA_BASE}/test_sequences.csv"
        DEFAULT_TRAIN_CSV = f"{DATA_BASE}/train_sequences.csv"

        def render_eda():
            try:
                train_df = pd.read_csv(DEFAULT_TRAIN_CSV)
                test_df = pd.read_csv(DEFAULT_TEST_CSV)
            except FileNotFoundError:
                print("Dataset not found. Skipping EDA.")
                return

            train_lens = train_df['sequence'].apply(len)
            test_lens = test_df['sequence'].apply(len)

            plt.figure(figsize=(12, 6))
            sns.histplot(train_lens, color='royalblue', alpha=0.6, label='Train Sequences', stat='density', bins=50)
            sns.histplot(test_lens, color='crimson', alpha=0.6, label='Test Sequences', stat='density', bins=50)
            
            plt.axvline(x=512, color='green', linestyle='--', label='Max Seq Len Threshold (512)')
            plt.title('Sequence Length Distribution: Train vs Test', fontsize=14, fontweight='bold')
            plt.xlabel('Length (nucleotides)', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()

            print(f"Train sequence stats:\n{train_lens.describe()}\n")
            print(f"Test sequence stats:\n{test_lens.describe()}\n")

        render_eda()
    """)

    # 4. Paths, Constants and Utilities
    md("""
        ## 4. Paths, Constants and Utilities

        N_SAMPLE is set to 5, which satisfies the submission specification while constraining diffusion overhead. MAX_SEQ_LEN is set to 512 tokens to balance context length with GPU memory. 
    """)
    code(r"""
        import gc
        import json
        import time

        os.environ["LAYERNORM_TYPE"] = "torch"
        os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

        from pathlib import Path
        import numpy as np
        import torch
        from Bio.Align import PairwiseAligner
        from tqdm import tqdm

        DEFAULT_TRAIN_LBLS = f"{DATA_BASE}/train_labels.csv"
        DEFAULT_VAL_CSV    = f"{DATA_BASE}/validation_sequences.csv"
        DEFAULT_VAL_LBLS   = f"{DATA_BASE}/validation_labels.csv"
        DEFAULT_OUTPUT     = "/kaggle/working/submission.csv"

        DEFAULT_CODE_DIR = (
            "/kaggle/input/datasets/qiweiyin/protenix-v1-adjusted"
            "/Protenix-v1-adjust-v2/Protenix-v1-adjust-v2/Protenix-v1"
        )
        DEFAULT_ROOT_DIR = DEFAULT_CODE_DIR

        MODEL_NAME    = "protenix_base_20250630_v1.0.0"
        N_SAMPLE      = 5 
        SEED          = 42
        MAX_SEQ_LEN   = int(os.environ.get("MAX_SEQ_LEN",   "512"))
        CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP",  "128"))

        MIN_SIMILARITY       = float(os.environ.get("MIN_SIMILARITY",       "0.0"))
        MIN_PERCENT_IDENTITY = float(os.environ.get("MIN_PERCENT_IDENTITY", "50.0"))

        USE_PROTENIX = True

        def parse_bool(value: str, default: bool = False) -> str:
            v = str(value).strip().lower()
            if v in {"1", "true", "t", "yes", "y", "on"}:
                return "true"
            if v in {"0", "false", "f", "no", "n", "off"}:
                return "false"
            return "true" if default else "false"

        USE_MSA      = parse_bool(os.environ.get("USE_MSA",      "false"))
        USE_TEMPLATE = parse_bool(os.environ.get("USE_TEMPLATE", "false"))
        USE_RNA_MSA  = parse_bool(os.environ.get("USE_RNA_MSA",  "true"))
        MODEL_N_SAMPLE = int(os.environ.get("MODEL_N_SAMPLE", str(N_SAMPLE)))

        def seed_everything(seed: int) -> None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = True
            torch.use_deterministic_algorithms(True)

        def resolve_paths():
            test_csv   = os.environ.get("TEST_CSV",           DEFAULT_TEST_CSV)
            output_csv = os.environ.get("SUBMISSION_CSV",     DEFAULT_OUTPUT)
            code_dir   = os.environ.get("PROTENIX_CODE_DIR",  DEFAULT_CODE_DIR)
            root_dir   = os.environ.get("PROTENIX_ROOT_DIR",  DEFAULT_ROOT_DIR)
            return test_csv, output_csv, code_dir, root_dir

        def ensure_required_files(root_dir: str) -> None:
            for p, name in [
                (Path(root_dir) / "checkpoint" / f"{MODEL_NAME}.pt",          "checkpoint"),
                (Path(root_dir) / "common" / "components.cif",                "CCD file"),
                (Path(root_dir) / "common" / "components.cif.rdkit_mol.pkl",  "CCD cache"),
            ]:
                if not p.exists():
                    raise FileNotFoundError(f"Missing {name}: {p}")

        def build_input_json(df: pd.DataFrame, json_path: str) -> None:
            data = [
                {
                    "name": row["target_id"],
                    "covalent_bonds": [],
                    "sequences": [{"rnaSequence": {"sequence": row["sequence"], "count": 1}}],
                }
                for _, row in df.iterrows()
            ]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f)

        def build_configs(input_json_path: str, dump_dir: str, model_name: str):
            from configs.configs_base import configs as configs_base
            from configs.configs_data import data_configs
            from configs.configs_inference import inference_configs
            from configs.configs_model_type import model_configs
            from protenix.config.config import parse_configs

            base = {**configs_base, **{"data": data_configs}, **inference_configs}

            def deep_update(t, p):
                for k, v in p.items():
                    if isinstance(v, dict) and k in t and isinstance(t[k], dict):
                        deep_update(t[k], v)
                    else:
                        t[k] = v

            deep_update(base, model_configs[model_name])
            arg_str = " ".join([
                f"--model_name {model_name}",
                f"--input_json_path {input_json_path}",
                f"--dump_dir {dump_dir}",
                f"--use_msa {USE_MSA}",
                f"--use_template {USE_TEMPLATE}",
                f"--use_rna_msa {USE_RNA_MSA}",
                f"--sample_diffusion.N_sample {MODEL_N_SAMPLE}",
                f"--seeds {SEED}",
            ])
            return parse_configs(configs=base, arg_str=arg_str, fill_required_with_null=True)

        def get_c1_mask(data: dict, atom_array) -> torch.Tensor:
            if atom_array is not None:
                try:
                    if hasattr(atom_array, "centre_atom_mask"):
                        m = atom_array.centre_atom_mask == 1
                        if hasattr(atom_array, "is_rna"):
                            m = m & atom_array.is_rna
                        return torch.from_numpy(m).bool()
                    if hasattr(atom_array, "atom_name"):
                        base = atom_array.atom_name == "C1'"
                        if hasattr(atom_array, "is_rna"):
                            base = base & atom_array.is_rna
                        return torch.from_numpy(base).bool()
                except Exception:
                    pass
            f = data["input_feature_dict"]
            if "centre_atom_mask" in f:
                return (f["centre_atom_mask"] == 1).bool()
            if "center_atom_mask" in f:
                return (f["center_atom_mask"] == 1).bool()
            n_tokens = data.get("N_token", torch.tensor(0)).item()
            mask11 = (f["atom_to_tokatom_idx"] == 11).bool()
            mask12 = (f["atom_to_tokatom_idx"] == 12).bool()
            if abs(mask11.sum().item() - n_tokens) < abs(mask12.sum().item() - n_tokens):
                return mask11
            return mask12

        def coords_to_rows(target_id: str, seq: str, coords: np.ndarray) -> list:
            rows = []
            for i in range(len(seq)):
                row = {"ID": f"{target_id}_{i + 1}", "resname": seq[i], "resid": i + 1}
                for s in range(N_SAMPLE):
                    if s < coords.shape[0] and i < coords.shape[1]:
                        x, y, z = coords[s, i]
                    else:
                        x, y, z = 0.0, 0.0, 0.0
                    row[f"x_{s + 1}"] = float(x)
                    row[f"y_{s + 1}"] = float(y)
                    row[f"z_{s + 1}"] = float(z)
                rows.append(row)
            return rows

        def split_into_chunks(seq_len: int, max_len: int, overlap: int) -> list:
            if seq_len <= max_len:
                return [(0, seq_len)]
            chunks, step, pos = [], max_len - overlap, 0
            while pos < seq_len:
                end = min(pos + max_len, seq_len)
                chunks.append((pos, end))
                if end == seq_len:
                    break
                pos += step
            return chunks

        def kabsch_align(P: np.ndarray, Q: np.ndarray):
            centroid_P = P.mean(axis=0)
            centroid_Q = Q.mean(axis=0)
            Pc, Qc = P - centroid_P, Q - centroid_Q
            U, _, Vt = np.linalg.svd(Pc.T @ Qc)
            d = np.linalg.det(Vt.T @ U.T)
            S = np.eye(3)
            if d < 0:
                S[2, 2] = -1
            R = Vt.T @ S @ U.T
            t = centroid_Q - R @ centroid_P
            return R, t

        def stitch_chunk_coords(chunk_coords_list, chunk_ranges, seq_len):
            if len(chunk_coords_list) == 1:
                coords = chunk_coords_list[0]
                if coords.shape[0] >= seq_len:
                    return coords[:seq_len]
                out = np.zeros((seq_len, 3), dtype=coords.dtype)
                out[:coords.shape[0]] = coords
                return out
            aligned = [chunk_coords_list[0].copy()]
            for i in range(1, len(chunk_coords_list)):
                prev_start, prev_end = chunk_ranges[i - 1]
                cur_start, cur_end = chunk_ranges[i]
                ov_start, ov_end = cur_start, min(prev_end, cur_end)
                ov_len = ov_end - ov_start
                if ov_len < 3:
                    aligned.append(chunk_coords_list[i].copy())
                    continue
                prev_ov = aligned[i - 1][ov_start - prev_start: ov_end - prev_start]
                cur_ov = chunk_coords_list[i][ov_start - cur_start: ov_end - cur_start]
                valid = ~(np.isnan(prev_ov).any(axis=1) | np.isnan(cur_ov).any(axis=1))
                if valid.sum() < 3:
                    aligned.append(chunk_coords_list[i].copy())
                    continue
                R, t = kabsch_align(cur_ov[valid], prev_ov[valid])
                aligned.append((chunk_coords_list[i] @ R.T) + t)
            full = np.zeros((seq_len, 3), dtype=np.float64)
            weights = np.zeros(seq_len, dtype=np.float64)
            for i, ((s, e), coords) in enumerate(zip(chunk_ranges, aligned)):
                actual_end = min(s + coords.shape[0], seq_len)
                used_len = actual_end - s
                w = np.ones(used_len, dtype=np.float64)
                if i > 0:
                    ov_end = min(chunk_ranges[i - 1][1], e)
                    ramp_len = ov_end - s
                    if ramp_len > 0:
                        w[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)
                if i < len(chunk_ranges) - 1:
                    next_s = chunk_ranges[i + 1][0]
                    ramp_start = next_s - s
                    ramp_len = actual_end - next_s
                    if ramp_len > 0 and ramp_start < used_len:
                        w[ramp_start:used_len] = np.linspace(1.0, 0.0, ramp_len)
                full[s:actual_end] += coords[:used_len] * w[:, None]
                weights[s:actual_end] += w
            mask = weights > 0
            full[mask] /= weights[mask, None]
            return full

        print("Utilities loaded.")
    """)

    # 5. TBM
    md("""
        ## 5. Template-Based Modeling (TBM)

        Targets are mapped against the training database using basic pairwise sequence alignments. When a template exceeds the MIN_PERCENT_IDENTITY threshold, it avoids running the computationally expensive Protenix fallback sequence. To enforce topological diversity, geometric transformations (hinge rotation, jitter) are sequentially applied.
    """)
    code(r"""
        def _make_aligner() -> PairwiseAligner:
            import warnings
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            al = PairwiseAligner()
            al.mode                           = "global"
            al.match_score                    = 2
            al.mismatch_score                 = -1.5
            al.open_gap_score                 = -8
            al.extend_gap_score               = -0.4
            al.query_left_open_gap_score      = -8
            al.query_left_extend_gap_score    = -0.4
            al.query_right_open_gap_score     = -8
            al.query_right_extend_gap_score   = -0.4
            al.target_left_open_gap_score     = -8
            al.target_left_extend_gap_score   = -0.4
            al.target_right_open_gap_score    = -8
            al.target_right_extend_gap_score  = -0.4
            return al

        _aligner = _make_aligner()

        def parse_stoichiometry(stoich: str) -> list:
            if pd.isna(stoich) or str(stoich).strip() == "":
                return []
            return [(ch.strip(), int(cnt)) for part in str(stoich).split(";")
                    for ch, cnt in [part.split(":")]]

        def parse_fasta(fasta_content: str) -> dict:
            out, cur, parts = {}, None, []
            for line in str(fasta_content).splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if cur is not None:
                        out[cur] = "".join(parts)
                    cur = line[1:].split()[0]
                    parts = []
                else:
                    parts.append(line.replace(" ", ""))
            if cur is not None:
                out[cur] = "".join(parts)
            return out

        def get_chain_segments(row) -> list:
            seq    = row["sequence"]
            stoich = row.get("stoichiometry", "")
            all_sq = row.get("all_sequences", "")
            if (pd.isna(stoich) or pd.isna(all_sq)
                    or str(stoich).strip() == "" or str(all_sq).strip() == ""):
                return [(0, len(seq))]
            try:
                chain_dict = parse_fasta(all_sq)
                order = parse_stoichiometry(stoich)
                segs, pos = [], 0
                for ch, cnt in order:
                    base = chain_dict.get(ch)
                    if base is None:
                        return [(0, len(seq))]
                    for _ in range(cnt):
                        segs.append((pos, pos + len(base)))
                        pos += len(base)
                return segs if pos == len(seq) else [(0, len(seq))]
            except Exception:
                return [(0, len(seq))]

        def build_segments_map(df: pd.DataFrame) -> tuple:
            seg_map, stoich_map = {}, {}
            for _, r in df.iterrows():
                tid               = r["target_id"]
                seg_map[tid]      = get_chain_segments(r)
                raw_s             = r.get("stoichiometry", "")
                stoich_map[tid]   = "" if pd.isna(raw_s) else str(raw_s)
            return seg_map, stoich_map

        def process_labels(labels_df: pd.DataFrame) -> dict:
            coords = {}
            prefixes = labels_df["ID"].str.rsplit("_", n=1).str[0]
            for prefix, grp in labels_df.groupby(prefixes):
                coords[prefix] = grp.sort_values("resid")[["x_1", "y_1", "z_1"]].values
            return coords

        def _build_aligned_strings(query_seq, template_seq, alignment):
            q_segs, t_segs = alignment.aligned
            aq, at, qi, ti = [], [], 0, 0
            for (qs, qe), (ts, te) in zip(q_segs, t_segs):
                while qi < qs: aq.append(query_seq[qi]);    at.append("-");              qi += 1
                while ti < ts: aq.append("-");              at.append(template_seq[ti]); ti += 1
                for qp, tp in zip(range(qs, qe), range(ts, te)):
                    aq.append(query_seq[qp]); at.append(template_seq[tp])
                qi, ti = qe, te
            while qi < len(query_seq):    aq.append(query_seq[qi]);    at.append("-");              qi += 1
            while ti < len(template_seq): aq.append("-");              at.append(template_seq[ti]); ti += 1
            return "".join(aq), "".join(at)

        def find_similar_sequences_detailed(query_seq, train_seqs_df, train_coords_dict, top_n=30):
            results = []
            for _, row in train_seqs_df.iterrows():
                tid, tseq = row["target_id"], row["sequence"]
                if tid not in train_coords_dict:
                    continue
                if abs(len(tseq) - len(query_seq)) / max(len(tseq), len(query_seq)) > 0.3:
                    continue
                aln       = next(iter(_aligner.align(query_seq, tseq)))
                norm_s    = aln.score / (2 * min(len(query_seq), len(tseq)))
                identical = sum(
                    1 for (qs, qe), (ts, te) in zip(*aln.aligned)
                    for qp, tp in zip(range(qs, qe), range(ts, te))
                    if query_seq[qp] == tseq[tp]
                )
                pct_id = 100 * identical / len(query_seq)
                aq, at = _build_aligned_strings(query_seq, tseq, aln)
                results.append((tid, tseq, norm_s, train_coords_dict[tid], pct_id, aq, at))
            results.sort(key=lambda x: x[2], reverse=True)
            return results[:top_n]

        def adapt_template_to_query(query_seq, template_seq, template_coords) -> np.ndarray:
            aln        = next(iter(_aligner.align(query_seq, template_seq)))
            new_coords = np.full((len(query_seq), 3), np.nan)
            for (qs, qe), (ts, te) in zip(*aln.aligned):
                chunk = template_coords[ts:te]
                if len(chunk) == (qe - qs):
                    new_coords[qs:qe] = chunk
            for i in range(len(new_coords)):
                if np.isnan(new_coords[i, 0]):
                    pv = next((j for j in range(i - 1, -1, -1) if not np.isnan(new_coords[j, 0])), -1)
                    nv = next((j for j in range(i + 1, len(new_coords)) if not np.isnan(new_coords[j, 0])), -1)
                    if pv >= 0 and nv >= 0:
                        w = (i - pv) / (nv - pv)
                        new_coords[i] = (1 - w) * new_coords[pv] + w * new_coords[nv]
                    elif pv >= 0:
                        new_coords[i] = new_coords[pv] + [3, 0, 0]
                    elif nv >= 0:
                        new_coords[i] = new_coords[nv] + [3, 0, 0]
                    else:
                        new_coords[i] = [i * 3, 0, 0]
            return np.nan_to_num(new_coords)

        def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=3) -> np.ndarray:
            X        = coords.copy()
            segments = segments_map.get(target_id, [(0, len(X))])
            strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
            for _ in range(passes):
                for s, e in segments:
                    C = X[s:e]; L = e - s
                    if L < 3:
                        continue
                    d    = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
                    adj  = d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
                    C[:-1] -= adj; C[1:] += adj
                    d2   = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
                    adj2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.10 * strength)
                    C[:-2] -= adj2; C[2:] += adj2
                    C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
                    if L >= 25:
                        idx  = np.linspace(0, L - 1, min(L, 160)).astype(int) if L > 220 else np.arange(L)
                        P    = C[idx]; diff = P[:, None, :] - P[None, :, :]
                        dm   = np.linalg.norm(diff, axis=2) + 1e-6
                        sep  = np.abs(idx[:, None] - idx[None, :])
                        mask = (sep > 2) & (dm < 3.2)
                        if np.any(mask):
                            vec = (diff * ((3.2 - dm) / dm)[:, :, None] * mask[:, :, None]).sum(axis=1)
                            C[idx] += (0.015 * strength) * vec
                    X[s:e] = C
            return X

        def _rotmat(axis, ang):
            a = np.asarray(axis, float); a /= np.linalg.norm(a) + 1e-12
            x, y, z = a; c, s = np.cos(ang), np.sin(ang); CC = 1 - c
            return np.array([[c+x*x*CC, x*y*CC-z*s, x*z*CC+y*s],
                             [y*x*CC+z*s, c+y*y*CC, y*z*CC-x*s],
                             [z*x*CC-y*s, z*y*CC+x*s, c+z*z*CC]])

        def apply_hinge(coords, seg, rng, deg=22):
            s, e = seg; L = e - s
            if L < 30: return coords
            pivot = s + int(rng.integers(10, L - 10))
            R = _rotmat(rng.normal(size=3), np.deg2rad(float(rng.uniform(-deg, deg))))
            X = coords.copy(); p0 = X[pivot].copy()
            X[pivot+1:e] = (X[pivot+1:e] - p0) @ R.T + p0
            return X

        def jitter_chains(coords, segs, rng, deg=12, trans=1.5):
            X = coords.copy(); gc_ = X.mean(0, keepdims=True)
            for s, e in segs:
                R     = _rotmat(rng.normal(size=3), np.deg2rad(float(rng.uniform(-deg, deg))))
                shift = rng.normal(size=3); shift = shift / (np.linalg.norm(shift) + 1e-12) * float(rng.uniform(0, trans))
                c     = X[s:e].mean(0, keepdims=True)
                X[s:e] = (X[s:e] - c) @ R.T + c + shift
            X -= X.mean(0, keepdims=True) - gc_
            return X

        def smooth_wiggle(coords, segs, rng, amp=0.8):
            X = coords.copy()
            for s, e in segs:
                L = e - s
                if L < 20: continue
                ctrl = np.linspace(0, L - 1, 6); disp = rng.normal(0, amp, (6, 3)); t = np.arange(L)
                X[s:e] += np.vstack([np.interp(t, ctrl, disp[:, k]) for k in range(3)]).T
            return X

        def generate_rna_structure(sequence: str, seed=None) -> np.ndarray:
            if seed is not None:
                np.random.seed(seed)
            n = len(sequence); coords = np.zeros((n, 3))
            for i in range(n):
                ang = i * 0.6
                coords[i] = [10.0 * np.cos(ang), 10.0 * np.sin(ang), i * 2.5]
            return coords

        print("TBM functions loaded.")
    """)

    # 6. Protenix Inference + Main
    md("""
        ## 6. Protenix Inference

        Targets not adequately resolved by TBM are scheduled for Protenix processing. Because Protenix consumes heavy memory overhead, sequences greater than MAX_SEQ_LEN are tiled to prevent CUDA Out Of Memory errors. A 128 residue overlap threshold buffers structural inconsistencies at chunk boundaries.
    """)
    code(r"""
        def tbm_phase(test_df, train_seqs_df, train_coords_dict, segments_map):
            print(f"\n{'='*60}")
            print(f"PHASE 1: Template-Based Modeling")
            print(f"{'='*60}")
            t0 = time.time()
            template_predictions = {}
            protenix_queue = {}
            for _, row in test_df.iterrows():
                tid = row["target_id"]
                seq = row["sequence"]
                segs = segments_map.get(tid, [(0, len(seq))])
                similar = find_similar_sequences_detailed(seq, train_seqs_df, train_coords_dict, top_n=30)
                preds, used = [], set()
                for i, (tmpl_id, tmpl_seq, sim, tmpl_coords, pct_id, _, _) in enumerate(similar):
                    if len(preds) >= N_SAMPLE:
                        break
                    if sim < MIN_SIMILARITY or pct_id < MIN_PERCENT_IDENTITY:
                        break
                    if tmpl_id in used:
                        continue
                    rng     = np.random.default_rng((row.name * 10000000000 + i * 10007) % (2**32))
                    adapted = adapt_template_to_query(seq, tmpl_seq, tmpl_coords)
                    slot = len(preds)
                    if slot == 0:
                        X = adapted
                    elif slot == 1:
                        X = adapted + rng.normal(0, max(0.01, (0.40 - sim) * 0.06), adapted.shape)
                    elif slot == 2:
                        longest = max(segs, key=lambda se: se[1] - se[0])
                        X = apply_hinge(adapted, longest, rng)
                    elif slot == 3:
                        X = jitter_chains(adapted, segs, rng)
                    else:
                        X = smooth_wiggle(adapted, segs, rng)
                    refined = adaptive_rna_constraints(X, tid, segments_map, confidence=sim)
                    preds.append(refined)
                    used.add(tmpl_id)
                template_predictions[tid] = preds
                n_needed = N_SAMPLE - len(preds)
                if n_needed > 0:
                    protenix_queue[tid] = (n_needed, seq)
                else:
                    print(f"  {tid} ({len(seq)} nt): all {N_SAMPLE} models generated from templates.")
            elapsed = time.time() - t0
            print(f"\nPhase 1 completed in {elapsed:.1f} seconds.")
            return template_predictions, protenix_queue
    """)

    # 7. Run Pipeline
    md("""
        ## 7. Run Pipeline

        We orchestrate the complete flow, running the TBM logic, appending Protenix diffusion results, filling gaps with geometric fallback data, and dumping a correctly formatted CSV.
    """)
    code(r"""
        def main() -> None:
            test_csv, output_csv, code_dir, root_dir = resolve_paths()
            if not os.path.isdir(code_dir):
                raise FileNotFoundError(f"Missing PROTENIX repo: {code_dir}")
            os.environ["PROTENIX_ROOT_DIR"] = root_dir
            sys.path.append(code_dir)
            ensure_required_files(root_dir)
            seed_everything(SEED)

            test_df_full = pd.read_csv(test_csv)
            test_df = (test_df_full.head(LOCAL_N_SAMPLES) if not IS_KAGGLE
                        else test_df_full).reset_index(drop=True)

            train_seqs   = pd.read_csv(DEFAULT_TRAIN_CSV)
            val_seqs     = pd.read_csv(DEFAULT_VAL_CSV)
            train_labels = pd.read_csv(DEFAULT_TRAIN_LBLS, low_memory=False)
            val_labels   = pd.read_csv(DEFAULT_VAL_LBLS, low_memory=False)
            combined_seqs   = pd.concat([train_seqs,   val_seqs],    ignore_index=True)
            combined_labels = pd.concat([train_labels, val_labels],  ignore_index=True)
            train_coords    = process_labels(combined_labels)
            segments_map, _ = build_segments_map(test_df)

            template_preds, protenix_queue = tbm_phase(
                test_df, combined_seqs, train_coords, segments_map
            )

            protenix_preds = {}
            if protenix_queue and USE_PROTENIX:
                print(f"\n{'='*60}")
                print(f"PHASE 2: Protenix Diffusion")
                print(f"{'='*60}")
                work_dir = Path("/kaggle/working")
                work_dir.mkdir(parents=True, exist_ok=True)
                tasks, chunk_info = [], {}
                for target_id, (n_needed, full_seq) in protenix_queue.items():
                    seq_len = len(full_seq)
                    if seq_len <= MAX_SEQ_LEN:
                        tasks.append({"target_id": target_id, "sequence": full_seq})
                        chunk_info[target_id] = [{"name": target_id, "range": (0, seq_len)}]
                    else:
                        chunks = split_into_chunks(seq_len, MAX_SEQ_LEN, CHUNK_OVERLAP)
                        chunk_info[target_id] = []
                        for ci, (cs, ce) in enumerate(chunks):
                            chunk_name = f"{target_id}_chunk{ci}"
                            tasks.append({"target_id": chunk_name, "sequence": full_seq[cs:ce]})
                            chunk_info[target_id].append({"name": chunk_name, "range": (cs, ce)})

                tasks_df = pd.DataFrame(tasks)
                input_json_path = str(work_dir / "protenix_queue_input.json")
                build_input_json(tasks_df, input_json_path)

                from protenix.data.inference.infer_dataloader import InferenceDataset
                from runner.inference import (InferenceRunner,
                                              update_gpu_compatible_configs,
                                              update_inference_configs)

                configs = build_configs(input_json_path, str(work_dir / "outputs"), MODEL_NAME)
                configs = update_gpu_compatible_configs(configs)
                runner  = InferenceRunner(configs)
                dataset = InferenceDataset(configs)

                raw_predictions = {}
                for i in tqdm(range(len(dataset)), desc="Protenix Iterations"):
                    data, atom_array, err = dataset[i]
                    sample_name = data.get("sample_name", f"sample_{i}")
                    if err:
                        raw_predictions[sample_name] = None
                        del data, atom_array, err
                        gc.collect(); torch.cuda.empty_cache()
                        continue
                    target_id_key = sample_name.split("_chunk")[0] if "_chunk" in sample_name else sample_name
                    n_needed = protenix_queue.get(target_id_key, (N_SAMPLE, ""))[0]
                    sub_seq_len = data["N_token"].item()
                    try:
                        new_cfg = update_inference_configs(configs, sub_seq_len)
                        new_cfg.sample_diffusion.N_sample = n_needed
                        runner.update_model_configs(new_cfg)
                        pred = runner.predict(data)
                        raw_coords = pred["coordinate"]
                        feat = data["input_feature_dict"]
                        if "centre_atom_mask" in feat:
                            c1mask = (feat["centre_atom_mask"] == 1).to(raw_coords.device)
                        elif "atom_to_tokatom_idx" in feat:
                            m11 = (feat["atom_to_tokatom_idx"] == 11).to(raw_coords.device)
                            m12 = (feat["atom_to_tokatom_idx"] == 12).to(raw_coords.device)
                            c1mask = m11 if abs(m11.sum() - sub_seq_len) < abs(m12.sum() - sub_seq_len) else m12
                        else:
                            c1mask = torch.zeros(raw_coords.shape[1], dtype=torch.bool, device=raw_coords.device)
                        coords = raw_coords[:, c1mask, :].detach().cpu().numpy()
                        if coords.shape[1] > 1:
                            diffs = np.linalg.norm(coords[0, 1:] - coords[0, :-1], axis=-1)
                            if np.all(diffs < 1e-4):
                                coords = None
                        if coords is not None and coords.shape[1] != sub_seq_len:
                            if coords.shape[1] == 1 and sub_seq_len > 1:
                                coords = None
                            elif coords is not None:
                                padded = np.zeros((coords.shape[0], sub_seq_len, 3), dtype=np.float32)
                                ml = min(coords.shape[1], sub_seq_len)
                                padded[:, :ml, :] = coords[:, :ml, :]
                                coords = padded
                        raw_predictions[sample_name] = coords
                    except Exception as exc:
                        raw_predictions[sample_name] = None
                    finally:
                        try: del pred, data, atom_array, raw_coords
                        except: pass
                        gc.collect(); torch.cuda.empty_cache()

                for target_id, (n_needed, full_seq) in protenix_queue.items():
                    seq_len = len(full_seq)
                    chunks = chunk_info.get(target_id, [])
                    if not chunks:
                        continue
                    if len(chunks) == 1:
                        protenix_preds[target_id] = raw_predictions.get(target_id)
                    else:
                        chunk_results_per_sample = {s: [] for s in range(n_needed)}
                        all_ok = True
                        for cinfo in chunks:
                            ccoords = raw_predictions.get(cinfo["name"])
                            if ccoords is None:
                                all_ok = False; break
                            for s_idx in range(n_needed):
                                idx = s_idx if s_idx < ccoords.shape[0] else -1
                                chunk_results_per_sample[s_idx].append((ccoords[idx], cinfo["range"]))
                        if not all_ok:
                            protenix_preds[target_id] = None; continue
                        stitched = []
                        for s_idx in range(n_needed):
                            items = chunk_results_per_sample[s_idx]
                            full_c = stitch_chunk_coords([c for c, _ in items], [r for _, r in items], seq_len)
                            stitched.append(full_c)
                        protenix_preds[target_id] = np.stack(stitched, axis=0)

            print(f"\n{'='*60}")
            print("PHASE 3: Aggregate Valid Target Submissions")
            print(f"{'='*60}")

            all_rows = []
            for _, row in test_df.iterrows():
                tid, seq = row["target_id"], row["sequence"]
                combined = list(template_preds.get(tid, []))
                ptx = protenix_preds.get(tid)
                if ptx is not None and ptx.ndim == 3:
                    for j in range(ptx.shape[0]):
                        if len(combined) >= N_SAMPLE:
                            break
                        combined.append(ptx[j])
                while len(combined) < N_SAMPLE:
                    seed_val = row.name * 1000000 + len(combined) * 1000
                    dn = generate_rna_structure(seq, seed=seed_val)
                    combined.append(adaptive_rna_constraints(dn, tid, segments_map, confidence=0.2))
                stacked = np.stack(combined[:N_SAMPLE], axis=0)
                all_rows.extend(coords_to_rows(tid, seq, stacked))

            sub = pd.DataFrame(all_rows)
            cols = ["ID", "resname", "resid"] + [
                f"{c}_{i}" for i in range(1, N_SAMPLE + 1) for c in ["x", "y", "z"]
            ]
            coord_cols = [c for c in cols if c.startswith(("x_", "y_", "z_"))]
            sub[coord_cols] = sub[coord_cols].clip(-999.999, 9999.999)
            sub[cols].to_csv(output_csv, index=False)
            print(f"\nWritten {len(sub):,} rows to {output_csv}")

        if __name__ == "__main__":
            main()
    """)

    # 8. Verify Submission
    md("## 8. Verify Submission\n\nSanity check on the generated coordinate data.")
    code("""
        submission_path = "/kaggle/working/submission.csv"
        submission_df = pd.read_csv(submission_path)
        print(f"Final shape: {submission_df.shape}")
        submission_df.head(15)
    """)

    # 9. Analysis Summary
    md("""
        ## 9. Analysis Summary

        This guide detailed a production-grade approach to Stanford RNA 3D Folding prediction:

        1. **Template-Based Modeling**: Utilized pairwise sequence alignment against the structural database to establish high-fidelity topological templates for identical sequence matches.
        2. **Diversity Transforms**: Injected geometric deformations including hinging, jittering, and wiggling functions to expand the structural permutations required for metric maximization.
        3. **Protenix-v1 Synthesis**: Scheduled unresolved target segments for deep coordinate sampling utilizing diffusion models governed by adaptive overlap boundaries.
        4. **Kabsch Alignment**: Realigned synthesized overlapping coordinate blocks across multi-tile predictions to stitch elongated 512-residue bounds.
        5. **De-Novo Fallback**: Filled any remaining prediction gaps safely using idealized A-form structural fallbacks to guarantee robust prediction files.

        ---
        **Citation:**
        Rhiju Das, Youhan Lee, Chris Munley, Przemek Porebski, Walter Reade, Theo Viel, and Ashley Oldacre. Stanford RNA 3D Folding Part 2. https://kaggle.com/competitions/stanford-rna-3d-folding-2, 2026. Kaggle.
    """)

    # Notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.12"},
            "kaggle": {
                "accelerator": "nvidiaTeslaP100",
                "dataSources": [
                    {"sourceType": "competition", "sourceId": 118765, "databundleVersionId": 15231210},
                    {"sourceType": "datasetVersion", "sourceId": 14604295, "datasetId": 9328538, "databundleVersionId": 15440074},
                    {"sourceType": "datasetVersion", "sourceId": 14962460, "datasetId": 9577079, "databundleVersionId": 15833819},
                    {"sourceType": "datasetVersion", "sourceId": 14874339, "datasetId": 9502242, "databundleVersionId": 15736806},
                    {"sourceType": "datasetVersion", "sourceId": 14962495, "datasetId": 9577097, "databundleVersionId": 15833858},
                    {"sourceType": "datasetVersion", "sourceId": 10855324, "datasetId": 6742586, "databundleVersionId": 11219268},
                ],
                "dockerImageVersionId": 31287,
                "isGpuEnabled": True,
                "isInternetEnabled": False,
                "language": "python",
                "sourceType": "notebook"
            }
        },
        "nbformat": 4, "nbformat_minor": 5
    }

    target = r"d:\GitHub\Kaggle-Competitions\Stanford RNA 3D Folding Part 2" \
             r"\stanford-rna-3d-folding-part-2-tbm-protenix-gpu-p100.ipynb"
    with open(target, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"Notebook written to {target}")

if __name__ == "__main__":
    build()
