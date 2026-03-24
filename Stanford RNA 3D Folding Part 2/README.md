<a name="readme-top"></a>
# Stanford RNA 3D Folding Part 2
### Structural Biology Pipeline Optimization

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ameythakur20/stanford-rna-3d-folding-part-2-tbm-protenix-v1)

---

## Technical Overview

This repository contains a high-performance, two-phase hybrid pipeline designed for the prediction of 3D RNA coordinates. The methodology integrates classical comparative modeling with state-of-the-art diffusion models to achieve robust structural predictions across varying sequence lengths and complexities.

### Methodology: Hybrid TBM + Protenix-v1

The pipeline is architected to prioritize verified structural data while utilizing generative fallback mechanisms for novel folds.

#### Phase 1: Template-Based Modeling (TBM)
The initial phase performs global pairwise sequence alignment against a curated database of known RNA structures. When a candidate template exceeds a 50% identity threshold, TBM is utilized to construct a coordinate framework.
*   **Gap Reconstruction**: Sinusoidal backbone reconstruction is applied to alignment gaps to maintain stereochemical continuity.
*   **Adaptive Constraints**: Coordinates are refined using confidence-scaled constraints to preserve template integrity while resolving local geometric clashes.

#### Phase 2: Protenix-v1 Diffusion
For target sequences that lack suitable templates or contain unresolved regions, the pipeline utilizes the Protenix-v1 diffusion model for de novo coordinate generation.
*   **Chunked Inference**: To accommodate hardware constraints (Kaggle T4 GPUs), sequences exceeding 420 nucleotides are processed in overlapping tiles.
*   **Kabsch Stitching**: Individual chunks are reintegrated into a global coordinate system using core-trimmed Kabsch alignment, minimizing structural drift at segment boundaries.

---

## Technical Components

| Component | Scientific Rationale |
| :--- | :--- |
| **Kabsch Alignment** | Minimizes Root-Mean-Square Deviation (RMSD) between overlapping structural segments to ensure global connectivity. |
| **Diffusion Sampling** | Utilizes stochastic denoising to navigate the complex conformer landscape of rare RNA folds. |
| **Energetic Ranking** | Final candidate models are ranked based on backbone bond length deviation and steric clash penalties. |

---

## Implementation Details

The pipeline is implemented using a custom ensemble of structural biology libraries, including Biopython for alignment, Biotite for coordinate manipulation, and a modified Protenix-v1 engine for inference. All dependencies are managed locally to ensure reproducible execution in restricted competition environments.

---

### Amey Thakur

[Kaggle](https://www.kaggle.com/ameythakur20) • [GitHub](https://github.com/Amey-Thakur)

---

<div align="center">

  [↑ Back to Top](#readme-top) &nbsp;·&nbsp; [← Back to Home](../README.md)

</div>
