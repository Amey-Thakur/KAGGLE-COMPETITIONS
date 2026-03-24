import numpy as np
from scipy.spatial import distance_matrix as _dm

def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2):
    """
    Optimized RNA constraint module with confidence-scaled bond lengths 
    and steric clash prevention (3.8A threshold).
    
    This utility script provides structural auditing for the Stanford RNA 3D Folding competition.
    """
    X        = coords.copy()
    segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3: continue
            
            # Primary Bond: 5.95A
            d = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj = d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
            C[:-1] -= adj; C[1:] += adj
            
            # Neighbor Angle: 10.2A
            d2 = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            adj2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.10 * strength)
            C[:-2] -= adj2; C[2:] += adj2
            
            # Laplacian Smoothing
            C[1:-1] += (0.06 * strength) * (0.5 * (C[:-2] + C[2:]) - C[1:-1])
            X[s:e] = C
            
    # Steric Clash Block (3.8A)
    # Provides physical sanity checking for low-confidence TBM or Protenix predictions.
    if strength > 0.3:
        d_mat = _dm(X, X)
        clashes = np.where((d_mat < 3.8) & (d_mat > 0))
        for ci in range(len(clashes[0])):
            ii, jj = clashes[0][ci], clashes[1][ci]
            if abs(ii - jj) <= 1 or ii >= jj: continue
            
            dd = d_mat[ii, jj]
            dr = (X[jj] - X[ii]) / (dd + 1e-10)
            fix = (3.8 - dd) * strength
            X[ii] -= dr * (fix / 2)
            X[jj] += dr * (fix / 2)
            
    return X

def get_backbone_energy(c):
    """Calculates bond length error and collision penalty for candidate ranking."""
    d        = np.linalg.norm(c[1:] - c[:-1], axis=1)
    bond_err = np.mean((d - 5.95) ** 2)
    clash    = 0.0
    if len(c) > 20:
        sub = c[::max(1, len(c) // 80)]
        dd  = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
        np.fill_diagonal(dd, 99.0)
        # 3.2A clash threshold for energy-based sorting
        clash = float(np.sum(np.maximum(0, 3.2 - dd) ** 2))
    return bond_err + 0.1 * clash
