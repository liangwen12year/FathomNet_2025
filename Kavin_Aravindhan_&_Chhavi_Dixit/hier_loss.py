"""
Key Idea:
----------
- Avoid external taxonomy APIs or files.
- Approximate a hierarchical distance matrix from label names.

Heuristic:
- Distance = 0 → identical class (species / genus / family …)
- Distance = 1 → different classes but same first token (≈ family)
- Distance = 6 → different families (worst-case in Kaggle metric)

If taxonomy_map.json becomes available, you can use exact hierarchy.
For now, this fallback works for any label list.
"""

import numpy as np
import torch
from functools import lru_cache

# ---------------------------------------------------------------------------
# Distance Matrix Computation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def distance_matrix(names_tuple):
    """
    Compute an approximate hierarchical distance matrix.

    Parameters:
    ----------
    names_tuple : tuple of strings
        List of class names (fine labels).

    Returns:
    ----------
    D : numpy.ndarray of shape (n, n)
        Distance matrix where:
        - D[i, j] = 0 if same class
        - D[i, j] = 1 if first tokens match (family level)
        - D[i, j] = 6 otherwise (different family)
    """
    names = list(names_tuple)
    first = [n.split()[0] for n in names]  # Extract first token (≈ family)
    n = len(names)
    D = np.full((n, n), 6, np.float32)     # Initialize with worst-case distance (6)

    for i in range(n):
        D[i, i] = 0  # Same class → distance 0
        for j in range(i + 1, n):
            if first[i] == first[j]:
                D[i, j] = D[j, i] = 1  # Same family → distance 1
    return D

# ---------------------------------------------------------------------------
# Expected Distance Computation (Soft version of classification loss)
# ---------------------------------------------------------------------------

def expected_distance(logits, target, D):
    """
    Compute expected hierarchical distance between predictions and ground truth.

    Parameters:
    ----------
    logits : torch.Tensor of shape [B, n]
        Raw model logits for n classes.
    target : torch.Tensor of shape [B]
        Ground truth fine class indices.
    D : numpy.ndarray of shape [n, n]
        Precomputed distance matrix.

    Returns:
    ----------
    expected_distance : torch.Tensor (scalar)
        Mean expected hierarchical distance (soft-label version).
    """
    # Convert logits to softmax probabilities
    P = torch.softmax(logits, dim=1)

    # Select distance vectors for each target label
    Dt = torch.from_numpy(D[target.cpu()]).to(logits.device)

    # Compute expected distance weighted by predicted probabilities
    # Scale by Dt.max() to bring values in ~[0,1] range for loss balance
    expected_dist = (P * Dt).sum(1) / Dt.max()
    return expected_dist.mean()

# ---------------------------------------------------------------------------
# Optional: Exact Distance Matrix from taxonomy_map.json (commented for now)
# ---------------------------------------------------------------------------

# This part builds a more precise hierarchy if taxonomy_map.json is provided.
# It's optional and disabled by default.

# import json, pathlib

# _MAP = pathlib.Path(__file__).with_name("taxonomy_map.json")  # Look for JSON next to this file

# def _matrix_from_json(names):
#     """
#     Builds an exact distance matrix using taxonomy_map.json.
#     Matches competition hierarchy tree.

#     Parameters:
#     ----------
#     names : list of class names.

#     Returns:
#     ----------
#     D : numpy.ndarray of shape (n, n)
#         Exact hierarchical distance matrix based on taxonomy tree.
#     """
#     tax = json.loads(_MAP.read_text())
#     ranks = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
#     paths = {n: [tax[n][r] for r in ranks if tax[n][r]] for n in names}

#     n = len(names)
#     D = np.zeros((n, n), np.float32)

#     for i, a in enumerate(names):
#         pa = paths[a]
#         for j, b in enumerate(names):
#             if i == j:
#                 continue
#             pb = paths[b]
#             lca = sum(u == v for u, v in zip(pa, pb))  # Lowest Common Ancestor depth
#             D[i, j] = len(pa) + len(pb) - 2 * lca      # Tree distance formula

#     return D

# Alternative fallback matrix builder (robust heuristic, always works)
# def _matrix_fast(names):
#     n = len(names)
#     first = [s.split()[0] for s in names]               # First token ~ family
#     D = np.full((n, n), 6, np.float32)
#     for i in range(n):
#         D[i, i] = 0
#         for j in range(i + 1, n):
#             if first[i] == first[j]:
#                 D[i, j] = D[j, i] = 1
#     return D

# @lru_cache(maxsize=1)
# def distance_matrix(tuple_names):
#     names = list(tuple_names)
#     if _MAP.exists():
#         try:
#             return _matrix_from_json(names)
#         except Exception as e:
#             print("taxonomy_map.json unreadable – falling back:", e)
#     return _matrix_fast(names)
