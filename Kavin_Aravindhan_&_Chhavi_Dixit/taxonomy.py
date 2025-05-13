"""
Taxonomy mapping utilities for FathomNet hierarchical classification.

Provides:
----------
- Fine to index mapping
- Index to fine label mapping
- Fine to coarse index mapping (using family-level heuristic)
- Index to coarse label mapping
"""

import pandas as pd
import json
import pathlib

def build_maps(csv_path: str):
    """
    Builds taxonomy mappings for fine and coarse labels.

    Parameters:
    ----------
    csv_path : str
        Path to the CSV file containing 'label' column (fine-grained labels).

    Returns:
    ----------
    name_to_idx : dict
        Mapping from fine label string ➜ integer index (0‥78).
    idx_to_name : list
        List where index ➜ fine label string.
    coarse_of_idx : list
        List of length = number of fine labels (79), where each entry is
        the coarse class index (0‥C-1).
    coarse_names : list
        List where index ➜ coarse label string.
    """
    df = pd.read_csv(csv_path)
    fine_names = sorted(df["label"].unique().tolist())

    # --------------------------------------------------------
    # Simple heuristic for grouping into coarse categories:
    # - Use the taxonomic rank *family* (first token in label string).
    #   e.g., "Gadidae Gadus morhua"  ➜  "Gadidae"
    # --------------------------------------------------------
    def coarse_name(label):
        return label.split()[0]  # Extract family-level name

    # Get sorted list of unique coarse names (families)
    coarse_names = sorted({coarse_name(name) for name in fine_names})

    # Map each coarse name to a unique ID
    coarse_to_id = {name: idx for idx, name in enumerate(coarse_names)}

    # Map fine label names to indices (0‥78)
    name_to_idx = {name: idx for idx, name in enumerate(fine_names)}

    # Map each fine label index to its corresponding coarse label index
    coarse_of_idx = [coarse_to_id[coarse_name(name)] for name in fine_names]

    return name_to_idx, fine_names, coarse_of_idx, coarse_names
