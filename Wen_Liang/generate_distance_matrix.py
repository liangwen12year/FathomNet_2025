#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetches the WoRMS ancestry for each of the 79 FathomNet categories via the
fathomnet API, builds an ete3 tree, prunes to the 7 accepted ranks, and
computes the full 79×79 pairwise taxonomic distance matrix.

Usage:
    pip install ete3 fathomnet tqdm pandas numpy
    python generate_distance_matrix.py \
      --json_path ~/.cache/kagglehub/competitions/fathomnet-2025/dataset_train.json \
      --output_npy distance_matrix.npy \
      --output_csv distance_matrix.csv
"""
import argparse
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
from ete3 import Tree

from fathomnet.api import worms

# Ranks that count as unit‐distance edges
ACCEPTED_RANKS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]


def recursive_child_snatcher(node):
    """
    Given a WormsNode whose .children form a single‐chain path down to the species,
    return (list_of_child_names, list_of_child_ranks) along that chain.
    """
    children = [c.name for c in node.children]
    ranks    = [c.rank for c in node.children]
    # We expect exactly one child at each level for a lineage tree
    assert len(children) == 1, f"Expected 1 child, got {len(children)}"
    # If that child has its own children, recurse further
    if node.children[0].children:
        sub_children, sub_ranks = recursive_child_snatcher(node.children[0])
        return children + sub_children, ranks + sub_ranks
    else:
        return children, ranks


def build_tree(category_names):
    """
    Build an ete3 Tree by fetching each category's WoRMS ancestor tree via worms.get_ancestors().
    Returns the root of the assembled taxonomy.
    """
    # Create a dummy root node
    root = Tree(name="")
    root.rank = ""
    root.dist = 0

    # Track which names we've already added
    added = set([""])

    for cat in tqdm(category_names, desc="Fetching WoRMS lineages"):
        anc = worms.get_ancestors(cat)
        # Get the down‐chain of names & ranks under this ancestor
        children, ranks = recursive_child_snatcher(anc)
        # Prepend the root placeholder
        path_names = [""] + children
        path_ranks = [""] + ranks

        # Walk the path and add any new nodes
        for parent_name, child_name, parent_rank, child_rank in zip(
            path_names, path_names[1:], path_ranks, path_ranks[1:]
        ):
            if child_name in added:
                continue
            # Find the parent node in our ete3 tree
            parent_node = next(node for node in root.traverse() if node.name == parent_name)
            # Create the child node
            child_node = Tree(name=child_name)
            child_node.rank = child_rank
            # Edge length = 1 if this rank is one of the 7 accepted, else 0
            child_node.dist = 1 if child_rank in ACCEPTED_RANKS else 0
            parent_node.add_child(child_node)
            added.add(child_name)

    return root


def tree_to_distance_matrix(tree, labels):
    """
    Given an ete3 Tree and the list of 79 category labels (leaf names),
    compute the symmetric pairwise distance matrix where each edge weight is node.dist.
    """
    labels_sorted = sorted(labels)
    # Map from name → ete3 node
    name2node = {n.name: n for n in tree.traverse()}
    n = len(labels_sorted)
    D = np.zeros((n, n), dtype=np.float32)

    for i, a in enumerate(labels_sorted):
        node_a = name2node[a]
        for j in range(i, n):
            b = labels_sorted[j]
            node_b = name2node[b]
            dist = node_a.get_distance(node_b)
            D[i, j] = dist
            D[j, i] = dist

    return pd.DataFrame(D, index=labels_sorted, columns=labels_sorted)


def main(args):
    # 1) Load the 79 category names from COCO JSON
    with open(args.json_path, "r") as f:
        data = json.load(f)
    categories = [c["name"] for c in data["categories"]]

    # 2) Build the taxonomy tree
    tree = build_tree(categories)

    # 3) Compute the 79×79 distance matrix
    df = tree_to_distance_matrix(tree, categories)

    # 4) Save outputs
    df.to_csv(args.output_csv)
    np.save(args.output_npy, df.values)

    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved NPY: {args.output_npy}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate FathomNet taxonomic distance matrix")
    p.add_argument(
        "--json_path",   type=str,
        default="dataset_train.json",
        help="Path to FathomNet train JSON"
    )
    p.add_argument(
        "--output_npy",  type=str,
        default="distance_matrix.npy",
        help="Where to save the .npy distance matrix"
    )
    p.add_argument(
        "--output_csv",  type=str,
        default="distance_matrix.csv",
        help="Where to save the CSV distance matrix"
    )
    args = p.parse_args()
    main(args)

