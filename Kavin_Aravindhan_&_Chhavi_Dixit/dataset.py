import pathlib
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from timm.data import create_transform

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def build_label_maps(df):
    """
    Build label mappings.

    Parameters:
    ----------
    df : pandas DataFrame with a 'label' column.

    Returns:
    ----------
    name_to_idx : dict
        Mapping from label names to integer indices.
    idx_to_name : list
        List of label names ordered by index.
    """
    classes = sorted(df["label"].unique().tolist())
    name_to_idx = {c: i for i, c in enumerate(classes)}
    return name_to_idx, classes  # idx_to_name is simply 'classes' list

# ---------------------------------------------------------------------------
# Data Augmentation Transforms
# ---------------------------------------------------------------------------

def build_transforms(split="train", input_size=224):
    """
    Create torchvision transforms using timm's create_transform utility.

    Parameters:
    ----------
    split : str
        'train' for training transforms (with augmentation),
        otherwise validation/test transforms.
    input_size : int
        Target image resolution.

    Returns:
    ----------
    A torchvision transform pipeline.
    """
    if split == "train":
        return create_transform(
            input_size=input_size,
            is_training=True,
            no_aug=False,
            scale=(0.7, 1.0),
            ratio=(0.75, 1.33),
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )
    else:
        return create_transform(
            input_size=input_size,
            is_training=False,
            interpolation='bicubic',
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

# ---------------------------------------------------------------------------
# Main Dataset Class for FathomNet 2025
# ---------------------------------------------------------------------------

class FathomNetDataset(Dataset):
    """
    Custom Dataset for FathomNet hierarchical marine-taxon classification.

    Parameters:
    ----------
    csv_file : str or Path
        Path to annotations CSV file with columns ['file_name', 'annotation_id', 'label'].
    root_dir : str or Path
        Root directory containing 'images/' and 'rois/' folders.
    name_to_idx : dict
        Mapping from label names to integer indices.
    use_roi : bool
        If True, load ROI-cropped organism chips. Else, load full images.
    split : str
        'train' or 'val' to determine which augmentation to apply.
    input_size : int
        Target image resolution.
    """

    def __init__(self, csv_file, root_dir, name_to_idx,
                 use_roi=True, split="train", input_size=224):
        self.df = pd.read_csv(csv_file)
        self.root_dir = pathlib.Path(root_dir)
        self.use_roi = use_roi
        self.name2idx = name_to_idx
        self.tfm = build_transforms(split, input_size)

        # Determine which column provides the image path
        if use_roi:
            # Prefer explicit 'roi_path' column if present
            if "roi_path" in self.df.columns:
                self.path_col = "roi_path"
            elif "path" in self.df.columns:  # fallback for older scripts
                self.path_col = "path"
            else:
                # If missing, build roi_path from annotation_id
                assert "annotation_id" in self.df.columns, \
                    "CSV must contain 'roi_path', 'path', or 'annotation_id'"
                self.df["roi_path"] = (
                    self.root_dir / "rois" /
                    (self.df["annotation_id"].astype(str) + ".png")
                )
                self.path_col = "roi_path"
        else:
            # Use full frame images
            if "image_path" in self.df.columns:
                self.path_col = "image_path"
            elif "file_name" in self.df.columns:
                self.df["image_path"] = (
                    self.root_dir / "images" / self.df["file_name"]
                )
                self.path_col = "image_path"
            else:
                raise ValueError("CSV must contain 'image_path' or 'file_name' columns")

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns the transformed image and label for a given index.

        Parameters:
        ----------
        idx : int
            Index of the sample.

        Returns:
        ----------
        img : Tensor
            Transformed image tensor.
        label : int (optional)
            Label index if available, else only image is returned.
        """
        row = self.df.iloc[idx]
        img = Image.open(row[self.path_col]).convert("RGB")
        img = self.tfm(img)

        if "label" in row and pd.notna(row["label"]):
            label = self.name2idx[row["label"]]
            return img, label
        else:
            return img
