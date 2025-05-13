"""
Sanity check script for verifying dataset integrity and basic statistics.

This script checks:
- Number of rows in the annotations CSV
- Number of unique classes (labels)
- Number of image files and ROI (Region of Interest) files present
- Basic distribution statistics of samples per class
"""

import pandas as pd
import pathlib

# Path to the annotations CSV file
csv_path = "fgvc-comp-2025/data/train/annotations.csv"
root = pathlib.Path(csv_path).parent

# Load the annotations CSV into a DataFrame
df = pd.read_csv(csv_path)

# Print basic dataset statistics
print("Total training samples:", len(df))                        # Expected: 23,699
print("Number of unique classes (labels):", df['label'].nunique())  # Expected: 79

# Count the number of image files present in the 'images' directory
print("Number of image files:", len(list((root / 'images').glob('*'))))

# Count the number of ROI (Region of Interest) files present in the 'rois' directory
print("Number of ROI files:", len(list((root / 'rois').glob('*'))))

# Display descriptive statistics of the number of samples per class
print(df.groupby('label').size().describe())
