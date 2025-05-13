#!/usr/bin/env python3
import os
import kagglehub
import zipfile
import subprocess

def safe_unzip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

def download_fathomnet_2025():
    print("[INFO] Logging into KaggleHub...")
    kagglehub.login()

    print("[INFO] Downloading FathomNet 2025 competition files...")
    comp_dir = kagglehub.competition_download('fathomnet-2025')
    print(f"[INFO] Dataset downloaded to: {comp_dir}")

    expected_files = [
        "dataset_train.json",
        "dataset_test.json",
        "annotations.csv",
        "download.py",
        "sample_submission.csv"
    ]

    print("[INFO] Unzipping all zip files if any...")
    for item in os.listdir(comp_dir):
        full_path = os.path.join(comp_dir, item)
        if item.endswith(".zip"):
            print(f"[INFO] Unzipping: {item}")
            safe_unzip(full_path, comp_dir)

    print("[INFO] Contents of dataset directory:")
    for item in os.listdir(comp_dir):
        print(f"  - {item}")
        if item not in expected_files and not item.endswith(".zip"):
            print(f"  [!] Unexpected file: {item}")

    print("[INFO] Done. Dataset ready in:", comp_dir)

if __name__ == "__main__":
    download_fathomnet_2025()

