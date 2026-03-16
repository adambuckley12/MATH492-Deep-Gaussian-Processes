"""
data/data_loader.py — Load the Kin8nm dataset from OpenML.

Kin8nm: 8,192 observations, 8 features (forward kinematics of an 8-link robot arm).
Target: distance of the end-effector from a desired position.
"""

from __future__ import annotations

import os
import pandas as pd


def load_kin8nm(dataset_dir: str) -> pd.DataFrame:
    """Load Kin8nm dataset, downloading from OpenML if not cached."""
    csv_path = os.path.join(dataset_dir, "kin8nm.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    from sklearn.datasets import fetch_openml
    data = fetch_openml(name="kin8nm", version=1, as_frame=True, parser="auto")
    df = data.data.copy()
    df["y"] = data.target.astype(float)
    os.makedirs(dataset_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"  Downloaded Kin8nm → {csv_path}")
    return df
