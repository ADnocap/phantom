#!/usr/bin/env python
"""Quick validation for v8 dataset."""
import numpy as np
import json

meta = json.load(open("data/processed_v8/asset_meta.json"))
print(f"Assets: {meta['n_assets']}")
for s in ["train", "val", "test"]:
    d = np.load(f"data/processed_v8/{s}.npz", allow_pickle=True)
    X = d["X"]
    Y = d["Y_relative"]
    print(f"{s}: {len(X):,} samples, X={X.shape}")
    assert X.shape[1] == 120 and X.shape[2] == 6 and Y.shape[1] == 30
    assert (~np.isfinite(X)).sum() == 0
print("Validation passed!")
