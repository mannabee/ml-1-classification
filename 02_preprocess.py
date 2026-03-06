"""
Step 3: Data Preprocessing

Before feeding data into a neural network, we need to:
1. Split into train/test sets (so we can evaluate on unseen data)
2. Scale features (neural networks train better when features are on similar scales)

Why scaling matters:
- Our features have very different ranges (e.g., mean area: 143-2501 vs mean smoothness: 0.05-0.16)
- Without scaling, features with large values dominate the learning process
- StandardScaler transforms each feature to mean=0, std=1

Why we fit the scaler on training data only:
- If we scale on the full dataset, information from the test set "leaks" into training
- The test set should simulate truly unseen data
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED, TEST_SIZE

# =============================================================================
# 1. Load the data
# =============================================================================
data = load_breast_cancer()
X = data.data       # shape: (569, 30)
y = data.target      # shape: (569,)  — 0=malignant, 1=benign

print("=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)
print(f"Full dataset: {X.shape[0]} samples, {X.shape[1]} features")

# =============================================================================
# 2. Train/test split
# =============================================================================
# 80% train, 20% test
# stratify=y ensures both sets have the same class ratio
# random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# Verify stratification preserved the class ratio
for name, labels in [("Train", y_train), ("Test", y_test)]:
    n_benign = (labels == 1).sum()
    n_malignant = (labels == 0).sum()
    print(f"  {name} — benign: {n_benign}, malignant: {n_malignant}, "
          f"ratio: {n_benign/n_malignant:.2f}")

# =============================================================================
# 3. Feature scaling (StandardScaler)
# =============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train, transform train
X_test_scaled = scaler.transform(X_test)          # only transform test (no fit!)

# Show the effect of scaling
print(f"\nBefore scaling:")
print(f"  Train mean range: [{X_train.mean(axis=0).min():.4f}, {X_train.mean(axis=0).max():.4f}]")
print(f"  Train std range:  [{X_train.std(axis=0).min():.4f}, {X_train.std(axis=0).max():.4f}]")

print(f"\nAfter scaling:")
print(f"  Train mean range: [{X_train_scaled.mean(axis=0).min():.6f}, {X_train_scaled.mean(axis=0).max():.6f}]")
print(f"  Train std range:  [{X_train_scaled.std(axis=0).min():.6f}, {X_train_scaled.std(axis=0).max():.6f}]")

print(f"\nTest set (scaled using train statistics):")
print(f"  Test mean range:  [{X_test_scaled.mean(axis=0).min():.4f}, {X_test_scaled.mean(axis=0).max():.4f}]")
print(f"  Test std range:   [{X_test_scaled.std(axis=0).min():.4f}, {X_test_scaled.std(axis=0).max():.4f}]")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
print("Data is ready for the neural network.")
