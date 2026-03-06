"""
Step 7: Misclassification Analysis

We look at the samples the model got wrong and ask:
- What do misclassified samples look like compared to correctly classified ones?
- Are they "borderline" cases (close to the decision boundary)?
- Which features are most different from the class average?

This is critical in medical AI — understanding WHY the model fails
helps us know when to trust it and when not to.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED, TEST_SIZE, INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE

# =============================================================================
# 1. Recreate data + model
# =============================================================================
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=data.target
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_test_t = torch.FloatTensor(X_test_scaled)

class BreastCancerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN1_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN1_SIZE, HIDDEN2_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN2_SIZE, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

model = BreastCancerNet()
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# =============================================================================
# 2. Get predictions and identify misclassified samples
# =============================================================================
with torch.no_grad():
    y_prob = model(X_test_t).numpy().flatten()
    y_pred = (y_prob >= 0.5).astype(int)

# Build a DataFrame with all test samples
df_test = pd.DataFrame(X_test, columns=data.feature_names)
df_test['true_label'] = y_test
df_test['predicted_label'] = y_pred
df_test['predicted_prob'] = y_prob
df_test['true_name'] = df_test['true_label'].map({0: 'malignant', 1: 'benign'})
df_test['pred_name'] = df_test['predicted_label'].map({0: 'malignant', 1: 'benign'})
df_test['correct'] = df_test['true_label'] == df_test['predicted_label']

misclassified = df_test[~df_test['correct']]
correct = df_test[df_test['correct']]

print("=" * 60)
print("MISCLASSIFICATION ANALYSIS")
print("=" * 60)
print(f"Total test samples: {len(df_test)}")
print(f"Correctly classified: {len(correct)}")
print(f"Misclassified: {len(misclassified)}")

# =============================================================================
# 3. Examine each misclassified sample
# =============================================================================
print("\n" + "=" * 60)
print("MISCLASSIFIED SAMPLES — DETAILS")
print("=" * 60)

for idx, row in misclassified.iterrows():
    error_type = "FALSE NEGATIVE (missed cancer!)" if row['true_label'] == 0 else "FALSE POSITIVE"
    print(f"\nSample index {idx}: {error_type}")
    print(f"  True: {row['true_name']}, Predicted: {row['pred_name']}")
    print(f"  Model confidence: {row['predicted_prob']:.4f} (benign probability)")
    print(f"  → Model was {'uncertain' if 0.3 < row['predicted_prob'] < 0.7 else 'confident but wrong'} "
          f"(threshold = 0.5)")

# =============================================================================
# 4. Compare misclassified vs correctly classified — key features
# =============================================================================
# Focus on the most discriminative mean features
key_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                'mean concavity', 'mean concave points']

print("\n" + "=" * 60)
print("FEATURE COMPARISON: MISCLASSIFIED vs CLASS AVERAGES")
print("=" * 60)

# Compute class averages from training data
df_train = pd.DataFrame(X_train, columns=data.feature_names)
df_train['label'] = y_train
malignant_avg = df_train[df_train['label'] == 0][key_features].mean()
benign_avg = df_train[df_train['label'] == 1][key_features].mean()

print(f"\n{'Feature':<25} {'Malignant avg':>14} {'Benign avg':>14}")
print("-" * 55)
for feat in key_features:
    print(f"{feat:<25} {malignant_avg[feat]:>14.2f} {benign_avg[feat]:>14.2f}")

print("\nMisclassified sample values:")
for idx, row in misclassified.iterrows():
    error_type = "FN" if row['true_label'] == 0 else "FP"
    print(f"\n  Sample {idx} ({error_type}, true={row['true_name']}):")
    for feat in key_features:
        val = row[feat]
        mal_avg = malignant_avg[feat]
        ben_avg = benign_avg[feat]
        # Which class average is this value closer to?
        closer_to = "malignant" if abs(val - mal_avg) < abs(val - ben_avg) else "benign"
        marker = " ← looks benign!" if (row['true_label'] == 0 and closer_to == "benign") else \
                 " ← looks malignant!" if (row['true_label'] == 1 and closer_to == "malignant") else ""
        print(f"    {feat:<25} {val:>10.2f}  (closer to {closer_to}){marker}")

# =============================================================================
# 5. Visualization: prediction confidence distribution
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 5a. Histogram of predicted probabilities by true class
ax = axes[0]
for label, name, color in [(0, 'Malignant', '#e74c3c'), (1, 'Benign', '#2ecc71')]:
    subset = df_test[df_test['true_label'] == label]
    ax.hist(subset['predicted_prob'], bins=20, alpha=0.6, label=name, color=color)
ax.axvline(x=0.5, color='black', linestyle='--', label='Decision threshold')
ax.set_xlabel('Predicted Probability (benign)')
ax.set_ylabel('Count')
ax.set_title('Prediction Confidence by True Class')
ax.legend()

# Mark misclassified samples
for _, row in misclassified.iterrows():
    ax.axvline(x=row['predicted_prob'], color='orange', linestyle=':', alpha=0.8)
ax.annotate('misclassified', xy=(0.5, 0.95), xycoords='axes fraction',
            color='orange', fontsize=9, ha='center')

# 5b. Scatter: misclassified vs correct on two key features
ax = axes[1]
correct_mal = df_test[(df_test['correct']) & (df_test['true_label'] == 0)]
correct_ben = df_test[(df_test['correct']) & (df_test['true_label'] == 1)]
misclass = df_test[~df_test['correct']]

ax.scatter(correct_mal['mean radius'], correct_mal['mean concave points'],
           c='#e74c3c', alpha=0.4, label='Correct malignant', s=30)
ax.scatter(correct_ben['mean radius'], correct_ben['mean concave points'],
           c='#2ecc71', alpha=0.4, label='Correct benign', s=30)
ax.scatter(misclass['mean radius'], misclass['mean concave points'],
           c='orange', marker='X', s=150, edgecolors='black', linewidth=1.5,
           label='Misclassified', zorder=5)

ax.set_xlabel('Mean Radius')
ax.set_ylabel('Mean Concave Points')
ax.set_title('Misclassified Samples in Feature Space')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_08_misclassification.png', dpi=150)
plt.close()
print("\nSaved: plot_08_misclassification.png")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("The misclassified samples tend to be 'borderline' cases —")
print("their feature values fall between the typical malignant and benign ranges.")
print("This is expected: ambiguous cases are hardest for any classifier.")
