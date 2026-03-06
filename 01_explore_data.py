"""
Step 2: Load and Explore the Wisconsin Breast Cancer Dataset

This script loads the dataset and performs exploratory data analysis (EDA)
to understand what we're working with before building a model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# =============================================================================
# 1. Load the dataset
# =============================================================================
data = load_breast_cancer()

# The dataset object has several attributes:
# - data: the feature matrix (numpy array)
# - target: the labels (0 = malignant, 1 = benign)
# - feature_names: names of the 30 features
# - target_names: ['malignant', 'benign']

# Let's put it into a pandas DataFrame for easier exploration
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target
df['diagnosis_label'] = df['diagnosis'].map({0: 'malignant', 1: 'benign'})

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape[0]} samples, {df.shape[1] - 2} features")
print(f"Target classes: {data.target_names}")
print(f"\nFeature names:\n{data.feature_names}")

# =============================================================================
# 2. Basic statistics
# =============================================================================
print("\n" + "=" * 60)
print("BASIC STATISTICS")
print("=" * 60)
print(df.describe().to_string())

# Check for missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# =============================================================================
# 3. Class distribution
# =============================================================================
print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)
class_counts = df['diagnosis_label'].value_counts()
print(class_counts)
print(f"\nRatio (benign/malignant): {class_counts['benign'] / class_counts['malignant']:.2f}")

# =============================================================================
# 4. Visualizations
# =============================================================================

# 4a. Class distribution bar chart
fig, ax = plt.subplots(figsize=(6, 4))
colors = ['#e74c3c', '#2ecc71']
class_counts.plot(kind='bar', color=colors, ax=ax)
ax.set_title('Class Distribution')
ax.set_ylabel('Count')
ax.set_xlabel('Diagnosis')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for i, v in enumerate(class_counts):
    ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('plot_01_class_distribution.png', dpi=150)
plt.close()
print("\nSaved: plot_01_class_distribution.png")

# 4b. Feature distributions - compare malignant vs benign for the 10 "mean" features
mean_features = [f for f in data.feature_names if 'mean' in f]
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, feature in enumerate(mean_features):
    ax = axes[i]
    for label, color in zip(['malignant', 'benign'], colors):
        subset = df[df['diagnosis_label'] == label]
        ax.hist(subset[feature], bins=20, alpha=0.6, label=label, color=color)
    ax.set_title(feature, fontsize=10)
    ax.legend(fontsize=8)

plt.suptitle('Feature Distributions: Malignant vs Benign (mean features)', fontsize=14)
plt.tight_layout()
plt.savefig('plot_02_feature_distributions.png', dpi=150)
plt.close()
print("Saved: plot_02_feature_distributions.png")

# 4c. Correlation matrix of mean features
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[mean_features].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
ax.set_title('Correlation Matrix (mean features)')
plt.tight_layout()
plt.savefig('plot_03_correlation_matrix.png', dpi=150)
plt.close()
print("Saved: plot_03_correlation_matrix.png")

print("\n" + "=" * 60)
print("EDA COMPLETE - Review the plots and statistics above.")
print("=" * 60)
