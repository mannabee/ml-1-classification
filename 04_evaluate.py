"""
Step 5: Evaluate Model Performance

We compute the standard metrics for binary classification:
1. Confusion Matrix — shows true/false positives and negatives
2. Classification Report — precision, recall, F1-score per class
3. ROC Curve — plots true positive rate vs false positive rate at all thresholds
4. AUC Score — area under the ROC curve (1.0 = perfect, 0.5 = random guessing)

In a medical context:
- False Negative (predicting benign when actually malignant) is DANGEROUS — missed cancer
- False Positive (predicting malignant when actually benign) causes unnecessary anxiety/procedures
- We care especially about recall for malignant class (catching all cancers)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)

from config import RANDOM_SEED, TEST_SIZE, INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE

# =============================================================================
# 1. Recreate data + model (same setup as training script)
# =============================================================================
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=data.target
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

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

# Load trained model
model = BreastCancerNet()
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# =============================================================================
# 2. Get predictions
# =============================================================================
with torch.no_grad():
    y_prob = model(X_test_t).numpy().flatten()   # probabilities
    y_pred = (y_prob >= 0.5).astype(int)          # binary predictions

# =============================================================================
# 3. Confusion Matrix
# =============================================================================
cm = confusion_matrix(y_test, y_pred)
# Labels: 0=malignant, 1=benign

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
print(f"                  Predicted")
print(f"                  Malignant  Benign")
print(f"Actual Malignant     {cm[0,0]:3d}      {cm[0,1]:3d}")
print(f"Actual Benign        {cm[1,0]:3d}      {cm[1,1]:3d}")

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'],
            annot_kws={'size': 16})
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('Confusion Matrix', fontsize=14)
plt.tight_layout()
plt.savefig('plot_05_confusion_matrix.png', dpi=150)
plt.close()
print("\nSaved: plot_05_confusion_matrix.png")

# =============================================================================
# 4. Classification Report
# =============================================================================
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
report = classification_report(y_test, y_pred,
                                target_names=['Malignant', 'Benign'])
print(report)

# =============================================================================
# 5. ROC Curve and AUC
# =============================================================================
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig('plot_06_roc_curve.png', dpi=150)
plt.close()

print("=" * 60)
print("ROC / AUC")
print("=" * 60)
print(f"AUC Score: {roc_auc:.4f}")
print("Saved: plot_06_roc_curve.png")

# =============================================================================
# 6. Clinical interpretation
# =============================================================================
tn, fp, fn, tp = cm.ravel()
print("\n" + "=" * 60)
print("CLINICAL INTERPRETATION")
print("=" * 60)
print(f"True Negatives  (correctly identified malignant): {tn}")
print(f"True Positives  (correctly identified benign):    {tp}")
print(f"False Positives (benign predicted as malignant):  {fp} — unnecessary further testing")
print(f"False Negatives (malignant predicted as benign):  {fn} — MISSED CANCER (most dangerous)")
print(f"\nMalignant recall (sensitivity): {tn/(tn+fp):.4f} — how many cancers we catch")
print(f"Benign recall (specificity):    {tp/(tp+fn):.4f} — how many healthy we correctly clear")
