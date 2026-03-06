"""
Step 6: Cross-Validation

Why cross-validation?
- A single train/test split can be "lucky" or "unlucky" depending on which samples
  end up in which set
- K-fold CV splits the data into K parts, trains K separate models, each time
  using a different part as the test set
- This gives us K performance scores → we can compute mean and std deviation
- More reliable estimate of how the model truly performs

We use Stratified K-Fold (K=5):
- "Stratified" preserves the class ratio in each fold (important with imbalanced data)
- 5 folds means each model trains on 80% and tests on 20% — same ratio as before
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from config import RANDOM_SEED, INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, NUM_EPOCHS, LEARNING_RATE

# =============================================================================
# 1. Load data
# =============================================================================
data = load_breast_cancer()
X = data.data
y = data.target

# =============================================================================
# 2. Define model factory (fresh model for each fold)
# =============================================================================
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

def train_and_evaluate(X_train, y_train, X_test, y_test, epochs=100, seed=RANDOM_SEED):
    """Train a fresh model on one fold and return metrics."""
    torch.manual_seed(seed)
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_s)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test_s)

    # Fresh model
    model = BreastCancerNet()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_prob = model(X_test_t).numpy().flatten()
        y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    return acc, auc_score, f1

# =============================================================================
# 3. Run 5-fold cross-validation
# =============================================================================
K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=RANDOM_SEED)

accuracies = []
aucs = []
f1_scores = []

print("=" * 60)
print(f"{K}-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 60)
print(f"{'Fold':<6} {'Accuracy':<12} {'AUC':<12} {'F1 Score':<12}")
print("-" * 42)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    acc, auc_score, f1 = train_and_evaluate(X_train, y_train, X_test, y_test, seed=RANDOM_SEED + fold)

    accuracies.append(acc)
    aucs.append(auc_score)
    f1_scores.append(f1)

    print(f"  {fold:<4} {acc:<12.4f} {auc_score:<12.4f} {f1:<12.4f}")

print("-" * 42)
print(f"  Mean {np.mean(accuracies):<12.4f} {np.mean(aucs):<12.4f} {np.mean(f1_scores):<12.4f}")
print(f"  Std  {np.std(accuracies):<12.4f} {np.std(aucs):<12.4f} {np.std(f1_scores):<12.4f}")

# =============================================================================
# 4. Visualize results
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(1, K + 1)
width = 0.25

bars1 = ax.bar(x - width, accuracies, width, label='Accuracy', color='#3498db')
bars2 = ax.bar(x, aucs, width, label='AUC', color='#2ecc71')
bars3 = ax.bar(x + width, f1_scores, width, label='F1 Score', color='#e74c3c')

ax.set_xlabel('Fold')
ax.set_ylabel('Score')
ax.set_title(f'{K}-Fold Cross-Validation Results')
ax.set_xticks(x)
ax.set_ylim([0.9, 1.01])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add mean lines
ax.axhline(y=np.mean(accuracies), color='#3498db', linestyle='--', alpha=0.5)
ax.axhline(y=np.mean(aucs), color='#2ecc71', linestyle='--', alpha=0.5)
ax.axhline(y=np.mean(f1_scores), color='#e74c3c', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('plot_07_cross_validation.png', dpi=150)
plt.close()
print("\nSaved: plot_07_cross_validation.png")

print(f"\nConclusion: The model consistently achieves ~{np.mean(accuracies)*100:.1f}% accuracy")
print(f"across all {K} folds, confirming the single-split result was not a fluke.")
