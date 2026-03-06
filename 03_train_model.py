"""
Step 4: Build and Train the Neural Network

We build a simple feedforward (fully connected) neural network in PyTorch.

Architecture:
    Input (30 features)
    → Hidden Layer 1 (64 neurons, ReLU activation)
    → Hidden Layer 2 (32 neurons, ReLU activation)
    → Output (1 neuron, Sigmoid activation → probability of being benign)

Why this architecture:
- 30 input features → we start wider (64) and compress down (32 → 1)
- ReLU is the standard activation for hidden layers (simple, effective, avoids vanishing gradient)
- Sigmoid at output squashes the result to [0, 1] — interpreted as probability
- Binary Cross Entropy loss is the standard for binary classification
- Adam optimizer adapts learning rates per-parameter (works well out of the box)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED, TEST_SIZE, INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, NUM_EPOCHS, LEARNING_RATE

# =============================================================================
# 1. Prepare data (same as step 3)
# =============================================================================
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=data.target
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set PyTorch seed for reproducible weight initialization
torch.manual_seed(RANDOM_SEED)

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)  # shape: (455, 1)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

# =============================================================================
# 2. Define the neural network
# =============================================================================
class BreastCancerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN1_SIZE),    # input → hidden 1
            nn.ReLU(),
            nn.Linear(HIDDEN1_SIZE, HIDDEN2_SIZE),  # hidden 1 → hidden 2
            nn.ReLU(),
            nn.Linear(HIDDEN2_SIZE, 1),             # hidden 2 → output
            nn.Sigmoid()                             # probability output
        )

    def forward(self, x):
        return self.network(x)

model = BreastCancerNet()
print("Model architecture:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total trainable parameters: {total_params}")

# =============================================================================
# 3. Set up training
# =============================================================================
criterion = nn.BCELoss()          # Binary Cross Entropy
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
test_losses = []
train_accs = []
test_accs = []

print(f"\nTraining for {NUM_EPOCHS} epochs...")
print("-" * 60)

for epoch in range(NUM_EPOCHS):
    # --- Training ---
    model.train()
    optimizer.zero_grad()
    y_pred_train = model(X_train_t)
    loss = criterion(y_pred_train, y_train_t)
    loss.backward()
    optimizer.step()

    # --- Evaluation (no gradient needed) ---
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_t)
        test_loss = criterion(y_pred_test, y_test_t)

        # Accuracy: round predictions to 0 or 1, compare with true labels
        train_acc = ((y_pred_train.round() == y_train_t).sum() / len(y_train_t)).item()
        test_acc = ((y_pred_test.round() == y_test_t).sum() / len(y_test_t)).item()

    train_losses.append(loss.item())
    test_losses.append(test_loss.item())
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f} | "
              f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# =============================================================================
# 5. Plot training curves
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
ax1.plot(train_losses, label='Train Loss', color='#3498db')
ax1.plot(test_losses, label='Test Loss', color='#e74c3c')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Binary Cross Entropy Loss')
ax1.set_title('Loss Over Training')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy curves
ax2.plot(train_accs, label='Train Accuracy', color='#3498db')
ax2.plot(test_accs, label='Test Accuracy', color='#e74c3c')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy Over Training')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_04_training_curves.png', dpi=150)
plt.close()
print("\nSaved: plot_04_training_curves.png")

# =============================================================================
# 6. Save the model
# =============================================================================
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_mean': scaler.mean_,
    'scaler_scale': scaler.scale_,
}, 'model.pt')
print("Saved: model.pt")

print(f"\nFinal Test Accuracy: {test_accs[-1]:.4f}")
