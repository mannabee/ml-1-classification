# Breast Cancer Classification with Neural Networks

A PyTorch neural network that classifies breast tumors as malignant or benign using the Wisconsin Breast Cancer Diagnostic dataset.

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 97.2% (5-fold CV mean) |
| AUC | 0.996 |
| F1 Score | 0.978 |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the scripts in order:

```bash
python 01_explore_data.py       # EDA and visualizations
python 02_preprocess.py         # Data splitting and scaling demo
python 03_train_model.py        # Train the neural network (saves model.pt)
python 04_evaluate.py           # Confusion matrix, ROC curve, AUC
python 05_cross_validation.py   # 5-fold stratified cross-validation
python 06_misclassification.py  # Analysis of misclassified samples
```

Scripts 04 and 06 require `model.pt` from script 03. All other scripts are independent.

## Configuration

All hyperparameters are in `config.py`:

```python
RANDOM_SEED = 42
TEST_SIZE = 0.2
INPUT_SIZE = 30
HIDDEN1_SIZE = 64
HIDDEN2_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
```

## Model Architecture

```
Input (30) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1) → Sigmoid
```

4,097 trainable parameters.

## Generated Outputs

| File | Description |
|------|-------------|
| `model.pt` | Trained model weights |
| `plot_01_class_distribution.png` | Class balance bar chart |
| `plot_02_feature_distributions.png` | Feature histograms by class |
| `plot_03_correlation_matrix.png` | Feature correlation heatmap |
| `plot_04_training_curves.png` | Loss and accuracy over epochs |
| `plot_05_confusion_matrix.png` | Confusion matrix |
| `plot_06_roc_curve.png` | ROC curve with AUC |
| `plot_07_cross_validation.png` | Per-fold metric comparison |
| `plot_08_misclassification.png` | Misclassified sample analysis |
| `jelentes.md` | Written report (Hungarian) |
