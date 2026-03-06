This is my homework:

Feladat: Építs egy egyszerű neurális hálózat alapú klasszifikációs modellt egy publikus egészségügyi adatbázison (pl. Wisconsin Breast Cancer Dataset, MNIST medical imaging subset).



Implementáld a modellt Python-ban (Keras/TensorFlow vagy PyTorch)

Számítsd ki a teljesítménymutatókat (confusion matrix, ROC curve, AUC)

Végezz cross-validation-t

Elemezd a hibásan klasszifikált eseteket

Készíts 1000-1500 szavas jelentést az eredményekről és a klinikai következményekről



Értékelési szempontok: Kód minősége és dokumentáltsága, modell teljesítménye, kritikus értékelés, vizualizációk

This is a learning objective, so instead of creating it as a whole, we should go step by step.

## Decisions Made

- **Dataset**: Wisconsin Breast Cancer Diagnostic (built-in via `sklearn.datasets.load_breast_cancer`)
- **Framework**: PyTorch (torch 2.4.1)
- **Python**: 3.8.10 via pyenv, venv in `.venv/`

## Dataset Summary

- 569 samples, 30 features (10 mean + 10 SE + 10 worst), 0 missing values
- Binary classification: benign (357) vs malignant (212), ratio ~1.68:1
- Best separating features: radius, perimeter, area, concavity, concave points
- Highly correlated feature groups: radius/perimeter/area (~0.99)

## Model

- Architecture: 30 → 64 (ReLU) → 32 (ReLU) → 1 (Sigmoid), 4,097 parameters
- Training: 100 epochs, Adam optimizer (lr=0.001), BCE loss
- Preprocessing: 80/20 stratified split, StandardScaler (fit on train only)

## Results

- Accuracy: 95.6%, AUC: 0.9964
- Confusion matrix (on 114 test samples): 41 TN, 68 TP, 1 FP, 4 FN
- Malignant recall (sensitivity): 97.6%, Benign recall (specificity): 94.4%
- Clinical concern: 4 false negatives = missed cancers (most dangerous error type)
- Cross-validation (5-fold): accuracy 97.2% ± 1.16%, AUC 99.6% ± 0.46%, F1 97.8% ± 0.96%

## Misclassification Findings

- 5 errors total: 1 false negative (missed cancer), 4 false positives
- False negative (sample 53): small, smooth malignant tumor — all key features look benign, model was 82% confident (wrong). This is the most dangerous error type.
- False positives: unusually large benign tumors or rough texture — would trigger extra tests but not miss cancer (safer error)
- All misclassified samples sit in the overlap zone between classes (borderline cases)

## Project Structure

- `01_explore_data.py` — EDA: loads data, prints stats, generates plots
- `02_preprocess.py` — data preprocessing demo (split, scaling)
- `03_train_model.py` — defines and trains the neural network, saves model.pt
- `04_evaluate.py` — confusion matrix, classification report, ROC/AUC
- `05_cross_validation.py` — 5-fold stratified cross-validation
- `06_misclassification.py` — analysis of misclassified samples
- `model.pt` — saved trained model weights + scaler params
- `plot_01_class_distribution.png` — bar chart of class counts
- `plot_02_feature_distributions.png` — histograms of mean features by class
- `plot_03_correlation_matrix.png` — heatmap of mean feature correlations
- `plot_04_training_curves.png` — loss and accuracy over epochs
- `plot_05_confusion_matrix.png` — confusion matrix heatmap
- `plot_06_roc_curve.png` — ROC curve with AUC
- `plot_07_cross_validation.png` — per-fold metric comparison
- `plot_08_misclassification.png` — confidence distribution + feature space scatter
- `jelentes.md` — final report (Hungarian, ~1050 words)
- `requirements.txt` — frozen pip dependencies

## Steps

1. ~~Project setup~~ (venv, dependencies)
2. ~~Load & explore data~~ (EDA, visualizations)
3. ~~Data preprocessing~~ (train/test split, scaling)
4. ~~Build neural network~~ (PyTorch, feedforward)
5. ~~Evaluate performance~~ (confusion matrix, ROC, AUC)
6. ~~Cross-validation~~ (5-fold, consistent results)
7. ~~Misclassification analysis~~ (5 errors, borderline cases)
8. ~~Written report~~ (jelentes.md, ~1050 words, Hungarian)