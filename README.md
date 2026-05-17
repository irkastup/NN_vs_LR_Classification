# Fitness Classification — EDA, Classical ML, Neural Network & Dimensionality Reduction

This project builds a complete machine-learning workflow for a **binary fitness classification** task using a synthetic real-world style fitness dataset.

The target variable is:

```text
is_fit
1 = fit
0 = not fit
```

The project is split into two notebook parts and three supporting Python modules implementing selected algorithms from scratch with NumPy.

---

## Project Overview

The goal is to predict whether a person is fit based on health and lifestyle features such as:

- age
- height
- weight
- heart rate
- blood pressure
- sleep hours
- nutrition quality
- activity index
- smoking status
- gender

The workflow includes:

1. Exploratory data analysis
2. Data cleaning and preprocessing
3. Feature engineering and feature selection
4. PCA and t-SNE visualisation
5. Baseline modelling with Dummy Classifier and Logistic Regression
6. Custom neural-network training
7. Regularisation and early stopping experiments
8. Hyperparameter tuning
9. Representation-learning analysis

---

## Repository Structure

```text
.
├── part1_eda_pipeline.ipynb       # EDA, preprocessing, feature engineering, PCA/t-SNE visualisation
├── part2_model.ipynb              # Modelling, neural network training, evaluation, tuning
├── nn_multiclass_student.py       # Custom NumPy neural network implementation
├── pca_svd_student.py             # PCA and Truncated SVD from scratch
├── dim_viz_student.py             # t-SNE implementation from scratch
├── fitness_dataset.csv            # Dataset file expected by the notebooks
└── artifacts/                     # Generated intermediate files
    └── part1_data.pkl             # Saved processed data from Part 1, used in Part 2
```

> Note: `fitness_dataset.csv` is expected to be available in the project root before running Part 1.

---

## Dataset

Dataset used in the notebooks:

```text
Real-World Style Fitness Classification Dataset (Synthetic)
Source: Kaggle
Task: Binary classification
Target: is_fit
```

The dataset contains **2,000 rows and 11 columns**. During EDA, the project identifies:

- mixed values in the `smokes` column, such as `yes`, `no`, `1`, and `0`
- missing values in `sleep_hours`
- potential outliers in `weight_kg`
- a moderately imbalanced target distribution of approximately 60/40

---

## Main Steps

### 1. Exploratory Data Analysis

The first notebook performs an initial inspection of the dataset, including:

- dataset shape and column types
- target distribution
- numerical feature distributions
- categorical feature analysis
- missing-value analysis
- outlier checks

The `smokes` column is normalised into a binary numerical feature.

---

### 2. Train/Validation/Test Split

The data is split using a stratified split:

```text
Train:      1,400 samples
Validation:   300 samples
Test:         300 samples
```

The class distribution is preserved across the splits.

---

### 3. Preprocessing

The preprocessing pipeline handles:

- numerical columns
- categorical encoding for `gender`
- missing-value treatment
- scaling
- feature engineering

After preprocessing, the model input contains **14 features**.

---

### 4. Feature Engineering and Selection

The project adds engineered features and uses mutual information to inspect feature relevance.

---

### 5. Dimensionality Reduction

The project includes custom implementations of:

- **PCA** using covariance matrix eigen-decomposition
- **Truncated SVD** using `np.linalg.svd`
- **t-SNE** using Gaussian high-dimensional affinities, Student-t low-dimensional affinities, and gradient descent

These are used for visualising the processed data and comparing raw input representations with learned neural-network representations.

---

## Models

### Dummy Classifier

A `most_frequent` Dummy Classifier is used as a basic baseline.

Validation results:

```text
Accuracy:  0.6000
Precision: 0.0000
Recall:    0.0000
F1:        0.0000
ROC-AUC:   0.5000
```

---

### Logistic Regression

Logistic Regression provides a strong classical ML baseline.

Validation results:

```text
Accuracy:  0.7633
Precision: 0.7290
Recall:    0.6500
F1:        0.6872
ROC-AUC:   0.8520
```

---

### Custom Neural Network

The neural network is implemented from scratch in NumPy and supports:

- arbitrary depth
- ReLU, tanh, sigmoid, and softmax activations
- binary classification with sigmoid output
- multiclass classification with softmax output
- cross-entropy loss
- backpropagation
- gradient-descent parameter updates
- sklearn-compatible `get_params`, `set_params`, and `score`

The best validation neural-network variant from the regularisation experiments achieved:

```text
Accuracy:  0.7900
Precision: 0.7568
Recall:    0.7000
F1:        0.7273
ROC-AUC:   0.8606
```

---

## Final Test Results

The best baseline neural network from Part 6 performed better than the tuned grid-search neural network on the test set.

### Baseline Neural Network

```text
Accuracy:  0.7833
Precision: 0.7723
Recall:    0.6500
F1:        0.7059
ROC-AUC:   0.8422
```

### Tuned Neural Network

Best grid-search parameters:

```python
{
    "layer_sizes": [14, 128, 64, 1],
    "activations": ["tanh", "tanh", "sigmoid"],
    "lr": 0.01,
    "n_epochs": 500
}
```

Test results:

```text
Accuracy:  0.7567
Precision: 0.7640
Recall:    0.5667
F1:        0.6507
ROC-AUC:   0.8426
```

The tuned model slightly improved ROC-AUC but reduced F1, recall, and accuracy compared with the baseline neural network.

---

## Representation Learning

The project compares raw processed features with neural-network hidden-layer representations.

A logistic-regression probe was trained on both representations:

| Representation | Test Accuracy | Test ROC-AUC | Test F1 |
|---|---:|---:|---:|
| Raw input after pipeline | 0.7800 | 0.8470 | 0.6972 |
| NN last hidden layer | 0.7733 | 0.8406 | 0.6881 |

The raw processed input performed slightly better than the last hidden layer for the linear probe.

The t-SNE comparison produced:

```text
Final KL on raw features:     0.9597
Final KL on hidden features:  0.8443
```

This suggests that the learned hidden representation produced a somewhat cleaner t-SNE embedding according to KL divergence, even though it did not improve the linear-probe metrics.

---

## From-Scratch Implementations

### `nn_multiclass_student.py`

Implements a general neural-network classifier with:

- Xavier-style weight initialisation
- forward propagation
- categorical cross-entropy and binary cross-entropy
- backpropagation
- gradient descent updates
- prediction and probability methods
- sklearn compatibility

### `pca_svd_student.py`

Implements:

- PCA with covariance matrix eigen-decomposition
- dimensionality projection
- inverse reconstruction
- reconstruction error
- Truncated SVD
- explained variance ratios

### `dim_viz_student.py`

Implements a simplified t-SNE pipeline:

- pairwise squared Euclidean distances
- Gaussian affinities
- perplexity-based bandwidth search
- symmetrised joint probability matrix `P`
- Student-t low-dimensional similarity matrix `Q`
- KL-divergence gradient
- gradient descent with momentum

---

## Installation

Create and activate a virtual environment, then install the required packages.

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
```

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

If the notebooks use UMAP comparison, also install:

```bash
pip install umap-learn
```

---

## How to Run

1. Place `fitness_dataset.csv` in the project root.
2. Start Jupyter:

```bash
jupyter notebook
```

3. Run the notebooks in this order:

```text
1. part1_eda_pipeline.ipynb
2. part2_model.ipynb
```

Part 1 creates the processed-data artifact used by Part 2:

```text
artifacts/part1_data.pkl
```

---

## Example Usage of the Custom Neural Network

```python
from nn_multiclass_student import NeuralNetwork

model = NeuralNetwork(
    layer_sizes=[14, 64, 1],
    activations=["tanh", "sigmoid"],
    random_state=42
)

model.fit(X_train, y_train, lr=0.01, n_epochs=1000)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

---

## Key Findings

- Logistic Regression is a strong baseline for this dataset.
- The custom neural network improves validation F1 compared with Logistic Regression.
- Hyperparameter tuning does not always improve final test-set performance.
- The baseline neural network achieved the best test F1 among the neural-network variants tested.
- Learned hidden-layer representations gave lower t-SNE KL divergence but did not outperform raw processed features in the linear-probe experiment.
- Accuracy alone is not sufficient because the target is moderately imbalanced; F1, recall, precision, and ROC-AUC provide a fuller evaluation.

---

## License

This project was created for educational purposes.
