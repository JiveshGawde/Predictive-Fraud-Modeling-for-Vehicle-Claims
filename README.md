# Predictive Fraud Modeling for Vehicle Claims

An end-to-end machine learning pipeline for detecting vehicle insurance fraud on imbalanced tabular data, comparing deep learning (PyTorch) against optimized gradient boosting (XGBoost).

## Project Overview
Handling severe class imbalance (94:6 non-fraud vs. fraud) in a 15,000+ record claims dataset. The project explores the trade-offs between custom PyTorch architectures using entity embeddings and production-grade tree models tuned via Bayesian optimization (Optuna).

## Key Results & Progression

| Model Iteration | ROC-AUC | PR-AUC | Fraud Recall | Fraud Precision |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch Baseline (Weighted Sampler)** | 0.8087 | 0.1652 | 49% | 16% |
| **Un-tuned XGBoost Baseline** | 0.9588 | 0.6494 | 64% | 62% |
| **Final Optuna XGBoost** | **0.9661** | **0.6751** | **80%** | **56%** |

## Technical Highlights
* **Data Engineering:** Processed mixed tabular data by mapping categorical features into integer embeddings for neural networks while preserving ordinal/continuous columns.
* **Deep Learning Baseline:** Architected a custom PyTorch MLP (10,881 parameters) using entity embeddings and batch-level `WeightedRandomSampler` to manage class imbalance.
* **Gradient Boosting Pivot:** Transitioned to XGBoost utilizing `scale_pos_weight` to better process imbalanced tabular structures.
* **Bayesian Optimization:** Deployed an Optuna study across a 5-fold Stratified CV, heavily constraining parameters like `max_delta_step` and `min_child_weight` to prevent overfitting and achieve a production-ready 0.67 PR-AUC.

## Project Structure
```text
├── X_train.csv / X_test.csv      # Preprocessed feature splits
├── y_train.csv / y_test.csv      # Target labels
├── fraud_model.pt                # Saved PyTorch checkpoint & metadata
└── README.md
