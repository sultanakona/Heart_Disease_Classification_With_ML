# ❤️ Heart Disease Classification with ML

[![Project](https://img.shields.io/badge/project-Heart%20Disease%20ML-blue)](https://github.com/SabihaMishu/Heart_Disease_Classification_with-_ML)
[![Language](https://img.shields.io/github/languages/top/SabihaMishu/Heart_Disease_Classification_with-_ML)](https://github.com/SabihaMishu/Heart_Disease_Classification_with-_ML)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A clean, well-documented machine learning project that predicts the presence of heart disease from patient data. This repository walks through data preprocessing, model training, evaluation, and visualizations — designed for learners, data scientists, and healthcare ML enthusiasts.

Live demo / notebook-ready for exploration — perfect for learning end-to-end ML workflows and model explainability.

---

## 🚀 Highlights

- End-to-end pipeline: data cleaning → feature engineering → model selection → evaluation → interpretation
- Multiple models: Logistic Regression, Random Forest, XGBoost, SVM, Neural Network
- Clear visualizations: feature importance, confusion matrix, ROC curves
- Reproducible experiments via notebooks and scripts
- Focus on interpretability (SHAP / feature importance)

---

## 📖 Table of Contents

- [About](#-about)
- [Dataset](#-dataset)
- [What’s included](#-whats-included)
- [Quick start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Modeling & Evaluation](#-modeling--evaluation)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 💡 About

Cardiovascular disease is one of the leading causes of death globally. Early detection using machine learning can help prioritize high-risk patients for timely intervention. This repository builds classification models to predict whether a patient has heart disease based on clinical and diagnostic features.

---

## 📦 Dataset

This project uses the widely-adopted "Heart Disease" dataset from the UCI Machine Learning Repository (often referred to as the Cleveland dataset). It contains patient attributes such as age, sex, chest pain type, blood pressure, cholesterol, max heart rate, and others.

Dataset source: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

Note: The repo includes a cleaned local copy for convenience (see the `data/` folder).

---

## 🔧 What’s included

- Notebooks:
  - `notebooks/01_exploration.ipynb` — EDA and preprocessing
  - `notebooks/02_modeling.ipynb` — Model training and evaluation
  - `notebooks/03_interpretability.ipynb` — SHAP and feature explanations
- Scripts:
  - `src/data_preprocessing.py` — Data cleaning and feature engineering
  - `src/train.py` — Train models, save artifacts
  - `src/predict.py` — Load model and predict for new samples
- Folders:
  - `data/` — datasets (cleaned)
  - `models/` — saved model artifacts
  - `results/` — evaluation reports and plots
- Config & requirements:
  - `requirements.txt` — Python dependencies
  - `environment.yml` (optional) — conda environment

---

## ⚡ Quick Start

Prerequisites:
- Python 3.8+
- pip (or conda)

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the exploratory notebook (recommended):
- Open `notebooks/01_exploration.ipynb` in Jupyter or VS Code and follow the cells.

Train models from script:
```bash
python src/train.py --data_path data/heart_cleveland_clean.csv --out_dir models/
```

Predict with a trained model:
```bash
python src/predict.py --model models/best_model.pkl --input_json sample_patient.json
```

---

## 🧠 Modeling & Evaluation

Models included:
- Logistic Regression (baseline)
- Decision Tree / Random Forest
- Support Vector Machine
- XGBoost
- Multi-Layer Perceptron (simple NN)

Evaluation metrics produced:
- Accuracy, Precision, Recall, F1-score
- ROC AUC
- Confusion Matrix
- Feature importance & SHAP explanations

To reproduce results and visualize metrics, run:
```bash
python src/train.py --data_path data/heart_cleveland_clean.csv --out_dir results/ --plot
```

Interpreting results:
- Look at `results/` for plots like ROC curves, feature importance, and SHAP summary plots to understand which features drive predictions.

---

## 🧩 Project Structure

- README.md — project overview
- data/ — raw and cleaned datasets
- notebooks/ — interactive notebooks for exploration & modeling
- src/
  - data_preprocessing.py
  - train.py
  - predict.py
  - utils.py
- models/ — saved model files (.pkl, .joblib)
- results/ — evaluation reports and visualizations

---

## ✅ Tips for Improvement

- Add cross-validation and hyperparameter tuning (GridSearchCV / Optuna)
- Add calibration plots and class imbalance handling
- Deploy as a small Flask or FastAPI service for demoing predictions
- Add unit tests for preprocessing and inference code
- Expand dataset or use external clinical datasets for better generalization

---

## 🤝 Contributing

Contributions are welcome! If you'd like to:
- Improve preprocessing
- Add new models or metrics
- Create a deployment example (API / Streamlit)
Open an issue or submit a pull request. Please follow the guidelines in `CONTRIBUTING.md` (if present).

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## ✉️ Contact

Maintainer: [SabihaMishu](https://github.com/SabihaMishu)

Feel free to open issues, feedback, or feature requests. If you find this repository useful, give it a ⭐!

---

Thank you for exploring this project — let's make heart disease prediction easier, interpretable, and more actionable with ML. ❤️
