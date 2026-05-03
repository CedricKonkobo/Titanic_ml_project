# Titanic Survival Prediction  ML Pipeline

End-to-end machine learning project that predicts passenger survival on the Titanic dataset: exploratory analysis, feature engineering, model comparison (, evaluation artifacts, and a small **REST API** for inference.


---

## Highlights

- **Pipeline**: Raw data → engineered features (`ColumnTransformer` + imputation + scaling / encoding) → cross-validated model selection → hyperparameter tuning → persisted model + metadata.
- **Models**: Baselines (logistic regression, random forest, SVM, XGBoost) with **GridSearchCV** and stratified splits.
- **Quality**: Metrics and plots saved under `reports/figures/` (confusion matrix, ROC, feature importance).
- **Serving**: Flask API with health check and `/predict` endpoint.

---

## Tech Stack

| Area | Tools |
|------|--------|
| Core | Python, pandas, NumPy |
| ML | scikit-learn, XGBoost |
| Viz | Matplotlib, Seaborn |
| API | Flask, Flask-CORS |
| Optional | MLflow, Evidently, SHAP (see `requirements.txt`) |

---

## Repository Structure

```
├── data/
│   ├── raw/           # Place Kaggle `train.csv` here (not committed)
│   └── processed/     # Processed CSVs written by scripts
├── models/            # Trained artifacts (*.joblib gitignored by default)
├── notebooks/         # EDA and experiments
├── reports/figures/   # Evaluation plots
├── src/
│   ├── api/           # Flask inference service
│   ├── data/          # Load / save helpers
│   ├── features/      # Feature engineering
│   └── models/        # Training & evaluation
├── requirements.txt
├── setup.py
└── README.md
```

---

## Setup

### 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
pip install -e .
```

### 2. Data

Download the Titanic dataset from [Kaggle — Titanic Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data) and save **`train.csv`** as:

`data/raw/train.csv`

(CSV files are ignored by `.gitignore` to keep the repo lightweight.)

---

## Train & Evaluate

From the project root:

```bash
python src/models/train_model.py
```

This loads data, builds features, compares models, tunes hyperparameters, saves `models/best_model.joblib` and `models/model_metadata.json`, and refreshes figures under `reports/figures/`.

---

## Run the API

Requires a trained `models/best_model.joblib` (run training first).

```bash
python src/api/api.py
```

- **Health**: `GET http://localhost:5000/health`
- **Predict**: `POST http://localhost:5000/predict` with JSON body (see docstring in `src/api/api.py` for expected fields).


---

## License

This project is released under the **MIT License** — see `LICENSE`.

---

## Author

Portfolio / learning project. Replace the placeholder author in `setup.py` with your name when publishing.
