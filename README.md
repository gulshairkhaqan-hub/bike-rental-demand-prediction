# 🚲 Bike Rental Demand Prediction

> Predict daily bike rental demand from weather and seasonal conditions using a Decision Tree Regressor — served through a clean, interactive Streamlit dashboard.

---

## Overview

Urban bike-sharing systems are sensitive to weather and time-of-year.  
This project builds a **Decision Tree Regression** model that estimates how many bikes will be rented on a given day based on four inputs: temperature, humidity, wind speed, and season.

The entire workflow — data exploration, model training, evaluation, and live prediction — is available through a multi-page Streamlit application.

---

## Features

- **Dataset Explorer** — preview rows, inspect data types, check for missing values and duplicates, view statistical summary
- **Visualizations** — histograms, box plots, scatter plots with trend line, and an annotated correlation heatmap
- **Model Training** — one-click training with MAE / RMSE / R² metrics, feature importance chart, and actual-vs-predicted scatter plot
- **Live Prediction** — number input form with instant demand forecast displayed in a styled card
- **Download Results** — export any prediction as a timestamped CSV
- **Assignment Example Loader** — pre-fills the required test case (`temp=0.5, humidity=0.6, windspeed=0.2, season=3`) in one click
- **Clean, responsive UI** — sidebar navigation, wide layout, consistent seaborn theme

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Core language |
| Streamlit | 1.57.0 | Web UI |
| scikit-learn | 1.8.0 | ML model, metrics, train/test split |
| pandas | 3.0.2 | Data handling |
| NumPy | 2.4.4 | Numerical computation |
| Plotly | 6.7.0 | Interactive charts |
| joblib | 1.5.3 | Model serialization |

---

## Project Structure

```
bike-rental-demand-prediction/
├── dataset/
│   └── bike_rental_100_rows.csv   # 100-record sample dataset
├── app.py                          # Streamlit UI (6 pages)
├── train_model.py                  # ML pipeline (train, evaluate, predict)
├── utils.py                        # Data utilities (load, clean, info)
├── model.pkl                       # Saved model (generated at runtime, git-ignored)
├── requirements.txt
└── README.md
```

---

## Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/bike-rental-demand-prediction.git
cd bike-rental-demand-prediction

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Train the model from the command line first
python train_model.py

# 5. Launch the Streamlit app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Dataset

| Property | Value |
|---|---|
| File | `dataset/bike_rental_100_rows.csv` |
| Records | 100 rows |
| Features used | `temp`, `humidity`, `windspeed`, `season` |
| Target | `count` (number of bike rentals) |
| Missing values | None |
| Duplicate rows | None |

**Column notes:**
- `temp` and `humidity` are normalised floats in `[0, 1]`
- `windspeed` is normalised in `[0, 1]`
- `season` is an integer: `1=Spring, 2=Summer, 3=Fall, 4=Winter`
- `count` is an integer representing total rentals for that record

> This is a **sample dataset** (100 rows). It is sufficient for demonstrating the workflow but too small for production-grade predictions.

---

## Model

**Algorithm:** `DecisionTreeRegressor` (scikit-learn)

**Hyperparameters:**

| Parameter | Value | Reason |
|---|---|---|
| `max_depth` | 5 | Prevents the tree from memorising all 80 training rows |
| `min_samples_leaf` | 3 | Requires at least 3 samples per leaf — reduces noise sensitivity |
| `random_state` | 42 | Reproducible splits and node decisions |
| `test_size` | 0.2 | 80 / 20 train-test split |

The tree is regularised because an unconstrained tree on 100 rows would perfectly overfit the training data and generalise poorly.

---

## Results

| Metric | Value |
|---|---|
| MAE | **34.17** |
| RMSE | **42.07** |
| R² Score | **0.8527** |

*Evaluated on 20 held-out test records (20 % of 100).*

**Custom prediction** (`temp=0.5, humidity=0.6, windspeed=0.2, season=3`): **~206 rentals**

---

## Screenshots

> Screenshots will be added after deployment.

---

## Future Improvements

- Collect a significantly larger dataset (thousands of records) for reliable generalisation
- Compare against Random Forest and XGBoost to quantify the benefit of ensemble methods
- Hyperparameter tuning via `GridSearchCV` or `RandomizedSearchCV`
- Add time-series features (hour of day, day of week, holiday flag)
- Cross-validation instead of a single train/test split
- Deploy to [Streamlit Cloud](https://streamlit.io/cloud) or [Hugging Face Spaces](https://huggingface.co/spaces)

---

## Author

Gul Shair  
[LinkedIn](https://www.linkedin.com/in/gulshair/) · [GitHub](https://github.com/gulshairkhaqan-hub)
