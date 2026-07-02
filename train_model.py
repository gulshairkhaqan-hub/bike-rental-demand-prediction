"""Machine learning pipeline for the Bike Rental Demand Prediction project.

This module provides modular, reusable functions covering the full training
workflow: feature/target preparation, train/test splitting, model training,
evaluation, feature importance, single-sample prediction, and model
persistence. Data-handling helpers are imported from :mod:`utils`.

Run this file directly to train a model on the bundled dataset and persist it
to ``model.pkl``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

import utils

# Columns used to train the model and the target variable.
FEATURE_COLUMNS: list[str] = ["temp", "humidity", "windspeed", "season"]
TARGET_COLUMN: str = "count"


def prepare_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix X and target series y.

    Args:
        df: The cleaned input DataFrame.

    Returns:
        A tuple ``(X, y)`` where ``X`` is a DataFrame of feature columns and
        ``y`` is the target Series.

    Raises:
        ValueError: If any required feature or target column is missing.
    """
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(required_columns)}."
        )

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into train and test sets.

    Args:
        X: Feature matrix.
        y: Target series.
        test_size: Proportion of the data to allocate to the test set.
        random_state: Seed for reproducible splitting.

    Returns:
        A tuple ``(X_train, X_test, y_train, y_test)``.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int = 5,
    min_samples_leaf: int = 3,
    random_state: int = 42,
) -> DecisionTreeRegressor:
    """Train a DecisionTreeRegressor on the training data.

    Args:
        X_train: Training feature matrix.
        y_train: Training target series.
        max_depth: Maximum depth of the tree.
        min_samples_leaf: Minimum number of samples required at a leaf node.
        random_state: Seed for reproducibility.

    Returns:
        The trained DecisionTreeRegressor.
    """
    # This dataset only has 100 rows. An unconstrained decision tree would grow
    # until it perfectly memorizes the training data (each leaf a single row),
    # overfitting badly and generalizing poorly. Capping ``max_depth`` and
    # requiring a minimum number of samples per leaf regularizes the tree and
    # keeps it from fitting noise in such a small dataset.
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: DecisionTreeRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate a trained model on the test set.

    Args:
        model: The trained regressor.
        X_test: Test feature matrix.
        y_test: True target values for the test set.

    Returns:
        A dictionary with keys:
            - ``mae``: Mean absolute error.
            - ``rmse``: Root mean squared error.
            - ``r2``: R-squared score.
            - ``y_pred``: The array of predictions (for plotting later).
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    # Compute RMSE via np.sqrt(MSE) rather than relying on the ``squared``
    # argument, which is not available across all sklearn versions.
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
        "y_pred": y_pred,
    }


def get_feature_importance(
    model: DecisionTreeRegressor,
    feature_names: list,
) -> pd.DataFrame:
    """Build a sorted feature importance DataFrame.

    Args:
        model: The trained model exposing ``feature_importances_``.
        feature_names: Names of the features, in training order.

    Returns:
        A DataFrame with columns ``feature`` and ``importance``, sorted in
        descending order of importance.
    """
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )
    return importance_df.sort_values(
        by="importance", ascending=False
    ).reset_index(drop=True)


def predict_single(
    model: DecisionTreeRegressor,
    temp: float,
    humidity: float,
    windspeed: float,
    season: int,
) -> float:
    """Predict bike rental count for a single set of input features.

    Args:
        model: The trained regressor.
        temp: Normalized temperature, expected in [0, 1].
        humidity: Normalized humidity, expected in [0, 1].
        windspeed: Normalized wind speed, expected in [0, 1].
        season: Season indicator, an integer in [1, 4].

    Returns:
        The predicted rental count as a float.

    Raises:
        ValueError: If any input fails validation.
    """
    for name, value in (("temp", temp), ("humidity", humidity), ("windspeed", windspeed)):
        if not 0 <= value <= 1:
            raise ValueError(
                f"'{name}' must be between 0 and 1, got {value}."
            )

    if not isinstance(season, int) or isinstance(season, bool) or not 1 <= season <= 4:
        raise ValueError(
            f"'season' must be an integer between 1 and 4, got {season!r}."
        )

    # Build a single-row DataFrame with the exact column order used in training.
    sample = pd.DataFrame(
        [[temp, humidity, windspeed, season]],
        columns=FEATURE_COLUMNS,
    )
    prediction = model.predict(sample)
    return float(prediction[0])


def save_model(model: DecisionTreeRegressor, filepath: str = "model.pkl") -> None:
    """Persist a trained model to disk using joblib.

    Args:
        model: The trained model to save.
        filepath: Destination path for the serialized model.
    """
    joblib.dump(model, filepath)


def load_model(filepath: str = "model.pkl") -> DecisionTreeRegressor:
    """Load a trained model from disk using joblib.

    Args:
        filepath: Path to the serialized model file.

    Returns:
        The loaded model.

    Raises:
        FileNotFoundError: If no model file exists at ``filepath``.
    """
    try:
        return joblib.load(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model file not found at: {filepath}. Train the model first."
        )


if __name__ == "__main__":
    DATASET_PATH = "dataset/bike_rental_100_rows.csv"
    MODEL_PATH = "model.pkl"

    # Load and clean the dataset using shared utilities.
    df = utils.load_data(DATASET_PATH)
    df = utils.clean_data(df)

    # Prepare features/target and split into train/test sets.
    X, y = prepare_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the regularized decision tree.
    model = train_decision_tree(X_train, y_train)

    # Evaluate and report metrics.
    metrics = evaluate_model(model, X_test, y_test)
    print("Model evaluation on test set:")
    print(f"  MAE : {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R2  : {metrics['r2']:.4f}")

    # Persist the trained model.
    save_model(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # Run a sample custom prediction.
    sample_prediction = predict_single(
        model, temp=0.5, humidity=0.6, windspeed=0.2, season=3
    )
    print(
        f"\nPredicted rental count for "
        f"(temp=0.5, humidity=0.6, windspeed=0.2, season=3): "
        f"{sample_prediction:.2f}"
    )
    print(
        "This is the model's estimated number of bike rentals for the given "
        "weather and season inputs."
    )
