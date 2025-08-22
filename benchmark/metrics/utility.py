from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error


def _prepare_ml_data(
    df: pd.DataFrame,
    target_col: str,
    categorical_columns: List[str],
    numerical_columns: List[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare data for machine learning models."""
    # Handle categorical columns
    df_encoded = df.copy()
    label_encoders = {}
    
    for col in categorical_columns:
        if col != target_col:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
    
    # Handle target column
    target_le = None
    if target_col in categorical_columns:
        target_le = LabelEncoder()
        df_encoded[target_col] = target_le.fit_transform(df_encoded[target_col].astype(str))
    
    # Separate features and target
    feature_cols = [c for c in df_encoded.columns if c != target_col]
    X = df_encoded[feature_cols].values
    y = df_encoded[target_col].values
    
    return X, y


def _train_classifier(X_train: np.ndarray, y_train: np.ndarray, 
                     X_test: np.ndarray, y_test: np.ndarray, 
                     model_type: str = "rf") -> Dict[str, float]:
    """Train classifier and return metrics."""
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "lr":
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return {
        f"{model_type}_accuracy": accuracy,
        f"{model_type}_f1": f1
    }


def _train_regressor(X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, y_test: np.ndarray, 
                    model_type: str = "rf") -> Dict[str, float]:
    """Train regressor and return metrics."""
    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "lr":
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return {
        f"{model_type}_r2": r2,
        f"{model_type}_rmse": rmse
    }


def _is_classification_task(y: np.ndarray) -> bool:
    """Determine if this is a classification or regression task."""
    unique_values = len(np.unique(y))
    total_values = len(y)
    
    # If less than 10 unique values or unique ratio < 5%, treat as classification
    if unique_values < 10 or (unique_values / total_values) < 0.05:
        return True
    return False


def compute_tstr(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth_train: pd.DataFrame,
    target_col: str,
    categorical_columns: List[str],
    numerical_columns: List[str],
    model_types: List[str] = ["rf", "lr"]
) -> Dict[str, float]:
    """Train on Synthetic, Test on Real (TSTR)."""
    try:
        # Prepare data
        X_synth, y_synth = _prepare_ml_data(synth_train, target_col, categorical_columns, numerical_columns)
        X_real, y_real = _prepare_ml_data(real_test, target_col, categorical_columns, numerical_columns)
        
        # Ensure same number of features
        min_features = min(X_synth.shape[1], X_real.shape[1])
        X_synth = X_synth[:, :min_features]
        X_real = X_real[:, :min_features]
        
        metrics = {}
        is_classification = _is_classification_task(y_real)
        
        for model_type in model_types:
            try:
                if is_classification:
                    model_metrics = _train_classifier(X_synth, y_synth, X_real, y_real, model_type)
                else:
                    model_metrics = _train_regressor(X_synth, y_synth, X_real, y_real, model_type)
                
                # Add TSTR prefix
                for key, value in model_metrics.items():
                    metrics[f"tstr_{key}"] = value
            except Exception as e:
                print(f"Error in TSTR {model_type}: {e}")
                continue
        
        return metrics
    except Exception as e:
        print(f"Error in TSTR: {e}")
        return {}


def compute_trts(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth_test: pd.DataFrame,
    target_col: str,
    categorical_columns: List[str],
    numerical_columns: List[str],
    model_types: List[str] = ["rf", "lr"]
) -> Dict[str, float]:
    """Train on Real, Test on Synthetic (TRTS)."""
    try:
        # Prepare data
        X_real, y_real = _prepare_ml_data(real_train, target_col, categorical_columns, numerical_columns)
        X_synth, y_synth = _prepare_ml_data(synth_test, target_col, categorical_columns, numerical_columns)
        
        # Ensure same number of features
        min_features = min(X_real.shape[1], X_synth.shape[1])
        X_real = X_real[:, :min_features]
        X_synth = X_synth[:, :min_features]
        
        metrics = {}
        is_classification = _is_classification_task(y_real)
        
        for model_type in model_types:
            try:
                if is_classification:
                    model_metrics = _train_classifier(X_real, y_real, X_synth, y_synth, model_type)
                else:
                    model_metrics = _train_regressor(X_real, y_real, X_synth, y_synth, model_type)
                
                # Add TRTS prefix
                for key, value in model_metrics.items():
                    metrics[f"trts_{key}"] = value
            except Exception as e:
                print(f"Error in TRTS {model_type}: {e}")
                continue
        
        return metrics
    except Exception as e:
        print(f"Error in TRTS: {e}")
        return {}


def compute_utility(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth_train: pd.DataFrame,
    target_col: str,
    categorical_columns: List[str],
    numerical_columns: List[str],
    model_types: List[str] = ["rf", "lr"]
) -> Dict[str, float]:
    """Compute all utility metrics."""
    metrics = {}
    
    # Generate synthetic test set
    synth_test = synth_train.sample(n=len(real_test), replace=True, random_state=42).reset_index(drop=True)
    
    # TSTR metrics
    tstr_metrics = compute_tstr(
        real_train, real_test, synth_train,
        target_col, categorical_columns, numerical_columns, model_types
    )
    metrics.update(tstr_metrics)
    
    # TRTS metrics
    trts_metrics = compute_trts(
        real_train, real_test, synth_test,
        target_col, categorical_columns, numerical_columns, model_types
    )
    metrics.update(trts_metrics)
    
    return metrics