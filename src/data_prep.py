"""Data loading, cleaning and preprocessing for the UCI Heart Disease (Cleveland) dataset."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "heart_disease.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "heart_clean.csv"

RANDOM_STATE = 42

# Feature groups (UCI data dictionary)
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]  # ca = # major vessels (0-3)
CATEGORICAL = ["cp", "restecg", "slope", "thal"]                   # nominal / ordinal codes
BINARY = ["sex", "fbs", "exang"]
FEATURES = NUMERIC + CATEGORICAL + BINARY
TARGET = "target"

LABELS = {
    "age": "Age (years)",
    "sex": "Sex (1 = male)",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG result",
    "thalach": "Max heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (exercise vs rest)",
    "slope": "Slope of peak-exercise ST segment",
    "ca": "Major vessels coloured by fluoroscopy",
    "thal": "Thalassemia test result",
}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV (original target column is `num`, values 0-4)."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Binarise the target (0 = no disease, 1 = disease) and remove duplicates.

    Missing values are intentionally *kept* here (only `ca` and `thal` have any);
    they are imputed inside the modelling pipeline so that imputation statistics
    are learned from training folds only (no leakage).
    """
    df = df.copy()
    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"]).drop_duplicates().reset_index(drop=True)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Optional features explored in the EDA notebook (not in the default model)."""
    df = df.copy()
    df["hr_reserve_pct"] = df["thalach"] / (220 - df["age"])
    df["age_group"] = pd.cut(df["age"], [0, 44, 54, 64, 120], labels=["<45", "45-54", "55-64", "65+"])
    return df


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + scale numerics, mode-impute + one-hot categoricals, pass binaries."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL), ("bin", "passthrough", BINARY)],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def get_splits(test_size: float = 0.2, random_state: int = RANDOM_STATE):
    """Stratified train/test split on the cleaned data."""
    df = clean(load_raw())
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def main() -> None:
    raw = load_raw()
    df = clean(raw)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Raw shape: {raw.shape} -> cleaned shape: {df.shape}")
    miss = df.isna().sum()
    print("Missing values:\n", miss[miss > 0].to_string())
    print("Target balance:\n", df[TARGET].value_counts(normalize=True).round(3).to_string())
    print(f"Saved {PROCESSED_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
