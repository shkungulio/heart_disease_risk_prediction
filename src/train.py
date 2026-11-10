"""Train, compare and tune classifiers; save the best pipeline.

Run:  python -m src.train
"""
import json
import warnings

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.data_prep import RANDOM_STATE, ROOT, build_preprocessor, get_splits

warnings.filterwarnings("ignore")
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
SCORING = {"roc_auc": "roc_auc", "recall": "recall", "precision": "precision", "f1": "f1", "accuracy": "accuracy"}


def candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM (RBF)": SVC(probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1, verbosity=0),
    }


PARAM_GRIDS = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10]},
    "SVM (RBF)": {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.05]},
    "Random Forest": {
        "clf__max_depth": [3, 5, None],
        "clf__min_samples_leaf": [1, 3, 5],
        "clf__max_features": ["sqrt", 0.5],
    },
    "XGBoost": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [2, 3],
        "clf__learning_rate": [0.03, 0.1],
        "clf__subsample": [0.8, 1.0],
    },
}


def make_pipeline(model) -> Pipeline:
    """Preprocessing lives *inside* the pipeline so CV never leaks information."""
    return Pipeline([("prep", build_preprocessor()), ("clf", model)])


def compare_models(X_train, y_train) -> pd.DataFrame:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    rows = []
    for name, model in candidate_models().items():
        res = cross_validate(make_pipeline(model), X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        row = {"model": name}
        for m in SCORING:
            row[f"{m}_mean"] = res[f"test_{m}"].mean()
            row[f"{m}_std"] = res[f"test_{m}"].std()
        rows.append(row)
        print(f"{name:22s} AUC={row['roc_auc_mean']:.3f}±{row['roc_auc_std']:.3f}  recall={row['recall_mean']:.3f}")
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)


def tune_models(X_train, y_train, names) -> dict:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
    tuned = {}
    for name in names:
        gs = GridSearchCV(
            make_pipeline(candidate_models()[name]), PARAM_GRIDS[name], scoring="roc_auc", cv=cv, n_jobs=-1
        )
        gs.fit(X_train, y_train)
        tuned[name] = {"estimator": gs.best_estimator_, "cv_auc": gs.best_score_, "params": gs.best_params_}
        print(f"Tuned {name:20s} CV AUC={gs.best_score_:.3f}  {gs.best_params_}")
    return tuned


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test = get_splits()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

    print("== Baseline comparison (5-fold x 3 repeats, train set only) ==")
    comparison = compare_models(X_train, y_train)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    print("\n== Hyper-parameter tuning ==")
    tuned = tune_models(X_train, y_train, list(PARAM_GRIDS))
    best_name = max(tuned, key=lambda k: tuned[k]["cv_auc"])
    print(f"\nBest tuned model: {best_name}")

    joblib.dump(tuned[best_name]["estimator"], MODELS_DIR / "best_model.joblib")
    # Keep the tuned logistic regression too: it is the interpretable reference model.
    joblib.dump(tuned["Logistic Regression"]["estimator"], MODELS_DIR / "logistic_regression.joblib")
    meta = {
        "best_model": best_name,
        "cv_auc": round(float(tuned[best_name]["cv_auc"]), 4),
        "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                        for k, v in tuned[best_name]["params"].items()},
        "tuned_cv_auc": {k: round(float(v["cv_auc"]), 4) for k, v in tuned.items()},
        "random_state": RANDOM_STATE,
    }
    (MODELS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
