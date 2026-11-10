"""Evaluate the tuned model on the held-out test set and produce figures.

Run:  python -m src.evaluate
"""
import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data_prep import LABELS, RANDOM_STATE, ROOT, get_splits

warnings.filterwarnings("ignore")
FIG = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
TARGET_RECALL = 0.90


def metrics_at(y_true, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round(accuracy_score(y_true, pred), 3),
        "precision": round(precision_score(y_true, pred), 3),
        "recall": round(recall_score(y_true, pred), 3),
        "specificity": round(tn / (tn + fp), 3),
        "f1": round(f1_score(y_true, pred), 3),
    }


def pick_threshold(model, X_train, y_train, target_recall=TARGET_RECALL) -> float:
    """Highest threshold whose out-of-fold recall on the TRAIN set is >= target (no test leakage)."""
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    best = 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        if recall_score(y_train, (oof >= t).astype(int)) >= target_recall:
            best = t
    return float(best)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    X_train, X_test, y_train, y_test = get_splits()

    best = joblib.load(MODELS / "best_model.joblib")
    meta = json.loads((MODELS / "model_meta.json").read_text())
    models = {meta["best_model"] + " (best)": best}
    if meta["best_model"] != "Logistic Regression":
        models["Logistic Regression"] = joblib.load(MODELS / "logistic_regression.joblib")

    proba = best.predict_proba(X_test)[:, 1]
    thr_default, thr_tuned = 0.5, pick_threshold(best, X_train, y_train)
    results = {
        "model": meta["best_model"],
        "n_test": int(len(y_test)),
        "roc_auc": round(roc_auc_score(y_test, proba), 3),
        "pr_auc": round(average_precision_score(y_test, proba), 3),
        "default_threshold": metrics_at(y_test.values, proba, thr_default),
        f"recall>={TARGET_RECALL:.0%}_threshold": metrics_at(y_test.values, proba, thr_tuned),
    }
    # Bootstrap CI for test AUC (the test set is only 61 rows, so uncertainty is large)
    rng = np.random.default_rng(RANDOM_STATE)
    aucs = []
    yv = y_test.values
    for _ in range(2000):
        idx = rng.integers(0, len(yv), len(yv))
        if len(np.unique(yv[idx])) == 2:
            aucs.append(roc_auc_score(yv[idx], proba[idx]))
    results["roc_auc_95ci"] = [round(float(np.percentile(aucs, 2.5)), 3), round(float(np.percentile(aucs, 97.5)), 3)]
    print(classification_report(y_test, (proba >= thr_default).astype(int), target_names=["No disease", "Disease"]))
    print(json.dumps(results, indent=2))

    # --- ROC + PR curves for every tuned model ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, m in models.items():
        RocCurveDisplay.from_estimator(m, X_test, y_test, ax=ax[0], name=name)
        PrecisionRecallDisplay.from_estimator(m, X_test, y_test, ax=ax[1], name=name)
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax[0].set_title("ROC curve (test set)")
    ax[1].set_title("Precision-recall curve (test set)")
    ax[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "roc_pr_curves.png")
    plt.close(fig)

    # --- Confusion matrices at both thresholds ---
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for a, (t, title) in zip(ax, [(thr_default, "Threshold = 0.50"), (thr_tuned, f"Threshold = {thr_tuned:.2f} (recall-tuned)")]):
        ConfusionMatrixDisplay.from_predictions(
            y_test, (proba >= t).astype(int), display_labels=["No disease", "Disease"], cmap="Blues", ax=a, colorbar=False
        )
        a.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrices.png")
    plt.close(fig)

    # --- Calibration ---
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=5, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", label=meta["best_model"])
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction positive", title="Calibration (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png")
    plt.close(fig)

    # --- Threshold trade-off ---
    ts = np.arange(0.05, 0.96, 0.01)
    rec = [recall_score(y_test, (proba >= t).astype(int)) for t in ts]
    prec = [precision_score(y_test, (proba >= t).astype(int), zero_division=0) for t in ts]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, rec, label="Recall (sensitivity)")
    ax.plot(ts, prec, label="Precision")
    ax.axvline(thr_tuned, color="grey", ls=":", label=f"Chosen threshold ({thr_tuned:.2f})")
    ax.set(xlabel="Decision threshold", ylabel="Score", title="Recall / precision trade-off (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "threshold_tradeoff.png")
    plt.close(fig)

    # --- Interpretability: coefficients / odds ratios (if linear) ---
    prep, clf = best.named_steps["prep"], best.named_steps["clf"]
    feat_names = list(prep.get_feature_names_out())
    if hasattr(clf, "coef_"):
        coefs = pd.Series(clf.coef_[0], index=feat_names).sort_values()
        odds = pd.DataFrame({"coef": coefs, "odds_ratio": np.exp(coefs)}).sort_values("coef", ascending=False)
        odds.round(3).to_csv(ROOT / "reports" / "odds_ratios.csv")
        fig, ax = plt.subplots(figsize=(6.5, 6))
        coefs.plot.barh(ax=ax, color=["#c0392b" if v > 0 else "#2471a3" for v in coefs])
        ax.set(title="Logistic regression coefficients (standardised)", xlabel="Log-odds of heart disease")
        fig.tight_layout()
        fig.savefig(FIG / "coefficients.png")
        plt.close(fig)

    # --- Permutation importance on raw features (test set, AUC) ---
    pi = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=RANDOM_STATE)
    pi_s = pd.Series(pi.importances_mean, index=X_test.columns).sort_values()
    pi_err = pd.Series(pi.importances_std, index=X_test.columns).reindex(pi_s.index)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh([LABELS[c] for c in pi_s.index], pi_s.values, xerr=pi_err.values, color="#5d6d7e")
    ax.set(title="Permutation importance (drop in test AUC)", xlabel="Mean decrease in ROC-AUC")
    fig.tight_layout()
    fig.savefig(FIG / "permutation_importance.png")
    plt.close(fig)
    pi_s.sort_values(ascending=False).round(4).to_csv(ROOT / "reports" / "permutation_importance.csv", header=["auc_drop"])

    # --- SHAP (on the transformed feature space) ---
    Xtr_t, Xte_t = prep.transform(X_train), prep.transform(X_test)
    if hasattr(clf, "coef_"):
        explainer = shap.LinearExplainer(clf, Xtr_t)
        sv = explainer.shap_values(Xte_t)
    else:
        explainer = shap.TreeExplainer(clf) if hasattr(clf, "feature_importances_") else shap.KernelExplainer(
            clf.predict_proba, shap.sample(Xtr_t, 50)
        )
        sv = explainer.shap_values(Xte_t)
        if isinstance(sv, list):
            sv = sv[1]
        elif getattr(sv, "ndim", 2) == 3:
            sv = sv[:, :, 1]
    plt.figure()
    shap.summary_plot(sv, Xte_t, show=False, plot_size=(7.5, 6))
    plt.title("SHAP summary (test set)")
    plt.tight_layout()
    plt.savefig(FIG / "shap_summary.png")
    plt.close()
    plt.figure()
    shap.summary_plot(sv, Xte_t, plot_type="bar", show=False, plot_size=(7, 5))
    plt.title("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(FIG / "shap_bar.png")
    plt.close()

    # --- Fairness check: performance by sex and age band ---
    df = X_test.copy()
    df["y"], df["p"] = y_test.values, proba
    df["age_band"] = pd.cut(df["age"], [0, 54, 64, 120], labels=["<55", "55-64", "65+"])
    rows = []
    for col, name in [("sex", "Sex"), ("age_band", "Age band")]:
        for g, sub in df.groupby(col, observed=True):
            if sub["y"].nunique() < 2 or len(sub) < 5:
                continue
            pred = (sub["p"] >= thr_tuned).astype(int)
            rows.append({
                "group": f"{name}: {'Male' if (col == 'sex' and g == 1) else 'Female' if col == 'sex' else g}",
                "n": len(sub),
                "disease_rate": round(sub["y"].mean(), 2),
                "roc_auc": round(roc_auc_score(sub["y"], sub["p"]), 3),
                "recall": round(recall_score(sub["y"], pred), 3),
                "precision": round(precision_score(sub["y"], pred, zero_division=0), 3),
            })
    fair = pd.DataFrame(rows)
    fair.to_csv(ROOT / "reports" / "fairness_by_group.csv", index=False)
    print(fair.to_string(index=False))

    results["fairness_note"] = "See reports/fairness_by_group.csv (small subgroups; interpret with caution)"
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(results, indent=2))
    print("Figures saved to", FIG)


if __name__ == "__main__":
    main()
