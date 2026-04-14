"""
visualization.py
----------------
All plotting functions for the Alberta wildfire project.

Each function saves a PNG to disk (unless save=False) and calls plt.show().
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

MODEL_COLORS = {
    "Random Forest":       "#E85D20",
    "Logistic Regression": "#378ADD",
    "XGBoost":             "#1D9E75",
    "Isolation Forest":    "#7F77DD",
    # legacy two-model keys
    "RF": "#E85D20",
    "LR": "#378ADD",
}


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    roc_data: dict,
    results: dict,
    mean_fpr: np.ndarray,
    colors: dict = None,
    save: bool = True,
    filename: str = "fig_roc_curves.png",
) -> None:
    """Plot mean ROC curves with ±1 SD bands for each model.

    Parameters
    ----------
    roc_data : dict  {model_name: [interp_tpr arrays]}
    results  : dict  {model_name: {'auroc': [...]}}
    mean_fpr : np.ndarray
    colors   : dict, optional  Defaults to MODEL_COLORS.
    save     : bool
    filename : str
    """
    if colors is None:
        colors = MODEL_COLORS

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier (AUC = 0.500)")

    for name in roc_data:
        tprs = np.array(roc_data[name])
        m_tpr = tprs.mean(axis=0)
        s_tpr = tprs.std(axis=0)
        ar_m = np.mean(results[name]["auroc"])
        ar_s = np.std(results[name]["auroc"])
        color = colors.get(name, None)

        ax.plot(mean_fpr, m_tpr, color=color, lw=2,
                label=f"{name} (AUC = {ar_m:.3f} ± {ar_s:.3f})")
        ax.fill_between(
            mean_fpr,
            np.clip(m_tpr - s_tpr, 0, 1),
            np.clip(m_tpr + s_tpr, 0, 1),
            alpha=0.12, color=color,
        )

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        "ROC Curves — Large Fire Escalation\n"
        "(10-Fold Cross-Validation, mean ± 1 SD)",
        fontsize=13,
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=200, bbox_inches="tight")
        print(f"Saved: {filename}")
    plt.show()


# ---------------------------------------------------------------------------
# Precision-Recall curves
# ---------------------------------------------------------------------------

def plot_prc_curves(
    prc_data: dict,
    results: dict,
    mean_rec: np.ndarray,
    baseline: float,
    colors: dict = None,
    save: bool = True,
    filename: str = "fig_prc_curves.png",
) -> None:
    """Plot mean Precision-Recall curves with ±1 SD bands.

    Parameters
    ----------
    prc_data : dict  {model_name: [interp_precision arrays]}
    results  : dict  {model_name: {'auprc': [...]}}
    mean_rec : np.ndarray
    baseline : float  Random classifier AUPRC (= positive class prevalence).
    colors   : dict, optional
    save     : bool
    filename : str
    """
    if colors is None:
        colors = MODEL_COLORS

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axhline(
        y=baseline, color="gray", linestyle="--", lw=1.5,
        label=f"Random classifier (AP = {baseline:.3f})",
    )

    for name in prc_data:
        prcs = np.array(prc_data[name])
        m_prc = prcs.mean(axis=0)
        s_prc = prcs.std(axis=0)
        ap_m = np.mean(results[name]["auprc"])
        ap_s = np.std(results[name]["auprc"])
        color = colors.get(name, None)

        ax.plot(mean_rec, m_prc, color=color, lw=2,
                label=f"{name} (AUPRC = {ap_m:.3f} ± {ap_s:.3f})")
        ax.fill_between(
            mean_rec,
            np.clip(m_prc - s_prc, 0, 1),
            np.clip(m_prc + s_prc, 0, 1),
            alpha=0.12, color=color,
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(
        "Precision-Recall Curves — Large Fire Escalation\n"
        "(10-Fold Cross-Validation, mean ± 1 SD)",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=200, bbox_inches="tight")
        print(f"Saved: {filename}")
    plt.show()


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(
    results: dict,
    iso_ap: float,
    baseline: float,
    colors: dict = None,
    save: bool = True,
    filename: str = "fig_model_comparison.png",
) -> None:
    """Bar chart comparing AUPRC across all models including Isolation Forest.

    Parameters
    ----------
    results  : dict
    iso_ap   : float
    baseline : float
    colors   : dict, optional
    save     : bool
    filename : str
    """
    if colors is None:
        colors = MODEL_COLORS

    model_names = list(results.keys()) + ["Isolation Forest"]
    auprc_means = [np.mean(results[m]["auprc"]) for m in results] + [iso_ap]
    auprc_stds = [np.std(results[m]["auprc"]) for m in results] + [0.0]
    bar_colors = [colors.get(m, "#999999") for m in model_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        model_names, auprc_means, color=bar_colors, alpha=0.88,
        yerr=auprc_stds, capsize=6, error_kw={"linewidth": 2},
    )
    ax.axhline(
        y=baseline, color="gray", linestyle="--", lw=1.5,
        label=f"Random baseline ({baseline:.3f})",
    )

    for bar, m, s in zip(bars, auprc_means, auprc_stds):
        label = f"{m:.3f}±{s:.3f}" if s > 0 else f"{m:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + 0.005,
            label, ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("AUPRC (higher = better)", fontsize=11)
    ax.set_title(
        "Model Comparison — AUPRC for Large Fire Prediction\n"
        "(10-Fold CV mean ± SD; Isolation Forest = unsupervised)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(auprc_means) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=200, bbox_inches="tight")
        print(f"Saved: {filename}")
    plt.show()


# ---------------------------------------------------------------------------
# SHAP bar chart
# ---------------------------------------------------------------------------

def plot_shap_bar(
    mean_shap: np.ndarray,
    feature_names: list,
    save: bool = True,
    filename: str = "fig_shap.png",
) -> None:
    """Horizontal bar chart of mean absolute SHAP values.

    Parameters
    ----------
    mean_shap     : np.ndarray  shape (n_features,)
    feature_names : list[str]
    save          : bool
    filename      : str
    """
    sorted_idx = np.argsort(mean_shap)
    n = len(sorted_idx)
    bar_c = [
        "#E85D20" if i >= n - 3
        else "#F28C4A" if i >= n - 6
        else "#AABBD0"
        for i in range(n)
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        mean_shap[sorted_idx],
        color=bar_c, alpha=0.9,
    )
    ax.set_xlabel(
        "Mean |SHAP Value| — contribution to large fire prediction", fontsize=11
    )
    ax.set_title(
        "Feature Importance for Large Fire Escalation Prediction\n"
        "(Random Forest, SHAP mean absolute values)",
        fontsize=12,
    )
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=200, bbox_inches="tight")
        print(f"Saved: {filename}")
    plt.show()


# ---------------------------------------------------------------------------
# Fire size class distribution
# ---------------------------------------------------------------------------

def plot_size_distribution(
    df: pd.DataFrame,
    save: bool = True,
    filename: str = "fig_size_distribution.png",
) -> None:
    """Bar chart of fire size class counts (A–E).

    Parameters
    ----------
    df       : pd.DataFrame  Must contain 'SIZE_CLASS' column.
    save     : bool
    filename : str
    """
    if "SIZE_CLASS" not in df.columns:
        print("SIZE_CLASS column not found; skipping size distribution plot.")
        return

    size_order = ["A", "B", "C", "D", "E"]
    size_labels = {
        "A": "A\n(0–0.1 ha)",
        "B": "B\n(0.1–4 ha)",
        "C": "C\n(4–40 ha)",
        "D": "D\n(40–200 ha)",
        "E": "E\n(200+ ha)",
    }
    counts = df["SIZE_CLASS"].value_counts().reindex(size_order).fillna(0)
    bar_cols = ["#378ADD", "#1D9E75", "#EF9F27", "#E85D20", "#E24B4A"]
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [size_labels[s] for s in size_order],
        counts.values,
        color=bar_cols, alpha=0.88, edgecolor="white",
    )
    for bar, val in zip(bars, counts.values):
        pct = val / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 80,
            f"{int(val):,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xlabel("Fire Size Class", fontsize=11)
    ax.set_ylabel("Number of fires", fontsize=11)
    ax.set_title(
        "Fire Size Class Distribution — Alberta FPA, 2006–2024\n"
        "Target: Classes D & E = Large Fire",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=200, bbox_inches="tight")
        print(f"Saved: {filename}")
    plt.show()


# ---------------------------------------------------------------------------
# Annual trends (fire count vs. area burned)
# ---------------------------------------------------------------------------

def plot_annual_trends(
    df: pd.DataFrame,
    save: bool = True,
    filename: str = "fig_annual_trends.png",
) -> None:
    """Dual-axis chart: annual fire count (bars) vs. area burned (line).

    Parameters
    ----------
    df       : pd.DataFrame  Must contain 'YEAR' and 'CURRENT_SIZE'.
    save     : bool
    filename : str
    """
    if "YEAR" not in df.columns or "CURRENT_SIZE" not in df.columns:
        print("YEAR or CURRENT_SIZE column not found; skipping annual trends plot.")
        return

    yearly = df.groupby("YEAR").agg(
        fire_count=("FIRE_NUMBER", "count"),
        area_burned=("CURRENT_SIZE", "sum"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    ax1.bar(yearly["YEAR"], yearly["fire_count"],
            color="#E85D20", alpha=0.6, label="Fire count")
    ax2.plot(
        yearly["YEAR"], yearly["area_burned"] / 1000,
        color="#1A2332", lw=2.5, marker="o", markersize=5,
        label="Area burned (000s ha)",
    )

    ax1.set_xlabel("Year", fontsize=11)
    ax1.set_ylabel("Number of fires", fontsize=11, color="#E85D20")
    ax2.set_ylabel("Area burned (thousands of hectares)", fontsize=11, color="#1A2332")
    ax1.set_title(
        "Annual Wildfire Frequency vs. Area Burned\n"
        "Alberta Forest Protection Area, 2006–2024",
        fontsize=13,
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
    ax1.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=200, bbox_inches="tight")
        print(f"Saved: {filename}")
    plt.show()
