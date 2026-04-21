
import math
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no GUI windows
import matplotlib.pyplot as plt

# Column groups (readable names after DataCleaner.clean())
_MONTHS = ["SEP", "AUG", "JUL", "JUN", "MAY", "APR"]
_PAY_COLS     = [f"PAY_{m}" for m in _MONTHS]
_BILL_COLS    = [f"BILL_{m}" for m in _MONTHS]
_PAY_AMT_COLS = [f"PAY_AMT_{m}" for m in _MONTHS]
_CONTINUOUS   = ["LIMIT_BAL", "AGE"] + _BILL_COLS + _PAY_AMT_COLS
_CATEGORICAL  = ["SEX", "EDUCATION", "MARRIAGE"]


def _grid_figure(n_cols, n_items, col_width=4, row_height=3):
    """Return (fig, axes_flat) for a tight subplot grid."""
    n_rows = math.ceil(n_items / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(col_width * n_cols, row_height * n_rows))
    axes_flat = axes.flatten() if n_items > 1 else [axes]
    for ax in axes_flat[n_items:]:
        ax.set_visible(False)
    return fig, axes_flat


def _save(fig, name: str, output_dir: str) -> None:
    """Save *fig* to *output_dir*/<name>.png and close it."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] saved → {path}")


class DataVisualisations:
    def __init__(self, output_dir: str = "results/eda"):
        self.output_dir = output_dir

    # 1. Class balance 

    def target_distribution(self, y: pd.Series):
        """Bar chart showing class balance (default vs no-default)."""
        counts = y.value_counts().sort_index()
        pcts   = counts / counts.sum() * 100

        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(["No Default (0)", "Default (1)"], counts.values,
                      color=["steelblue", "tomato"], edgecolor="white")
        for bar, pct in zip(bars, pcts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 150,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=10)
        ax.set_title("Class Distribution")
        ax.set_ylabel("Count")
        ax.set_ylim(0, counts.max() * 1.12)
        fig.tight_layout()
        _save(fig, "target_distribution", self.output_dir)

    # 2. Correlation heatmap (continuous features only)

    def correlation_heatmap(self, X: pd.DataFrame):
        """Heatmap of Pearson correlations among continuous features."""
        import seaborn as sns

        cont_cols = [c for c in _CONTINUOUS if c in X.columns]
        corr = X[cont_cols].corr()

        fig, ax = plt.subplots(figsize=(14, 11))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, linewidths=0.4, ax=ax,
                    annot_kws={"size": 7})
        ax.set_title("Correlation Heatmap — Continuous Features")
        fig.tight_layout()
        _save(fig, "correlation_heatmap", self.output_dir)

    # 3. Continuous feature distributions (grid)

    def feature_distributions(self, X: pd.DataFrame):
        """Histogram + KDE grid for all continuous features."""
        import seaborn as sns

        cont_cols = [c for c in _CONTINUOUS if c in X.columns]
        fig, axes = _grid_figure(n_cols=4, n_items=len(cont_cols))
        fig.suptitle("Continuous Feature Distributions", fontsize=13, y=1.01)
        for ax, col in zip(axes, cont_cols):
            sns.histplot(X[col], kde=True, ax=ax, color="steelblue",
                         edgecolor="none", bins=40)
            ax.set_title(col, fontsize=9)
            ax.set_xlabel("")
        fig.tight_layout()
        _save(fig, "feature_distributions", self.output_dir)

    # 4. Categorical feature counts

    def categorical_distributions(self, X: pd.DataFrame, y: pd.Series):
        """Count-plot grid for categorical features, coloured by default label."""
        import seaborn as sns

        cat_cols = [c for c in _CATEGORICAL if c in X.columns]
        fig, axes = _grid_figure(n_cols=len(cat_cols), n_items=len(cat_cols),
                                 col_width=5, row_height=4)
        fig.suptitle("Categorical Features by Default Label", fontsize=13)
        combined = X[cat_cols].copy()
        combined["default"] = y.values
        for ax, col in zip(axes, cat_cols):
            sns.countplot(data=combined, x=col, hue="default",
                          palette=["steelblue", "tomato"], ax=ax)
            ax.set_title(col)
            ax.legend(title="Default", labels=["No", "Yes"])
        fig.tight_layout()
        _save(fig, "categorical_distributions", self.output_dir)

    # 5. Boxplots of key continuous features by target

    def boxplots_by_target(self, X: pd.DataFrame, y: pd.Series):
        """Box-plots for LIMIT_BAL, AGE, and the most recent BILL/PAY_AMT."""
        import seaborn as sns

        key_cols = ["LIMIT_BAL", "AGE", "BILL_SEP", "PAY_AMT_SEP"]
        key_cols = [c for c in key_cols if c in X.columns]

        combined = X[key_cols].copy()
        combined["default"] = y.values

        fig, axes = _grid_figure(n_cols=2, n_items=len(key_cols),
                                 col_width=5, row_height=4)
        fig.suptitle("Key Feature Distributions by Default Label", fontsize=13)
        for ax, col in zip(axes, key_cols):
            sns.boxplot(data=combined, x="default", y=col,
                        palette=["steelblue", "tomato"], ax=ax,
                        showfliers=False)
            ax.set_xticklabels(["No Default (0)", "Default (1)"])
            ax.set_title(col)
            ax.set_xlabel("")
        fig.tight_layout()
        _save(fig, "boxplots_by_target", self.output_dir)

    # 6. Temporal trends across months

    def temporal_trends(self, X: pd.DataFrame, y: pd.Series):
        """
        Line plots showing the mean of PAY_*, BILL_*, and PAY_AMT_* columns
        over the 6 reference months (Sep → Apr), split by default label.
        """
        groups = {
            "Mean Repayment Status (PAY_*)": _PAY_COLS,
            "Mean Bill Amount (BILL_*)": _BILL_COLS,
            "Mean Payment Amount (PAY_AMT_*)": _PAY_AMT_COLS,
        }
        months_labels = _MONTHS  # Sep → Apr

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle("Temporal Trends by Default Label (mean ± 1 SD)", fontsize=13)

        for ax, (title, cols) in zip(axes, groups.items()):
            cols_present = [c for c in cols if c in X.columns]
            if not cols_present:
                continue
            for label_val, color, name in [(0, "steelblue", "No Default"),
                                           (1, "tomato",    "Default")]:
                subset = X.loc[y == label_val, cols_present]
                means = subset.mean()
                sds   = subset.std()
                xs    = range(len(cols_present))
                ax.plot(xs, means.values, marker="o", color=color, label=name)
                ax.fill_between(xs,
                                means.values - sds.values,
                                means.values + sds.values,
                                alpha=0.15, color=color)
            ax.set_xticks(range(len(cols_present)))
            ax.set_xticklabels(months_labels[:len(cols_present)], rotation=45)
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8)
        fig.tight_layout()
        _save(fig, "temporal_trends", self.output_dir)

    #  7. Repayment status heatmap

    def repayment_status_heatmap(self, X: pd.DataFrame, y: pd.Series):
        """
        Heatmap of the proportion of each repayment status code per month,
        separated by default label — shows the most discriminative pattern.
        """
        import seaborn as sns

        pay_cols = [c for c in _PAY_COLS if c in X.columns]
        if not pay_cols:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Repayment Status Distribution per Month", fontsize=13)

        for ax, (label_val, name) in zip(axes,
                                         [(0, "No Default"), (1, "Default")]):
            subset = X.loc[y == label_val, pay_cols]
            # Count each status code per month, then normalise to proportions
            codes = sorted(X[pay_cols].stack().unique())
            prop_matrix = pd.DataFrame(
                {col: subset[col].value_counts(normalize=True)
                              .reindex(codes, fill_value=0)
                 for col in pay_cols}
            )
            sns.heatmap(prop_matrix, ax=ax, cmap="YlOrRd",
                        annot=True, fmt=".2f", linewidths=0.3,
                        cbar_kws={"label": "Proportion"})
            ax.set_title(f"{name} (n={int(y.eq(label_val).sum()):,})")
            ax.set_xlabel("Month")
            ax.set_ylabel("Repay Code")
        fig.tight_layout()
        _save(fig, "repayment_status_heatmap", self.output_dir)
