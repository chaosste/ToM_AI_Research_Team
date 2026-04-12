# %%
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd

# %%
ROOT = Path("/Users/stephenbeale/Projects/ToM_AI_Research_Team")
EXPORT_DIR = ROOT / "logs" / "quickviz_exports"

RUN_SPECS = [
    {
        "label": "local800",
        "seed": 7,
        "metrics_path": ROOT / "logs" / "local-run-v2-modal-seed7" / "candidate_metrics" / "metrics.json",
        "curve_path": ROOT / "logs" / "local-run-v2-modal-seed7" / "candidate_metrics" / "learning_curve.csv",
        "group": "baseline",
    },
    {
        "label": "local800",
        "seed": 11,
        "metrics_path": ROOT / "logs" / "local-run-v2-modal-seed11" / "candidate_metrics" / "metrics.json",
        "curve_path": ROOT / "logs" / "local-run-v2-modal-seed11" / "candidate_metrics" / "learning_curve.csv",
        "group": "baseline",
    },
    {
        "label": "postevidence_800",
        "seed": 7,
        "metrics_path": ROOT / "logs" / "v3omx_postevidence_reengage_seed7" / "candidate_metrics" / "metrics.json",
        "curve_path": ROOT / "logs" / "v3omx_postevidence_reengage_seed7" / "candidate_metrics" / "learning_curve.csv",
        "group": "candidate_800",
    },
    {
        "label": "postevidence_800",
        "seed": 11,
        "metrics_path": ROOT / "logs" / "v3omx_postevidence_reengage_seed11" / "candidate_metrics" / "metrics.json",
        "curve_path": ROOT / "logs" / "v3omx_postevidence_reengage_seed11" / "candidate_metrics" / "learning_curve.csv",
        "group": "candidate_800",
    },
    {
        "label": "delayedtrust_800",
        "seed": 7,
        "metrics_path": ROOT / "logs" / "v3omx_delayedtrust_promo_seed7" / "candidate_metrics" / "metrics.json",
        "curve_path": ROOT / "logs" / "v3omx_delayedtrust_promo_seed7" / "candidate_metrics" / "learning_curve.csv",
        "group": "candidate_800",
    },
    {
        "label": "delayedtrust_800",
        "seed": 11,
        "metrics_path": ROOT / "logs" / "v3omx_delayedtrust_promo_seed11" / "candidate_metrics" / "metrics.json",
        "curve_path": ROOT / "logs" / "v3omx_delayedtrust_promo_seed11" / "candidate_metrics" / "learning_curve.csv",
        "group": "candidate_800",
    },
    {
        "label": "old140",
        "seed": 7,
        "metrics_path": ROOT / "modal" / "tom-140k-modal-results" / "seed7" / "target-140000" / "run_summary.json",
        "curve_path": None,
        "group": "longrun_140k",
    },
    {
        "label": "old140",
        "seed": 11,
        "metrics_path": ROOT / "modal" / "tom-140k-modal-results" / "seed11" / "target-140000" / "run_summary.json",
        "curve_path": None,
        "group": "longrun_140k",
    },
    {
        "label": "v2old140",
        "seed": 7,
        "metrics_path": ROOT / "modal" / "tom-140k-modal-results-v2" / "seed7" / "target-140000" / "run_summary.json",
        "curve_path": None,
        "group": "longrun_140k",
    },
    {
        "label": "v2old140",
        "seed": 11,
        "metrics_path": ROOT / "modal" / "tom-140k-modal-results-v2" / "seed11" / "target-140000" / "run_summary.json",
        "curve_path": None,
        "group": "longrun_140k",
    },
    {
        "label": "check2_140k",
        "seed": 7,
        "metrics_path": ROOT / "modal" / "tom-experiment-incumbent" / "check2-140k-20260410" / "seed7" / "run_summary.json",
        "curve_path": ROOT / "modal" / "tom-experiment-incumbent" / "check2-140k-20260410" / "seed7" / "learning_curve.csv",
        "group": "longrun_140k",
    },
    {
        "label": "check2_140k",
        "seed": 11,
        "metrics_path": ROOT / "modal" / "tom-experiment-incumbent" / "check2-140k-20260410" / "seed11" / "run_summary.json",
        "curve_path": ROOT / "modal" / "tom-experiment-incumbent" / "check2-140k-20260410" / "seed11" / "learning_curve.csv",
        "group": "longrun_140k",
    },
    {
        "label": "check3_140k",
        "seed": 7,
        "metrics_path": ROOT / "modal" / "tom-experiment-incumbent" / "check3-140k-20260410" / "seed7" / "target-140000" / "run_summary.json",
        "curve_path": None,
        "group": "longrun_140k",
    },
    {
        "label": "check3_140k",
        "seed": 11,
        "metrics_path": ROOT / "modal" / "tom-experiment-incumbent" / "check3-140k-20260410" / "seed11" / "target-140000" / "run_summary.json",
        "curve_path": None,
        "group": "longrun_140k",
    },
    {
        "label": "v5dt_140k_salvage",
        "seed": 7,
        "metrics_path": ROOT / "logs" / "v2-duplicate-v5dt-<seed7>" / "salvaged_seed7_target140000_eval_metrics.json",
        "curve_path": None,
        "group": "longrun_140k",
    },
    {
        "label": "v5dt_140k_salvage",
        "seed": 11,
        "metrics_path": ROOT / "logs" / "v2-duplicate-v5dt-20260411-031103 : App Logs" / "salvaged_seed11_target140000_eval_metrics.json",
        "curve_path": None,
        "group": "longrun_140k",
    },
]

LABEL_ORDER = [
    "local800",
    "postevidence_800",
    "delayedtrust_800",
    "old140",
    "v2old140",
    "check2_140k",
    "check3_140k",
    "v5dt_140k_salvage",
]

METRIC_SPECS = [
    ("ToMCoordScore", "higher"),
    ("SuccessRate", "higher"),
    ("CollisionRate", "lower"),
    ("DeadlockRate", "lower"),
    ("IntentionPredictionF1", "higher"),
    ("StrategySwitchAccuracy", "higher"),
    ("AmbiguityEfficiency", "higher"),
    ("AverageDelay", "context"),
]

SEED_COLORS = {7: "#1f77b4", 11: "#d62728"}

# %%
def load_metric_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["eval_metrics"] if "eval_metrics" in payload else payload


def build_metrics_frame(run_specs: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in run_specs:
        path = Path(spec["metrics_path"])
        if not path.exists():
            continue
        metrics = load_metric_payload(path)
        row = {
            "label": spec["label"],
            "seed": spec["seed"],
            "group": spec["group"],
            "metrics_path": str(path),
            "curve_path": str(spec["curve_path"]) if spec["curve_path"] else None,
        }
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                row[key] = float(value)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["label"] = pd.Categorical(df["label"], categories=LABEL_ORDER, ordered=True)
    return df.sort_values(["label", "seed"]).reset_index(drop=True)


def load_curve(path_str: str) -> pd.DataFrame | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["episode_rel"] = df["episode"] - df["episode"].iloc[0] + 1
    return df


metrics_df = build_metrics_frame(RUN_SPECS)
metrics_df

# %%
def plot_metric_lines(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(METRIC_SPECS), 1, figsize=(14, 3.3 * len(METRIC_SPECS)), sharex=True)
    if len(METRIC_SPECS) == 1:
        axes = [axes]

    for ax, (metric, direction) in zip(axes, METRIC_SPECS):
        for seed, color in SEED_COLORS.items():
            subset = df[df["seed"] == seed].sort_values("label")
            ax.plot(
                subset["label"].astype(str),
                subset[metric],
                marker="o",
                linewidth=2.2,
                markersize=6,
                color=color,
                label=f"seed {seed}",
            )
            baseline = subset[subset["label"].astype(str) == "local800"]
            if not baseline.empty:
                ax.axhline(
                    baseline[metric].iloc[0],
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.4,
                )
        title_suffix = {
            "higher": " (higher is better)",
            "lower": " (lower is better)",
            "context": " (context-dependent)",
        }[direction]
        ax.set_title(f"{metric}{title_suffix}", fontsize=12)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    axes[-1].tick_params(axis="x", rotation=35)
    fig.suptitle("Per-metric run comparison", fontsize=16, y=1.0)
    plt.tight_layout()
    plt.show()
    return fig


metric_lines_fig = plot_metric_lines(metrics_df)

# %%
def build_absolute_heatmap_frame(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    heatmap_df = df.copy()
    heatmap_df["run_seed"] = heatmap_df["label"].astype(str) + "_s" + heatmap_df["seed"].astype(str)
    return heatmap_df.set_index("run_seed")[metrics]


absolute_heatmap = build_absolute_heatmap_frame(
    metrics_df,
    ["ToMCoordScore", "SuccessRate", "CollisionRate", "DeadlockRate", "IntentionPredictionF1", "StrategySwitchAccuracy"],
)
absolute_heatmap

# %%
def plot_heatmap(data: pd.DataFrame, title: str, cmap: str = "viridis", center: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(12, max(4, 0.42 * len(data))))
    values = data.to_numpy(dtype=float)

    if center is None:
        norm = None
    else:
        vmax = np.nanmax(np.abs(values))
        norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=center, vmax=vmax)

    im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(data.index)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = values[i, j]
            text_color = "white" if abs(value) > (np.nanmax(np.abs(values)) * 0.55) else "black"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8, color=text_color)

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.show()
    return fig


absolute_heatmap_fig = plot_heatmap(absolute_heatmap, "Absolute final metrics by run/seed", cmap="YlGnBu")

# %%
def build_better_delta_frame(df: pd.DataFrame, metric_specs: list[tuple[str, str]]) -> pd.DataFrame:
    baselines = (
        df[df["label"].astype(str) == "local800"]
        .set_index("seed")
    )
    rows: list[dict] = []
    for _, row in df.iterrows():
        if row["label"] == "local800":
            continue
        baseline = baselines.loc[row["seed"]]
        delta_row = {
            "run_seed": f"{row['label']}_s{int(row['seed'])}",
        }
        for metric, direction in metric_specs:
            raw_delta = row[metric] - baseline[metric]
            delta_row[metric] = raw_delta if direction != "lower" else -raw_delta
        rows.append(delta_row)
    return pd.DataFrame(rows).set_index("run_seed")


better_delta_heatmap = build_better_delta_frame(
    metrics_df,
    [
        ("ToMCoordScore", "higher"),
        ("SuccessRate", "higher"),
        ("CollisionRate", "lower"),
        ("DeadlockRate", "lower"),
        ("IntentionPredictionF1", "higher"),
        ("StrategySwitchAccuracy", "higher"),
    ],
)
better_delta_heatmap

# %%
better_delta_heatmap_fig = plot_heatmap(
    better_delta_heatmap,
    "Baseline-relative advantage vs local800 (positive is better)",
    cmap="RdYlGn",
    center=0.0,
)

# %%
def plot_learning_curves(df: pd.DataFrame, metric: str, title: str) -> None:
    fig, axes = plt.subplots(1, len(SEED_COLORS), figsize=(16, 5), sharey=False)
    if len(SEED_COLORS) == 1:
        axes = [axes]

    for ax, (seed, color) in zip(axes, SEED_COLORS.items()):
        subset = df[df["seed"] == seed]
        for _, row in subset.iterrows():
            curve = load_curve(row["curve_path"])
            if curve is None or metric not in curve.columns:
                continue
            ax.plot(
                curve["episode_rel"],
                curve[metric],
                linewidth=2,
                label=str(row["label"]),
            )
        ax.set_title(f"{title} - seed {seed}")
        ax.set_xlabel("relative episode")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)

    axes[0].set_ylabel(metric)
    plt.tight_layout()
    plt.show()
    return fig


curve_df = metrics_df[metrics_df["curve_path"].notna()].copy()
reward_curve_fig = plot_learning_curves(curve_df, "reward_ma_10", "Learning curves")

# %%
loss_curve_fig = plot_learning_curves(curve_df, "total_loss", "Loss curves")

# %%
# Optional: tighten the view to the lines you care about most.
focus_labels = ["local800", "check2_140k", "v5dt_140k_salvage"]
focus_df = metrics_df[metrics_df["label"].astype(str).isin(focus_labels)].copy()
focus_metric_lines_fig = plot_metric_lines(focus_df)

# %%
# Optional: a tiny helper table for quick reading in the notebook.
summary_cols = [
    "label",
    "seed",
    "ToMCoordScore",
    "SuccessRate",
    "CollisionRate",
    "DeadlockRate",
    "IntentionPredictionF1",
    "StrategySwitchAccuracy",
    "AmbiguityEfficiency",
]
metrics_df[summary_cols].round(4)

# %%
def export_figures(figures: dict[str, plt.Figure], export_dir: Path = EXPORT_DIR, dpi: int = 180) -> list[Path]:
    export_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, fig in figures.items():
        path = export_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    return written


written_paths = export_figures(
    {
        "metric_lines_all": metric_lines_fig,
        "heatmap_absolute": absolute_heatmap_fig,
        "heatmap_delta_vs_local800": better_delta_heatmap_fig,
        "learning_curves_reward_ma_10": reward_curve_fig,
        "learning_curves_total_loss": loss_curve_fig,
        "metric_lines_focus": focus_metric_lines_fig,
    }
)
written_paths
