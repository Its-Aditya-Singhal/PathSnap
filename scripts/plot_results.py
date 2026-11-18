"""Plots from the real outputs of train_models.py (no numbers are typed in here).

    python scripts/plot_results.py
Writes PNGs to results/plots/.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.config import PLOTS_DIR, RESULTS_DIR

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#52514e"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "legend.frameon": False,
})


def save(fig, name):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / name)
    plt.close(fig)
    print("wrote", PLOTS_DIR / name)


def label_bars(ax, bars, fmt="{:.3f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=7, color=MUTED, xytext=(0, 2),
                    textcoords="offset points")


def main():
    mc = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    base = mc[mc.model == "hmm_baseline"].iloc[0] if (mc.model == "hmm_baseline").any() else None
    m = mc[mc.model != "hmm_baseline"].sort_values("cv_macro_f1", ascending=False)
    x = np.arange(len(m))

    # 1. CV Macro F1 (GroupKFold by trip) with fold std
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    bars = ax.bar(x, m.cv_macro_f1, yerr=m.cv_macro_f1_std, color=BLUE, width=0.6,
                  error_kw=dict(ecolor=GRAY, lw=1, capsize=3))
    label_bars(ax, bars)
    ax.set_xticks(x, m.model)
    ax.set_ylim(max(0, m.cv_macro_f1.min() - 0.1), 1)
    ax.set_ylabel("Macro F1 (mean ± std over 5 folds)")
    ax.set_title("Candidate classifier: 5-fold GroupKFold CV on train (80 trips)")
    save(fig, "cv_macro_f1.png")

    # 2. Student-test Top-1: classifier-only vs classifier + HMM
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.4))
    b1 = ax.bar(x - w / 2 - 0.01, m.top1_clf, w, color=BLUE, label="Classifier only (argmax per point)")
    b2 = ax.bar(x + w / 2 + 0.01, m.top1_hmm, w, color=ORANGE, label="Classifier + HMM/Viterbi")
    label_bars(ax, b1); label_bars(ax, b2)
    if base is not None:
        ax.axhline(base.top1_hmm, color=GRAY, ls="--", lw=1)
        ax.annotate(f"HMM, Gaussian emission: {base.top1_hmm:.3f}", (len(m) - 0.5, base.top1_hmm),
                    ha="right", va="bottom", fontsize=7, color=MUTED, xytext=(0, 2),
                    textcoords="offset points")
    ax.set_xticks(x, m.model)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Top-1 point accuracy")
    ax.set_title("Student test (10 trips): per-point Top-1 accuracy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=7)
    save(fig, "student_test_top1.png")

    # 3. Route F1: classifier-only vs + HMM
    fig, ax = plt.subplots(figsize=(7, 3.4))
    b1 = ax.bar(x - w / 2 - 0.01, m.route_f1_clf, w, color=BLUE, label="Classifier only")
    b2 = ax.bar(x + w / 2 + 0.01, m.route_f1_hmm, w, color=ORANGE, label="Classifier + HMM/Viterbi")
    label_bars(ax, b1); label_bars(ax, b2)
    if base is not None:
        ax.axhline(base.route_f1_hmm, color=GRAY, ls="--", lw=1)
        ax.annotate(f"HMM, Gaussian emission: {base.route_f1_hmm:.3f}", (-0.5, base.route_f1_hmm),
                    ha="left", va="bottom", fontsize=7, color=MUTED, xytext=(0, 8),
                    textcoords="offset points")
    ax.set_xticks(x, m.model)
    ax.set_ylim(0.8, 1.02)
    ax.set_ylabel("Length-weighted route F1")
    ax.set_title("Student test: reconstructed-route F1 (mean over trips)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=7)
    save(fig, "student_test_route_f1.png")

    # 4. Log loss: CV vs student test
    fig, ax = plt.subplots(figsize=(7, 3.2))
    b1 = ax.bar(x - w / 2 - 0.01, m.cv_log_loss, w, color=BLUE, label="CV (train)")
    b2 = ax.bar(x + w / 2 + 0.01, m.test_log_loss, w, color=ORANGE, label="Student test")
    label_bars(ax, b1); label_bars(ax, b2)
    ax.set_xticks(x, m.model)
    ax.set_ylabel("Log loss (lower is better)")
    ax.set_title("Probability calibration matters: the HMM consumes these probabilities")
    ax.legend(fontsize=7)
    save(fig, "log_loss.png")

    # 5. Confusion matrices (student test, threshold 0.5)
    names = [n for n in m.model if (RESULTS_DIR / "confusion" / f"{n}.json").exists()]
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.3 * rows), squeeze=False)
    for ax, n in zip(axes.flat, names):
        cm = np.array(json.loads((RESULTS_DIR / "confusion" / f"{n}.json").read_text()))
        rn = cm / cm.sum(1, keepdims=True)
        ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{rn[i, j]:.1%}", ha="center", va="center", fontsize=7,
                        color="white" if rn[i, j] > 0.6 else INK)
        ax.set_title(n, fontsize=9)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.grid(False)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.suptitle("Student test confusion matrices (candidate = true arc?, threshold 0.5)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "confusion_matrices.png")

    # 6. Real loss curves (only models that produce them)
    hist = {p.stem: json.loads(p.read_text()) for p in sorted((RESULTS_DIR / "histories").glob("*.json"))}
    if hist:
        fig, axes = plt.subplots(1, len(hist), figsize=(3.3 * len(hist), 2.8), squeeze=False)
        for ax, (n, h) in zip(axes[0], hist.items()):
            unit = "epoch" if n == "mlp" else "boosting round"
            ax.plot(np.arange(1, len(h["train_logloss"]) + 1), h["train_logloss"], color=BLUE, lw=2, label="train")
            if "val_logloss" in h:
                ax.plot(np.arange(1, len(h["val_logloss"]) + 1), h["val_logloss"], color=ORANGE, lw=2,
                        label="validation (held-out trips)")
            ax.set_title(n); ax.set_xlabel(unit); ax.set_ylabel("log loss")
            ax.legend(fontsize=7)
        fig.suptitle("Training curves, CV fold 0", fontsize=10, fontweight="bold")
        fig.tight_layout()
        save(fig, "loss_curves.png")


if __name__ == "__main__":
    main()
