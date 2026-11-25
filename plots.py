"""
Shared chart helpers for the notebooks: one style, bars with 95% CI whiskers for model comparisons,
and line charts for ordered settings (hyperparameters, true rating, validation -> test).
"""
import numpy as np
import matplotlib.pyplot as plt

# categorical slots in fixed order; GRAY for reference lines, INK for secondary text
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BLUE, ORANGE = SERIES[0], SERIES[1]
GRAY, INK = "#8a8984", "#52514e"


def setup():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": "#e6e5e0", "axes.axisbelow": True, "lines.linewidth": 2,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "figure.dpi": 110,
    })


def _label(name):
    """'B2 user features' -> 'B2\\nuser features' so long model names fit under a bar."""
    return name.replace(" ", "\n", 1)


def _reference(ax, ref, ref_label):
    if ref is not None:
        ax.axhline(ref, color=GRAY, linestyle="--", linewidth=1)
        ax.text(ax.get_xlim()[1], ref, f" {ref_label}", va="center", ha="left", fontsize=8, color=INK)


def bars(ax, names, values, low=None, high=None, ref=None, ref_label="random", fmt="{:.3f}", title=None, ylabel=None,
         short=False):
    """Vertical bars (one per model) with optional 95% CI whiskers and value labels above each bar.
    short=True labels each bar with the model code only ('B2'), for narrow panels."""
    values = np.asarray(values, float)
    x = np.arange(len(values))
    err = None if low is None else [values - np.asarray(low, float), np.asarray(high, float) - values]
    ax.bar(x, values, width=0.6, color=BLUE, yerr=err,
           error_kw=dict(ecolor=INK, elinewidth=1.2, capsize=4))
    top = values if high is None else np.maximum(values, np.asarray(high, float))
    pad = 0.03 * np.nanmax(np.abs(top))
    for xi, v, t in zip(x, values, top):
        ax.text(xi, max(t, 0) + pad, fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [n.split(" ")[0] if short else _label(n) for n in names], fontsize=8)
    # y range covers the whiskers too, so a CI that crosses 0 stays visible
    bottom = np.nanmin(values if low is None else np.minimum(values, np.asarray(low, float)))
    upper = np.nanmax(top) * 1.18 if np.nanmax(top) > 0 else 0.15 * abs(bottom)
    ax.set_ylim(min(0, bottom - 0.08 * (upper - bottom)) if bottom < 0 else 0, upper)
    _reference(ax, ref, ref_label)
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)


def metric_bars(table, metrics, refs=None, suptitle=None, low="{} low", high="{} high", width=4.2):
    """One bar panel per metric; `table` has a row per model and columns metric, '<metric> low', '<metric> high'."""
    refs = refs or {}
    fig, axes = plt.subplots(1, len(metrics), figsize=(width * len(metrics), 3.6), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        bars(ax, table.index, table[m], table[low.format(m)], table[high.format(m)], ref=refs.get(m), title=m, short=True)
    if suptitle:
        fig.suptitle(suptitle, fontweight="bold")
    fig.text(0.5, 0.01, "   ".join(table.index), ha="center", fontsize=8, color=INK)   # code -> model name
    plt.tight_layout(rect=(0, 0.05, 1, 1)); plt.show()


def diff_bars(table, col, title, ylabel):
    """Paired differences against a reference model, with 95% CI whiskers; 0 = no difference."""
    fig, ax = plt.subplots(figsize=(1.8 * len(table) + 2, 3.4))
    bars(ax, table.index, table[col], table["CI low"], table["CI high"], ref=0, ref_label="no difference",
         fmt="{:+.3f}", title=title, ylabel=ylabel)
    plt.tight_layout(); plt.show()


def lines(ax, x, series, xlabel, ylabel, title=None, xlog=False, xticklabels=None):
    """One line per series (dict name -> y values), markers at each x, value labels at the right end.
    Earlier series get larger markers, so identical series stay visible on top of each other."""
    for j, (color, (name, y)) in enumerate(zip(SERIES, series.items())):
        ax.plot(x, y, "o-", color=color, markersize=9 - 1.3 * j, label=name)
    if xlog:
        ax.set_xscale("log")
    # end labels, nudged apart so that close values do not overlap (identical values share one label)
    lo, hi = ax.get_ylim()
    gap, placed = 0.06 * (hi - lo), []
    for v in sorted({round(float(y[-1]), 3) for y in series.values()}):
        placed.append(max(v, placed[-1] + gap) if placed else v)
        ax.annotate(f"{v:.3f}", (x[-1], v), xytext=(8, (placed[-1] - v) / (hi - lo) * ax.bbox.height * 72 / ax.figure.dpi),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_xticks(x, xticklabels if xticklabels is not None else [str(v) for v in x])
    ax.minorticks_off()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8)


def val_test_lines(val, test, metric, lower_is_better=False, labels=("validation (2013)", "test (2014-18)")):
    """How each model's metric moves from validation to test (one line per model)."""
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    series = {name: [val[name], test[name]] for name in test.index}
    lines(ax, [0, 1], series, "", metric, xticklabels=list(labels),
          title=f"{metric} from validation to test" + (" (lower is better)" if lower_is_better else ""))
    ax.set_xlim(-0.2, 1.45)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(); plt.show()


def grouped_bars(ax, categories, series, ylabel, title, fmt="{:.2f}"):
    """One group of bars per category, one bar per model (series: model -> values), value above each bar."""
    x, w = np.arange(len(categories)), 0.8 / len(series)
    for k, (color, (name, values)) in enumerate(zip(SERIES, series.items())):
        xs = x + (k - (len(series) - 1) / 2) * w
        ax.bar(xs, values, w * 0.92, color=color, label=name)
        for xi, v in zip(xs, values):
            ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.15 * (hi - lo))          # room for a one-row legend above the bars
    ax.legend(frameon=False, fontsize=8, ncol=len(series), loc="upper left")


def calibration(ax, y, scores, n_bins=10, title=None, base_rate=True):
    """Reliability curve: mean prediction vs. observed positive share in equal-size bins of the prediction
    (scores: model -> probabilities). A calibrated model lies on the diagonal."""
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1, label="perfect calibration")
    lo = 1.0
    for color, (name, s) in zip(SERIES, scores.items()):
        bins = np.array_split(np.argsort(s), n_bins)
        pred, obs = [s[b].mean() for b in bins], [y[b].mean() for b in bins]
        ax.plot(pred, obs, "o-", color=color, label=name, markersize=5)
        lo = min(lo, *pred, *obs)
    ax.set_xlim(lo - 0.05, 1); ax.set_ylim(lo - 0.05, 1)
    if base_rate:
        ax.axhline(np.mean(y), color=INK, linestyle=":", linewidth=1)
        ax.text(lo - 0.04, np.mean(y) + 0.01, f"actual positive share {np.mean(y):.1%}", fontsize=8, color=INK)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed share of positives")
    ax.grid(True, axis="both")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    if title:
        ax.set_title(title)


def heatmap(ax, corr, title=None, fmt="{:.2f}"):
    """Correlation matrix as a diverging heatmap (-1 blue, 0 white, +1 red) with the value in each cell.
    Returns the image, for a colorbar."""
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(corr.shape[1]), corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(corr.shape[0]), corr.index, fontsize=8)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            v = corr.values[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else INK)
    ax.grid(False)
    if title:
        ax.set_title(title)
    return im


def compare_medians(a, b, labels, title):
    """Small multiples: the median of each feature in two row sets (e.g. training vs evaluation rows)."""
    cols = list(a.index)
    fig, axes = plt.subplots(1, len(cols), figsize=(2.3 * len(cols), 3), squeeze=False)
    for ax, c in zip(axes[0], cols):
        v = [a[c], b[c]]
        ax.bar([0, 1], v, width=0.6, color=[BLUE, ORANGE])
        for xi, vi in enumerate(v):
            ax.text(xi, vi, f"{vi:.3g}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks([0, 1], labels, fontsize=8)
        ax.set_title(c, fontsize=9)
        ax.set_ylim(0, max(v) * 1.2)
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout(); plt.show()
