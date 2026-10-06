#!/usr/bin/env python3
"""Slide figure 5: registration uncertainty tr(Sigma) by collaboration scale (MILUV, sec 4.9).

    python scripts/slides/s5_trsigma.py           # covor env python

Data: results/results_49_collaboration.csv (conditions A/B/C anchor-free, D with
anchors as a reference). Checked against RESULTS_SUMMARY.md sec 7.2 (0.940 / 0.130 /
0.074 / 0.0015 m^2) before plotting. Right panel: IoU@1vox and false-free rate of
the same conditions -- plotted as they are, so the saturation after the first
pair (A->B large, B->C flat or slightly worse) is visible.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt

CSV = "/src/gs25058/cr_RNE/covor_slam/results/results_49_collaboration.csv"
OUT = "/src/gs25058/cr_RNE/covor_slam/paper/figures/slides/s5_trsigma.png"
DOC = {"A_1drone_af": 0.940, "B_2drone_af": 0.130, "C_3drone_af": 0.074, "D_3drone_anch": 0.0015}
LABEL = {"A_1drone_af": "1대\n(쌍 0)", "B_2drone_af": "2대\n(쌍 1)",
         "C_3drone_af": "3대\n(쌍 3)", "D_3drone_anch": "3대+앵커\n(참고)"}

plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False,
                     "font.size": 15, "mathtext.fontset": "dejavusans"})


def main():
    rows = {r["condition"]: r for r in csv.DictReader(open(CSV))}
    keys = list(DOC)
    tr = [float(rows[k]["tr_sigma_med"]) for k in keys]
    for k, v in zip(keys, tr):
        # the document rounds; a mismatch beyond rounding means a different run
        tol = 0.0005 if DOC[k] < 0.01 else 0.0006
        assert abs(v - DOC[k]) <= tol, (k, v, DOC[k])
    iou = [float(rows[k]["iou_1vox"]) for k in keys]
    ff = [float(rows[k]["false_free_rate_pct"]) for k in keys]

    f = plt.figure(figsize=(10, 6))
    ax = f.add_axes([0.09, 0.17, 0.47, 0.78])
    x = range(len(keys))
    bars = ax.bar(x, tr, width=0.62,
                  color=["#3a3a3a", "#3a3a3a", "#3a3a3a", "#d9d9d9"],
                  edgecolor=["#3a3a3a"] * 3 + ["#8a8a8a"], linewidth=1.0)
    bars[3].set_hatch("///")
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 4)
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels(["0.001", "0.01", "0.1", "1"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel(r"$\mathrm{tr}(\Sigma)$ 중앙값 [m$^2$]")
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABEL[k] for k in keys], fontsize=14)
    for b, v in zip(bars, tr):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.25,
                ("%.3f" % v) if v >= 0.01 else ("%.4f" % v),
                ha="center", va="bottom", fontsize=14)
    ratio = tr[0] / tr[2]
    ax.annotate("", xy=(2.0, tr[2] * 2.3), xytext=(0.25, tr[0] * 2.6),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2.0,
                                connectionstyle="arc3,rad=-0.2"))
    ax.text(1.55, 0.75, "%.0f배 감소" % ratio, ha="left", va="center",
            fontsize=16, color="#c0392b")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=14)

    # right: IoU and false-free, as measured (no smoothing, no axis tricks)
    xs = list(range(3))
    a1 = f.add_axes([0.68, 0.59, 0.31, 0.36])
    a2 = f.add_axes([0.68, 0.17, 0.31, 0.36])
    a1.plot(xs, iou[:3], "o-", color="#3a3a3a", lw=2, ms=7)
    a1.plot([3], [iou[3]], "o", mfc="white", mec="#8a8a8a", ms=7)
    a1.set_ylabel("IoU", fontsize=14)
    a1.set_ylim(0.25, 0.65)
    a2.plot(xs, ff[:3], "o-", color="#3a3a3a", lw=2, ms=7)
    a2.plot([3], [ff[3]], "o", mfc="white", mec="#8a8a8a", ms=7)
    a2.set_ylabel("false-free [%]", fontsize=14)
    a2.set_ylim(4.0, 6.5)
    for a, vals, fmt in ((a1, iou, "%.2f"), (a2, ff, "%.1f")):
        a.set_xticks(range(4))
        a.set_xlim(-0.4, 3.4)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(labelsize=14)
        for i, v in enumerate(vals):
            below = (a is a2 and i == 1) or (a is a1 and i == 0)
            a.annotate(fmt % v, (i, v), textcoords="offset points",
                       xytext=(7, -19 if below else 6), fontsize=14,
                       color="#3a3a3a" if i < 3 else "#8a8a8a")
    a1.set_xticklabels([])
    a2.set_xticklabels(["1대", "2대", "3대", "앵커"], fontsize=14)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    f.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    print("tr:", tr, "iou:", iou, "ff:", ff, "ratio A/C %.1f" % ratio, "->", OUT)


if __name__ == "__main__":
    main()
