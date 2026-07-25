"""Generate the paper's two figures from the COMMITTED result arrays (no GPU, no models).

    python paper/make_figures.py

Outputs .pdf (vector, for LaTeX) and .png (300 dpi) into paper/figures/.

Palette: dataviz categorical slots 1-3 (blue #2a78d6, orange #eb6834, aqua #1baf7a),
validated all-pairs in light mode (worst CVD dE 9.2, worst normal-vision dE 24.0).
Aqua sits below 3:1 on the light surface, so the relief rule applies -- every series
carries a visible direct label, and identity is additionally encoded by line style and
marker shape so both figures survive greyscale print and full CVD.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unlearn_audit.attacks.membership_inference import _retention_from_losses  # noqa: E402
from unlearn_audit.audit.calibration import Calibrator  # noqa: E402
from unlearn_audit.experiments import RESULTS  # noqa: E402
from unlearn_audit.experiments.exp2_matched_reference import CALIPER, caliper_match  # noqa: E402

OUT = Path(__file__).parent / "figures"

# ---- design tokens (dataviz reference palette, light surface) ----------------------
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # blue, orange, aqua

plt.rcParams.update({
    "font.family": "DejaVu Sans",                  # system sans; ships with matplotlib
    "font.size": 8.5,
    "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
        fig.savefig(OUT / f"{stem}.{ext}", **kw)
    print(f"  wrote {OUT / stem}.pdf and .png")


# ==================================================================== Figure 1
def figure1(e1):
    f, h, r = e1["forget"], e1["holdout"], e1["retain"]
    auc_fh = _retention_from_losses(f, h)[1]
    auc_fr = _retention_from_losses(f, r)[1]
    auc_rh = _retention_from_losses(r, h)[1]

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    grid = np.linspace(0.0, 4.2, 600)                 # NLL >= 0; clip KDE spill below 0
    series = [
        ("forget10 (members)", f, S1, "-"),
        ("holdout10 (reference)", h, S2, "--"),
        ("retain90 (control)", r, S3, ":"),
    ]
    for name, data, color, ls in series:
        dens = gaussian_kde(data)(grid)               # Scott bandwidth (default)
        ax.fill_between(grid, dens, color=color, alpha=0.08, linewidth=0)
        ax.plot(grid, dens, color=color, ls=ls, lw=1.8, solid_capstyle="round", label=name)

    # mean markers on the baseline. forget10 and retain90 nearly coincide (2.07 vs 2.11) --
    # that overlap IS the control result, so the marks are left to overlap rather than nudged.
    for name, data, color, _ in series:
        ax.plot([data.mean()], [0], marker="v", ms=4.5, color=color,
                markeredgecolor=SURFACE, markeredgewidth=1.2, clip_on=False, zorder=5)

    # Direct labels collide here (two of the three curves sit on top of each other), so
    # identity rides the legend + line style, per the marks-and-anatomy collision fallback.
    ax.legend(loc="upper right", bbox_to_anchor=(1.015, 1.02), labelcolor=INK_2,
              handlelength=2.2, borderaxespad=0, labelspacing=0.35)

    # AUC values sit below the axis: every in-axes placement collided with either the
    # legend or the rising flank of the curves at single-column width.
    fig.text(0.5, -0.055,
             f"pairwise AUC (never-trained model):   forget10–holdout10 {auc_fh:.3f}"
             f"   ·   retain90–holdout10 {auc_rh:.3f}\n"
             f"forget10–retain90 {auc_fr:.3f}  ← control: two original-generation splits, "
             "matched to noise",
             ha="center", va="top", fontsize=7.0, color=INK_2, linespacing=1.6)

    ax.set_xlabel("answer NLL per token (base Phi-1.5)")
    ax.set_ylabel("density")
    ax.set_xlim(0, 4.2)
    ax.set_ylim(0, 0.95)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    save(fig, "fig1_difficulty_distributions")
    plt.close(fig)
    return auc_fh, auc_fr, auc_rh


# ==================================================================== Figure 2
def figure2(e1, e2, e2b):
    f, h = e1["forget"], e1["holdout"]
    fi, hi = caliper_match(f, h, CALIPER)
    thr_m = Calibrator(target_far=0.05).calibrate(0.0, e2["null_match"]).threshold

    coarse = [0.0, 0.15, 0.3, 0.5, 0.7]
    a_c = np.array(coarse)
    ch = [_retention_from_losses(e2[f"f_{a}"], e2[f"h_{a}"]) for a in coarse]
    cm = [_retention_from_losses(e2[f"f_{a}"][fi], e2[f"h_{a}"][hi]) for a in coarse]
    rh_c, ah_c = np.array([x[0] for x in ch]), np.array([x[1] for x in ch])
    rm_c = np.array([x[0] for x in cm])
    a_f, rh_f, rm_f, ah_f = e2b["alphas"], e2b["ret_h"], e2b["ret_m"], e2b["auc_h"]
    a1 = np.array([1.0])
    r1h, a1h = _retention_from_losses(e2["f_1.0"], e2["h_1.0"])
    rh_1, ah_1 = np.array([r1h]), np.array([a1h])
    rm_1 = np.array([_retention_from_losses(e2["f_1.0"][fi], e2["h_1.0"][hi])[0]])

    A = np.concatenate([a_c, a_f, a1])
    RH = np.concatenate([rh_c, rh_f, rh_1])
    RM = np.concatenate([rm_c, rm_f, rm_1])
    AH = np.concatenate([ah_c, ah_f, ah_1])          # holdout AUC, for the field-rule crossing
    o = np.argsort(A)
    A, RH, RM, AH = A[o], RH[o], RM[o], AH[o]

    def cross(x, y, level):
        for i in range(len(x) - 1):
            if (y[i] - level) * (y[i + 1] - level) <= 0 and y[i] != y[i + 1]:
                return x[i] + (x[i + 1] - x[i]) * (y[i] - level) / (y[i] - y[i + 1])
        return float("nan")

    # The holdout boundary is where its AUC crosses 0.5. It must NOT be read off the
    # retention curve: retention = clip(2*(AUC-0.5),0,1) is floored at 0, so the retention
    # curve only registers the crossing once it has already landed on the floor (0.80),
    # overstating the boundary. AUC keeps the sub-0.5 information -> 0.769.
    x_hold = cross(A, AH, 0.5)
    x_match = cross(A, RM, thr_m)
    rungs = [0.80, 0.85, 0.90]

    fig = plt.figure(figsize=(7.0, 3.25))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.32, 1], height_ratios=[1, 0.15],
                          wspace=0.16, hspace=0.45)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    axS = fig.add_subplot(gs[1, 1])          # verdict strip, under panel (b)

    for ax, zoom in [(axL, False), (axR, True)]:
        ax.axvspan(x_hold, x_match, color=MUTED, alpha=0.13, linewidth=0, zorder=0)
        ax.axhline(thr_m, color=S1, lw=1.0, ls=(0, (5, 3)), alpha=0.85, zorder=1)
        ax.axhline(0.0, color=S2, lw=1.0, ls=(0, (5, 3)), alpha=0.85, zorder=1)
        for x in (x_hold, x_match):
            ax.axvline(x, color=MUTED, lw=0.8, ls=(0, (1, 2.5)), zorder=1)

        ax.plot(A, RM, color=S1, ls="-", lw=1.9, marker="o", ms=4.2,
                markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=3)
        ax.plot(A, RH, color=S2, ls="--", lw=1.9, marker="s", ms=4.0,
                markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=3)

        # sampled rungs where the flip was DIRECTLY OBSERVED
        for x in rungs:
            for series, color in [(RM, S1), (RH, S2)]:
                y = float(np.interp(x, A, series))
                ax.plot([x], [y], marker="o" if color == S1 else "s", ms=7.5, color=color,
                        markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4)

        ax.grid(True, axis="y")
        ax.set_axisbelow(True)

    # one shared x-label for both panels (two separate labels collide across the gutter)
    fig.supxlabel(r"interpolation coefficient $\alpha$   (synthetic retention gradient —"
                  " NOT an unlearning method)", fontsize=9, color=INK, y=-0.03)

    axL.set_ylabel("residual-retention score")
    axL.set_xlim(-0.02, 1.02)
    axL.set_ylim(-0.03, 1.06)
    axL.set_title("(a) full ladder", loc="left", color=INK_2, fontsize=8.5, pad=6)

    axR.set_xlim(0.695, 1.005)
    axR.set_ylim(-0.03, 0.62)
    axR.set_title("(b) zoom: the missed-detection band", loc="left", color=INK_2,
                  fontsize=8.5, pad=6)
    # Rule labels live in panel (a)'s empty lower-left quadrant: in the zoom panel every
    # placement grazed one of the two curves, and the rules only need stating once.
    axL.text(0.01, thr_m + 0.018, f"matched threshold {thr_m:.3f}  (FAR 0.05)",
             ha="left", va="bottom", fontsize=7.4, color=INK_2)
    axL.text(0.01, 0.022, "holdout rule:  AUC > 0.5", ha="left", va="bottom",
             fontsize=7.4, color=INK_2)
    axR.text(x_hold - 0.006, 0.605, f"$\\alpha\\approx${x_hold:.2f}", ha="right", va="top",
             fontsize=7.4, color=MUTED)
    axR.text(x_match + 0.006, 0.605, f"$\\alpha\\approx${x_match:.2f}", ha="left", va="top",
             fontsize=7.4, color=MUTED)
    axR.text((x_hold + x_match) / 2, 0.485,
             "matched: RETENTION DETECTED\nholdout10: “forgotten”",
             ha="center", va="center", fontsize=7.6, color=INK, linespacing=1.5)

    # ---- verdict strip: makes the FLIP a shape rather than a deduction -------------------
    # Two rows, one per reference, filled where that auditor says RETENTION DETECTED and
    # blank where it says forgotten. The rows agree left, DISAGREE across the band, agree
    # right. Segment edges are the crossings themselves (x_hold / x_match), so the
    # disagreement region lines up with the shaded band by construction, not by eye.
    # Fill-vs-blank carries the verdict independently of hue, so the strip survives
    # greyscale print; hue only says which reference the row belongs to.
    x0, x1 = 0.695, 1.005
    axS.set_xlim(x0, x1)
    axS.set_ylim(0, 1)
    axS.axvspan(x_hold, x_match, color=MUTED, alpha=0.13, linewidth=0, zorder=0)
    for x in (x_hold, x_match):
        axS.axvline(x, color=MUTED, lw=0.8, ls=(0, (1, 2.5)), zorder=1)

    # An inline label is drawn ONLY where it fits with padding -- a segment label that
    # overflows its own mark is worse than no label (the holdout DETECTED segment is far
    # too narrow). The fill/blank pattern carries the verdict regardless; the caption below
    # states the mapping.
    IN_PER_DATA = 2.40 / (x1 - x0)                # strip axes is ~2.40in wide
    def fits(text, width_data, pt):
        return len(text) * 0.58 * pt / 72.0 * 1.25 <= width_data * IN_PER_DATA

    for (lo, hi), color, boundary, label in [((0.55, 0.95), S1, x_match, "matched"),
                                             ((0.06, 0.46), S2, x_hold, "holdout10")]:
        axS.add_patch(plt.Rectangle((x0, lo), boundary - x0, hi - lo, facecolor=color,
                                    alpha=0.88, linewidth=0, zorder=2))
        axS.add_patch(plt.Rectangle((boundary, lo), x1 - boundary, hi - lo, facecolor="#efeeea",
                                    edgecolor=AXIS, linewidth=0.5, zorder=2))
        axS.text(x0 - 0.006, (lo + hi) / 2, label, ha="right", va="center",
                 fontsize=7.0, color=INK_2)
        if fits("RETENTION DETECTED", boundary - x0, 6.3):
            axS.text((x0 + boundary) / 2, (lo + hi) / 2, "RETENTION DETECTED", ha="center",
                     va="center", fontsize=6.3, color="#ffffff", zorder=3)
        if fits("forgotten", x1 - boundary, 6.3):
            axS.text((boundary + x1) / 2, (lo + hi) / 2, "forgotten", ha="center",
                     va="center", fontsize=6.3, color=INK_2, zorder=3)

    axS.text(x0 - 0.006, 1.32, "verdict", ha="right", va="center", fontsize=7.0, color=MUTED)
    axS.text(x1, -0.42, "filled = RETENTION DETECTED    ·    blank = forgotten",
             ha="right", va="top", fontsize=6.6, color=MUTED)
    axS.set_xticks([])
    axS.set_yticks([])
    for sp in axS.spines.values():
        sp.set_visible(False)

    handles = [
        Line2D([], [], color=S1, ls="-", lw=1.9, marker="o", ms=4.2,
               markeredgecolor=SURFACE, label="difficulty-matched reference (n=287)"),
        Line2D([], [], color=S2, ls="--", lw=1.9, marker="s", ms=4.0,
               markeredgecolor=SURFACE, label="holdout10 reference (field default)"),
        Line2D([], [], color=MUTED, marker="o", ls="none", ms=7.5, alpha=0.9,
               markeredgecolor=SURFACE, label="sampled rungs (flip directly observed)"),
    ]
    # Identity rides the legend + line style + marker shape, never colour alone. Floating
    # direct labels were dropped here: the two curves converge on the right and the labels
    # collided with the data (marks-and-anatomy sanctions the legend fallback on collision).
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.155),
               labelcolor=INK_2, handlelength=2.4, columnspacing=1.6)

    save(fig, "fig2_missed_detection")
    plt.close(fig)
    return x_hold, x_match, thr_m


def main() -> int:
    e1 = np.load(RESULTS / "exp1_probe_nll.npz")
    e2 = np.load(RESULTS / "exp2_ladder.npz")
    e2b = np.load(RESULTS / "exp2b_fine.npz")
    print("Figure 1 ...")
    a = figure1(e1)
    print(f"  AUCs on plot: {a[0]:.3f} / {a[1]:.3f} / {a[2]:.3f}")
    print("Figure 2 ...")
    xh, xm, thr = figure2(e1, e2, e2b)
    print(f"  band {xh:.3f}-{xm:.3f}, matched threshold {thr:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
