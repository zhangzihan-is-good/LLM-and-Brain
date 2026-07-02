import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

brain_matrix_list = np.load("brain_matrix_list.npy")
llm_matrix_list = np.load("llm_matrix_list.npy")

plt.rcParams.update({
    'font.family': 'sans-serif',
    "font.family": "serif",        
    "font.serif": ["STIXGeneral"], 
    "mathtext.fontset": "stix",    
    'font.size': 7,        
    'axes.titlesize': 8,    
    'axes.labelsize': 10,    
    'xtick.labelsize': 10,   
    'ytick.labelsize': 10,
    'legend.fontsize': 6,
    'figure.dpi': 300,       
    'pdf.fonttype': 42,     
    'ps.fonttype': 42
})

def frobenius_similarity_list(matrix_list1, matrix_list2, name=None):
   
    result = []
    def frobenius_similarity(S1, S2, eps=1e-8):
        diff_norm = np.dot(S1.flatten(), S2.flatten()) / (np.linalg.norm(S1) * np.linalg.norm(S2))
        return diff_norm

    for i, matrix1 in enumerate(matrix_list1):
        for j, matrix2 in enumerate(matrix_list2):
            if name == "bb":
                if j > i:
                    sim = frobenius_similarity(matrix1, matrix2)
                    result.append(sim)
            else:
                sim = frobenius_similarity(matrix1, matrix2)
                result.append(sim)

    return result

def mean_ci_95(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    ci = 1.96 * se
    return mean, mean - ci, mean + ci


def permutation_pvalue(a, b, n_perm=10000, seed=42):
    rng = np.random.default_rng(seed)

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    observed = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    n_a = len(a)

    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        new_a = perm[:n_a]
        new_b = perm[n_a:]
        stat = new_a.mean() - new_b.mean()
        if abs(stat) >= abs(observed):
            count += 1

    return count / n_perm


def p_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def add_sig_bar(ax, x1, x2, y, h, text):
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        color="black",
        linewidth=1.1,
        clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2,
        y + h + 0.006,
        text,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="black",
    )


def plot_reactivation_similarity_box(
    bb_sim,
    bl_sim,
    bn_sim,
    save_path="cross_system_reactivation_similarity_box.png",
    title="Cross-system reactivation similarity",
    show_points=True,
):
    groups = [
        np.asarray(bb_sim, dtype=float),
        np.asarray(bl_sim, dtype=float),
        np.asarray(bn_sim, dtype=float),
    ]

    group_names = [
        "Brain-Brain",
        "Brain-LLM",
        "Brain-Noise",
    ]

    colors = ["#2f6fc0", "#f28e2b", "#8e44ad"]
    positions = [1, 2, 3]

    plt.rcParams.update({
        'font.family': 'sans-serif',
        "font.family": "serif",     
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",   
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=300)

    box = ax.boxplot(
        groups,
        positions=positions,
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops=dict(color="black", linewidth=1.4),
        boxprops=dict(linewidth=1.3),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)
        patch.set_edgecolor(color)

    for i, color in enumerate(colors):

        for w in box["whiskers"][2*i:2*i+2]:
            w.set_color(color)
        for c in box["caps"][2*i:2*i+2]:
            c.set_color(color)

    rng = np.random.default_rng(7)
    stats = []

    for x, color, pos in zip(groups, colors, positions):
        x = x[~np.isnan(x)]

        if show_points:
            jitter = rng.normal(loc=0, scale=0.055, size=len(x))
            ax.scatter(
                np.full_like(x, pos, dtype=float) + jitter,
                x,
                s=14,
                color=color,
                alpha=0.55,
                edgecolors="none",
                zorder=3,
            )

        mean, ci_low, ci_high = mean_ci_95(x)
        stats.append((mean, ci_low, ci_high))

        ax.axhline(
            mean,
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
            zorder=1,
        )

        ax.errorbar(
            pos,
            mean,
            yerr=[[mean - ci_low], [ci_high - mean]],
            fmt="s",
            color="black",
            ecolor="black",
            elinewidth=1.2,
            capsize=3,
            markersize=4.5,
            zorder=6,
        )

    p_bb_bl = permutation_pvalue(groups[0], groups[1], n_perm=10000, seed=1)
    p_bb_bn = permutation_pvalue(groups[0], groups[2], n_perm=10000, seed=2)
    p_bl_bn = permutation_pvalue(groups[1], groups[2], n_perm=10000, seed=3)

    ax.set_ylim(-0.20, 0.32)
    ax.set_xlim(0.55, 3.75)

    add_sig_bar(ax, 1, 2, 0.255, 0.015, p_to_stars(p_bb_bl))
    add_sig_bar(ax, 1, 3, 0.295, 0.015, p_to_stars(p_bb_bn))
    add_sig_bar(ax, 2, 3, 0.215, 0.015, p_to_stars(p_bl_bn))

    ax.set_ylabel("Cross-system Similarity", fontweight="bold", fontsize=18)
    ax.set_xlabel("")

    ax.set_xticks(positions)
    ax.set_xticklabels(group_names, fontsize=16)

    for tick, color in zip(ax.get_xticklabels(), colors):
        tick.set_color(color)
        tick.set_fontweight("bold")
        tick.set_fontsize(16)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
    ax.grid(axis="x", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", transparent=False, dpi=300)
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()

noise_list = [np.random.randn(28, 28) for _ in range(len(llm_matrix_list))]
bb_sim = frobenius_similarity_list(brain_matrix_list, brain_matrix_list, name="bb")
bl_sim = frobenius_similarity_list(brain_matrix_list, llm_matrix_list)
bn_sim= frobenius_similarity_list(brain_matrix_list, noise_list)

S_all = np.concatenate([bb_sim, bl_sim, bn_sim])
labels = np.array([0]*len(bb_sim) + [1]*len(bl_sim) + [2]*len(bn_sim))


plot_reactivation_similarity_box(
    bb_sim,
    bl_sim,
    bn_sim,
    save_path="cross_system_reactivation_similarity_box.png",
    show_points=True,
)