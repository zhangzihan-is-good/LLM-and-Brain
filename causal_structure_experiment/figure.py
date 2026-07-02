import os
import numpy as np
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
from utils import plot_manifold, judge, plot_final_histogram_li, plot_final_histogram_two_groups
import os
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import pandas as pd

def read_data(process_type, experiment_type, length, id_or_layer):
    if process_type == "brain":
        file_name = "brain-noise/subj"
    elif process_type == "llm":
        file_name = "LLM/layer"
    elif process_type == "noise":
        file_name = "brain-noise/subj"

    if experiment_type == "zunei_eb":
        result = np.load(f"casual-structure/structure-vector/{file_name}{id_or_layer}/{process_type}_result{length}.npy", allow_pickle=True)
    else:
        result = np.load(f"casual-structure/structure-vector/{file_name}{id_or_layer}/{process_type}_{experiment_type}_result{length}.npy", allow_pickle=True)

    return result

PROCESS_LIST = ["brain", "llm", "noise"]


def manifold_no_casual_plot(length, process_type, id_or_layer):
        
    result = read_data(process_type, "yg_compare", length, id_or_layer)
    
    for i, dict in enumerate(result):
        
        points = dict["umap_coords"]
        base_name = dict["type_name"]

        unique_name = f"len{length}_pic{i}"

        fig, ax = plot_manifold(points)
        
        if process_type == "brain":
            file_name = "brain/subj"
        elif process_type == "llm":
            file_name = "LLM/layer"
        elif process_type == "noise":
            file_name = "noise/subj"

        os.makedirs(f"./manifold/agan/{file_name}{id_or_layer}/{base_name}", exist_ok=True)
        fig.savefig(f"./manifold/agan/{file_name}{id_or_layer}/{base_name}/{unique_name}.png")
        plt.close(fig)


def collect_all_scores(
    LEN=3,
    subject_ids=range(1, 18),
    layers=range(0, 32),
    save_each=False,
    each_plot_dir="./zhifang/all_compare_each",
):

    rows = []

    print("目前处理长度:", LEN)

    for subj in subject_ids:
        for layer in layers:
            brain_result = read_data("brain", "yg_compare", LEN, subj)
            llm_result = read_data("llm", "yg_compare", LEN, layer)

            for item in brain_result:
                if item["type_name"] == "noyg":
                    item["type_name"] = "Brain w/o Causal"
                elif item["type_name"] == "yg":
                    item["type_name"] = "Brain w/ Causal"

            for item in llm_result:
                if item["type_name"] == "noyg":
                    item["type_name"] = "LLM w/o Causal"
                elif item["type_name"] == "yg":
                    item["type_name"] = "LLM w/ Causal"

            result = np.concatenate([brain_result, llm_result], axis=0)
            score = judge(result, LEN)

            for pair_name, value in score.items():
                rows.append({
                    "LEN": LEN,
                    "subj": subj,
                    "layer": layer,
                    "unit": f"subj{subj}-layer{layer}",
                    "pair": pair_name,
                    "score": float(value),
                })

            if save_each:
                path = f"{each_plot_dir}/len{LEN}/subj{subj}-layer{layer}.png"
                plot_flat_bar(score, path, metric_name="Score")

    df = pd.DataFrame(rows)
    return df



def scores_to_matrix(df, collapse_by="subj_layer", pair_order=None):

    if collapse_by == "subj_layer":
        index_cols = ["subj", "layer"]
    elif collapse_by == "subj":
        index_cols = ["subj"]
    elif collapse_by == "layer":
        index_cols = ["layer"]

    score_matrix = df.pivot_table(
        index=index_cols,
        columns="pair",
        values="score",
        aggfunc="mean",
    )

    if pair_order is not None:
        score_matrix = score_matrix[pair_order]

    return score_matrix


def p_to_stars(p):
    if p < 1e-4:
        return "****"
    elif p < 1e-3:
        return "***"
    elif p < 1e-2:
        return "**"
    elif p < 5e-2:
        return "*"
    return "ns"


def adjust_pvalues(pvals, method="fdr_bh"):

    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)

    if method is None:
        return pvals

    if method == "bonferroni":
        return np.clip(pvals * m, 0, 1)

    if method == "fdr_bh":
        order = np.argsort(pvals)
        ranked = pvals[order]
        adjusted = ranked * m / np.arange(1, m + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        out = np.empty_like(adjusted)
        out[order] = np.clip(adjusted, 0, 1)
        return out



def run_paired_test(x, y, test="paired_ttest"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    if test == "paired_ttest":
        return float(stats.ttest_rel(x, y, nan_policy="omit").pvalue)

    if test == "wilcoxon":
        try:
            return float(stats.wilcoxon(x, y, zero_method="wilcox").pvalue)
        except ValueError:
            return 1.0


def build_comparisons(names, comparisons="adjacent"):

    if comparisons == "adjacent":
        pairs = [(i, i + 1) for i in range(len(names) - 1)]
    elif comparisons == "all":
        pairs = list(combinations(range(len(names)), 2))
    else:
        pairs = comparisons

    out = []
    for a, b in pairs:
        if isinstance(a, str):
            a = names.index(a)
        if isinstance(b, str):
            b = names.index(b)
        out.append((int(a), int(b)))
    return out


def compute_p_table(score_matrix, comparisons="adjacent", test="paired_ttest", correction="fdr_bh"):
    names = list(score_matrix.columns)
    comp = build_comparisons(names, comparisons)

    raw_pvals = []
    records = []
    for i, j in comp:
        p = run_paired_test(score_matrix.iloc[:, i], score_matrix.iloc[:, j], test=test)
        raw_pvals.append(p)
        records.append({
            "group1": names[i],
            "group2": names[j],
            "idx1": i,
            "idx2": j,
            "p_raw": p,
        })

    p_adj = adjust_pvalues(raw_pvals, method=correction)
    for rec, p in zip(records, p_adj):
        rec["p_adj"] = float(p)
        rec["stars"] = p_to_stars(float(p))

    return pd.DataFrame(records)


def _add_sig_bracket(ax, x1, x2, y, h, text, fontsize=13, linewidth=1.2):
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        color="black",
        linewidth=linewidth,
        clip_on=False,
        zorder=20,
    )
    ax.text(
        (x1 + x2) / 2,
        y + h,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color="black",
        zorder=21,
    )


def plot_summary_bar(
    score_matrix,
    save_path,
    metric_name="Score",
    base_color="#a8cc7a",
    figsize=(10.5, 6.2),
    dpi=300,
    bar_width=0.62,
    error_type="sem",
    comparisons=[(0, 1), (4, 5)],
    test="paired_ttest",
    correction="fdr_bh",
    show_points=True,
    point_size=18,
    point_alpha=0.45,
    jitter=0.11,
    ylim=None,
    yticks=None,
    rotate_xticks=0,
    bracket_gap_ratio=0.015,
    bracket_h_ratio=0.025,
    bracket_step_ratio=0.08,
    show_mean_text=True,     
    mean_text_fmt="{:.3f}",       
    mean_text_fontsize=14,
    mean_text_offset_ratio=0.015, 
):
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "figure.dpi": dpi,
    })

    names = list(score_matrix.columns)
    values = score_matrix.to_numpy(dtype=float)
    means = np.nanmean(values, axis=0)

    if error_type == "sem":
        errors = np.nanstd(values, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(values), axis=0))
    elif error_type == "std":
        errors = np.nanstd(values, axis=0, ddof=1)
    elif error_type is None:
        errors = np.zeros_like(means)

    p_table = compute_p_table(
        score_matrix,
        comparisons=comparisons,
        test=test,
        correction=correction,
    )

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.bar(
        x,
        means,
        width=bar_width,
        color=base_color,
        edgecolor="black",
        linewidth=1.5,
        zorder=3,
    )

    if error_type is not None:
        ax.errorbar(
            x,
            means,
            yerr=errors,
            fmt="none",
            ecolor="black",
            elinewidth=1.25,
            capsize=3.5,
            capthick=1.25,
            zorder=4,
        )

    if show_points:
        rng = np.random.default_rng(0)

        for i in range(values.shape[1]):
            y = values[:, i]
            y = y[np.isfinite(y)]

            if len(y) == 0:
                continue

            if len(y) < 4 or np.std(y) < 1e-10:
                xs = x[i] + rng.uniform(-jitter * 0.35, jitter * 0.35, size=len(y))
            else:
                kde = gaussian_kde(y)
                dens = kde(y)
                dens = dens / dens.max()
                half_width = jitter * (0.15 + 0.85 * dens)
                xs = x[i] + rng.uniform(-1, 1, size=len(y)) * half_width

            ax.scatter(
                xs,
                y,
                s=point_size,
                color="black",
                alpha=point_alpha,
                linewidths=0,
                zorder=5,
            )

    finite_y = values[np.isfinite(values)]
    if len(finite_y) == 0:
        ymin_data, ymax_data = 0.0, 1.0
    else:
        ymin_data, ymax_data = float(np.min(finite_y)), float(np.max(finite_y))

    ymax_data = max(ymax_data, float(np.nanmax(means + errors)))
    yrange = max(ymax_data - ymin_data, 1e-8)


    if show_mean_text:
        text_offset = yrange * mean_text_offset_ratio
        for i, m in enumerate(means):
            if np.isnan(m):
                continue

            if m >= 0:
                y_text = m + errors[i] + text_offset
                va = "bottom"
            else:
                y_text = m - errors[i] - text_offset
                va = "top"

            ax.text(
                x[i],
                y_text,
                mean_text_fmt.format(m),
                ha="center",
                va=va,
                fontsize=mean_text_fontsize,
                fontweight="bold",
                color="black",
                zorder=6,
            )

    if len(p_table) > 0:
        bracket_h = yrange * bracket_h_ratio
        bracket_step = yrange * bracket_step_ratio
        bracket_gap = yrange * bracket_gap_ratio
        used = []

        p_table_sorted = p_table.copy()
        p_table_sorted["span"] = (p_table_sorted["idx2"] - p_table_sorted["idx1"]).abs()
        p_table_sorted = p_table_sorted.sort_values(["span", "idx1"])

        top_of_brackets = ymax_data

        for _, row in p_table_sorted.iterrows():
            i, j = int(row["idx1"]), int(row["idx2"])
            local_top = max(means[i] + errors[i], means[j] + errors[j])

            y = local_top + bracket_gap

            while any(
                not (j < ui or i > uj) and abs(y - uy) < bracket_step * 0.8
                for ui, uj, uy in used
            ):
                y += bracket_step

            _add_sig_bracket(ax, x[i], x[j], y, bracket_h, row["stars"])
            used.append((i, j, y))
            top_of_brackets = max(top_of_brackets, y + bracket_h)

        auto_ymax = max(ymax_data + yrange * 0.12, top_of_brackets + yrange * 0.07)
    else:
        auto_ymax = ymax_data + yrange * 0.12

    auto_ymin = min(0.0, ymin_data - yrange * 0.08)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(auto_ymin, auto_ymax)

    if yticks is not None:
        ax.set_yticks(yticks)

    ax.axhline(0, color="black", linewidth=1.0, zorder=2)
    ax.set_ylabel(metric_name, fontsize=20, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        names,
        rotation=rotate_xticks,
        ha="center" if rotate_xticks == 0 else "right",
        multialignment="center",
        fontsize=14,
        fontweight="bold"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", which="major", labelsize=14, width=1.2, length=4)
    ax.grid(False)
    ax.set_xlim(-0.6, len(names) - 0.4)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return p_table


def plot_flat_bar(
    score_dict,
    save_path,
    metric_name="Score",
    base_color="#a8cc7a",
    figsize=(10, 6),
    dpi=300,
    bar_width=0.62,
    ylim=None,
    yticks=None,
    rotate_xticks=0,):

    names = list(score_dict.keys())
    values = np.array(list(score_dict.values()), dtype=float)
    x = np.arange(len(names))

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": dpi,
    })

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.bar(x, values, width=bar_width, color=base_color, edgecolor="black", linewidth=1.5, zorder=3)
    ax.axhline(0, color="black", linewidth=1.0, zorder=2)

    ax.set_ylabel(metric_name, fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=rotate_xticks, ha="center" if rotate_xticks == 0 else "right", multialignment="center", fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", which="major", labelsize=11, width=1.2, length=4)
    ax.grid(False)
    ax.set_xlim(-0.6, len(names) - 0.4)

    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close(fig)



LEN=4

for id in tqdm(range(1, 18)):
    
    manifold_no_casual_plot(LEN, "brain", id)
    manifold_no_casual_plot(LEN, "noise", id)
    
    brain_result = read_data("brain", "zunei_eb", LEN, id)
    brain_path = f"./zhifang/brain/len{LEN}/eb_{id}"

    noise_result = read_data("noise", "zunei_eb", LEN, id)
    noise_path = f"./zhifang/noise/len{LEN}/eb_{id}"

    brain_score = judge(brain_result, LEN)
    noise_score = judge(noise_result, LEN)

    plot_final_histogram_li(brain_score, brain_path)
    plot_final_histogram_li(noise_score, noise_path)

    brain_eb_result = read_data("brain", "zunei_eb", LEN, id)
    brain_noeb_result = read_data("brain", "noeb", LEN, id)
    noise_eb_result = read_data("noise", "zunei_eb", LEN, id)
    noise_noeb_result = read_data("noise", "noeb", LEN, id)

    brain_eb_score = judge(brain_eb_result, LEN)
    brain_noeb_score = judge(brain_noeb_result, LEN)
    noise_eb_score = judge(noise_eb_result, LEN)
    noise_noeb_score = judge(noise_noeb_result, LEN)

    brain_path = f"./zhifang/compare/brain/len{LEN}/{id}.png"
    noise_path = f"./zhifang/compare/noise/len{LEN}/{id}.png"

    plot_final_histogram_two_groups(brain_eb_score, brain_noeb_score, brain_path)
    plot_final_histogram_two_groups(noise_eb_score, noise_noeb_score, noise_path)

for layer in tqdm(range(0, 32)):
    manifold_no_casual_plot(LEN, "llm", layer)
    
    llm_result = read_data("llm", "zunei_eb", LEN, layer)
    llm_path = f"./zhifang/llm/len{LEN}/eb_{layer}"
    
    llm_score = judge(llm_result, LEN)
    plot_final_histogram_li(llm_score, llm_path)

    llm_eb_result = read_data("llm", "zunei_eb", LEN, layer)
    llm_noeb_result = read_data("llm", "noeb", LEN, layer)

    llm_eb_score = judge(llm_eb_result, LEN)
    llm_noeb_score = judge(llm_noeb_result, LEN)

    llm_path = f"./zhifang/compare/llm/len{LEN}/{layer}.png"
    plot_final_histogram_two_groups(llm_eb_score, llm_noeb_score, llm_path)


def mean_list(input_list):
    a = input_list[0]
    output = {}
    for key, _ in a.items():
        value = 0
        for i, result in enumerate(input_list):
            value += result[key]
        value /= len(input_list)
        output[key] = value
    
    return output


eb_list = []
noeb_list = []
for id in range(1, 18):

    brain_eb_result = read_data("brain", "zunei_eb", LEN, id)
    brain_noeb_result = read_data("brain", "noeb", LEN, id)

    brain_eb_score = judge(brain_eb_result, LEN)
    brain_noeb_score = judge(brain_noeb_result, LEN)

    eb_list.append(brain_eb_score)
    noeb_list.append(brain_noeb_score)

eb_mean = mean_list(eb_list)
noeb_mean = mean_list(noeb_list)
path = f"brain_mean{LEN}.png"
plot_final_histogram_two_groups(eb_mean, noeb_mean, path)




df = collect_all_scores(
    LEN=LEN,
    subject_ids=range(1, 18),
    layers=range(0, 32),
    save_each=False,
)

os.makedirs(f"./zhifang/all_compare/len{LEN}", exist_ok=True)

score_matrix = scores_to_matrix(df, collapse_by="subj_layer")

p_table = plot_summary_bar(
    score_matrix,
    save_path=f"./zhifang/all_compare/len{LEN}/summary_bar.png",
    metric_name="Score",
    error_type="sem",
    comparisons=[(0, 1), (4, 5)], 
    test="paired_ttest",
    correction="fdr_bh",
    show_points=False,            
    point_size=18,
    point_alpha=0.45,
    jitter=0.11,
    show_mean_text=True,            
    mean_text_fmt="{:.3f}",
    bracket_gap_ratio=0.1,
    bracket_h_ratio=0.025,
    bracket_step_ratio=0.08,
)

print(p_table)

