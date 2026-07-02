import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import random
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import ttest_rel

plt.rcParams.update({
    'font.family': 'sans-serif',
    "font.family": "serif",     
    "font.serif": ["STIXGeneral"], 
    "mathtext.fontset": "stix",    
    'font.size': 7,      
    'axes.titlesize': 8,  
    'axes.labelsize': 7,   
    'xtick.labelsize': 6,    
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.dpi': 300,      
    'pdf.fonttype': 42,     
    'ps.fonttype': 42
})

def nature_smooth_saturation_noise(data, max_std=0.025, sensitivity=20, window=4, order=2):
    data = np.array(data)
    
    std_dev = max_std * np.tanh(sensitivity * np.abs(data))
    
    noise = np.random.normal(0, std_dev)
    noisy_data = data + noise
    
    if window >= len(noisy_data):
        window = len(noisy_data) // 2 * 2 - 1 
    
    smoothed_data = savgol_filter(noisy_data, window_length=window, polyorder=order)
    
    return smoothed_data

def load_braindata(id):

    brain_story_path = f"story_{id}.npy"
    brain_eb_path = f"eb_{id}.npy"

    brain_story = np.load(brain_story_path)
    brain_eb = np.load(brain_eb_path)

    mask_B_row = ~np.isnan(brain_story).all(axis=1)
    mask_E_row = ~np.isnan(brain_eb).all(axis=1)

    mask_row = mask_B_row & mask_E_row

    brain_story_mask_row = brain_story[mask_row]
    brain_eb_mask_row = brain_eb[mask_row]

    mask_B_col = ~np.isnan(brain_story_mask_row).any(axis=0)
    mask_E_col = ~np.isnan(brain_eb_mask_row).any(axis=0)
    mask_col = mask_B_col & mask_E_col

    brain_story_mask = brain_story_mask_row[:, mask_col]
    brain_eb_mask = brain_eb_mask_row[:, mask_col]

    return brain_story_mask, brain_eb_mask, mask_row

def load_dict(is_mask=False):

    guo_react = np.load("result/llama3.1-8B/new_guo_react.npy", allow_pickle=True).item()
    yin_react = np.load("result/llama3.1-8B/new_yin_react.npy", allow_pickle=True).item()
    ind_react = np.load("result/llama3.1-8B/new_ind_react.npy", allow_pickle=True).item()
    other_react = np.load("result/llama3.1-8B/new_other_react.npy", allow_pickle=True).item()
    yg_react = np.load("result/llama3.1-8B/new_yg_react.npy", allow_pickle=True).item()

    return guo_react, yin_react, ind_react, other_react, yg_react

def plot_matrix(brain_or_llm, is_mask=False):

    guo_react, _, _, _, _ = load_dict(is_mask)

    def plot_figure(matrix, color_max, num, brain_or_llm, is_mask):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        
        im0 = ax.imshow(matrix, vmin=-color_max, vmax=color_max, cmap="coolwarm", interpolation='nearest')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1) 

        for spine in ax.spines.values():
                spine.set_linewidth(0.5)
        ax.tick_params(direction='out', length=3, width=0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

        cbar = fig.colorbar(im0, cax=cax)

        if is_mask is False:
            if brain_or_llm == "brain":
                os.makedirs(f"result/llama3.1-8B/matrix/{brain_or_llm}", exist_ok=True)
                plt.savefig(f"llama3.1-8B/matrix/{brain_or_llm}/subj{num+1}.png", bbox_inches='tight', dpi=400)
                plt.close()
            elif brain_or_llm == "llm":
                os.makedirs(f"result/llama3.1-8B/matrix/{brain_or_llm}", exist_ok=True)
                plt.savefig(f"result/llama3.1-8B/matrix/{brain_or_llm}/layer{num}.png", bbox_inches='tight', dpi=400)
                plt.close()
        
    max_list = []
    matrix_list = []

    if brain_or_llm == "brain":
        for id in tqdm(range(1, 18)):
            brain_matrix = guo_react[f'subj{id}'][f"layer0"]["brain_matrix"].T
            brain_matrix_abs = np.abs(brain_matrix)
            brain_matrix_vis = (brain_matrix_abs - brain_matrix_abs.mean()) / brain_matrix_abs.std()
            brain_max = abs(brain_matrix_vis).max()
            
            matrix_list.append(brain_matrix_vis)
            max_list.append(brain_max)

    elif brain_or_llm == "llm":
        for layer in tqdm(range(0, 32)):
            llm_matrix = guo_react[f'subj1'][f"layer{layer}"]["llm_matrix"].T
            llm_matrix_abs = np.abs(llm_matrix)
            llm_matrix_vis = (llm_matrix_abs - llm_matrix_abs.mean()) / llm_matrix_abs.std()
            llm_max = abs(llm_matrix_vis).max()

            matrix_list.append(llm_matrix_vis)
            max_list.append(llm_max)

    color_max = min(max_list)
    for i, matrix in enumerate(tqdm(matrix_list)):
        plot_figure(matrix, color_max, i, brain_or_llm, is_mask)
    
    return matrix_list, max_list

def plot_reactnum_line(is_mask, brain_or_llm):

    guo_react, yin_react, ind_react, other_react, _ = load_dict(is_mask)
    
    if brain_or_llm == "brain":
        guo_list = []
        yin_list = []
        ind_list = []
        other_list = []
        for id in tqdm(range(1, 18)):
            guo_react_num = guo_react[f'subj{id}'][f"layer0"]["brain_react"]
            yin_react_num = yin_react[f'subj{id}'][f"layer0"]["brain_react"]
            ind_react_num = ind_react[f'subj{id}'][f"layer0"]["brain_react"]
            other_react_num = other_react[f'subj{id}'][f"layer0"]["brain_react"]

            guo_list.append(guo_react_num)
            yin_list.append(yin_react_num)
            ind_list.append(ind_react_num)
            other_list.append(other_react_num)

        x = range(1, 18)
        labels = ['Result events', 'Cause events', 'Independent events', 'Other events']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        for i in range(4):
            ax.plot(x, lists[i], 
                    label=labels[i], 
                    color=colors[i], 
                    linewidth=1.0,
                    alpha=0.9, 
                    antialiased=True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor('black')
        ax.tick_params(which='major', direction='in', length=3.5, width=0.8, labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, 
                fontsize=12, 
                loc='upper right', 
                handlelength=1.5,
                labelspacing=0.2)
        if brain_or_llm == "brain":
            ax.set_xlabel('Subject Index', fontsize=14, fontweight='bold')
        else:
            ax.set_xlabel('LLM Layer Index', fontsize=14, fontweight='bold')
        ax.set_ylabel('Reactivation Index', fontsize=14, fontweight='bold')
        plt.tight_layout(pad=0.2)
        plt.savefig(f"reactnum_{brain_or_llm}.png", bbox_inches='tight', transparent=True, dpi=300)

    elif brain_or_llm == "llm":
        guo_list = []
        yin_list = []
        ind_list = []
        other_list = []
        for layer in range(0, 32):
            guo_react_num = guo_react[f'subj1'][f"layer{layer}"]["llm_react"]
            yin_react_num = yin_react[f'subj1'][f"layer{layer}"]["llm_react"]
            ind_react_num = ind_react[f'subj1'][f"layer{layer}"]["llm_react"]
            other_react_num = other_react[f'subj1'][f"layer{layer}"]["llm_react"]

            guo_list.append(guo_react_num)
            yin_list.append(yin_react_num)
            ind_list.append(ind_react_num)
            other_list.append(other_react_num)

        x = range(1, 33)
        lists = [guo_list, yin_list, ind_list, other_list]
        labels = ['Result events', 'Cause events', 'Independent events', 'Other events']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

        for i in range(4):
            ax.plot(x, lists[i], 
                    label=labels[i], 
                    color=colors[i], 
                    linewidth=1.0,    
                    alpha=0.9, 
                    antialiased=True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor('black')
        ax.tick_params(which='major', direction='in', length=3.5, width=0.8, labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, 
                fontsize=12, 
                loc='upper right', 
                handlelength=1.5,
                labelspacing=0.2)

        if brain_or_llm == "brain":
            ax.set_xlabel('Subject Index', fontsize=14, fontweight='bold')
        else:
            ax.set_xlabel('LLM Layer Index', fontsize=14, fontweight='bold')
        ax.set_ylabel('Reactivation Index', fontsize=14, fontweight='bold')
        plt.tight_layout(pad=0.2)
        plt.savefig(f"reactnum_{brain_or_llm}.png", bbox_inches='tight', transparent=True, dpi=300)

    return lists


def p_to_star(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'


def add_sig_bar(ax, x1, x2, y, h, text, fontsize=13):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=1.1, c='black', clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text,
            ha='center', va='bottom',
            fontsize=fontsize, fontweight='bold')


def plot_reactnum(is_mask, brain_or_llm):

    guo_react, yin_react, ind_react, other_react, _ = load_dict(is_mask)

    guo_list = []
    yin_list = []
    ind_list = []
    other_list = []

    if brain_or_llm == "brain":
        for id in tqdm(range(1, 18)):
            guo_react_num = guo_react[f'subj{id}']["layer0"]["brain_react"]
            yin_react_num = yin_react[f'subj{id}']["layer0"]["brain_react"]
            ind_react_num = ind_react[f'subj{id}']["layer0"]["brain_react"]
            other_react_num = other_react[f'subj{id}']["layer0"]["brain_react"]

            guo_list.append(float(np.asarray(guo_react_num).squeeze()))
            yin_list.append(float(np.asarray(yin_react_num).squeeze()))
            ind_list.append(float(np.asarray(ind_react_num).squeeze()))
            other_list.append(float(np.asarray(other_react_num).squeeze()))

        save_name = f"reactnum_{brain_or_llm}_category_scatter.png"

    elif brain_or_llm == "llm":
        for layer in range(0, 32):
            guo_react_num = guo_react['subj1'][f"layer{layer}"]["llm_react"]
            yin_react_num = yin_react['subj1'][f"layer{layer}"]["llm_react"]
            ind_react_num = ind_react['subj1'][f"layer{layer}"]["llm_react"]
            other_react_num = other_react['subj1'][f"layer{layer}"]["llm_react"]

            guo_list.append(float(np.asarray(guo_react_num).squeeze()))
            yin_list.append(float(np.asarray(yin_react_num).squeeze()))
            ind_list.append(float(np.asarray(ind_react_num).squeeze()))
            other_list.append(float(np.asarray(other_react_num).squeeze()))

        save_name = f"reactnum_{brain_or_llm}_category_scatter.png"

    lists = [
        np.asarray(guo_list, dtype=float),
        np.asarray(yin_list, dtype=float),
        np.asarray(ind_list, dtype=float),
        np.asarray(other_list, dtype=float)
    ]

    labels = ['Result', 'Cause', 'Independent', 'Other']
    colors = ['#4C78A8', '#F58518', '#54A24B', '#E45756']
    x_pos = np.arange(1, 5)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=300)

    rng = np.random.default_rng(0)

    for i, values in enumerate(lists):
        x = x_pos[i]

        jitter = rng.normal(loc=0.0, scale=0.045, size=len(values))

        ax.scatter(
            np.full(len(values), x) + jitter,
            values,
            s=45,
            color=colors[i],
            alpha=0.45,
            edgecolor='none',
            zorder=2
        )

        mean = np.nanmean(values)
        sem = np.nanstd(values, ddof=1) / np.sqrt(np.sum(~np.isnan(values)))

        ax.errorbar(
            x,
            mean,
            yerr=sem,
            fmt='s',
            markersize=6,
            color=colors[i],
            markerfacecolor=colors[i],
            markeredgecolor=colors[i],
            elinewidth=2.0,
            capsize=4,
            capthick=1.5,
            zorder=4
        )

    ax.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.65, zorder=1)


    sig_pairs = [
        (0, 1), 
        (0, 2),  
        (0, 3),  
        (1, 3),  
        (2, 3),  
    ]

    all_values = np.concatenate(lists)
    y_min = np.nanmin(all_values)
    y_max = np.nanmax(all_values)
    y_range = y_max - y_min
    if y_range == 0:
        y_range = 1.0

    bar_h = y_range * 0.035
    start_y = y_max + y_range * 0.08
    step_y = y_range * 0.12

    for idx, (i, j) in enumerate(sig_pairs):
        stat, p = ttest_rel(lists[i], lists[j], nan_policy='omit')
        star = p_to_star(p)

        if star == 'n.s.':
            continue

        y = start_y + idx * step_y
        add_sig_bar(ax, x_pos[i], x_pos[j], y, bar_h, star)

    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(y_min - y_range * 0.12, start_y + len(sig_pairs) * step_y + y_range * 0.05)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=16, fontweight='bold')

    for tick_label, color in zip(ax.get_xticklabels(), colors):
        tick_label.set_color(color)

    ax.set_ylabel('Reactivation index', fontsize=18, fontweight='bold')

    ax.tick_params(which='major', direction='out', length=3.5, width=0.8, labelsize=16)

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor('black')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=0.3)
    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    plt.show()

    return lists


def frobenius_similarity_list(matrix_list1, matrix_list2):
   
    result = np.zeros((len(matrix_list1), len(matrix_list2)))

    def frobenius_similarity(S1, S2, eps=1e-8):
        diff_norm = np.linalg.norm(S1 - S2, ord='fro')
        return diff_norm

    for i, matrix1 in enumerate(matrix_list1):
        for j, matrix2 in enumerate(matrix_list2):
            sim = frobenius_similarity(matrix1, matrix2)
            result[i, j] = sim

    return result



is_mask = False
brain_react_lists = plot_reactnum(is_mask=is_mask, brain_or_llm="brain")
llm_react_lists = plot_reactnum(is_mask=is_mask, brain_or_llm="llm")

brain_matrix_list, brain_max_list = plot_matrix(is_mask=is_mask, brain_or_llm="brain")
llm_matrix_list, llm_max_list = plot_matrix(is_mask=is_mask, brain_or_llm="llm")
np.save("brain_matrix_list.npy", brain_matrix_list)
np.save("llm_matrix_list.npy", llm_matrix_list)

brain_matrix_stack = np.array(brain_matrix_list)
average_brain_matrix = np.mean(brain_matrix_stack, axis=0)
llm_matrix_stack = np.array(llm_matrix_list)
average_llm_matrix = np.mean(llm_matrix_stack, axis=0)


def plot_figure_all(matrix, color_max, save_path):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    
    im0 = ax.imshow(matrix, vmin=-color_max, vmax=color_max, cmap="coolwarm", interpolation='nearest')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1) 

    for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    ax.tick_params(direction='out', length=3, width=0.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    cbar = fig.colorbar(im0, cax=cax)
    plt.savefig(save_path, bbox_inches='tight', dpi=400)
    plt.close()

plot_figure_all(average_brain_matrix, min(brain_max_list), save_path="result/llama3.1-8B/matrix/brain/subj_all.png")
plot_figure_all(average_llm_matrix, min(llm_max_list), save_path="result/llama3.1-8B/matrix/llm/layer_all.png")





