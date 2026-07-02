import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import wasserstein_distance


np.random.seed(0)

C_noshort = np.load("C_noshort.npy")

matrix_dict = {"C_noshort":C_noshort}
matrix = matrix_dict["C_noshort"]


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

    return brain_story_mask, brain_eb_mask, mask_row, mask_col

def load_llmdata(layer, mask):

    llm_story_path = f"story/{layer}/hidden_states.npy"
    llm_eb_path = f"eb/{layer}/hidden_states.npy"

    llm_story = np.load(llm_story_path)
    llm_eb = np.load(llm_eb_path)

    llm_story_mask = llm_story[mask]
    llm_eb_mask = llm_eb[mask]

    return llm_story_mask, llm_eb_mask

def compute_brain_error(matrix, new=None, new_delta=None):
    R_list = []
    delta_R_list = []
    mse_list = []
    delta_mse_list = []
    id_list = range(1, 18)
    for id in id_list:
        _, _, mask, _ = load_braindata(id)
        R = np.load(f"matrix_result/subj{id}/layer0/R_brain.npy")
        delta_R = np.load(f"matrix_result/subj{id}/layer0/delta_brain.npy")
        R_list.append(R)
        delta_R_list.append(delta_R)

        if new is None and new_delta is None:
            if matrix.shape[0] > 32:
                mse = np.mean((R- matrix[mask][:, mask])**2)
                delta_mse = np.mean((delta_R- matrix[mask][:, mask])**2)
            else:
                mse = np.mean((R- matrix)**2)
                delta_mse = np.mean((delta_R- matrix)**2)
        else:
            mse = np.mean((R- new)**2)
            delta_mse = np.mean((delta_R- new_delta)**2)

        mse_list.append(mse)
        delta_mse_list.append(delta_mse)

    return R_list, delta_R_list, mse_list, delta_mse_list

def compute_llm_error(matrix, new=None, new_delta=None):

    _, _, mask, _ = load_braindata(1)
    R_list = []
    delta_R_list = []
    mse_list = []
    delta_mse_list = []
    for layer in range(0, 32):
        R = np.load(f"matrix_result/subj1/layer{layer}/R_llm.npy")
        delta_R = np.load(f"matrix_result/subj1/layer{layer}/delta_llm.npy")
        R_list.append(R)
        delta_R_list.append(delta_R)

        mask_matrix = None
        if new is None and new_delta is None:
            if matrix.shape[0] > 32:
                mse = np.mean((R- matrix[mask][:, mask])**2)
                delta_mse = np.mean((delta_R- matrix[mask][:, mask])**2)
                mask_matrix = matrix[mask][:, mask]
            else:
                mse = np.mean((R- matrix)**2)
                delta_mse = np.mean((delta_R- matrix)**2)
                mask_matrix = matrix
        else:
            mse = np.mean((R- new)**2)
            delta_mse = np.mean((delta_R- new_delta)**2)


        mse_list.append(mse)
        delta_mse_list.append(delta_mse)

    return R_list, delta_R_list, mse_list, delta_mse_list, mask_matrix

def matrix_distance(A, B):
    return wasserstein_distance(A.flatten(), B.flatten())

def generate_lower_permutation(mat, lower_idx, eps=1e-8, seed=None):

    if seed is not None:
        np.random.seed(seed)

    vals = mat[lower_idx]
    nz_mask = np.abs(vals) > eps
    nz_vals = vals[nz_mask]
    K = len(nz_vals)

    perm_pos = np.random.choice(len(vals), size=K, replace=False)
    perm_vals = np.random.permutation(K)

    return perm_pos, perm_vals, nz_mask

def apply_nonzero_permutation(mat, lower_idx, perm_pos, perm_vals, nz_mask):
    vals = mat[lower_idx]
    nz_vals = vals[nz_mask]

    new_mat = np.zeros_like(mat)
    new_mat[
        lower_idx[0][perm_pos],
        lower_idx[1][perm_pos]
    ] = nz_vals[perm_vals]

    return new_mat


def plot(brain_list, shuffle, name):
    save_dir = f"./agan/error"
    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "figure.dpi": 300
    })

    def draw_3d_points(ax, x, y, base_color, edge_color, mid_color, inner_color,
                       label=None, zbase=5):

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        x_range = max(x.max() - x.min(), 1.0)
        y_range = max(y.max() - y.min(), 1e-6)

        dx = 0.008 * x_range
        dy = 0.012 * y_range

        sizes_outer = np.full(len(x), 160.0)
        sizes_mid = np.full(len(x), 110.0)
        sizes_inner = np.full(len(x), 60.0)
        sizes_shadow = np.full(len(x), 170.0)

        ax.scatter(
            x + dx, y - dy,
            s=sizes_shadow,
            c='k',
            alpha=0.10,
            linewidths=0,
            zorder=zbase
        )

        ax.scatter(
            x, y,
            s=sizes_outer,
            c=base_color,
            edgecolors=edge_color,
            linewidths=0.9,
            alpha=0.98,
            zorder=zbase + 2,
            label=label
        )

        ax.scatter(
            x, y,
            s=sizes_mid,
            c=mid_color,
            linewidths=0,
            alpha=0.95,
            zorder=zbase + 3
        )

        ax.scatter(
            x, y,
            s=sizes_inner,
            c=inner_color,
            linewidths=0,
            alpha=0.92,
            zorder=zbase + 4
        )

    def draw_3d_hline(ax, y, xmin, xmax, main_color, highlight_color, zbase=1):

        x_range = max(xmax - xmin, 1.0)
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        line_dx = 0.004 * x_range
        line_dy = 0.006 * max(abs(y_range), 1e-6)


        ax.plot(
            [xmin + line_dx, xmax + line_dx],
            [y - line_dy, y - line_dy],
            color='k',
            linewidth=4.8,
            alpha=0.10,
            solid_capstyle='round',
            zorder=zbase
        )

        ax.plot(
            [xmin, xmax],
            [y, y],
            color=main_color,
            linewidth=3.6,
            solid_capstyle='round',
            zorder=zbase + 1
        )

        ax.plot(
            [xmin, xmax],
            [y, y],
            color=highlight_color,
            linewidth=1.8,
            alpha=0.95,
            solid_capstyle='round',
            zorder=zbase + 2
        )

    y = np.asarray(brain_list, dtype=float)
    y_shuf = np.asarray(shuffle, dtype=float)

    n = len(y)

    x = np.arange(n, dtype=float)+1

    mean_y = y.mean()
    mean_shuffle = y_shuf.mean()

    x_range = max(x.max() - x.min(), 1.0)
    offset = 0.06 * x_range / max(n - 1, 1)

    x_o = x - offset
    x_s = x + offset

    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    all_y = np.concatenate([y, y_shuf])
    y_min, y_max = all_y.min(), all_y.max()
    y_pad = max(0.08 * (y_max - y_min + 1e-8), 1e-4)

    padding = (x[-1] - x[0]) * 0.1 

    ax.set_xlim(x[0] - padding, x[-1] + padding)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    draw_3d_hline(
        ax, mean_y, x[0] - 0.3, x[-1] + 0.3,
        main_color='#1F5FA8',   
        highlight_color='#5FA8FF',
        zbase=1
    )
    draw_3d_hline(
        ax, mean_shuffle, x[0] - 0.3, x[-1] + 0.3,
        main_color='#B22222',    
        highlight_color='#FF8A8A', 
        zbase=1
    )

    draw_3d_points(
        ax, x_o, y,
        base_color='#2C6DB2',
        edge_color='#1E4F85',
        mid_color='#5FA8FF',
        inner_color='#B9DCFF',
        label='Original',
        zbase=5
    )

    draw_3d_points(
        ax, x_s, y_shuf,
        base_color='#C0392B',
        edge_color='#8E2B20',
        mid_color='#FF7F6E',
        inner_color='#FFD0C8',
        label='Shuffle',
        zbase=5
    )

    ax.set_xlabel("Layer Index", fontsize=18, fontweight="bold")
    ax.set_ylabel("Causal Consistency",fontsize=18, fontweight="bold")
    # ax.set_title(f"{name}_change")

    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(5))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    ax.tick_params(axis='both', labelsize=16, width=1.0, length=4)

    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.25)

    ax.legend(frameon=False, fontsize=16)

    save_path = os.path.join(save_dir, f"{name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()



brain_R_list, brain_delta_R_list, brain_mse_list, brain_delta_mse_list = compute_brain_error(matrix)
llm_R_list, llm_delta_R_list, llm_mse_list, llm_delta_mse_list, mask_matrix = compute_llm_error(matrix)


lower_idx = np.tril_indices(mask_matrix.shape[0], k=-1)
perm_pos, perm_vals, nz_mask = generate_lower_permutation(mask_matrix, lower_idx, seed=42)

sh_mat = apply_nonzero_permutation(mask_matrix, lower_idx, perm_pos, perm_vals, nz_mask)

_, _, brain_sh_mse_list, brain_sh_delta_mse_list = compute_brain_error(sh_mat)
_, _, llm_sh_mse_list, llm_sh_delta_mse_list, _ = compute_llm_error(sh_mat)


plot(brain_mse_list, brain_sh_mse_list, "brain_R")
plot(brain_delta_mse_list, brain_sh_delta_mse_list, "brain_delta_R")


plot(llm_mse_list, llm_sh_mse_list, "llm_R")
plot(llm_delta_mse_list, llm_sh_delta_mse_list, "llm_delta_R")


print("已完成")