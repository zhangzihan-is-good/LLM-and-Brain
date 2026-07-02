import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def matrix_figure(A, save_path):
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "figure.dpi": 300
    })

    n = A.shape[0]
    eps = 1e-10

    upper_mask = np.triu(np.ones_like(A, dtype=bool), k=1)
    zero_mask = np.abs(A) < eps
    mask = upper_mask | zero_mask
    masked_A = np.ma.masked_where(mask, A)

    vals = A[~mask]
    vmax = np.percentile(np.abs(vals), 98) if len(vals) > 0 else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad((1, 1, 1, 0))

    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=300)

    mesh = ax.pcolormesh(
        np.arange(n + 1),
        np.arange(n + 1),
        masked_A,
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="#f0f0f0",
        linewidth=0.4
    )

    triangle_vertices = [
        (0, 0),  
        (0, n),  
        (n, n), 
        (0, 0)  
    ]
    triangle_codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY
    ]
    triangle_path = Path(triangle_vertices, triangle_codes)
    clip_patch = PathPatch(triangle_path, transform=ax.transData, facecolor='none')
    mesh.set_clip_path(clip_patch)

    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")

    tick_idx = np.arange(0, n, 5)
    ax.set_xticks(tick_idx + 0.5)
    ax.set_yticks(tick_idx + 0.5)
    ax.set_xticklabels(tick_idx, fontsize=8)
    ax.set_yticklabels(tick_idx, fontsize=8)
    ax.tick_params(length=0, labelsize=12)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.plot([0, 0], [0, n], color="#bdbdbd", linewidth=1.0)   
    ax.plot([0, n], [n, n], color="#bdbdbd", linewidth=1.0)   
    ax.plot([0, n], [0, n], color="#bdbdbd", linewidth=1.0)   

    cbar = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    cbar.outline.set_linewidth(0.6)

    ax.set_facecolor((1, 1, 1, 0))
    fig.patch.set_alpha(0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()

id_list = range(1, 18)
for id in id_list:
    brain_matrix = np.load(f"matrix_result/subj{id}/layer0/R_brain.npy")
    brain_delta_matrix = np.load(f"matrix_result/subj{id}/layer0/delta_brain.npy")
    brain_path = f"matrix_figure/brain/subj{id}/R_brain.png"
    brain_delta_path = f"matrix_figure/brain/subj{id}/delta_brain.png"
    os.makedirs(f"matrix_figure/brain/subj{id}", exist_ok=True)
    matrix_figure(brain_matrix, brain_path)
    matrix_figure(brain_delta_matrix, brain_delta_path)

for layer in tqdm(range(0, 32)):
    llm_matrix = np.load(f"matrix_result/subj1/layer{layer}/R_llm.npy")
    llm_delta_matrix = np.load(f"matrix_result/subj1/layer{layer}/delta_llm.npy")
    llm_path = f"matrix_figure/LLM/layer{layer}/R_llm.png"
    llm_delta_path = f"matrix_figure/LLM/layer{layer}/delta_llm.png"
    os.makedirs(f"matrix_figure/LLM/layer{layer}", exist_ok=True)
    matrix_figure(llm_matrix, llm_path)
    matrix_figure(llm_delta_matrix, llm_delta_path)