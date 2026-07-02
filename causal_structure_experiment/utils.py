import numpy as np
import os
import gudhi as gd
from gudhi.representations import PersistenceImage
import time
from sklearn.decomposition import PCA
from tqdm import tqdm
import sys
import random
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import silhouette_score
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle, Polygon, Patch
from matplotlib.lines import Line2D
import plotly.graph_objects as go

seed = 42
random.seed(seed)
np.random.seed(seed)

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

def load_struct(id, struct_len):
   
    C_noshort_structure = np.load(f"casual-structure/dict{struct_len}/subj{id}/C_noshort_5.npy", allow_pickle=True).item()
    matrix_dict = {"C_noshort":C_noshort_structure}

    return matrix_dict

def process_struct_noeb(story, structs, name, struct_len):
    data_list = []
    if struct_len == 3:
        if name == "A->B->C":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if (i != real_nodes[1]) and (i != real_nodes[2]):
                        process_list.append(story[i])
                    elif (i == real_nodes[1]) or (i == real_nodes[2]):
                        process_list.append(story[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))
        elif name == "A->B<-C":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if i != real_nodes[1]:
                        process_list.append(story[i])
                    elif i == real_nodes[1]:
                        process_list.append(story[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))
        elif name == "A<-B->C":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if (i != real_nodes[0]) and (i != real_nodes[2]):
                        process_list.append(story[i])
                    elif (i == real_nodes[0]) and (i == real_nodes[2]):
                        process_list.append(story[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))

    elif struct_len == 4:
        if name == "A->B->C->D":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if (i != real_nodes[1]) and (i != real_nodes[2]) and (i != real_nodes[3]):
                        process_list.append(story[i])
                    elif (i == real_nodes[1]) or (i == real_nodes[2]) or (i == real_nodes[3]):
                        process_list.append(story[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))
        elif name == "A->B<-C<-D":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if (i != real_nodes[1]) and (i != real_nodes[2]):
                        process_list.append(story[i])
                    elif (i == real_nodes[1]) and (i == real_nodes[2]):
                        process_list.append(story[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))
        elif name == "A<-B->C->D":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if (i != real_nodes[0]) and (i != real_nodes[2]) and (i != real_nodes[3]):
                        process_list.append(story[i])
                    elif (i == real_nodes[0]) or (i == real_nodes[2]) or (i == real_nodes[3]):
                        process_list.append(story[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))
        elif name == "A<-B->C<-D":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if (i != real_nodes[0]) and (i != real_nodes[2]):
                        process_list.append(story[i])
                    elif (i == real_nodes[0]) or (i == real_nodes[2]):
                        process_list.append(story[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))
        elif name == "A->B->C B->D":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if (i != real_nodes[1]) and (i != real_nodes[2]) and (i != real_nodes[3]):
                        process_list.append(story[i])
                    elif (i != real_nodes[1]) and (i != real_nodes[2]) and (i != real_nodes[3]):
                        process_list.append(story[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))
        elif name == "A->B->C D->B":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if (i != real_nodes[1]) and (i != real_nodes[2]):
                        process_list.append(story[i])
                    elif (i == real_nodes[1]) or (i == real_nodes[2]):
                        process_list.append(story[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))
        elif name == "A->B<-C B<-D":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if i != real_nodes[1]:
                        process_list.append(story[i])
                    elif i == real_nodes[1]:
                        process_list.append(story[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))
        elif name == "A<-B->C B->D":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if (i != real_nodes[0]) and (i != real_nodes[2]) and (i != real_nodes[3]):
                        process_list.append(story[i])
                    elif (i == real_nodes[0]) or (i == real_nodes[2]) or (i == real_nodes[3]):
                        process_list.append(story[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))
    return data_list

def process_struct(story, eb, structs, name, struct_len):
    data_list = []
    if struct_len == 3:
        if name == "A->B->C":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if (i != real_nodes[1]) and (i != real_nodes[2]):
                        process_list.append(story[i])
                    elif (i == real_nodes[1]) or (i == real_nodes[2]):
                        process_list.append(story[i])
                        process_list.append(eb[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))
        elif name == "A->B<-C":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if i != real_nodes[1]:
                        process_list.append(story[i])
                    elif i == real_nodes[1]:
                        process_list.append(story[i])
                        process_list.append(eb[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))
        elif name == "A<-B->C":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if (i != real_nodes[0]) and (i != real_nodes[2]):
                        process_list.append(story[i])
                    elif (i == real_nodes[0]) and (i == real_nodes[2]):
                        process_list.append(story[i])
                        process_list.append(eb[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))

    elif struct_len == 4:
        if name == "A->B->C->D":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if (i != real_nodes[1]) and (i != real_nodes[2]) and (i != real_nodes[3]):
                        process_list.append(story[i])
                    elif (i == real_nodes[1]) or (i == real_nodes[2]) or (i == real_nodes[3]):
                        process_list.append(story[i])
                        process_list.append(eb[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))
        elif name == "A->B<-C<-D":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if (i != real_nodes[1]) and (i != real_nodes[2]):
                        process_list.append(story[i])
                    elif (i == real_nodes[1]) and (i == real_nodes[2]):
                        process_list.append(story[i])
                        process_list.append(eb[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))
        elif name == "A<-B->C->D":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if (i != real_nodes[0]) and (i != real_nodes[2]) and (i != real_nodes[3]):
                        process_list.append(story[i])
                    elif (i == real_nodes[0]) or (i == real_nodes[2]) or (i == real_nodes[3]):
                        process_list.append(story[i])
                        process_list.append(eb[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))
        elif name == "A<-B->C<-D":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if (i != real_nodes[0]) and (i != real_nodes[2]):
                        process_list.append(story[i])
                    elif (i == real_nodes[0]) or (i == real_nodes[2]):
                        process_list.append(story[i])
                        process_list.append(eb[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))
        elif name == "A->B->C B->D":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if (i != real_nodes[1]) and (i != real_nodes[2]) and (i != real_nodes[3]):
                        process_list.append(story[i])
                    elif (i != real_nodes[1]) and (i != real_nodes[2]) and (i != real_nodes[3]):
                        process_list.append(story[i])
                        process_list.append(eb[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))
        elif name == "A->B->C D->B":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if (i != real_nodes[1]) and (i != real_nodes[2]):
                        process_list.append(story[i])
                    elif (i == real_nodes[1]) or (i == real_nodes[2]):
                        process_list.append(story[i])
                        process_list.append(eb[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))
        elif name == "A->B<-C B<-D":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if i != real_nodes[1]:
                        process_list.append(story[i])
                    elif i == real_nodes[1]:
                        process_list.append(story[i])
                        process_list.append(eb[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))
        elif name == "A<-B->C B->D":
            for j, struct in enumerate(structs):
                process_list = []
                all_nodes = struct[0]
                real_nodes = struct[1]
                for i in sorted(all_nodes):
                    if (i != real_nodes[0]) and (i != real_nodes[2]) and (i != real_nodes[3]):
                        process_list.append(story[i])
                    elif (i == real_nodes[0]) or (i == real_nodes[2]) or (i == real_nodes[3]):
                        process_list.append(story[i])
                        process_list.append(eb[i])
                process_vector = np.stack(process_list)
                data_list.append((process_vector, all_nodes, real_nodes, name))
    return data_list

def process_struct_nocasual(story, length):
    
    datas = np.load(f"casual-structure/structure-vector/brain-noise/subj1/brain_result{length}.npy", allow_pickle=True)
    matrix = np.load("casual_matrix/C_noshrot.npy")

    len_num = len(datas)
    
    full = list(range(len(story)))

    def judge(all_num, M):
        for i in all_num:
            for j in all_num:
                if i != j and (M[i, j] == 1 or M[j, i] == 1):
                    return False
        return True

    result_list = []
    num = 0
    for _ in range(1000000):
        if length == 3:
            k = random.randint(5, 13)
        elif length == 4:
            k = random.randint(8, 16)
        result = random.sample(full, k)
        if judge(result, matrix):
            result = sorted(result)
            result_list.append(result)
            num += 1
        if num == len_num:
            break

    all_data_list = []
    for noyg in result_list:
        new_data_list = []
        for i in noyg:
            story_i = story[i]
            new_data_list.append(story_i)
        process_vector = np.stack(new_data_list)
        all_data_list.append((process_vector, i))


    return all_data_list

def struct_name_dict(struct_len):
    if struct_len == 3:
        a = {"A->B->C noc":"linear",
            "A->B<-C noc":"central",
            "A<-B->C noc":"decentral"}
    elif struct_len == 4:
        a = {"A->B->C->D noc":"linear",
            "A->B<-C<-D noc":"central",
            "A<-B->C->D noc":"decentral",
            "A<-B->C<-D noc":"strange",
            "A->B->C B->D noc":"linearBD",
            "A->B->C D->B noc":"linearDB",
            "A->B<-C B<-D noc":"centralDB",
            "A<-B->C B->D noc":"decentralBD"}
    return a

def rips_compute(st, p1=15, p2=0.5):
    diag = st.persistence()
    diag_H0 = st.persistence_intervals_in_dimension(0)
    diag_H1 = st.persistence_intervals_in_dimension(1)

    H1_birth = diag_H1[:, 0]
    H0_birth = diag_H0[:, 0]
    H1_persistence = diag_H1[:, 1] - diag_H1[:, 0]
    H0_persistence = diag_H0[:, 1] - diag_H0[:, 0]

    diagram_H1 = np.vstack([H1_birth, H1_persistence]).T
    diagram_H0 = np.vstack([H0_birth, H0_persistence]).T
    
    diagram_H1 = diagram_H1[~np.isinf(diagram_H1).any(axis=1)]
    diagram_H0 = diagram_H0[~np.isinf(diagram_H0).any(axis=1)]

    p_image = PersistenceImage(resolution=[p1, p1], bandwidth=p2)
    pi_vector_H0 = p_image.fit_transform([diagram_H0])[0]
    pi_vector_H1 = p_image.fit_transform([diagram_H1])[0]
    pi_vector = np.hstack([pi_vector_H0, pi_vector_H1])
    return pi_vector

def plot_manifold(
    points,
    color="#ff5724",
    radius=3,
    sphere_resolution=20,
    figsize=(10, 7),
    dpi=300,
    elev=30,
    azim=45,
    alpha=0.9,
    bg_color="white",
    pane_color=(1, 1, 1, 0.08),
    grid_color=(0.72, 0.72, 0.72, 0.42),
    xlabel=r'Axis$_1$',
    ylabel=r'Axis$_2$',
    zlabel=r'Axis$_3$',
    title=None,
    ambient=0.78,        
    diffuse_strength=0.4,   
    specular_strength=0.80,  
    shininess=22,           
    white_mix=0.18):

    plt.rcParams.update({
        "text.usetex": False,          
        "font.family": "sans-serif",     
        "font.sans-serif": ["Arial"],         
        "mathtext.fontset": "dejavusans",     
        "font.size": 18,              
        "axes.unicode_minus": False,   
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": True,
        "grid.alpha": 0.2,
        "axes.titlesize": 16,
        "axes.labelsize": 26,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 20,
        "figure.dpi": 300
    })

    points = np.asarray(points, dtype=float)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    if radius is None:
        ranges = np.ptp(points, axis=0)
        scale = np.max(ranges) if np.max(ranges) > 0 else 1.0
        radius = scale * 0.025

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    try:
        ax.xaxis.set_pane_color(pane_color)
        ax.yaxis.set_pane_color(pane_color)
        ax.zaxis.set_pane_color(pane_color)
    except Exception:
        pass

    u = np.linspace(0, 2 * np.pi, sphere_resolution)
    v = np.linspace(0, np.pi, sphere_resolution)
    uu, vv = np.meshgrid(u, v)

    sx = np.cos(uu) * np.sin(vv)
    sy = np.sin(uu) * np.sin(vv)
    sz = np.cos(vv)

    xs = radius * sx
    ys = radius * sy
    zs = radius * sz

    normals = np.stack([sx, sy, sz], axis=-1)

    light_dir = np.array([-0.55, -0.35, 0.75], dtype=float)
    light_dir /= np.linalg.norm(light_dir)

    view_dir = np.array([0.0, 0.0, 1.0], dtype=float)
    view_dir /= np.linalg.norm(view_dir)

    half_vec = light_dir + view_dir
    half_vec /= np.linalg.norm(half_vec)

    lambert = np.maximum(np.sum(normals * light_dir, axis=-1), 0.0)

    spec = np.maximum(np.sum(normals * half_vec, axis=-1), 0.0) ** shininess

    base = np.array(to_rgb(color), dtype=float)
    base = base * (1 - white_mix) + np.ones(3) * white_mix

    rgb = base[None, None, :] * (ambient + diffuse_strength * lambert[..., None])

    rgb = rgb + specular_strength * spec[..., None] * (1.0 - rgb)

    rgb = rgb * 0.92 + 0.08 * np.ones_like(rgb)

    rgb = np.clip(rgb, 0, 1)

    facecolors = np.concatenate(
        [rgb, np.full(rgb.shape[:2] + (1,), alpha)],
        axis=-1
    )

    for cx, cy, cz in points:
        ax.plot_surface(
            xs + cx,
            ys + cy,
            zs + cz,
            facecolors=facecolors,
            linewidth=0,
            antialiased=True,
            shade=False 
        )

    ax.set_xlabel(xlabel, fontsize=20, labelpad=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=20, labelpad=10, fontweight='bold')
    ax.set_zlabel(zlabel, fontsize=20, labelpad=8, fontweight='bold')

    if title is not None:
        ax.set_title(title, fontsize=14, pad=12)

    ax.grid(True)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        try:
            axis._axinfo["grid"]["color"] = grid_color
            axis._axinfo["grid"]["linewidth"] = 0.8
            axis._axinfo["grid"]["linestyle"] = "-"
        except Exception:
            pass

    ax.view_init(elev=elev, azim=azim)

    x_range = np.ptp(x) if np.ptp(x) > 1e-12 else 1.0
    y_range = np.ptp(y) if np.ptp(y) > 1e-12 else 1.0
    z_range = np.ptp(z) if np.ptp(z) > 1e-12 else 1.0

    max_range = max(x_range, y_range, z_range)
    x_mid = (x.min() + x.max()) / 2
    y_mid = (y.min() + y.max()) / 2
    z_mid = (z.min() + z.max()) / 2

    half = max_range / 2 + 2 * radius
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)

    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass

    plt.tight_layout()

    return fig, ax

def struct_name_dict(struct_len):
    if struct_len == 3:
        a = {"A->B->C":"Chain",
            "A->B<-C":"Merge",
            "A<-B->C":"Branch"}
    elif struct_len == 4:
        a = {"A->B->C->D":"Chain",
            "A->B<-C<-D":"Merge",
            "A<-B->C->D":"Branch",
            "A<-B->C<-D":"Mixed",
            "A->B->C B->D":"Fork",
            "A->B->C D->B":"Funnel",
            "A->B<-C B<-D":"Sink",
            "A<-B->C B->D":"Source"}
    return a

def judge(result, length):
    land_dict = {}
    vector_pair = {}
    y_pair = {}
    name_dict = struct_name_dict(length)
    for item in result:
        base_name = item["type_name"]
        vec = item["persistence_vector"]
        # name = name_dict[base_name]
        name = base_name
        
        if name not in land_dict:
            land_dict[name] = vec
        else:
            land_dict[name] = np.vstack([land_dict[name], vec])
    
    for a, b in combinations(land_dict.keys(), 2):
        stacked = np.vstack([land_dict[a], land_dict[b]])
        vector_pair[f"{a}\nvs\n{b}"] = stacked
        y_pair[f"{a}\nvs\n{b}"] = np.array([0]*len(land_dict[a]) + [1]*len(land_dict[b]))

    def score(vp, yp):
        score_pair = {}
        for key in vp.keys():
            score_pair[key] = silhouette_score(vp[key], yp[key])
        return score_pair

    score_dict = score(vector_pair, y_pair)
    return score_dict

def _adjust_color(color, factor=1.0):

    rgb = np.array(to_rgb(color))
    if factor >= 1:
        rgb = 1 - (1 - rgb) / factor
    else:
        rgb = rgb * factor
    return np.clip(rgb, 0, 1)

def plot_final_histogram_li(
    score_dict,
    save_path,
    metric_name="Score",
    base_color="#b3d67e",
    figsize=(10, 6),
    dpi=300,
    bar_width=0.62,
    depth_x=0.10,
    depth_y_ratio=0.035):

    plt.rcParams.update({
        "text.usetex": False,     
        "font.family": "sans-serif",  
        "font.sans-serif": ["Arial"],        
        "mathtext.fontset": "dejavusans",      
        "axes.unicode_minus": False,  
        "figure.dpi": 300
    })
    names = list(score_dict.keys())
    values = np.array(list(score_dict.values()), dtype=float)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    n = len(names)
    x = np.arange(n)

    vmin, vmax = values.min(), values.max()
    vrange = max(vmax - vmin, 1e-6)
    depth_y = vrange * depth_y_ratio

    factors = np.linspace(1.18, 0.92, n)
    front_colors = [_adjust_color(base_color, f) for f in factors]

    ax.axhline(0, color="black", linewidth=0.8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)

    for i, (xi, val) in enumerate(zip(x, values)):
        left = xi - bar_width / 2
        right = xi + bar_width / 2

        y0 = 0 if val >= 0 else val
        h = abs(val)

        front_color = front_colors[i]
        top_color = _adjust_color(front_color, 1.35)
        side_color = _adjust_color(front_color, 0.72)
        back_color = _adjust_color(front_color, 0.88)

        if val >= 0:

            back = Rectangle(
                (left + depth_x, depth_y),
                bar_width,
                h,
                facecolor=back_color,
                edgecolor="black",
                linewidth=0.45,
                alpha=0.35,
                zorder=1,
            )
            ax.add_patch(back)


            side = Polygon(
                [
                    (right, 0),
                    (right, val),
                    (right + depth_x, val + depth_y),
                    (right + depth_x, depth_y),
                ],
                closed=True,
                facecolor=side_color,
                edgecolor="black",
                linewidth=0.6,
                zorder=2,
            )
            ax.add_patch(side)

            top = Polygon(
                [
                    (left, val),
                    (right, val),
                    (right + depth_x, val + depth_y),
                    (left + depth_x, val + depth_y),
                ],
                closed=True,
                facecolor=top_color,
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )
            ax.add_patch(top)

            front = Rectangle(
                (left, 0),
                bar_width,
                h,
                facecolor=front_color,
                edgecolor="black",
                linewidth=0.7,
                zorder=4,
            )
            ax.add_patch(front)


        else:
            back = Rectangle(
                (left + depth_x, val - depth_y),
                bar_width,
                h,
                facecolor=back_color,
                edgecolor="black",
                linewidth=0.45,
                alpha=0.35,
                zorder=1,
            )
            ax.add_patch(back)

            bottom = Polygon(
                [
                    (left, val),
                    (right, val),
                    (right + depth_x, val - depth_y),
                    (left + depth_x, val - depth_y),
                ],
                closed=True,
                facecolor=top_color,
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )
            ax.add_patch(bottom)

            side = Polygon(
                [
                    (right, val),
                    (right, 0),
                    (right + depth_x, -depth_y),
                    (right + depth_x, val - depth_y),
                ],
                closed=True,
                facecolor=side_color,
                edgecolor="black",
                linewidth=0.6,
                zorder=2,
            )
            ax.add_patch(side)

            front = Rectangle(
                (left, val),
                bar_width,
                h,
                facecolor=front_color,
                edgecolor="black",
                linewidth=0.7,
                zorder=4,
            )
            ax.add_patch(front)


    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylabel(metric_name, fontsize=16, fontweight="bold")

    ax.set_xticks(x)

    ax.set_xticklabels(
    names,
    rotation=45,
    ha="center",
    multialignment="center",
    fontsize=12,
    fontweight="bold")

    ax.tick_params(axis='y', labelsize=12)

    ax.set_xlim(-0.6, n - 0.4 + depth_x + 0.12)

    if np.all(values >= 0):
        ymin = 0
        ymax = vmax + abs(vrange) * 0.12 + depth_y * 1.8

    elif np.all(values <= 0):
        ymin = vmin - abs(vrange) * 0.12 - depth_y * 1.8
        ymax = 0

    else:
        ymin = vmin - abs(vrange) * 0.08 - depth_y * 1.2
        ymax = vmax + abs(vrange) * 0.12 + depth_y * 1.6

    if abs(ymax - ymin) < 1e-8:
        ymax += 1
        ymin -= 1

    ax.set_ylim(ymin, ymax)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_final_histogram_ping(
    score_dict,
    save_path,
    metric_name="Score",
    base_color="#a8cc7a",
    figsize=(10, 6),
    dpi=300,
    bar_width=0.62,
    ylim=None,
    yticks=None,
    rotate_xticks=0,
    show_values=False,
    value_fmt="{:.3f}",
):

    plt.rcParams.update({
        "text.usetex": False,       
        "font.family": "sans-serif",  
        "font.sans-serif": ["Arial"],         
        "mathtext.fontset": "dejavusans",   
        "axes.unicode_minus": False,  
        "figure.dpi": 300
    })

    names = list(score_dict.keys())
    values = np.array(list(score_dict.values()), dtype=float)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x = np.arange(len(names))

    bars = ax.bar(
        x,
        values,
        width=bar_width,
        color=base_color,
        edgecolor="black",
        linewidth=1.5,
        zorder=3,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", which="major", labelsize=14, width=1.2, length=4)
    ax.grid(False)

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


    ax.axhline(0, color="black", linewidth=1.0, zorder=2)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        vmin, vmax = float(np.min(values)), float(np.max(values))
        vrange = max(vmax - vmin, 1e-8)
        if np.all(values >= 0):
            ymin = 0
            ymax = vmax + vrange * 0.15 + 1e-3
        elif np.all(values <= 0):
            ymin = vmin - vrange * 0.15 - 1e-3
            ymax = 0
        else:
            ymin = vmin - vrange * 0.10 - 1e-3
            ymax = vmax + vrange * 0.15 + 1e-3
        ax.set_ylim(ymin, ymax)

    if yticks is not None:
        ax.set_yticks(yticks)

    ax.set_xlim(-0.6, len(names) - 0.4)

    if show_values:
        y0, y1 = ax.get_ylim()
        yrange = y1 - y0
        offset = yrange * 0.02
        for rect, val in zip(bars, values):
            x_text = rect.get_x() + rect.get_width() / 2
            if val >= 0:
                y_text = val + offset
                va = "bottom"
            else:
                y_text = val - offset
                va = "top"
            ax.text(
                x_text,
                y_text,
                value_fmt.format(val),
                ha="center",
                va=va,
                fontsize=10,
                fontweight="bold",
            )

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)



def plot_manifold_multi(
    groups,
    radius=None,
    sphere_resolution=20,
    figsize=(10, 7),
    dpi=300,
    elev=20,
    azim=45,
    alpha=0.9,
    bg_color="white",
    pane_color=(1, 1, 1, 0.08),
    grid_color=(0.72, 0.72, 0.72, 0.42),
    xlabel=r'Axis$_1$',
    ylabel=r'Axis$_2$',
    zlabel=r'Axis$_3$',
    title=None,
    ambient=0.78,
    diffuse_strength=0.4,
    specular_strength=0.80,
    shininess=22,
    white_mix=0.18,
    show_legend=True,
    legend_loc="upper right",

    show_separator_plane=False,
    separator_mode="centroid", 
    separator_color="#999999",
    separator_alpha=0.16,
    separator_grid_size=18,
    show_group_center_line=False,
    center_line_color="black",
    center_line_style="--",

    auto_view=True,                      
    preferred_view_dir=(1.0, 1.0, 0.65), 
    min_auto_elev=10,                 
    max_auto_elev=35,              
    plane_visibility=0.15,

    remove_outliers=False,        
    outlier_method="mad",         
    outlier_thresh=3.5,           
    min_points_after_filter=10,   
    ):

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 18,
        "axes.unicode_minus": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": True,
        "grid.alpha": 0.2,
        "axes.titlesize": 16,
        "axes.labelsize": 26,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "figure.dpi": dpi
    })

    def _normalize(v):
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        if n < 1e-12:
            return None
        return v / n

    def _remove_outliers_mad(points, thresh=3.5, min_points_after_filter=10):
        points = np.asarray(points, dtype=float)
        if len(points) <= min_points_after_filter:
            return points

        center = np.median(points, axis=0)
        dist = np.linalg.norm(points - center, axis=1)

        med = np.median(dist)
        mad = np.median(np.abs(dist - med))

        if mad < 1e-12:
            return points

        robust_z = 0.6745 * (dist - med) / mad
        mask = np.abs(robust_z) <= thresh

        if mask.sum() < min_points_after_filter:
            return points
        return points[mask]

    def _remove_outliers_zscore(points, thresh=3.0, min_points_after_filter=10):
        points = np.asarray(points, dtype=float)
        if len(points) <= min_points_after_filter:
            return points

        mean = points.mean(axis=0, keepdims=True)
        std = points.std(axis=0, keepdims=True)
        std[std < 1e-12] = 1.0

        z = (points - mean) / std
        mask = (np.abs(z) <= thresh).all(axis=1)

        if mask.sum() < min_points_after_filter:
            return points
        return points[mask]

    def _maybe_filter_outliers(points):
        if not remove_outliers:
            return points

        if outlier_method == "mad":
            return _remove_outliers_mad(
                points,
                thresh=outlier_thresh,
                min_points_after_filter=min_points_after_filter,
            )
        elif outlier_method == "zscore":
            return _remove_outliers_zscore(
                points,
                thresh=outlier_thresh,
                min_points_after_filter=min_points_after_filter,
            )

    def _auto_view_from_normal(normal,
                               preferred_dir=(1.0, 1.0, 0.65),
                               min_elev=10,
                               max_elev=35,
                               plane_visibility=0.35):

        n = _normalize(normal)
        if n is None:
            return None, None

        pref = _normalize(preferred_dir)
        if pref is None:
            pref = np.array([1.0, 1.0, 0.65], dtype=float)
            pref = pref / np.linalg.norm(pref)

        v_in_plane = pref - np.dot(pref, n) * n
        v_in_plane = _normalize(v_in_plane)

        if v_in_plane is None:

            helper = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(helper, n)) > 0.9:
                helper = np.array([0.0, 1.0, 0.0])
            v_in_plane = _normalize(np.cross(n, helper))

        view_dir = (1.0 - plane_visibility) * v_in_plane + plane_visibility * n
        view_dir = _normalize(view_dir)

        if view_dir[2] < 0:
            view_dir = -view_dir

        vx, vy, vz = view_dir
        az = np.degrees(np.arctan2(vy, vx))
        el = np.degrees(np.arctan2(vz, np.sqrt(vx ** 2 + vy ** 2)))
        el = float(np.clip(el, min_elev, max_elev))

        return el, az

    parsed_groups = []

    if isinstance(groups, dict):
        for label, item in groups.items():
            pts = np.asarray(item["points"], dtype=float)
            pts = _maybe_filter_outliers(pts)
            color = item.get("color", "#ff5724")
            parsed_groups.append({
                "label": label,
                "points": pts,
                "color": color
            })
    elif isinstance(groups, list):
        for i, item in enumerate(groups):
            pts = np.asarray(item["points"], dtype=float)
            pts = _maybe_filter_outliers(pts)
            color = item.get("color", "#ff5724")
            label = item.get("label", f"Group {i+1}")
            parsed_groups.append({
                "label": label,
                "points": pts,
                "color": color
            })


    all_points = []
    for g in parsed_groups:
        pts = g["points"]
        if len(pts) == 0:
            continue
        all_points.append(pts)

    all_points = np.vstack(all_points)
    x_all, y_all, z_all = all_points[:, 0], all_points[:, 1], all_points[:, 2]

    if radius is None:
        ranges = np.ptp(all_points, axis=0)
        scale = np.max(ranges) if np.max(ranges) > 0 else 1.0
        radius = scale * 0.025

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    try:
        ax.xaxis.set_pane_color(pane_color)
        ax.yaxis.set_pane_color(pane_color)
        ax.zaxis.set_pane_color(pane_color)
    except Exception:
        pass

    u = np.linspace(0, 2 * np.pi, sphere_resolution)
    v = np.linspace(0, np.pi, sphere_resolution)
    uu, vv = np.meshgrid(u, v)

    sx = np.cos(uu) * np.sin(vv)
    sy = np.sin(uu) * np.sin(vv)
    sz = np.cos(vv)

    xs = radius * sx
    ys = radius * sy
    zs = radius * sz

    normals = np.stack([sx, sy, sz], axis=-1)

    light_dir = np.array([-0.55, -0.35, 0.75], dtype=float)
    light_dir /= np.linalg.norm(light_dir)

    view_dir = np.array([0.0, 0.0, 1.0], dtype=float)
    view_dir /= np.linalg.norm(view_dir)

    half_vec = light_dir + view_dir
    half_vec /= np.linalg.norm(half_vec)

    lambert = np.maximum(np.sum(normals * light_dir, axis=-1), 0.0)
    spec = np.maximum(np.sum(normals * half_vec, axis=-1), 0.0) ** shininess

    legend_handles = []

    for g in parsed_groups:
        pts = g["points"]
        color = g["color"]
        label = g["label"]

        if len(pts) == 0:
            continue

        base = np.array(to_rgb(color), dtype=float)
        base = base * (1 - white_mix) + np.ones(3) * white_mix

        rgb = base[None, None, :] * (ambient + diffuse_strength * lambert[..., None])
        rgb = rgb + specular_strength * spec[..., None] * (1.0 - rgb)
        rgb = rgb * 0.92 + 0.08 * np.ones_like(rgb)
        rgb = np.clip(rgb, 0, 1)

        facecolors = np.concatenate(
            [rgb, np.full(rgb.shape[:2] + (1,), alpha)],
            axis=-1
        )

        for cx, cy, cz in pts:
            ax.plot_surface(
                xs + cx,
                ys + cy,
                zs + cz,
                facecolors=facecolors,
                linewidth=0,
                antialiased=True,
                shade=False
            )

        legend_handles.append(
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                label=label,
                markerfacecolor=color,
                markeredgecolor='black',
                markersize=10,
                linewidth=0
            )
        )

    ax.set_xlabel(xlabel, fontsize=20, labelpad=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=20, labelpad=10, fontweight='bold')
    ax.set_zlabel(zlabel, fontsize=20, labelpad=8, fontweight='bold')

    if title is not None:
        ax.set_title(title, fontsize=14, pad=12)

    ax.grid(True)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        try:
            axis._axinfo["grid"]["color"] = grid_color
            axis._axinfo["grid"]["linewidth"] = 0.8
            axis._axinfo["grid"]["linestyle"] = "-"
        except Exception:
            pass

    if auto_view and len(parsed_groups) == 2:
        pts1 = parsed_groups[0]["points"]
        pts2 = parsed_groups[1]["points"]

        c1 = pts1.mean(axis=0)
        c2 = pts2.mean(axis=0)

        if separator_mode in ("centroid", "pca_normal"):
            view_normal = c2 - c1
        else:
            view_normal = c2 - c1

        auto_elev, auto_azim = _auto_view_from_normal(
            view_normal,
            preferred_dir=preferred_view_dir,
            min_elev=min_auto_elev,
            max_elev=max_auto_elev,
            plane_visibility=plane_visibility,
        )

        if auto_elev is not None and auto_azim is not None:
            elev, azim = auto_elev, auto_azim

    ax.view_init(elev=elev, azim=azim)

    x_range = np.ptp(x_all) if np.ptp(x_all) > 1e-12 else 1.0
    y_range = np.ptp(y_all) if np.ptp(y_all) > 1e-12 else 1.0
    z_range = np.ptp(z_all) if np.ptp(z_all) > 1e-12 else 1.0

    max_range = max(x_range, y_range, z_range)
    x_mid = (x_all.min() + x_all.max()) / 2
    y_mid = (y_all.min() + y_all.max()) / 2
    z_mid = (z_all.min() + z_all.max()) / 2

    half = max_range / 2 + 2 * radius
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)

    if len(parsed_groups) == 2 and (show_separator_plane or show_group_center_line):
        pts1 = parsed_groups[0]["points"]
        pts2 = parsed_groups[1]["points"]

        c1 = pts1.mean(axis=0)
        c2 = pts2.mean(axis=0)
        mid = (c1 + c2) / 2.0

        if separator_mode == "centroid":
            normal = c2 - c1
        elif separator_mode == "pca_normal":
            normal = c2 - c1
        else:
            raise ValueError("separator_mode 必须是 'centroid' 或 'pca_normal'")

        norm_val = np.linalg.norm(normal)
        if norm_val > 1e-12:
            normal = normal / norm_val

            if show_group_center_line:
                ax.plot(
                    [c1[0], c2[0]],
                    [c1[1], c2[1]],
                    [c1[2], c2[2]],
                    color=center_line_color,
                    linestyle=center_line_style,
                    linewidth=2.0,
                    alpha=0.9,
                )

            if show_separator_plane:
                helper = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(helper, normal)) > 0.9:
                    helper = np.array([0.0, 1.0, 0.0])

                u = np.cross(normal, helper)
                u = u / np.linalg.norm(u)
                v = np.cross(normal, u)
                v = v / np.linalg.norm(v)

                plane_half_size = max_range * 0.5

                s = np.linspace(-plane_half_size, plane_half_size, separator_grid_size)
                t = np.linspace(-plane_half_size, plane_half_size, separator_grid_size)
                ss, tt = np.meshgrid(s, t)

                plane = mid[None, None, :] + ss[..., None] * u[None, None, :] + tt[..., None] * v[None, None, :]

                px = plane[..., 0]
                py = plane[..., 1]
                pz = plane[..., 2]

                ax.plot_surface(
                    px, py, pz,
                    color=separator_color,
                    alpha=separator_alpha,
                    linewidth=0,
                    shade=False,
                    zorder=0
                )

    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass

    if show_legend and len(legend_handles) > 0:
        ax.legend(handles=legend_handles, loc=legend_loc, ncol=2, frameon=False, fontsize=14)

    plt.tight_layout()
    return fig, ax

def _draw_25d_bar(ax, center_x, val, width, depth_x, depth_y,
                  front_color, text_offset, zorder_base=1):
    left = center_x - width / 2
    right = center_x + width / 2

    h = abs(val)
    top_color = _adjust_color(front_color, 1.35)
    side_color = _adjust_color(front_color, 0.72)
    back_color = _adjust_color(front_color, 0.88)

    if val >= 0:
        # 背面
        back = Rectangle(
            (left + depth_x, depth_y),
            width,
            h,
            facecolor=back_color,
            edgecolor="black",
            linewidth=0.45,
            alpha=0.35,
            zorder=zorder_base,
        )
        ax.add_patch(back)

        # 侧面
        side = Polygon(
            [
                (right, 0),
                (right, val),
                (right + depth_x, val + depth_y),
                (right + depth_x, depth_y),
            ],
            closed=True,
            facecolor=side_color,
            edgecolor="black",
            linewidth=0.6,
            zorder=zorder_base + 1,
        )
        ax.add_patch(side)

        # 顶面
        top = Polygon(
            [
                (left, val),
                (right, val),
                (right + depth_x, val + depth_y),
                (left + depth_x, val + depth_y),
            ],
            closed=True,
            facecolor=top_color,
            edgecolor="black",
            linewidth=0.6,
            zorder=zorder_base + 2,
        )
        ax.add_patch(top)

        # 正面
        front = Rectangle(
            (left, 0),
            width,
            h,
            facecolor=front_color,
            edgecolor="black",
            linewidth=0.7,
            zorder=zorder_base + 3,
        )
        ax.add_patch(front)


    else:
        back = Rectangle(
            (left + depth_x, val - depth_y),
            width,
            h,
            facecolor=back_color,
            edgecolor="black",
            linewidth=0.45,
            alpha=0.35,
            zorder=zorder_base,
        )
        ax.add_patch(back)

        bottom = Polygon(
            [
                (left, val),
                (right, val),
                (right + depth_x, val - depth_y),
                (left + depth_x, val - depth_y),
            ],
            closed=True,
            facecolor=top_color,
            edgecolor="black",
            linewidth=0.6,
            zorder=zorder_base + 2,
        )
        ax.add_patch(bottom)

        side = Polygon(
            [
                (right, val),
                (right, 0),
                (right + depth_x, -depth_y),
                (right + depth_x, val - depth_y),
            ],
            closed=True,
            facecolor=side_color,
            edgecolor="black",
            linewidth=0.6,
            zorder=zorder_base + 1,
        )
        ax.add_patch(side)

        front = Rectangle(
            (left, val),
            width,
            h,
            facecolor=front_color,
            edgecolor="black",
            linewidth=0.7,
            zorder=zorder_base + 3,
        )
        ax.add_patch(front)


def plot_final_histogram_two_groups(
    score_dict_1,
    score_dict_2,
    save_path,
    group_labels=("Causal w/ Event Boundary vs Non-causal", "Causal w/o Event Boundary vs Non-causal"),
    metric_name="Score",
    base_colors=("#5B8FD9", "#F28E2B"),
    figsize=(10, 6),
    dpi=300,
    group_width=0.72,
    inner_gap=0.08,
    depth_x=0.10,
    depth_y_ratio=0.035
    ):

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "figure.dpi": 300
    })

    names_1 = list(score_dict_1.keys())
    names_2 = list(score_dict_2.keys())


    names = names_1
    values_1 = np.array([score_dict_1[k] for k in names], dtype=float)
    values_2 = np.array([score_dict_2[k] for k in names], dtype=float)

    all_values = np.concatenate([values_1, values_2])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    n = len(names)
    x = np.arange(n)

    vmin, vmax = all_values.min(), all_values.max()
    vrange = max(vmax - vmin, 1e-6)
    depth_y = vrange * depth_y_ratio
    text_offset = vrange * 0.02

    ax.axhline(0, color="black", linewidth=0.8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)

    bar_width = (group_width - inner_gap) / 2.0
    centers_1 = x - (bar_width + inner_gap) / 2.0
    centers_2 = x + (bar_width + inner_gap) / 2.0

    factors = np.linspace(1.10, 0.94, n)
    front_colors_1 = [_adjust_color(base_colors[0], f) for f in factors]
    front_colors_2 = [_adjust_color(base_colors[1], f) for f in factors]

    for i in range(n):
        _draw_25d_bar(
            ax=ax,
            center_x=centers_1[i],
            val=values_1[i],
            width=bar_width,
            depth_x=depth_x,
            depth_y=depth_y,
            front_color=front_colors_1[i],
            text_offset=text_offset,
            zorder_base=1,
        )

        _draw_25d_bar(
            ax=ax,
            center_x=centers_2[i],
            val=values_2[i],
            width=bar_width,
            depth_x=depth_x,
            depth_y=depth_y,
            front_color=front_colors_2[i],
            text_offset=text_offset,
            zorder_base=10,
        )

    mean_1 = values_1.mean()
    mean_2 = values_2.mean()

    ax.axhline(
        mean_1,
        color=base_colors[0],
        linestyle="--",
        linewidth=1.8,
        alpha=0.9,
        zorder=0,
    )

    ax.axhline(
        mean_2,
        color=base_colors[1],
        linestyle="--",
        linewidth=1.8,
        alpha=0.9,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylabel(metric_name, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9, fontweight="bold")

    ax.set_xlim(-0.8, n - 0.2 + depth_x + 0.15)

    all_for_ylim = np.concatenate([all_values, [mean_1, mean_2]])
    ymin_data, ymax_data = all_for_ylim.min(), all_for_ylim.max()
    yrange = max(ymax_data - ymin_data, 1e-6)

    if np.all(all_for_ylim >= 0):
        ymin = 0
        ymax = ymax_data + abs(yrange) * 0.12 + depth_y * 1.8
    elif np.all(all_for_ylim <= 0):
        ymin = ymin_data - abs(yrange) * 0.12 - depth_y * 1.8
        ymax = 0
    else:
        ymin = ymin_data - abs(yrange) * 0.08 - depth_y * 1.2
        ymax = ymax_data + abs(yrange) * 0.12 + depth_y * 1.6

    if abs(ymax - ymin) < 1e-8:
        ymax += 1
        ymin -= 1

    ax.set_ylim(ymin, ymax)

    legend_handles = [
        Patch(facecolor=base_colors[0], edgecolor="black", label=group_labels[0]),
        Patch(facecolor=base_colors[1], edgecolor="black", label=group_labels[1]),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="best",
        fontsize=12,
        handlelength=2.0,
        handleheight=1.5,
        labelspacing=0.8,
        borderpad=0.8
    )

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)