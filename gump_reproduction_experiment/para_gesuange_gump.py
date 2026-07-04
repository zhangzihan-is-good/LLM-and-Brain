"""
GUMP 数据集因果重激活分析（三条件对比版）
==========================================

基于 para_gesuange.py 改写，适配 GUMP 数据格式。

与原版 Sherlock 版本的关键差异：
- 数据来源：从目录加载多个 npz 文件并 stack 为 (nVoxel, nEvent, nSphere, nSub)
- 事件数：34（GUMP 35场景-1）而非 49
- 球体数：61566 而非 60917
- 因果事件索引：本地 data/gump/ 下的 npy 文件（各5个事件）
- NIfTI 空间尺寸由 sphereCenters 动态确定

用法：
    python main/para_gesuange_gump.py
"""

import os
import numpy as np
import scipy.io as sio
import nibabel as nib
from joblib import Parallel, delayed
import warnings

# ==================== 项目路径 ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "gump")
INDICE_DIR = DATA_DIR  # indice npy 文件也在 data/gump 下

# GUMP 参数
NUM_EVENTS = 34       # GUMP: 35场景 - 1
# IGNORE_EVENTS = [0, 13,14,26,27]  # 忽略的事件索引（短场景 + 不同run切分）
IGNORE_EVENTS = [0, 13,14,26,27]  # 忽略的事件索引（短场景 + 不同run切分）


def load_spheres_data(path):
    """从 spheres.mat 文件加载球体定义数据。"""
    data = sio.loadmat(os.path.join(path, "spheres.mat"))
    sc = data["sphereCenters"]
    if sc.shape == (1, 1):
        sc = sc[0, 0]
    sphereCenters = {
        "x": np.array(sc["x"]).flatten(),
        "y": np.array(sc["y"]).flatten(),
        "z": np.array(sc["z"]).flatten(),
    }
    inds = data["sphereInds"]
    if inds.dtype == "object":
        sphereInds = [inds.flat[i] for i in range(inds.size)]
    else:
        sphereInds = inds
    return sphereCenters, sphereInds


def fisher_z(r):
    """Fisher z 转换：z = arctanh(r)。"""
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)


def vectorized_corr(A, B):
    """向量化 Pearson 相关系数矩阵计算。NaN 安全。"""
    mask = (~np.isnan(A)) & (~np.isnan(B))
    A = np.where(mask, A, np.nan)
    B = np.where(mask, B, np.nan)
    A -= np.nanmean(A, axis=0)
    B -= np.nanmean(B, axis=0)
    denom = np.sqrt(np.nansum(A**2, axis=0)[:, None] * np.nansum(B**2, axis=0)[None, :])
    return np.nansum(A[:, :, None] * B[:, None, :], axis=0) / denom


def getTriangular(mat, Lower, idx):
    """提取相关矩阵的下三角或上三角指定子区域元素（Fortran 列优先）。"""
    if Lower == 1:
        m = np.tril(mat)
    else:
        m = np.triu(mat)
    m[m == 0] = np.nan
    if Lower == 1:
        m = m[idx]
    else:
        m = m[:, idx]
    vec = m.flatten(order="F")
    return vec[~np.isnan(vec)]


def build_bool_idx(event_indices, n_events):
    """将事件索引数组转换为布尔掩码。"""
    idx = np.zeros(n_events, dtype=bool)
    idx[event_indices] = True
    return idx


def process_sphere(iSphere, Data, sphereCenters, yin_idx, guo_idx, duli_idx,
                   numDiag, ISC, iSub, mask):
    """处理单个球体：计算三种因果条件下的重激活指标。

    与原版 para_gesuange.py 的 process_sphere 逻辑完全一致。

    Args:
        iSphere: 球体索引。
        Data: [DataEB, DataP, DataF, DataScenes]，每个形状 (nVoxel, nEvent, nSphere, nSub)。
        sphereCenters: 球体中心坐标 dict。
        yin_idx/guo_idx/duli_idx: (nEvents,) 布尔索引。
        numDiag: 去除对角线宽度。
        ISC: 0=个体内, 1=个体间。
        iSub: 当前被试索引。
        mask: 3D ROI 掩码，None 则不过滤。
    Returns:
        dict or None。
    """
    warnings.filterwarnings("ignore")
    x = int(sphereCenters['x'][iSphere]) - 1
    y = int(sphereCenters['y'][iSphere]) - 1
    z = int(sphereCenters['z'][iSphere]) - 1

    # 跳过不在 ROI 掩码内的球体
    if mask is not None and mask[x, y, z] == 0:
        return None
    if iSphere % 500 == 0:
        print(iSphere, iSub)

    DataEB, DataP, DataF, DataScenes = Data
    repEB = DataEB[:, :, iSphere, :]      # (nVoxel, nEvent, nSub)
    sceneAvg = DataScenes[:, :, iSphere, :]
    repP = DataP[:, :, iSphere, :]
    repF = DataF[:, :, iSphere, :]

    # 移除无效体素行（mean=999 表示填充）
    temp = repEB[:, :, iSub]
    idx = np.where(np.nanmean(temp, axis=1) == 999)[0]
    if len(idx) > 0:
        repEB = np.delete(repEB, idx, axis=0)
        sceneAvg = np.delete(sceneAvg, idx, axis=0)
        repP = np.delete(repP, idx, axis=0)
        repF = np.delete(repF, idx, axis=0)

    if repEB.shape[0] < 45:
        return None

    # 计算相关矩阵（与原版一致：ISC=0 用 vectorized_corr, ISC=1 用 corrcoef）
    if ISC != 1:
        boundE = vectorized_corr(repEB[:, :, iSub], sceneAvg[:, :, iSub])
        boundP = vectorized_corr(repP[:, :, iSub], sceneAvg[:, :, iSub])
        boundF = vectorized_corr(repF[:, :, iSub], sceneAvg[:, :, iSub])
    else:
        ss = sceneAvg[:, :, iSub]
        groupE = np.delete(repEB, iSub, axis=2)
        boundE = np.corrcoef(np.nanmean(groupE, axis=2).T, ss.T)[:groupE.shape[1], groupE.shape[1]:]

    # 构建对角线掩码
    m = repEB.shape[1]
    X = np.zeros((m, m))
    if numDiag > 1:
        lo, hi = -(numDiag // 2 - 1), numDiag // 2
        for k in range(lo, hi + 1):
            i = np.arange(m)
            j = i + k
            valid = (j >= 0) & (j < m)
            X[i[valid], j[valid]] = np.nan
    else:
        X = np.eye(m) * np.nan

    boundE += X; boundP += X; boundF += X
    boundE = fisher_z(boundE)
    boundP = fisher_z(boundP)
    boundF = fisher_z(boundF)

    # 将布尔掩码截断到实际事件数
    yin_m = yin_idx[:m]
    guo_m = guo_idx[:m]
    duli_m = duli_idx[:m]

    res = {}
    for tag, idx_mask in [("yin", yin_m), ("guo", guo_m), ("duli", duli_m)]:
        res[f"E_{tag}"] = np.nanmean(getTriangular(boundE, 1, idx_mask)) - np.nanmean(getTriangular(boundE, 0, idx_mask))
        res[f"P_{tag}"] = np.nanmean(getTriangular(boundP, 1, idx_mask)) - np.nanmean(getTriangular(boundP, 0, idx_mask))
        res[f"F_{tag}"] = np.nanmean(getTriangular(boundF, 1, idx_mask)) - np.nanmean(getTriangular(boundF, 0, idx_mask))

    res["coord"] = (x, y, z)
    return res


def check_path_exists(path, description):
    """检查路径是否存在，不存在则报错退出。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{description}] 路径不存在: {path}")


def reactivation_analysis_parallel(reps_dir, spheres_path, indice_dir, output_dir, n_jobs=8):
    """并行计算所有球体和被试的三种因果条件重激活 map。

    数据加载流程与原版完全一致：
    1. 从 reps_dir 加载所有 npz 文件
    2. 按 axis=-1 stack 为 (nVoxel, nEvent, nSphere, nSub)
    3. 屏蔽忽略事件
    4. for iSub: 并行处理所有球体
    5. 输出 (nSub, x, y, z) map

    Args:
        reps_dir: 包含所有被试 reps.npz 文件的目录。
        spheres_path: spheres.mat 所在目录。
        indice_dir: 三个 indice npy 文件所在目录。
        output_dir: 输出目录。
        n_jobs: 并行进程数。
    """
    # ===== 输入路径存在检查 =====
    check_path_exists(reps_dir, "reps目录")
    check_path_exists(spheres_path, "spheres目录")
    check_path_exists(os.path.join(spheres_path, "spheres.mat"), "spheres.mat")
    for name in ["cause_event_indice.npy", "result_event_indice.npy", "ind_event_indice.npy"]:
        check_path_exists(os.path.join(indice_dir, name), f"indice文件 {name}")

    os.makedirs(output_dir, exist_ok=True)

    # ===== Step 1: 加载并拼接所有被试数据（与原版 reactivation_analysis.py 一致）=====
    print("Loading npz files from", reps_dir)
    files = sorted(f for f in os.listdir(reps_dir) if f.endswith("reps.npz"))
    nSub = len(files)
    print(f"  Found {nSub} subjects")

    listEB, listP, listF, listS = [], [], [], []
    for f in files:
        d = np.load(os.path.join(reps_dir, f))
        listEB.append(d["repEB"])
        listP.append(d["repP"])
        listF.append(d["repF"])
        listS.append(d["sceneAvg"])

    # stack: (123, nEvents, nSphere, nSub) — 与原版 DataEB/DataP/DataF/DataScenes 格式一致
    DataEB = np.stack(listEB, axis=-1)
    DataP = np.stack(listP, axis=-1)
    DataF = np.stack(listF, axis=-1)
    DataScenes = np.stack(listS, axis=-1)
    print(f"  Data shape: {DataEB.shape}  (nVoxel, nEvents, nSphere, nSub)")

    # ===== Step 2: 屏蔽忽略事件 =====
    for arr in (DataEB, DataP, DataF, DataScenes):
        arr[:, IGNORE_EVENTS, :] = np.nan

    # ===== Step 3: 加载球体和索引 =====
    sphereCenters = load_spheres_data(spheres_path)[0]
    nSphere = DataEB.shape[2]

    cause_idx = build_bool_idx(np.load(os.path.join(indice_dir, "cause_event_indice.npy")), NUM_EVENTS)
    result_idx = build_bool_idx(np.load(os.path.join(indice_dir, "result_event_indice.npy")), NUM_EVENTS)
    ind_idx = build_bool_idx(np.load(os.path.join(indice_dir, "ind_event_indice.npy")), NUM_EVENTS)
    print(f"  yin(cause): {np.where(cause_idx)[0]}")
    print(f"  guo(result): {np.where(result_idx)[0]}")
    print(f"  duli(ind): {np.where(ind_idx)[0]}")

    numDiag, ISC = 10, 0

    # mask: GUMP 用 414ROI.nii（NIfTI 格式），非 npy
    mask = None
    mask_nii = None
    mask_path = os.path.join(spheres_path, "schaefer17n400p1mm+subctx_gump.nii")
    if os.path.exists(mask_path):
        print("mask exists")
        mask_nii = nib.load(mask_path)
        mask = mask_nii.get_fdata()

    Data = [DataEB, DataP, DataF, DataScenes]

    # ===== Step 4: 确定输出空间尺寸（用 mask NIfTI 实际尺寸，而非 sphereCenters 最大值）=====
    if mask_nii is not None:
        shape_3d = tuple(mask.shape)
    else:
        shape_3d = (
            int(sphereCenters["x"].max()),
            int(sphereCenters["y"].max()),
            int(sphereCenters["z"].max()),
        )
    print(f"  shape_3d shape: {shape_3d}")

    # ===== Step 5: 逐被试并行处理（与原版 for iSub in range(nSub) 一致）=====
    mapT1_yin = np.zeros((nSub,) + shape_3d)
    mapT2_yin = np.zeros_like(mapT1_yin)
    mapT1_guo = np.zeros_like(mapT1_yin)
    mapT2_guo = np.zeros_like(mapT1_yin)
    mapT1_duli = np.zeros_like(mapT1_yin)
    mapT2_duli = np.zeros_like(mapT1_yin)

    for iSub in range(nSub):
        print(f"Processing subject {iSub + 1}/{nSub}")
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_sphere)(
                iSphere, Data, sphereCenters,
                cause_idx, result_idx, ind_idx,
                numDiag, ISC, iSub, mask
            )
            for iSphere in range(nSphere)
        )
        for r in filter(None, results):
            x, y, z = r["coord"]
            mapT1_yin[iSub, x, y, z] = r["E_yin"]
            mapT2_yin[iSub, x, y, z] = r["E_yin"] - (r["P_yin"] + r["F_yin"]) / 2
            mapT1_guo[iSub, x, y, z] = r["E_guo"]
            mapT2_guo[iSub, x, y, z] = r["E_guo"] - (r["P_guo"] + r["F_guo"]) / 2
            mapT1_duli[iSub, x, y, z] = r["E_duli"]
            mapT2_duli[iSub, x, y, z] = r["E_duli"] - (r["P_duli"] + r["F_duli"]) / 2

    # ===== Step 6: 保存 =====
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "mapT1_yin.npy"), mapT1_yin)
    np.save(os.path.join(output_dir, "mapT2_yin.npy"), mapT2_yin)
    np.save(os.path.join(output_dir, "mapT1_guo.npy"), mapT1_guo)
    np.save(os.path.join(output_dir, "mapT2_guo.npy"), mapT2_guo)
    np.save(os.path.join(output_dir, "mapT1_duli.npy"), mapT1_duli)
    np.save(os.path.join(output_dir, "mapT2_duli.npy"), mapT2_duli)
    print(f"Results saved to {output_dir}")


def run_statistical_testing(output_dir, spheres_path):
    """对 para 阶段产出的 mapT1/mapT2_guo/yin 做配对 t 检验（guo vs yin）。

    Args:
        output_dir: 包含 mapT1_guo/yin.npy 与 mapT2_guo/yin.npy 的目录。
        spheres_path: spheres.mat 与 MNI152_resampled_mask.nii 所在目录。
    输出（写到 output_dir）:
        dRI_guo_yin.nii / T1_causality_1_graymasked.nii / T2_causality_1_graymasked.nii
    """
    from scipy.stats import ttest_rel

    for name in ["mapT1_guo.npy", "mapT1_yin.npy", "mapT2_guo.npy", "mapT2_yin.npy"]:
        check_path_exists(os.path.join(output_dir, name), f"统计检验输入 {name}")
    check_path_exists(os.path.join(spheres_path, "MNI152_resampled_mask.nii"), "MNI152_resampled_mask.nii")

    mapT1_guo = np.load(os.path.join(output_dir, "mapT1_guo.npy"))
    mapT1_yin = np.load(os.path.join(output_dir, "mapT1_yin.npy"))
    mapT2_guo = np.load(os.path.join(output_dir, "mapT2_guo.npy"))
    mapT2_yin = np.load(os.path.join(output_dir, "mapT2_yin.npy"))
    sphereCenters = load_spheres_data(spheres_path)[0]
    nSphere = sphereCenters["x"].shape[0]

    shape_3d = mapT1_guo.shape[1:]  # (x, y, z)
    print("shape of mapT1_guo: ", shape_3d)
    tmap_T1 = np.zeros(shape_3d + (2,))
    tmap_T2 = np.zeros(shape_3d + (2,))
    reac_map = np.zeros(shape_3d)
    reac_map_nofilter = np.zeros(shape_3d)

    for iSphere in range(nSphere):
        x = int(sphereCenters['x'][iSphere]) - 1
        y = int(sphereCenters['y'][iSphere]) - 1
        z = int(sphereCenters['z'][iSphere]) - 1
        tmap_T1[x, y, z] = ttest_rel(mapT1_guo[:, x, y, z], mapT1_yin[:, x, y, z])
        tmap_T2[x, y, z] = ttest_rel(mapT2_guo[:, x, y, z], mapT2_yin[:, x, y, z])
        if tmap_T2[x, y, z, 1] < 0.1:
            reac_map[x, y, z] = np.mean(mapT2_guo[:, x, y, z]) - np.mean(mapT2_yin[:, x, y, z])
        reac_map_nofilter[x, y, z] = np.mean(mapT2_guo[:, x, y, z]) - np.mean(mapT2_yin[:, x, y, z])

    affine = nib.load(os.path.join(spheres_path, "MNI152_resampled_mask.nii")).affine
    os.makedirs(output_dir, exist_ok=True)
    nib.save(nib.Nifti1Image(reac_map, affine), os.path.join(output_dir, "dRI_guo_yin.nii"))
    nib.save(nib.Nifti1Image(reac_map_nofilter, affine), os.path.join(output_dir, "dRI_guo_yin_nofilter.nii"))
    nib.save(nib.Nifti1Image(tmap_T1, affine), os.path.join(output_dir, "T1_causality_1_graymasked.nii"))
    nib.save(nib.Nifti1Image(tmap_T2, affine), os.path.join(output_dir, "T2_causality_1_graymasked.nii"))
    print("Statistical testing done.")


if __name__ == "__main__":
    # ===== 路径配置 =====
    reps_dir = "/path/to/gump_data/process_data/reps_v2/"
    spheres_path = "/path/to/gump_data/"
    indice_dir = "/path/to/gump_data/process_data/indices/gump_i2_c5_r5/"
    output_dir = os.path.join(spheres_path, "process_data", "gump_causality", "gump_i2_c5_r5")

    reactivation_analysis_parallel(
        reps_dir=reps_dir,
        spheres_path=spheres_path,
        indice_dir=indice_dir,
        output_dir=output_dir,
        n_jobs=8,
    )
    run_statistical_testing(output_dir=output_dir, spheres_path=spheres_path)
