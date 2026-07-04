"""
GUMP 数据集 — 因果重激活一致性分析
====================================

基于 consensus_copy.py 改写，适配 GUMP 数据格式。
加载 para_gesuange_gump.py 输出的 mapT1/mapT2 npy 文件，
在三种因果条件 (guo/yin/duli) 之间做统计分析。

分析模式（通过 ANALYSIS_MODE 切换）：
  1. "consensus"  — 一致性筛选：某条件同时显著高于另外两个条件
  2. "separate"   — 两两配对：不做一致性筛选，保留所有球体
  3. "tongshi"    — 同时筛选：以 guo 为基准，同时与 duli 和 yin 比较

用法：
    python main/consensus_gump.py
"""

import os
import numpy as np
import nibabel as nib
from scipy.stats import ttest_rel
from scipy import io as sio

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "gump")

# 服务器路径
SPHERES_PATH = ""
MAP_DIR = os.path.join(SPHERES_PATH, "process_data", "gump_causality")
AFFINE_PATH = os.path.join(SPHERES_PATH, "MNI152_resampled_mask.nii")
OUTPUT_DIR = os.path.join(SPHERES_PATH, "process_data", "gump_causality", "consensus")

# ==================== 分析模式 ====================
# "consensus" / "separate" / "tongshi"
ANALYSIS_MODE = "tongshi"


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


def run_consensus(mapT2_guo, mapT2_yin, mapT2_duli, sphereCenters, nSphere, affine):
    """一致性分析：某条件显著高于另外两个条件时保留。

    输出 NIfTI 形状: (x, y, z, 3)，第4维为 [t值, p值, 均值]。
    """
    shape_3d = mapT2_guo.shape[1:]
    guo_T2 = np.zeros(shape_3d + (3,))
    yin_T2 = np.zeros(shape_3d + (3,))
    duli_T2 = np.zeros(shape_3d + (3,))

    for iSphere in range(nSphere):
        if iSphere % 5000 == 0:
            print(f"  consensus: {iSphere}/{nSphere}")
        x = int(sphereCenters['x'][iSphere]) - 1
        y = int(sphereCenters['y'][iSphere]) - 1
        z = int(sphereCenters['z'][iSphere]) - 1

        guoseqT2 = mapT2_guo[:, x, y, z]
        yinseqT2 = mapT2_yin[:, x, y, z]
        duliseqT2 = mapT2_duli[:, x, y, z]

        tguo_yin, pguo_yin = ttest_rel(guoseqT2, yinseqT2)
        tduli_yin, pduli_yin = ttest_rel(duliseqT2, yinseqT2)
        tguo_duli, pguo_duli = ttest_rel(guoseqT2, duliseqT2)

        # guo 显著高于 yin 和 duli
        if tguo_yin * tguo_duli > 0 and pguo_duli < 0.025 and pguo_yin < 0.025:
            guo_T2[x, y, z, 2] = np.nanmean(guoseqT2)
            guo_T2[x, y, z, 0] = tguo_yin
            guo_T2[x, y, z, 1] = pguo_yin

        # yin 显著高于 guo 和 duli
        if -tduli_yin * (-tguo_yin) > 0 and pduli_yin < 0.025 and pguo_yin < 0.025:
            yin_T2[x, y, z, 2] = np.nanmean(yinseqT2)
            yin_T2[x, y, z, 0] = -tguo_yin
            yin_T2[x, y, z, 1] = pguo_yin

        # duli 显著高于 guo 和 yin
        if tduli_yin * (-tguo_duli) > 0 and pduli_yin < 0.025 and pguo_duli < 0.025:
            duli_T2[x, y, z, 2] = np.nanmean(duliseqT2)
            duli_T2[x, y, z, 0] = tduli_yin
            duli_T2[x, y, z, 1] = pduli_yin

    return guo_T2, yin_T2, duli_T2


def run_separate(mapT2_guo, mapT2_yin, mapT2_duli, sphereCenters, nSphere, affine):
    """两两配对分析：不做一致性筛选，保留所有球体的统计结果。

    输出 NIfTI 形状: (x, y, z, 4)，第4维为 [t值, p值, 条件A均值, 条件B均值]。
    """
    shape_3d = mapT2_guo.shape[1:]
    guo_yin = np.zeros(shape_3d + (4,))
    guo_duli = np.zeros(shape_3d + (4,))
    duli_yin = np.zeros(shape_3d + (4,))
    yin_duli = np.zeros(shape_3d + (4,))

    for iSphere in range(nSphere):
        if iSphere % 5000 == 0:
            print(f"  separate: {iSphere}/{nSphere}")
        x = int(sphereCenters['x'][iSphere]) - 1
        y = int(sphereCenters['y'][iSphere]) - 1
        z = int(sphereCenters['z'][iSphere]) - 1

        guoseqT2 = mapT2_guo[:, x, y, z]
        yinseqT2 = mapT2_yin[:, x, y, z]
        duliseqT2 = mapT2_duli[:, x, y, z]

        guo_yin[x, y, z, :2] = ttest_rel(guoseqT2, yinseqT2)
        guo_yin[x, y, z, 2] = np.nanmean(guoseqT2)
        guo_yin[x, y, z, 3] = np.nanmean(yinseqT2)

        guo_duli[x, y, z, :2] = ttest_rel(guoseqT2, duliseqT2)
        guo_duli[x, y, z, 2] = np.nanmean(guoseqT2)
        guo_duli[x, y, z, 3] = np.nanmean(duliseqT2)

        duli_yin[x, y, z, :2] = ttest_rel(duliseqT2, yinseqT2)
        duli_yin[x, y, z, 2] = np.nanmean(duliseqT2)
        duli_yin[x, y, z, 3] = np.nanmean(yinseqT2)

        yin_duli[x, y, z, :2] = ttest_rel(yinseqT2, duliseqT2)
        yin_duli[x, y, z, 2] = np.nanmean(yinseqT2)
        yin_duli[x, y, z, 3] = np.nanmean(duliseqT2)

    return guo_yin, guo_duli, duli_yin, yin_duli


def run_tongshi(mapT2_guo, mapT2_yin, mapT2_duli, sphereCenters, nSphere, affine):
    """同时筛选分析：以 guo 为基准，分别与 duli 和 yin 做配对 t 检验。

    筛选标准: |t_guo_duli| >= 2.12 且 |t_guo_yin| >= 2.12（近似 p < 0.05 未校正）。
    只有同时通过两个 t 检验的球体才记录三种条件的均值。

    输出 NIfTI 形状: (x, y, z, 5)，第4维为:
      [0] guo T2 均值, [1] yin T2 均值, [2] duli T2 均值,
      [3] t(guo vs duli), [4] t(guo vs yin)
    """
    shape_3d = mapT2_guo.shape[1:]
    yinguo_duli = np.zeros(shape_3d + (5,))

    for iSphere in range(nSphere):
        if iSphere % 5000 == 0:
            print(f"  tongshi: {iSphere}/{nSphere}")
        x = int(sphereCenters['x'][iSphere]) - 1
        y = int(sphereCenters['y'][iSphere]) - 1
        z = int(sphereCenters['z'][iSphere]) - 1

        guoseqT2 = mapT2_guo[:, x, y, z]
        yinseqT2 = mapT2_yin[:, x, y, z]
        duliseqT2 = mapT2_duli[:, x, y, z]

        t_guo_duli, _ = ttest_rel(guoseqT2, duliseqT2)
        t_guo_yin, _ = ttest_rel(guoseqT2, yinseqT2)

        # 同时通过阈值才记录均值
        if np.abs(t_guo_duli) >= 1.75 and np.abs(t_guo_yin) >= 1.75:
            yinguo_duli[x, y, z, 0] = np.nanmean(guoseqT2)
            yinguo_duli[x, y, z, 1] = np.nanmean(yinseqT2)
            yinguo_duli[x, y, z, 2] = np.nanmean(duliseqT2)
        # 无论是否通过阈值，都保存 t 值
        yinguo_duli[x, y, z, 3] = t_guo_duli
        yinguo_duli[x, y, z, 4] = t_guo_yin

    return yinguo_duli


def run_consensus_analysis(map_dir, output_dir, spheres_path, affine_path, modes):
    """对一个 para 阶段输出目录做 consensus 分析（支持多模式批量）。

    Args:
        map_dir: 包含 mapT2_guo/yin/duli.npy 的目录。
        output_dir: consensus 结果输出目录。
        spheres_path: spheres.mat 所在目录。
        affine_path: 提供 affine 的 NIfTI 路径。
        modes: 要执行的模式（str 或 list），元素 ∈ {'consensus','separate','tongshi'}。
    """
    if isinstance(modes, str):
        modes = [modes]

    print("Loading reactivation maps...")
    mapT2_guo = np.load(os.path.join(map_dir, "mapT2_guo.npy"))
    mapT2_yin = np.load(os.path.join(map_dir, "mapT2_yin.npy"))
    mapT2_duli = np.load(os.path.join(map_dir, "mapT2_duli.npy"))
    nSub = mapT2_guo.shape[0]
    shape_3d = mapT2_guo.shape[1:]
    print(f"  Subjects: {nSub}, Shape: {shape_3d}")

    sphereCenters = load_spheres_data(spheres_path)[0]
    nSphere = sphereCenters["x"].shape[0]
    print(f"  Spheres: {nSphere}")

    affine = nib.load(affine_path).affine
    os.makedirs(output_dir, exist_ok=True)

    for mode in modes:
        print(f"\nRunning analysis mode: {mode}")
        if mode == "consensus":
            guo_T2, yin_T2, duli_T2 = run_consensus(
                mapT2_guo, mapT2_yin, mapT2_duli, sphereCenters, nSphere, affine)
            nib.save(nib.Nifti1Image(guo_T2, affine),
                     os.path.join(output_dir, "T2_guo_consensus.nii"))
            nib.save(nib.Nifti1Image(yin_T2, affine),
                     os.path.join(output_dir, "T2_yin_consensus.nii"))
            nib.save(nib.Nifti1Image(duli_T2, affine),
                     os.path.join(output_dir, "T2_duli_consensus.nii"))
            print("Saved: T2_guo/yin/duli_consensus.nii")

        elif mode == "separate":
            guo_yin, guo_duli, duli_yin, yin_duli = run_separate(
                mapT2_guo, mapT2_yin, mapT2_duli, sphereCenters, nSphere, affine)
            nib.save(nib.Nifti1Image(guo_yin, affine),
                     os.path.join(output_dir, "guo_yin.nii"))
            nib.save(nib.Nifti1Image(guo_duli, affine),
                     os.path.join(output_dir, "guo_duli.nii"))
            nib.save(nib.Nifti1Image(duli_yin, affine),
                     os.path.join(output_dir, "duli_yin.nii"))
            nib.save(nib.Nifti1Image(yin_duli, affine),
                     os.path.join(output_dir, "yin_duli.nii"))
            print("Saved: guo_yin/guo_duli/duli_yin/yin_duli.nii")

        elif mode == "tongshi":
            yinguo_duli = run_tongshi(
                mapT2_guo, mapT2_yin, mapT2_duli, sphereCenters, nSphere, affine)
            nib.save(nib.Nifti1Image(yinguo_duli, affine),
                     os.path.join(output_dir, "guo_yinduli.nii"))
            print("Saved: guo_yinduli.nii")

        else:
            raise ValueError(f"Unknown mode: {mode}. "
                             f"Choose from: consensus, separate, tongshi")

    print(f"\nDone. Results saved to {output_dir}")


if __name__ == "__main__":
    run_consensus_analysis(
        map_dir=MAP_DIR,
        output_dir=OUTPUT_DIR,
        spheres_path=SPHERES_PATH,
        affine_path=AFFINE_PATH,
        modes=ANALYSIS_MODE,
    )
