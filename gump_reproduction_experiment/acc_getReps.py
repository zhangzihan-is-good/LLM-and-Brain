"""
fMRI Sphere-based Representation Extraction
==========================================

目的：
-----
从fMRI NIfTI数据中提取球体(sphere)区域的多时间点表示，为重激活分析准备数据

球体(Sphere)分析原理：
--------------------
1. 预定义一组脑区球体（基于先验知识或解剖分区）
2. 每个球体包含若干体素（通常45-123个）
3. 提取球体内所有体素的时间序列，形成局部神经活动模式

关键概念：
---------
- BOLD信号：血氧水平依赖信号，反映神经活动的间接指标
- TR(Repetition Time)：两次扫描的时间间隔
- iLag：血流动力学延迟的TR数，神经活动到BOLD响应约3-6秒
- distCtr：控制条件的距离(单位：TR)，用于选择基线时间点

提取的表示类型：
---------------
1. repEB(Event Boundary)：事件边界处的BOLD信号，+iLag TR后
2. repP(Past control)：事件边界前distCtr TR的控制点
3. repF(Future control)：事件边界后distCtr TR的控制点
4. sceneAvg：整个场景(排除边界附近)的平均活动模式

数据结构：
---------
输出形状：(体素数, 事件数, 球体数)
- 体素数：球体内的体素数量（不足123则用999填充）
- 事件数：电影中的事件数量（事件边界数-1）
- 球体数：预定义的球体总数

填充策略：
---------
- 球体内体素不足123个时，用999填充（标记为无效体素）
- 分析时通过np.nanmean == 999识别并移除无效体素
"""

import json

import numpy as np
import nibabel as nib
import scipy.io as sio
import os
from pathlib import Path


def load_spheres_data(path: str):
    """
    加载 MATLAB spheres.mat 文件并解析球体定义

    spheres.mat 包含：
    - sphereCenters：每个球体的中心坐标 (x, y, z)
    - sphereInds：每个球体包含的体素索引（相对于mask的非零体素）

    返回：
    - sphereCenters: {'x': [...], 'y': [...], 'z': [...]}
    - sphereInds: [[体素索引列表], ...]，每个列表对应一个球体

    用途：
    - sphereCenters用于将分析结果映射回3D脑空间
    - sphereInds用于从NIfTI数据中提取球体区域的时间序列
    """
    data = sio.loadmat(os.path.join(path, 'spheres.mat'))

    # 统一解析 sphereCenters
    sc_raw = data['sphereCenters']
    if sc_raw.shape == (1, 1):
        sc = sc_raw[0, 0]
        sphereCenters = {k: sc[k].flatten() for k in ('x', 'y', 'z')}
    else:
        sc = sc_raw[0][0]
        sphereCenters = {k: sc[k].flatten() for k in ('x', 'y', 'z')}

    # 解析 sphereInds（MATLAB cell array）
    si_raw = data['sphereInds']
    if si_raw.dtype == object:
        sphereInds = [np.asarray(si_raw.flat[i]).flatten() for i in range(si_raw.size)]
    else:
        sphereInds = si_raw

    return sphereCenters, sphereInds


def process_subject(iSub_idx: int, file_path: str, Mask: np.ndarray,
                    sphereInds: list, allBoundaries: np.ndarray,
                    DiffBoundaries: np.ndarray, distCtr: int,
                    iLag: int, numEvents: int, out_dir: str, ignore_list: list = None, iSub_movie_id: int = None):
    """
    处理单个被试的球体数据，提取多时间点表示

    核心流程：
    1. 加载NIfTI数据并转换为(time, voxel)格式
    2. 对每个球体，提取EB/P/F和sceneAvg表示
    3. 将结果保存为.npz文件

    参数说明：
    - file_path: NIfTI文件路径
    - Mask: 脑mask，定义有效体素范围
    - sphereInds: 球体定义，每个球体包含的体素索引
    - allBoundaries: 事件边界时间点(TR单位)
    - DiffBoundaries: 相邻事件边界的时间差
    - distCtr: 控制条件的距离(单位：TR)，通常为10
    - iLag: 血流动力学延迟(单位：TR)，GUMP=3, Film Festival=1
    - numEvents: 事件数量
    - ignore_list: 忽略的事件索引（如短场景、无关场景）

    时间点选择策略：
    --------------
    - EB: allBoundaries[i] + iLag - 1 (考虑延迟的事件边界时刻)
    - P: allBoundaries[i] - distCtr - 1 (事件边界前的控制点)
    - F: allBoundaries[i] + distCtr - 1 (事件边界后的控制点)
    - sceneAvg: 场景内所有TR的平均，排除边界前后boundary_cut_tr个TR
    """
    if iSub_movie_id is not None:
        print(f'Processing subject {iSub_idx + 1} movie {iSub_movie_id + 1}')
    else:
        print(f'Processing subject {iSub_idx + 1}')

    # 加载NIfTI数据，形状为(x, y, z, time)
    nii = nib.load(file_path)
    Data = nii.get_fdata().astype(np.float32)

    # 扁平化成 time × voxel（列优先布局，与MATLAB兼容）
    # 原始形状: (x, y, z, time)
    # moveaxis后: (time, x, y, z)
    # reshape后: (time, x*y*z) - 每行是一个时间点，每列是一个体素
    Data = np.moveaxis(Data, -1, 0).reshape(Data.shape[-1], -1, order='F')
    print(Data.shape)

    num_spheres = len(sphereInds)
    # 表示数组形状：(最大体素数=123, 事件数, 球体数)
    # 使用123作为最大体素数是为了统一格式，不足的用999填充
    rep_shape = (123, numEvents, num_spheres)
    repEB = np.full(rep_shape, np.nan, dtype=np.float32)  # 事件边界表示
    repP = np.full(rep_shape, np.nan, dtype=np.float32)   # 过去控制表示
    repF = np.full(rep_shape, np.nan, dtype=np.float32)   # 未来控制表示
    sceneAvg = np.full(rep_shape, np.nan, dtype=np.float32)  # 场景平均表示

    # 计算控制 TR
    ctrP = allBoundaries - distCtr - 1
    ctrF = allBoundaries + distCtr - 1

    # 提前计算非零 mask 索引
    nonZero = np.where(Mask.flatten(order='F') != 0)[0]

    # 主循环：遍历每个球体，提取其表示
    for iSphere, inds in enumerate(sphereInds):
        print(f'  Sphere {iSphere + 1}/{num_spheres}')

        # 将mask非零体素索引映射到完整数据空间的体素索引
        # sphereInds存储的是相对于mask非零体素的索引，需要转换为绝对索引
        sphere_idx = nonZero[inds.flatten() - 1]

        # 提取该球体的 time × voxel 数据
        # sphere_data[i, j] = 第i个时间点，第j个球体内体素的BOLD值
        sphere_data = Data[:, sphere_idx]

        # ---- 提取事件边界(EB)、过去控制(P)、未来控制(F)的表示 ----
        for iEvent in range(numEvents):
            eventBound = allBoundaries[iEvent] - 1

            for cond, offset, rep in (
                ("EB", eventBound, repEB),
                ("P", ctrP[iEvent], repP),
                ("F", ctrF[iEvent], repF),
            ):
                idx = offset + iLag
                if 0 <= idx < sphere_data.shape[0]:
                    temp = sphere_data[idx, :]
                    padded = np.pad(
                        temp, (0, max(0, 123 - len(temp))), constant_values=999
                    )[:123]
                    rep[:, iEvent, iSphere] = padded

        # ---- 提取场景平均表示 ----
        # 原理：场景平均反映整个场景期间的稳定活动模式
        # 排除边界附近以避免边界转换的影响
        boundary_cut_tr = 4  # 边界前后排除的TR数
        max_tr_for_mean = 50  # 超过此阈值时使用分块降采样
        n_bins = 50  # 降采样bin数
        for iEvent in range(numEvents):
            eventBound = allBoundaries[iEvent]

            # 确定场景的起止时间（排除边界附近）
            if iEvent == 0:
                # 第一个场景：从第1个TR到第一个事件边界
                start = 1 + boundary_cut_tr + iLag - 1
                end = eventBound - boundary_cut_tr + iLag
            else:
                # 其他场景：从上一个事件边界到当前事件边界
                start = allBoundaries[iEvent - 1] + boundary_cut_tr + iLag - 1
                end = eventBound - boundary_cut_tr + iLag

            # 只处理有效场景：时间范围合法，且场景持续>10 TR
            if 0 <= start < end <= sphere_data.shape[0] and (
                iEvent == 0 or DiffBoundaries[iEvent - 1] > 10
            ):
                duration = end - start
                if duration > max_tr_for_mean:
                    # 分块降采样：等分为n_bins个bin，bin内先均值降噪，再对bin均值求总均值
                    bin_edges = np.linspace(start, end, n_bins + 1, dtype=int)
                    temp = np.stack([
                        sphere_data[bin_edges[i]:bin_edges[i+1], :].mean(axis=0)
                        for i in range(n_bins)
                    ]).mean(axis=0)
                else:
                    # 短场景直接全量均值
                    temp = sphere_data[start:end, :].mean(axis=0)
                padded = np.pad(
                    temp, (0, max(0, 123 - len(temp))), constant_values=999
                )[:123]
                sceneAvg[:, iEvent, iSphere] = padded

        # 屏蔽特定事件
        for arr in (repEB, repP, repF, sceneAvg):
            arr[:, ignore_list, iSphere] = np.nan

    # 保存结果
    os.makedirs(out_dir, exist_ok=True)
    if iSub_movie_id is not None:
        out_file = Path(out_dir) / (f"sub-{iSub_idx + 1}_{iSub_movie_id}" + "_reps.npz")
    else:
        out_file = Path(out_dir) / (Path(file_path).stem + "_reps.npz")
    np.savez_compressed(out_file, repEB=repEB, sceneAvg=sceneAvg,
                        repP=repP, repF=repF)
    print(f'  Saved: {out_file}')


def read_boundaries(path, tr_duration):

    # 读取JSON文件
    with open(path, 'r') as json_file:
        time_series = json.load(json_file)

    # 将字典中的时间节点转换为list
    time_series_list = list(time_series)
    start_time = time_series_list[0]
    time_series_list = [int((t -  start_time)/ tr_duration) for t in time_series_list]
    # 转换为ndarray
    time_series_array = np.array(time_series_list[1:])
    return time_series_array



def find_subdirectories(base_directory):
    # 初始化存储符合条件的文件夹名称列表
    subdirectories = []

    # 遍历文件夹
    for root, dirs, files in os.walk(base_directory):
        # 过滤以"sub-"开头的文件夹
        for dir_name in dirs:
            if dir_name.startswith("sub-"):
                subdirectories.append(os.path.join(root, dir_name))

    return sorted(subdirectories)


def filter_event(arr):
    """
    过滤事件边界，移除间隔太小的"假"事件

    原理：相邻事件边界间隔<4 TR的可能是噪声，而非真实的事件转换
    这个函数确保每个事件之间有足够的持续时间

    参数：
    - arr: 原始事件边界时间点列表

    返回：
    - 过滤后的事件边界时间点列表（相邻间隔>=4 TR）
    """
    # 首先确保输入是一个ndarray
    arr = np.asarray(arr)  # 确保输入为ndarray
    arr = np.insert(arr, 0, 0)  # 在开头添加0作为起点
    # 创建一个新的列表（使用列表推导）
    filtered = [arr[0]]  # 添加第一个元素

    # 遍历数组，从第二个元素开始
    for i in range(1, len(arr)):
        # 检查与上一个元素的差值
        if abs(arr[i] - filtered[-1]) >= 4:
            filtered.append(arr[i])

    return np.array(filtered[1:])


def gump_process():
    """
    处理GUMP电影数据集的表示提取

    GUMP数据集特点：
    - TR = 2秒（扫描重复时间）
    - iLag = 3 TR（血流动力学延迟约6秒）
    - 共35个场景事件（减1得到34个可分析事件）
    - 需要忽略的场景：短场景[0]和无关场景[13,14,26,27]

    数据路径：
    - NIfTI文件：/path/to/gump_data/mri_data/niis/
    - 事件边界：/path/to/gump_data/mri_data/clipped_times.json
    - 输出：/path/to/gump_data/process_data/reps/
    """
    iLag = 3  # BOLD血流动力学延迟（3 TR ≈ 6秒）
    numEvents = 35 - 1 # 事件数
    distCtr = 7  # 控制条件的距离（7 TR ≈ 15秒）
    ignore_list = [0, 13, 14, 26, 27]

    # 防御性断言：ignore_list 必须与 main/utils/truncate_indices.py 的
    # IGNOR_1BASED_FROM_GETREPS=[1,14,15,27,28] 一一对应（0-based = 1-based - 1）。
    # 若因果候选事件（来自 truncate_indices.py 输出的 npy）落在 ignore_list 内，
    # 会在 para_gesuange_gump.py 中触发二次屏蔽。真正的"因果候选 vs ignore_list
    # 重叠检查"应在 para_gesuange_gump.py 加载 npy 后实现。
    assert all(0 <= i < numEvents for i in ignore_list), \
        f"ignore_list {ignore_list} 越界：必须在 [0, {numEvents}) 范围内"
    assert len(ignore_list) == 5, \
        f"ignore_list 应有 5 个事件（对应 truncate_indices.py 的 [1,14,15,27,28]），实际 {len(ignore_list)}"

    # base_path = Path('/disk/lqy_event_casualty')
    # path_mask_data = base_path / 'osfstorage-archive'

    root_path = "/path/to/gump_data/"
    path_data = os.path.join(root_path, "mri_data")
    time_path = os.path.join(path_data, "clipped_times_v2.json")

    path_func = os.path.join(path_data, "niis_v2")
    out_dir = os.path.join(root_path, "process_data", "reps_v2")
    #
    # from utils import file_tools
    # file_tools.move_nii_2_folder(path_data, path_func, [1,2,3,4,5,6,9,10,14,15,16,17,18,19,20])

    Mask = nib.load(Path(root_path) / 'MNI152_resampled_mask.nii').get_fdata()
    sphereCenters, sphereInds = load_spheres_data(str(root_path))
    allBoundaries = read_boundaries(time_path, tr_duration=2)
    DiffBoundaries = np.diff(allBoundaries)

    files = sorted(f for f in os.listdir(path_func) if f.endswith('.nii'))

    for iSub_idx, fname in enumerate(files[:15]):
        process_subject(
            iSub_idx=iSub_idx,
            file_path=str(Path(path_func) / fname),
            Mask=Mask,
            sphereInds=sphereInds,
            allBoundaries=allBoundaries,
            DiffBoundaries=DiffBoundaries,
            distCtr=distCtr,
            iLag=iLag,
            numEvents=numEvents,
            out_dir=str(out_dir),
            # ignore_list: 忽略的场景索引
            # - [0]: 第一个场景（作为初始化，无重激活效应）
            # - [13,14,26,27]: 不同run切分相关场景（控制实验设计因素）
            ignore_list=ignore_list
        )

def gump_process_multi_ctr(distCtr_list=None):
    """
    处理GUMP电影数据集 — 多 distCtr 对比实验

    依次尝试多个 distCtr 值，分别保存结果到不同文件夹（reps_v2_ctrX）。
    公共数据（Mask、sphereInds、allBoundaries）只加载一次。

    参数:
        distCtr_list: 要尝试的 distCtr 列表，默认 [3, 4, 5, 6, 7, 8]
    """
    if distCtr_list is None:
        distCtr_list = [3, 4, 5, 6, 7, 8]

    iLag = 3
    numEvents = 35 - 1
    ignore_list = [0, 13, 14, 26, 27]

    root_path = "/path/to/gump_data/"
    path_data = os.path.join(root_path, "mri_data")
    time_path = os.path.join(path_data, "clipped_times_v2.json")
    path_func = os.path.join(path_data, "niis_v2")

    # 公共数据只加载一次
    Mask = nib.load(Path(root_path) / 'MNI152_resampled_mask.nii').get_fdata()
    sphereCenters, sphereInds = load_spheres_data(str(root_path))
    allBoundaries = read_boundaries(time_path, tr_duration=2)
    DiffBoundaries = np.diff(allBoundaries)

    files = sorted(f for f in os.listdir(path_func) if f.endswith('.nii'))

    for distCtr in distCtr_list:
        out_dir = os.path.join(root_path, "process_data", f"reps_v2_ctr{distCtr}")
        print(f"\n{'='*60}")
        print(f"  distCtr = {distCtr}  →  {out_dir}")
        print(f"{'='*60}")
        for iSub_idx, fname in enumerate(files[:15]):
            process_subject(
                iSub_idx=iSub_idx,
                file_path=str(Path(path_func) / fname),
                Mask=Mask,
                sphereInds=sphereInds,
                allBoundaries=allBoundaries,
                DiffBoundaries=DiffBoundaries,
                distCtr=distCtr,
                iLag=iLag,
                numEvents=numEvents,
                out_dir=str(out_dir),
                ignore_list=ignore_list,
            )


if __name__ == "__main__":

    gump_process()
    # gump_process_multi_ctr()
