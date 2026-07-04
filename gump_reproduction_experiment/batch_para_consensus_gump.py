"""
GUMP 多配置批量因果重激活分析
================================

支持两种批量维度（通过 --mode 切换）：

  A. indice 模式（默认）— 不同 top-N 事件索引组合
     适配 main/utils/truncate_indices.py 的 --output-name 产出的子目录。
     默认 4 组与本地运行结果对齐：
       gump_i2_c5_r5 / gump_i2_c3_r3 / gump_i3_c3_r3 / gump_i3_c4_r4
     python main/batch_para_consensus_gump.py
     python main/batch_para_consensus_gump.py --indice-dirs gump_i2_c5_r5 gump_i3_c3_r3

  B. distctr 模式 — 不同控制距离
     适配 acc_getReps.py gump_process_multi_ctr() 产出的 reps_v2_ctr{X}/ 目录。
     python main/batch_para_consensus_gump.py --mode distctr
     python main/batch_para_consensus_gump.py --mode distctr --distctrs 3 5 7

对每个配置依次执行：
  1. para 阶段         (para_gesuange_gump.reactivation_analysis_parallel)
  2. para 统计检验     (para_gesuange_gump.run_statistical_testing)
  3. consensus 阶段    (consensus_gump.run_consensus_analysis, 支持多模式)

输出布局：
  gump_causality/{indice_name}/        ← indice 模式
  gump_causality/ctr{X}/               ← distctr 模式
    ├── mapT1_yin.npy, mapT2_yin.npy
    ├── mapT1_guo.npy, mapT2_guo.npy
    ├── mapT1_duli.npy, mapT2_duli.npy
    ├── dRI_guo_yin.nii, T1/T2_causality_1_graymasked.nii   ← para 统计
    └── consensus/
        ├── guo_yinduli.nii                                  ← tongshi
        └── guo_yin.nii, guo_duli.nii, duli_yin.nii, yin_duli.nii  ← separate

前置条件:
  - reps 数据已生成（indice 模式: reps_v2/; distctr 模式: reps_v2_ctr{X}/）
  - spheres.mat, MNI152_resampled_mask.nii 在 --spheres-path 下
  - cause/result/ind_event_indice.npy 在对应 indice_dir 下
    （indice 模式: {indice_base}/{indice_name}/；distctr 模式: {indice_base}/indices/）

部署提示:
  本地 truncate_indices.py 产出在 analyze_project/data/gump_*/，
  部署到服务器时把 4 个目录放到 --indice-base 下（默认 /path/to/gump_data/process_data/）。

用法:
    python main/batch_para_consensus_gump.py                          # 默认 4 组 indice
    python main/batch_para_consensus_gump.py --mode distctr           # distCtr 批量
    python main/batch_para_consensus_gump.py --modes tongshi          # 只跑 tongshi
    python main/batch_para_consensus_gump.py --indice-dirs gump_i2_c5_r5 gump_i3_c4_r4
"""

import argparse
import os
import sys


if __package__:
    from . import para_gesuange_gump as para
    from . import consensus_gump as cons
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import para_gesuange_gump as para
    import consensus_gump as cons


# ==================== 默认配置 ====================
SPHERES_PATH = ""
PROCESS_DATA = os.path.join(SPHERES_PATH, "process_data")
INDICE_BASE = os.path.join(PROCESS_DATA, "indices")                       # indice 子目录的根
REPS_BASE = PROCESS_DATA                          # reps 子目录的根
AFFINE_PATH = os.path.join(SPHERES_PATH, "MNI152_resampled_mask.nii")
CAUSALITY_BASE = os.path.join(PROCESS_DATA, "gump_causality")

# 4 组默认 indice（与 main/utils/truncate_indices.py 的 --output-name 一致）
DEFAULT_INDICE_DIRS = ["gump_i2_c5_r5" ]
DEFAULT_DISTCTRS = [3, 4, 5, 6, 7, 8]
DEFAULT_MODES = ["tongshi", "separate"]
DEFAULT_N_JOBS = 8


def process_one(reps_dir, indice_dir, output_dir, modes, n_jobs,
                spheres_path=SPHERES_PATH, affine_path=AFFINE_PATH):
    """通用单次处理：para + 统计检验 + consensus。

    Args:
        reps_dir: 包含所有被试 *_reps.npz 的目录。
        indice_dir: 包含 cause/result/ind_event_indice.npy 的目录。
        output_dir: para 输出目录（mapT*.npy + *.nii）。
        modes: consensus 模式列表，元素 ∈ {'consensus','separate','tongshi'}。
        n_jobs: para 阶段并行进程数。
        spheres_path: spheres.mat 与 MNI152_resampled_mask.nii 所在目录。
        affine_path: 提供 affine 的 NIfTI 路径。
    """
    consensus_output_dir = os.path.join(output_dir, "consensus")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(consensus_output_dir, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"#  reps_dir          = {reps_dir}")
    print(f"#  indice_dir        = {indice_dir}")
    print(f"#  para output       = {output_dir}")
    print(f"#  consensus output  = {consensus_output_dir}")
    print(f"#  modes             = {modes}")
    print(f"{'#' * 70}\n")

    # ===== Step 1: para 阶段 - 计算三种因果条件重激活 map =====
    print(f"Step 1/3: para reactivation analysis")
    para.reactivation_analysis_parallel(
        reps_dir=reps_dir,
        spheres_path=spheres_path,
        indice_dir=indice_dir,
        output_dir=output_dir,
        n_jobs=n_jobs,
    )

    # ===== Step 2: para 阶段 - guo vs yin 配对 t 检验 =====
    print(f"\nStep 2/3: para statistical testing (guo vs yin)")
    para.run_statistical_testing(
        output_dir=output_dir,
        spheres_path=spheres_path,
    )

    # ===== Step 3: consensus 阶段 - tongshi/separate 等模式 =====
    print(f"\nStep 3/3: consensus analysis ({', '.join(modes)})")
    cons.run_consensus_analysis(
        map_dir=output_dir,
        output_dir=consensus_output_dir,
        spheres_path=spheres_path,
        affine_path=affine_path,
        modes=modes,
    )


def main():
    parser = argparse.ArgumentParser(
        description="GUMP 批量 para + consensus 分析（支持 indice / distctr 两种维度）")
    parser.add_argument(
        "--mode", choices=["indice", "distctr"], default="indice",
        help="批量维度: indice (默认，遍历不同事件索引组合) 或 distctr (遍历不同控制距离)")
    parser.add_argument(
        "--indice-dirs", type=str, nargs="+", default=DEFAULT_INDICE_DIRS,
        help=f"[indice 模式] indice 子目录名列表 (相对 --indice-base)，"
             f"默认: {DEFAULT_INDICE_DIRS}")
    parser.add_argument(
        "--distctrs", type=int, nargs="+", default=DEFAULT_DISTCTRS,
        help=f"[distctr 模式] 要处理的 distCtr 列表，默认: {DEFAULT_DISTCTRS}")
    parser.add_argument(
        "--reps-name", type=str, default="reps_v2",
        help="[indice 模式] reps 子目录名 (相对 --reps-base)，默认 reps_v2")
    parser.add_argument(
        "--indice-base", type=str, default=INDICE_BASE,
        help=f"indice 根目录，默认: {INDICE_BASE}")
    parser.add_argument(
        "--reps-base", type=str, default=REPS_BASE,
        help=f"reps 根目录，默认: {REPS_BASE}")
    parser.add_argument(
        "--modes", type=str, nargs="+", default=DEFAULT_MODES,
        choices=["consensus", "separate", "tongshi"],
        help=f"consensus 模式，默认: {DEFAULT_MODES}")
    parser.add_argument(
        "--n-jobs", type=int, default=DEFAULT_N_JOBS,
        help=f"para 阶段并行进程数，默认: {DEFAULT_N_JOBS}")
    parser.add_argument(
        "--spheres-path", type=str, default=SPHERES_PATH,
        help=f"spheres.mat 所在目录，默认: {SPHERES_PATH}")
    parser.add_argument(
        "--causality-base", type=str, default=CAUSALITY_BASE,
        help=f"gump_causality 根目录，默认: {CAUSALITY_BASE}")
    args = parser.parse_args()

    affine_path = os.path.join(args.spheres_path, "MNI152_resampled_mask.nii")

    print(f"Batch mode:       {args.mode}")
    print(f"spheres_path:     {args.spheres_path}")
    print(f"causality_base:   {args.causality_base}")
    print(f"modes:            {args.modes}")
    print(f"n_jobs:           {args.n_jobs}")

    if args.mode == "indice":
        reps_dir = os.path.join(args.reps_base, args.reps_name)
        print(f"reps_dir (shared): {reps_dir}")
        print(f"indice_dirs:      {args.indice_dirs}")
        for name in args.indice_dirs:
            indice_dir = os.path.join(args.indice_base, name)
            output_dir = os.path.join(args.causality_base, name)
            process_one(
                reps_dir=reps_dir,
                indice_dir=indice_dir,
                output_dir=output_dir,
                modes=args.modes,
                n_jobs=args.n_jobs,
                spheres_path=args.spheres_path,
                affine_path=affine_path,
            )
        print(f"\nAll indice_dirs processed. Results under {args.causality_base}/")
    else:  # distctr
        indice_dir = args.indice_base
        print(f"indice_dir (shared): {indice_dir}")
        print(f"distctrs:         {args.distctrs}")
        for distCtr in args.distctrs:
            reps_dir = os.path.join(args.reps_base, f"reps_v2_ctr{distCtr}")
            output_dir = os.path.join(args.causality_base, f"ctr{distCtr}")
            process_one(
                reps_dir=reps_dir,
                indice_dir=indice_dir,
                output_dir=output_dir,
                modes=args.modes,
                n_jobs=args.n_jobs,
                spheres_path=args.spheres_path,
                affine_path=affine_path,
            )
        print(f"\nAll distctrs processed. Results under {args.causality_base}/ctr*/")


if __name__ == "__main__":
    main()
