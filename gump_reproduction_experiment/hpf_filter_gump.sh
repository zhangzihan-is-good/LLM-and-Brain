#!/bin/bash
# GUMP 数据高通滤波脚本
# 工具: AFNI 3dBandpass
# 高通截止: 128s (0.0078 Hz) — SPM 默认值，自然范式常用
# 低通: 不施加（仅高通滤波）
# 数据状态: 经功率谱确认未做高通滤波（低频/中频比=135）
# 用法: bash hpf_filter_gump.sh

# 切分后nii文件地址
ROOT=""

for subj in {01..20}; do
  func_dir="${ROOT}/sub-${subj}/ses-movie/func"

  if [ ! -d "$func_dir" ]; then
    echo "[SKIP] $func_dir 不存在"
    continue
  fi

  echo "========== sub-${subj} =========="

  for nii in "${func_dir}"/*.nii; do
    [ -e "$nii" ] || continue

    fname=$(basename "$nii")
    # 去掉 .nii 得到基础文件名
    bname="${fname%.nii}"
    out_nii="${func_dir}/${bname}.hpf.nii"

    # 跳过已经处理过的文件
    if [ -e "${out_nii}" ]; then
      echo "[SKIP] ${bname} 已处理"
      continue
    fi

    echo "[RUN]  ${bname}"
    3dBandpass -prefix "${out_nii}" -band 0.0078 9999 "$nii"
  done
done

echo "全部完成"
