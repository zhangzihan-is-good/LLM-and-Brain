#!/bin/bash

SRC_ROOT=""
DST_DIR="${SRC_ROOT}/nii_v2"

SUBJECTS=(1 2 3 4 5 6 9 10 14 15 16 17 18 19 20)

mkdir -p "$DST_DIR"

for sub in "${SUBJECTS[@]}"; do
    sub_tag=$(printf "sub-%02d" $sub)
    src="${SRC_ROOT}/${sub_tag}/ses-movie/func/cut_v2/${sub_tag}_trimmed_bold.nii"
    dst="${DST_DIR}/${sub_tag}_trimmed_bold.nii"
    if [ -f "$src" ]; then
        cp "$src" "$dst"
        echo "Copied: $src -> $dst"
    else
        echo "Not found: $src"
    fi
done

echo "Done. Files in nii_v2/:"
ls -lh "$DST_DIR"
