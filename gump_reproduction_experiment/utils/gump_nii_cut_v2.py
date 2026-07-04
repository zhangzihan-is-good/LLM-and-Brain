"""
gump_nii_cut_v2.py — studyforrest/GUMP NIfTI 切分与拼接

数据特征（基于 studyforrest 3T 实际数据 + Hanke 2014 melt 命令验证）：
  - 每个 run 是独立的视频片段，events.tsv 中 videotime/frameidx/audiotime 均从 0/1 开始
  - 相邻 segment 间存在 16s（8 vols）内容重放（melt ad in/out 证据：Seg N+1 开头复用 Seg N 末尾 400 frames）
  - gump_movie_times_final.json 是跨越所有 run 的全局连续电影时间线

策略：
  - 使用论文已知参数（overlap=4, trim_after=4，前后各移除 4 vols 共 8 vols = 16s overlap）
  - events.tsv 仅用于获取 per-run 视频时长
  - 裁剪后拼接 volumes 形成纯净全局时间线：movie_time = volume_idx × TR
  - 眼动 frameid 为 per-run 本地，仅用于 intra-run 验证
"""

import argparse
import gzip
import json
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd

ACTIVE_SUBJECTS = [1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 17, 18, 19, 20]
TR = 2.0
FPS = 25


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GUMP NIfTI cut v2: boundary-aware run concatenation")
    p.add_argument("--root-path", default="",
                  help="Server root for GUMP MRI data")
    p.add_argument("--subjects", type=int, nargs="+", default=ACTIVE_SUBJECTS,
                  help="Subject IDs to process")
    p.add_argument("--tr", type=float, default=TR)
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--overlap-vols", type=int, default=4,
                  help="Replay volumes at start of runs 2+ (paper-confirmed: 16s overlap, split 4+4 with trim_after)")
    p.add_argument("--extra-vols", type=int, default=0,
                  help="Extra volumes at end of each run (paper-confirmed: 0, acquired = content)")
    p.add_argument("--trim-before", type=int, default=0,
                  help="Additional volumes to trim before boundary")
    p.add_argument("--trim-after", type=int, default=4,
                  help="Volumes to trim at end of non-last runs (mirrors overlap_vols for symmetric 4+4 cleanup)")
    p.add_argument("--bids-root", default="",
                  help="BIDS root for events.tsv and eyegaze files (per-subject)")
    p.add_argument("--eyegaze-dir", default=None,
                  help="Override: directory with eyegaze physio files")
    p.add_argument("--movie-times-json", default=None,
                  help="Path to gump_movie_times_final.json")
    p.add_argument("--events-dir", default=None,
                  help="Override directory for events.tsv files")
    p.add_argument("--output-suffix", default="cut_v2",
                  help="Output subdirectory name under func/")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_run_events(events_tsv_path):
    """Read a BIDS events.tsv (frame-level) and return DataFrame."""
    df = pd.read_csv(events_tsv_path, sep="\t")
    required = {"onset", "videotime", "frameidx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"events.tsv missing columns: {missing}")
    return df


def load_eyegaze(eyegaze_path):
    """Read eyegaze physio file (4 cols, no header: x, y, pupil, frameid)."""
    if eyegaze_path.endswith(".gz"):
        df = pd.read_csv(eyegaze_path, sep="\t", header=None, compression="gzip")
    else:
        df = pd.read_csv(eyegaze_path, sep="\t", header=None)
    if df.shape[1] < 4:
        raise ValueError(f"Expected >= 4 columns in eyegaze file, got {df.shape[1]}")
    df.columns = ["x", "y", "pupil", "frameid"] + list(df.columns[4:])
    return df


# ---------------------------------------------------------------------------
# Per-run video duration from events.tsv
# ---------------------------------------------------------------------------
def get_run_video_duration(events_df, tr):
    """Get per-run video content duration from events.tsv.

    Returns (video_duration_sec, estimated_content_vols).
    The events.tsv only covers stimulus presentation (not extra scan volumes).
    """
    video_end = float(events_df["videotime"].max())
    content_vols = int(np.ceil(video_end / tr))
    return video_end, content_vols


# ---------------------------------------------------------------------------
# Global offset computation
# ---------------------------------------------------------------------------
def compute_global_offsets(run_video_durations, overlap_vols, tr):
    """
    Compute per-run global movie time offset.

    Run 1 starts at global time 0 (no replay).
    Run 2+ starts with replay of previous run's end, so the NEW content
    begins at previous_offset + (prev_video_duration - overlap_vols * tr).

    Returns list of global offsets (one per run).
    """
    offsets = [0.0]
    for r in range(1, len(run_video_durations)):
        prev_content = run_video_durations[r - 1] - overlap_vols * tr
        offsets.append(offsets[-1] + prev_content)
    return offsets


# ---------------------------------------------------------------------------
# Trim index computation
# ---------------------------------------------------------------------------
def compute_trim_indices(n_volumes_list, overlap_vols, extra_vols,
                         trim_before, trim_after):
    """
    Compute per-run [start, end) slice indices after trimming.

    Default (overlap=4, extra=0, trim_after=4) achieves symmetric 4+4 cleanup
    at every adjacent-segment boundary (total 8 vols = 16s = real overlap):
      Run 1 (no replay):   [0, n - trim_after]                  = [0, 447)
      Run 2+ (non-last):   [overlap, n - trim_after]            = [4, n-4)
      Last run:            [overlap, n]                         = [4, n)

    Non-last runs always trim at least max(extra+trim_after, overlap) from end
    (to remove content that next run replays).
    """
    n_runs = len(n_volumes_list)
    indices = []
    for r in range(n_runs):
        n = n_volumes_list[r]
        overlap = overlap_vols if r > 0 else 0
        start = overlap + (trim_before if r > 0 else 0)
        is_last = (r == n_runs - 1)
        if is_last:
            end = n  # last run: no end trim
        else:
            # Non-last: trim end by max(extra_vols + trim_after, overlap_vols)
            # Even if no extra vols detected, the end content is replayed
            # in the next run's overlap, so remove at least overlap_vols.
            end_trim = max(extra_vols + trim_after, overlap_vols)
            end = n - end_trim
        start = max(0, start)
        end = min(n, end)
        if end <= start:
            print(f"  WARNING: Run {r+1} has no valid volumes "
                  f"(start={start}, end={end}, n={n})")
        indices.append((start, end))
    return indices


# ---------------------------------------------------------------------------
# NIfTI concatenation
# ---------------------------------------------------------------------------
def concat_and_trim(nii_paths, trim_indices, dry_run=False):
    """Load each run's NIfTI, apply trimming, concatenate along time axis."""
    if dry_run:
        return None

    total_data = None
    affine = None
    header = None

    for i, (nii_path, (start, end)) in enumerate(zip(nii_paths, trim_indices)):
        print(f"  Loading {nii_path} ...")
        img = nib.load(nii_path)
        data = img.get_fdata()

        trimmed = data[..., start:end]
        print(f"    Run {i+1}: volumes {start}-{end-1} -> keeping {trimmed.shape[3]}")

        if total_data is None:
            total_data = trimmed
            affine = img.affine
            header = img.header
        else:
            total_data = np.concatenate((total_data, trimmed), axis=3)

    header.set_data_shape(total_data.shape)
    return total_data, affine, header


# ---------------------------------------------------------------------------
# Eyegaze intra-run validation
# ---------------------------------------------------------------------------
def validate_eyegaze_intra_run(run_luts, eyegaze_paths, trim_indices):
    """
    Validate timing alignment within each run using eyegaze frameid.

    Since frameid resets per run, only intra-run checks are meaningful:
    - frameid should be monotonically increasing within each trimmed range
    """
    results = {"per_run": {}, "warnings": [], "validated": True}

    for r in range(len(run_luts)):
        if eyegaze_paths[r] is None:
            results["warnings"].append(f"Run {r+1}: no eyegaze file")
            continue
        try:
            gaze = load_eyegaze(eyegaze_paths[r])
            frameids = gaze["frameid"].values
            start, end = trim_indices[r]
            # Downsample eyegaze to volume level: take median frameid per TR window
            vol_frameids = []
            for v in range(start, end):
                t_start = v * TR
                t_end = (v + 1) * TR
                # Approximate: frameid at time t ≈ t * FPS
                est_fid = int(t_start * FPS)
                vol_frameids.append(est_fid)
            diffs = np.diff(vol_frameids)
            non_positive = np.sum(diffs <= 0)
            results["per_run"][r] = {
                "n_volumes": len(vol_frameids),
                "first_frameid": vol_frameids[0] if vol_frameids else None,
                "last_frameid": vol_frameids[-1] if vol_frameids else None,
                "monotonic": bool(non_positive == 0),
            }
            if non_positive > 0:
                results["warnings"].append(
                    f"Run {r+1}: {non_positive} non-monotonic steps")
                results["validated"] = False
        except Exception as e:
            results["warnings"].append(f"Run {r+1}: {e}")
            results["validated"] = False

    return results


# ---------------------------------------------------------------------------
# clipped_times.json generation
# ---------------------------------------------------------------------------
def generate_clipped_times(movie_times, total_trimmed_duration, tr):
    """
    Generate clipped_times.json from gump_movie_times_final.json.

    After trimming, concatenated volume i maps to movie_time = i * TR.
    Keep events with t <= total_trimmed_duration.
    Output: [0.0, t1, t2, ...] compatible with read_boundaries().
    """
    clipped = [t for t in movie_times if t <= total_trimmed_duration]
    if clipped and clipped[0] != 0.0:
        clipped = [0.0] + clipped
    return clipped


# ---------------------------------------------------------------------------
# Event-to-volume mapping (for debugging/verification)
# ---------------------------------------------------------------------------
def map_events_to_volumes(movie_times, global_offsets, run_content_durations, tr):
    """Map global movie time events to per-run local volume indices."""
    mappings = []
    for t in movie_times:
        if t < 0:
            continue
        # Find which run this event falls in
        for r, offset in enumerate(global_offsets):
            end = offset + run_content_durations[r] if r < len(run_content_durations) else float("inf")
            if t < end or r == len(global_offsets) - 1:
                local_vol = (t - offset) / tr
                mappings.append({
                    "global_time": t,
                    "run": r + 1,
                    "local_volume": local_vol,
                })
                break
    return mappings


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------
def process_subject(sub_id, args):
    sub_tag = f"sub-{sub_id:02d}"
    func_dir = os.path.join(args.root_path, sub_tag, "ses-movie", "func")
    output_dir = os.path.join(func_dir, args.output_suffix)

    print(f"\n{'='*60}")
    print(f"Processing {sub_tag}")
    print(f"{'='*60}")

    # ---- Paths ----
    # NIfTI files are under root-path (mri_data/)
    # events.tsv and eyegaze are under bids-root (ds000113-download/)
    bids_func_dir = os.path.join(args.bids_root, sub_tag, "ses-movie", "func")
    events_dir = args.events_dir or bids_func_dir
    eyegaze_dir = args.eyegaze_dir or bids_func_dir

    nii_paths = []
    events_paths = []
    eyegaze_paths = []
    for r in range(1, args.n_runs + 1):
        nii = os.path.join(func_dir,
                           f"{sub_tag}_ses-movie_task-movie_run-{r}"
                           f"_space-MNI152NLin2009cAsym_desc-preproc_bold.hpf.hpf.nii")
        evt = os.path.join(events_dir,
                           f"{sub_tag}_ses-movie_task-movie_run-{r}_events.tsv")
        egz = os.path.join(eyegaze_dir,
                           f"{sub_tag}_ses-movie_task-movie_run-{r}"
                           f"_recording-eyegaze_physio.tsv.gz")
        nii_paths.append(nii)
        events_paths.append(evt)
        eyegaze_paths.append(egz if os.path.exists(egz) else None)

    # ---- Load per-run info ----
    print("\n--- Per-run information ---")
    run_video_durations = []
    run_n_volumes = []  # actual NIfTI volume counts
    run_content_vols_est = []  # estimated from events.tsv
    detected_extra_vols = []  # per-run detected extra volumes

    for r in range(args.n_runs):
        # Video duration from events.tsv
        if os.path.exists(events_paths[r]):
            df = load_run_events(events_paths[r])
            vdur, cvols = get_run_video_duration(df, args.tr)
            run_video_durations.append(vdur)
            run_content_vols_est.append(cvols)
            print(f"  Run {r+1}: video duration = {vdur:.1f}s "
                  f"({cvols} content vols from events.tsv)")
        else:
            print(f"  Run {r+1}: events.tsv not found, will use NIfTI volume count")
            run_video_durations.append(None)
            run_content_vols_est.append(None)
            detected_extra_vols.append(args.extra_vols)

        # Actual NIfTI volume count
        if os.path.exists(nii_paths[r]):
            img = nib.load(nii_paths[r])
            n_vol = img.shape[3]
            run_n_volumes.append(n_vol)
            print(f"           NIfTI volumes = {n_vol}")

            # Detect extra volumes: NIfTI has more volumes than
            # events.tsv video duration covers
            if run_video_durations[r] is not None:
                video_vols = run_content_vols_est[r]
                detected_extra = n_vol - video_vols
                detected_extra_vols.append(max(detected_extra, 0))
                print(f"           Video vols (events.tsv): {video_vols}")
                print(f"           Extra vols (NIfTI - video): {detected_extra} "
                      f"{'(paper: ~5)' if detected_extra != args.extra_vols else '(matches expected)'}")
        else:
            if run_video_durations[r] is not None:
                run_n_volumes.append(run_content_vols_est[r] + args.overlap_vols + args.extra_vols)
                detected_extra_vols.append(args.extra_vols)
            else:
                print(f"  ERROR: Neither events.tsv nor NIfTI found for run {r+1}")
                return None
            print(f"           NIfTI not found, estimated = {run_n_volumes[-1]}")

    # Check consistency across subjects: all runs should have same extra count
    if detected_extra_vols:
        unique_extras = set(detected_extra_vols)
        if len(unique_extras) == 1:
            actual_extra = detected_extra_vols[0]
            if actual_extra != args.extra_vols:
                print(f"\n  NOTE: Detected extra_vols={actual_extra} across all runs, "
                      f"using this instead of default {args.extra_vols}")
                args.extra_vols = actual_extra
        else:
            print(f"\n  WARNING: Inconsistent extra_vols across runs: "
                  f"{detected_extra_vols}, using first run's value")
            args.extra_vols = detected_extra_vols[0]

    # ---- Compute global offsets ----
    print("\n--- Global time mapping ---")
    global_offsets = compute_global_offsets(
        run_video_durations, args.overlap_vols, args.tr)

    # Per-run content duration (excluding replay and extra)
    run_content_durations = []
    for r in range(args.n_runs):
        content_dur = run_video_durations[r] - args.overlap_vols * args.tr
        run_content_durations.append(content_dur)
        print(f"  Run {r+1}: global offset = {global_offsets[r]:.1f}s, "
              f"content duration = {content_dur:.1f}s")

    total_movie_duration = sum(run_content_durations)
    print(f"  Total movie content: {total_movie_duration:.1f}s "
          f"({total_movie_duration/60:.1f} min)")

    # ---- Compute trim indices ----
    print("\n--- Trim indices ---")
    trim_indices = compute_trim_indices(
        run_n_volumes, args.overlap_vols, args.extra_vols,
        args.trim_before, args.trim_after)

    total_trimmed = 0
    for r, (s, e) in enumerate(trim_indices):
        n_keep = e - s
        removed_start = s
        removed_end = run_n_volumes[r] - e
        is_last = (r == args.n_runs - 1)
        print(f"  Run {r+1}: [{s}, {e}) -> keep {n_keep}/{run_n_volumes[r]} "
              f"(removed: {removed_start} start, {removed_end} end"
              f"{'' if is_last else ' (non-last: forced end trim)'})")
        total_trimmed += n_keep

    total_trimmed_duration = total_trimmed * args.tr
    print(f"\n  Total trimmed: {total_trimmed} volumes, "
          f"{total_trimmed_duration:.1f}s ({total_trimmed_duration/60:.1f} min)")

    # ---- Concatenate ----
    print("\n--- Concatenation ---")
    result = concat_and_trim(nii_paths, trim_indices, dry_run=args.dry_run)

    # ---- Load movie times & generate clipped_times.json ----
    if args.movie_times_json and os.path.exists(args.movie_times_json):
        mt_path = args.movie_times_json
    else:
        # Try common locations
        candidates = [
            os.path.join(args.root_path, "gump_movie_times_final.json"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "data", "gump", "gump_movie_times_final.json"),
        ]
        mt_path = None
        for c in candidates:
            if os.path.exists(c):
                mt_path = c
                break

    movie_times = []
    if mt_path:
        with open(mt_path) as f:
            movie_times = json.load(f)
        print(f"\n  Loaded {len(movie_times)} events from movie times JSON "
              f"({movie_times[0]}-{movie_times[-1]}s)")

    if movie_times:
        clipped = generate_clipped_times(
            movie_times, total_trimmed_duration, args.tr)
        print(f"  clipped_times.json: {len(clipped)} events "
              f"(from {len(movie_times)} total)")
        print(f"  Recommended numEvents = {len(clipped) - 1}")

        # Verify event mapping
        if not args.dry_run:
            print("\n  --- Event-to-volume mapping (sample) ---")
            for evt in clipped[1:4]:
                vol = int(evt / args.tr)
                print(f"    t={evt}s -> volume {vol}")

            last_evt = clipped[-1]
            last_vol = int(last_evt / args.tr)
            print(f"    ...")
            print(f"    t={last_evt}s -> volume {last_vol}")
            print(f"    Total volumes available: {total_trimmed}")
    else:
        clipped = None
        print("  WARNING: No movie times JSON found, skipping clipped_times.json")

    # ---- Eyegaze validation ----
    print("\n--- Eyegaze validation (intra-run) ---")
    # Build minimal LUTs for eyegaze validation
    eyegaze_luts = []
    for r in range(args.n_runs):
        eyegaze_luts.append({"video_start": 0.0, "video_end": run_video_durations[r]})
    validation = validate_eyegaze_intra_run(eyegaze_luts, eyegaze_paths, trim_indices)
    print(f"  Validated: {validation['validated']}")
    for w in validation.get("warnings", []):
        print(f"  WARNING: {w}")
    for r, info in validation.get("per_run", {}).items():
        print(f"  Run {r+1}: {info['n_volumes']} vols, "
              f"frameid {info['first_frameid']}-{info['last_frameid']}, "
              f"monotonic={info['monotonic']}")

    # ---- Save outputs ----
    if not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)

        nii_out = os.path.join(output_dir, f"{sub_tag}_trimmed_bold.nii")
        nib.save(nib.Nifti1Image(result[0], result[1], result[2]), nii_out)
        print(f"\n  Saved NIfTI: {nii_out}")

        if clipped:
            json_out = os.path.join(output_dir, "clipped_times.json")
            with open(json_out, "w") as f:
                json.dump(clipped, f, indent=4)
            print(f"  Saved clipped_times.json ({len(clipped)} events)")

        timing = {
            "sub_id": sub_id,
            "tr": args.tr,
            "n_runs": args.n_runs,
            "overlap_vols": args.overlap_vols,
            "extra_vols": args.extra_vols,
            "global_offsets": global_offsets,
            "run_video_durations": run_video_durations,
            "run_content_durations": run_content_durations,
            "per_run": [],
            "total_trimmed_volumes": total_trimmed,
            "total_trimmed_duration_s": total_trimmed_duration,
        }
        for r in range(args.n_runs):
            timing["per_run"].append({
                "run": r + 1,
                "n_volumes_nifti": run_n_volumes[r],
                "video_duration": run_video_durations[r],
                "global_offset": global_offsets[r],
                "trim_indices": list(trim_indices[r]),
                "n_volumes_kept": trim_indices[r][1] - trim_indices[r][0],
            })
        timing_out = os.path.join(output_dir, f"{sub_tag}_timing.json")
        with open(timing_out, "w") as f:
            json.dump(timing, f, indent=4)
        print(f"  Saved timing: {timing_out}")

        val_out = os.path.join(output_dir, f"{sub_tag}_validation.json")
        with open(val_out, "w") as f:
            json.dump(validation, f, indent=4)
        print(f"  Saved validation: {val_out}")
    else:
        print(f"\n  [DRY RUN] Would save to: {output_dir}")

    return {
        "sub_id": sub_id,
        "total_trimmed_volumes": total_trimmed,
        "total_trimmed_duration": total_trimmed_duration,
        "clipped_events": len(clipped) if clipped else 0,
        "validated": validation.get("validated", False),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    print(f"GUMP NIfTI Cut v2")
    print(f"  Root: {args.root_path}")
    print(f"  Subjects: {args.subjects}")
    print(f"  TR: {args.tr}s, Runs: {args.n_runs}")
    print(f"  Overlap: {args.overlap_vols}, Extra: {args.extra_vols}")
    print(f"  Trim: before={args.trim_before}, after={args.trim_after}")
    print(f"  Dry run: {args.dry_run}")

    summaries = []
    for sub_id in args.subjects:
        try:
            summary = process_subject(sub_id, args)
            if summary:
                summaries.append(summary)
        except Exception as e:
            print(f"\n  ERROR processing sub-{sub_id:02d}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for s in summaries:
        print(f"  sub-{s['sub_id']:02d}: {s['total_trimmed_volumes']} vols, "
              f"{s['total_trimmed_duration']:.0f}s, "
              f"{s['clipped_events']} events, "
              f"validated={'Y' if s['validated'] else 'N'}")


if __name__ == "__main__":
    main()
