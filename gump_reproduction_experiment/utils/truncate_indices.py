"""
截取 GUMP 因果事件索引，过滤忽略事件，转为 0-based 后保存到 data/gump 目录。

忽略事件来源（1-based，与 divide_into_low_high.py 输出一致）：
- getReps 中 ignore_list [0,13,14,26,27] (0-based) → 1-based [1,14,15,27,28]
- 从 gump_movie_times_final.json 计算时长 < 10 TR 的事件 → 1-based [1,23]

用法:
    python main/utils/truncate_indices.py                          # 默认各取前3
    python main/utils/truncate_indices.py --top 5                  # 全部取前5
    python main/utils/truncate_indices.py --top-ind 3 --top-cause 5 --top-result 5
                                                              # 独立取3，因果各取5
"""
import argparse
import json
import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
SRC_DIR = os.path.join(BASE_DIR, '..', 'behavior_analysis', 'data', 'gump_data')
DST_DIR = os.path.join(BASE_DIR, 'data', 'gump')
TIMES_JSON = os.path.join(BASE_DIR, 'data', 'gump', 'gump_movie_times_final.json')
AGG_CSV = os.path.join(SRC_DIR, 'dawid_skene_aggregate.csv')
# 三个指标列 → 对应的输出文件名
COL_FILE_MAP = [
    ('p_independent', 'ind_event_indice.npy'),
    ('p_result', 'result_event_indice.npy'),
    ('p_cause', 'cause_event_indice.npy'),
]

# 输出文件名 → 对应的 top-N 参数名
FILE_TOP_ARG = {
    'ind_event_indice.npy': 'top_ind',
    'result_event_indice.npy': 'top_result',
    'cause_event_indice.npy': 'top_cause',
}

# getReps gump_process() 的 ignore_list，0-based 转为 1-based
# 原始: [0, 13, 14, 26, 27]
IGNOR_1BASED_FROM_GETREPS = [1, 14, 15, 27, 28, 35]


def _event_num(event_id: str) -> int:
    m = re.search(r"(\d+)", str(event_id))
    return int(m.group(1)) if m else -1


def load_aggregate_long(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["_event_num"] = df["event_id"].map(_event_num)
    df = df.sort_values("_event_num").reset_index(drop=True)
    return df


def compute_short_events(json_path, tr_duration, min_tr):
    """从 JSON 时间文件计算时长 < min_tr 的事件，返回 1-based 事件编号。"""
    with open(json_path) as f:
        times = json.load(f)
    durations = [(times[i + 1] - times[i]) / tr_duration for i in range(len(times) - 1)]
    short_0based = [i for i, d in enumerate(durations) if d < min_tr]
    return [i + 1 for i in short_0based], durations


def print_prob_table(df, ignore_1based, top_n):
    """打印每个事件的概率占比表，标注忽略和选中状态。"""
    cols = ['p_independent', 'p_result', 'p_cause']
    print('\n' + '=' * 90)
    print('所有事件概率占比 (1-based 事件编号)')
    print('=' * 90)
    hdr = f'{"事件":>4}  {"p_ind":>6}  {"p_res":>6}  {"p_cau":>6}  {"p_unc":>6}  {"时长/TR":>7}  状态'
    print(hdr)
    print('-' * 90)

    with open(TIMES_JSON) as f:
        times = json.load(f)
    durations = [(times[i + 1] - times[i]) / 2.0 for i in range(len(times) - 1)]

    p_unc_col = 'p_uncertain' if 'p_uncertain' in df.columns else None

    for _, row in df.iterrows():
        eid = row["_event_num"]
        p_ind = row['p_independent']
        p_res = row['p_result']
        p_cau = row['p_cause']
        p_unc = row[p_unc_col] if p_unc_col else 1.0 - p_ind - p_res - p_cau
        dur = durations[eid - 1] if eid <= len(durations) else 0
        status = ''
        if eid in ignore_1based:
            status = 'IGN'
        print(f'  {eid:>3}  {p_ind:>6.3f}  {p_res:>6.3f}  {p_cau:>6.3f}  {p_unc:>6.3f}  {dur:>7.1f}  {status}')
    print('-' * 90)


def print_ranking_with_probs(df, ignore_1based, top_per_file):
    """打印各指标的 top-N 排名及其概率值。"""
    print('\n' + '=' * 90)
    print('事件筛选结果')
    print('=' * 90)

    for col, name in COL_FILE_MAP:
        n = top_per_file[name]
        # 按概率降序排列
        sorted_df = df.sort_values(col, ascending=False)
        # 过滤忽略事件
        sorted_df = sorted_df[~sorted_df['_event_num'].isin(ignore_1based)]
        truncated = sorted_df.head(n)

        print(f'\n--- {col} 降序 (top={n}) ---')
        print(f'{"排名":>4}  {"事件(1b)":>7}  {"事件(0b)":>7}  {"概率":>7}  选中')
        for rank, (_, row) in enumerate(truncated.iterrows(), 1):
            eid = row['_event_num']
            print(f'  {rank:>3}  {int(eid):>6}  {int(eid) - 1:>6}  {row[col]:>7.4f}  YES')


def truncate_indices(dst_dir, top_n=3, top_ind=None, top_cause=None, top_result=None, show_probs=True, force_ind_1based=None, force_cause_1based=None, force_result_1based=None):
    os.makedirs(dst_dir, exist_ok=True)

    short_1based, durations = compute_short_events(TIMES_JSON, 2.0, 10)
    print(f'Duration < 10 TR (1-based): {short_1based}')
    print(f'  详情: ' + ', '.join(
        f'scene {i + 1}={d:.1f}TR' for i, d in enumerate(durations) if d < 10))

    # 合并忽略列表（1-based）
    ignore_1based = sorted(set(IGNOR_1BASED_FROM_GETREPS + short_1based))
    ignore_0based = [x - 1 for x in ignore_1based]
    print(f'Combined ignore (0-based): {ignore_0based}')
    print(f'Combined ignore (1-based): {ignore_1based}')

    # 每个文件的 top-N：优先用独立参数，否则用全局 top_n
    top_per_file = {
        'ind_event_indice.npy': top_ind if top_ind is not None else top_n,
        'result_event_indice.npy': top_result if top_result is not None else top_n,
        'cause_event_indice.npy': top_cause if top_cause is not None else top_n,
    }

    # 从 CSV 按概率降序排序，不依赖上游 npy 文件的顺序
    if not os.path.exists(AGG_CSV):
        raise FileNotFoundError(f'未找到 {AGG_CSV}，请先运行 GPT_new.py 生成频率统计')

    df = load_aggregate_long(AGG_CSV)

    if show_probs:
        print_prob_table(df, ignore_1based, top_n)
        print_ranking_with_probs(df, ignore_1based, top_per_file)

    force_map = {
        'ind_event_indice.npy': force_ind_1based,
        'cause_event_indice.npy': force_cause_1based,
        'result_event_indice.npy': force_result_1based,
    }

    for col, name in COL_FILE_MAP:
        # force 路径：直接用用户指定的事件编号，跳过 top-N 排序
        forced = force_map.get(name)
        if forced is not None:
            bad = [e for e in forced if e in ignore_1based]
            if bad:
                raise ValueError(f'force for {name}: 1-based {bad} 与 ignore 列表 {ignore_1based} 冲突')
            forced_arr = np.array([e - 1 for e in forced], dtype=int)
            np.save(os.path.join(dst_dir, name), forced_arr)
            print(f'\n{name} (forced): 1-based {forced} -> 0-based {forced_arr.tolist()}')
            continue

        n = top_per_file[name]
        # 按概率降序
        sorted_df = df.sort_values(col, ascending=False)
        # 过滤忽略事件
        sorted_df = sorted_df[~sorted_df['_event_num'].isin(ignore_1based)]
        # 截取前 N
        truncated = sorted_df.head(n)
        event_ids_1based = truncated['_event_num'].values.astype(int)
        # 转为 0-based
        result = event_ids_1based - 1
        np.save(os.path.join(dst_dir, name), result)
        print(f'\n{name} (top={n}): {event_ids_1based.tolist()} -> 0-based {result.tolist()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=int, default=3, help='全局默认 top-N')
    parser.add_argument('--top-ind', type=int, default=None, help='独立事件 top-N (默认用 --top)')
    parser.add_argument('--top-cause', type=int, default=None, help='因事件 top-N (默认用 --top)')
    parser.add_argument('--top-result', type=int, default=None, help='果事件 top-N (默认用 --top)')
    parser.add_argument('--output-name', type=str, default='gump',
                        help='输出子目录名（相对 data/），默认 gump')
    parser.add_argument('--quiet', action='store_true', help='只打印最终选中的事件索引')
    parser.add_argument('--force-ind', type=int, nargs='+', default=None,
                        help='强制指定 ind 事件编号（1-based），覆盖 --top-ind 排序，'
                             '例如 --force-ind 22 24 → 0-based [21, 23]')
    parser.add_argument('--force-cause', type=int, nargs='+', default=None,
                        help='强制指定 cause 事件编号（1-based），覆盖 --top-cause 排序')
    parser.add_argument('--force-result', type=int, nargs='+', default=None,
                        help='强制指定 result 事件编号（1-based），覆盖 --top-result 排序')
    args = parser.parse_args()
    dst_dir = os.path.join(BASE_DIR, 'data', args.output_name)
    truncate_indices(dst_dir, args.top,
                    top_ind=args.top_ind, top_cause=args.top_cause, top_result=args.top_result,
                    show_probs=not args.quiet,
                    force_ind_1based=args.force_ind,
                    force_cause_1based=args.force_cause,
                    force_result_1based=args.force_result)
