"""
构建46维特征矩阵缓存 (生产级高效版)

策略:
  - 数据按student排序，每块~4个学生
  - 单块学生(大部分): groupby后直接计算特征
  - 跨块学生(~10个): 缓存到DataFrame，满chunk时与新数据拼接
  - 全部学生: 按student ID顺序组织，结果整齐

预计: 150-200秒 (2-3分钟)，受IO限制
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy

EVENT_TYPES_7 = ['focus_gained', 'focus_lost', 'run', 'submit',
                 'text_insert', 'text_paste', 'text_remove']
OUT_DIR = 'outputs/compare_7dim_vs_46dim'
CACHE_PATH = os.path.join(OUT_DIR, 'features_46dim.npz')
os.makedirs(OUT_DIR, exist_ok=True)


def _safe_float(val, default=0.0):
    if isinstance(val, (int, float)):
        return float(val) if np.isfinite(val) else default
    if isinstance(val, np.ndarray):
        return float(val.flat[0]) if val.size > 0 and np.isfinite(val.flat[0]) else default
    return default


def _entropy(counts):
    if len(counts) == 0: return 0.0
    counts = np.array(counts, dtype=float)
    if counts.sum() == 0: return 0.0
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def compute_46dim(df: pd.DataFrame) -> np.ndarray:
    """从单个学生DataFrame计算46维特征"""
    f = np.zeros(46, dtype=np.float32)
    i = 0

    # ── 1. 事件基础统计 (28维) ────────────────────
    for et in EVENT_TYPES_7:
        sub = df[df['eventType'] == et]
        if len(sub) > 0:
            t = (sub['timestamp'] - sub['timestamp'].min()).dt.total_seconds().values
            if len(t) < 2: t = np.array([0.0, 0.0])
        else:
            t = np.array([0.0, 0.0])
        f[i] = _safe_float(np.mean(t)); i += 1
        f[i] = _safe_float(np.std(t)); i += 1
        f[i] = _safe_float(np.std(t) / (np.mean(t) + 1e-10)); i += 1
        b = min(10, max(1, len(t) // 10))
        if b > 1:
            h, _ = np.histogram(t, bins=b)
            f[i] = _safe_float(_entropy(h + 1e-10)); i += 1
        else:
            f[i] = 0.0; i += 1

    # ── 2. 行为轨迹 (10维) ───────────────────────
    if len(df) < 2:
        i += 10
    else:
        ts = df['timestamp']
        ts_s = (ts - ts.min()).dt.total_seconds().values
        iv = np.diff(ts_s)
        if len(iv) == 0:
            i += 10
        else:
            x = np.arange(len(iv))
            f[i] = _safe_float(np.polyfit(x, iv, 1)[0]) if len(iv) >= 2 else 0.0; i += 1
            mn = np.mean(iv)
            f[i] = _safe_float(np.std(iv) / (mn + 1e-10)) if mn > 0 else 0.0; i += 1
            x2 = np.arange(len(ts_s))
            f[i] = _safe_float(np.polyfit(x2, ts_s, 1)[0]) if len(ts_s) >= 2 else 0.0; i += 1
            f[i] = _safe_float(np.mean(iv)); i += 1
            f[i] = _safe_float(np.std(iv)); i += 1
            f[i] = _safe_float(np.min(iv)); i += 1
            f[i] = _safe_float(np.max(iv)); i += 1
            f[i] = _safe_float((ts_s[-1] - ts_s[0]) / (len(ts_s) + 1e-10)); i += 1
            f[i] = _safe_float(np.median(iv)); i += 1
            q75, q25 = np.percentile(iv, [75, 25])
            f[i] = _safe_float(q75 - q25); i += 1

    # ── 3. 情绪复合 (6维) ────────────────────────
    ed = df[df['eventType'] == 'text_insert']
    dl = df[df['eventType'] == 'text_remove']
    fc = df[df['eventType'] == 'focus_gained']

    if len(ed) > 0 and len(dl) > 0:
        ed_ex = ed.groupby('exercise').size()
        dl_ex = dl.groupby('exercise').size()
        ex_set = set(ed_ex.index) | set(dl_ex.index)
        er = [ed_ex.get(ex, 0) / (ed_ex.get(ex, 0) + dl_ex.get(ex, 0) + 1e-10) for ex in ex_set]
        dr = [dl_ex.get(ex, 0) / (ed_ex.get(ex, 0) + dl_ex.get(ex, 0) + 1e-10) for ex in ex_set]
        f[i] = _safe_float(np.mean(er)); i += 1
        f[i] = _safe_float(np.std(er)); i += 1
        f[i] = _safe_float(np.mean(dr)); i += 1
        f[i] = _safe_float(np.std(dr)); i += 1
    else:
        i += 4

    if len(fc) > 0:
        fc_ex = fc.groupby('exercise').size()
        tt_ex = df.groupby('exercise').size()
        fr = [fc_ex.get(ex, 0) / (tt_ex.get(ex, 1) + 1e-10) for ex in tt_ex.index]
        f[i] = _safe_float(np.mean(fr)); i += 1
        f[i] = _safe_float(np.std(fr)); i += 1
    else:
        i += 2

    # ── 4. 元信息 (2维) ─────────────────────────
    f[i] = float(df['exercise'].nunique()) if len(df) > 0 else 0.0; i += 1
    f[i] = float(len(df)); i += 1

    return f


def build_cache():
    LOGS = '/tmp/IDE_logs/IDE_logs.csv'
    PASSED = '/tmp/IDE_logs/passed.csv'
    CHUNK_SIZE = 500_000

    t_start = time.time()

    # 加载标签
    print("Loading labels...")
    passed = pd.read_csv(PASSED)
    passed.columns = ['student', 'passed']
    label_dict = dict(zip(passed['student'], passed['passed']))
    valid_sids = set(label_dict.keys())

    # 第一步: 扫描每块有哪些学生 (用于识别跨块学生)
    print("First pass: identifying student boundaries...")
    chunk_iter = pd.read_csv(LOGS, usecols=['student'], chunksize=CHUNK_SIZE)
    student_last_chunk = {}   # sid -> last chunk index seen
    student_first_chunk = {}  # sid -> first chunk index seen
    n_chunks = 0
    for chunk in chunk_iter:
        for sid in chunk['student'].unique():
            if sid not in student_last_chunk:
                student_first_chunk[sid] = n_chunks
            student_last_chunk[sid] = n_chunks
        n_chunks += 1

    single_chunk_sids = sorted([s for s, last in student_last_chunk.items()
                                 if student_first_chunk[s] == last and s in valid_sids])
    multi_chunk_sids = sorted([s for s, last in student_last_chunk.items()
                                if student_first_chunk[s] != last and s in valid_sids])

    print(f"  {n_chunks} total chunks")
    print(f"  {len(single_chunk_sids)} single-chunk students")
    print(f"  {len(multi_chunk_sids)} multi-chunk students: {multi_chunk_sids[:5]}...")

    # 建立反向索引: chunk_idx -> [sids in that chunk]
    chunk_to_sids = {}
    for sid in single_chunk_sids:
        cidx = student_first_chunk[sid]
        chunk_to_sids.setdefault(cidx, []).append(sid)

    # 已知多块学生的完整学生ID集合
    multi_set = set(multi_chunk_sids)

    # ── 主循环 ─────────────────────────────────
    # 存储 (sid, features, label) 三元组，避免顺序错位
    results_list = []

    # 预分配多块学生缓冲区
    multi_buffer = {sid: [] for sid in multi_chunk_sids}
    multi_flushed = {sid: False for sid in multi_chunk_sids}

    reader = pd.read_csv(LOGS, dtype={
        'student': 'int32', 'part': 'str', 'exercise': 'str',
        'eventType': 'str', 'timestamp': 'str', 'timeToDeadline': 'float32'
    }, chunksize=CHUNK_SIZE)

    for cidx, chunk in enumerate(reader):
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce')

        # ── A. 处理单块学生 ──
        if cidx in chunk_to_sids:
            sids_here = chunk_to_sids[cidx]
            sub = chunk[chunk['student'].isin(sids_here)]
            for sid, grp in sub.groupby('student', sort=False):
                feat = compute_46dim(grp)
                label = 1 if label_dict.get(sid, False) in [True, 'True'] else 0
                results_list.append((sid, feat, label))

        # ── B. 追加到多块学生缓冲区 ──
        multi_rows = chunk[chunk['student'].isin(multi_set)]
        if len(multi_rows) > 0:
            for sid, grp in multi_rows.groupby('student', sort=False):
                multi_buffer[sid].append(grp)

        # ── C. 检查哪些多块学生在本块后变"完整" ──
        for sid in multi_chunk_sids:
            if not multi_flushed[sid] and student_last_chunk.get(sid, -1) == cidx:
                buf = pd.concat(multi_buffer[sid], ignore_index=True) if multi_buffer[sid] else pd.DataFrame()
                if len(buf) > 0:
                    feat = compute_46dim(buf)
                    label = 1 if label_dict.get(sid, False) in [True, 'True'] else 0
                    results_list.append((sid, feat, label))
                multi_flushed[sid] = True

        elapsed = time.time() - t_start
        done_single = sum(1 for s in single_chunk_sids if
                          student_first_chunk.get(s, 9999) <= cidx)
        done_multi = sum(1 for s in multi_chunk_sids if multi_flushed.get(s, False))
        total_done = done_single + done_multi
        if cidx % 10 == 0 or cidx == n_chunks - 1:
            rate = (cidx + 1) * CHUNK_SIZE / elapsed / 1e6
            eta = elapsed * (n_chunks - cidx - 1) / max(cidx + 1, 1)
            print(f"  Chunk {cidx+1}/{n_chunks}: {total_done}/{len(single_chunk_sids)+len(multi_chunk_sids)} "
                  f"students done, {rate:.1f} M rows/s, ETA={eta:.0f}s")

    # 按学生ID排序
    results_list.sort(key=lambda x: x[0])
    sorted_sids = np.array([r[0] for r in results_list], dtype=np.int32)
    X = np.array([r[1] for r in results_list], dtype=np.float32)
    y = np.array([r[2] for r in results_list], dtype=np.int64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    np.savez_compressed(CACHE_PATH, X=X, y=y, student_ids=sorted_sids)

    elapsed = time.time() - t_start
    print(f"\n✅ Done! X={X.shape}, passed={int(y.sum())}, failed={int((y==0).sum())}")
    print(f"   Time: {elapsed:.0f}s, Saved: {CACHE_PATH}")


if __name__ == '__main__':
    build_cache()
