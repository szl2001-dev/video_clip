"""
为 ASR 数据计算 segment 级别的视频-文本相似度分数（使用 VideoCLIP-XL）。

输入: filtered_clean parquet 文件（含 mms_url、asr_segments 字段）
输出: 同结构，新增以下列：
  - seg_count      : 该条视频 asr_segments 的总段数
  - seg_pass_count : 相似度 >= THRESHOLD 的段数
  - seg_pass_ratio : seg_pass_count / seg_count（无段时为 NaN）
  - seg_scores     : 每段的相似度分数列表（下载/推理失败时为 None）

多 GPU：每个 GPU 子进程独立处理分配到的文件。
修改顶部 CONFIG 后运行: python run_clip_score_asr.py
"""

import os
import sys
import glob
import math
import uuid
import tempfile
import traceback
import multiprocessing as mp
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import time

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import torch

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
"""
  单文件，输出到目录：
  INPUT_DIR  = "/path/to/filtered_clean/part_0/00000.parquet"
  OUTPUT_DIR = "/path/to/filtered_clip_scored"
  # → 输出到 filtered_clip_scored/00000.parquet

  单文件，指定输出文件名：
  INPUT_DIR  = "/path/to/filtered_clean/part_0/00000.parquet"
  OUTPUT_DIR = "/path/to/my_output.parquet"
  # → 输出到 my_output.parquet

  目录（原有行为不变）：
  INPUT_DIR  = "/path/to/filtered_clean"
  OUTPUT_DIR = "/path/to/filtered_clip_scored"
  #
"""
INPUT_DIR   = "/home/work/mllm_datas/zhengliang/asr/output/filtered_clean/pass/news/part_00.parquet"
OUTPUT_DIR  = "/home/work/mllm_datas/zhengliang/video_clip_for_asr/output/test"

MODEL_PATH  = "/home/work/mllm_datas/zhengliang/model/VideoCLIP-XL/VideoCLIP-XL-v2.bin"
MODEL_BASE  = "/home/work/mllm_datas/zhengliang/model/VideoCLIP-XL"   # dir with modeling.py

# 相似度阈值：>= THRESHOLD 视为"通过"
THRESHOLD   = 0.2
# 使用哪些 GPU（留空 [] 则单卡 GPU 0）
GPU_IDS     = [0, 1, 2, 3, 4, 5, 6, 7]
# 每个 segment 采样的帧数
FNUM        = 8
# True: 跳过已存在的输出文件；False: 重新处理
RESUME      = False
# 调试模式：每个文件只处理前 N 行（0 = 关闭，处理全部）
DEBUG_ROWS  = 100
# 下载超时（秒）
DOWNLOAD_TIMEOUT = 30
# 模型推理 batch size（segment 数量）
INFER_BATCH_SIZE = 32
# 生产者线程数：并行下载+提取帧的线程数（每个 GPU 进程内）
# 不宜过大：下载比GPU推理快，过多线程会导致帧数据在内存积压
DL_WORKERS = 8
# 临时文件目录
TMP_DIR     = "/home/work/tmp"

# --------------------------------------------------------------------------- #


# ============================================================================ #
#  视频下载 & 帧提取
# ============================================================================ #

def download_video(url: str, tmp_dir: str, timeout: int = DOWNLOAD_TIMEOUT) -> str:
    """把视频下载到临时 mp4 文件，返回本地路径。失败则抛异常。"""
    os.makedirs(tmp_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".mp4", dir=tmp_dir)
    try:
        resp = requests.get(url, timeout=(5, timeout), stream=True)
        resp.raise_for_status()
        with os.fdopen(fd, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return path
    except Exception:
        # 关闭 fd 并删除不完整文件
        try:
            os.close(fd)
        except Exception:
            pass
        if os.path.exists(path):
            os.remove(path)
        raise


def safe_remove(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def get_segment_frame_indices(start_sec: float, end_sec: float,
                              fps: float, total_frames: int, fnum: int) -> list:
    """
    计算某个时间段 [start_sec, end_sec] 内均匀采样 fnum 帧的帧索引列表。
    """
    start_frame = max(0, int(start_sec * fps))
    end_frame   = min(total_frames - 1, max(0, int(end_sec * fps)))

    if start_frame > end_frame:
        end_frame = start_frame

    n = end_frame - start_frame + 1
    if n <= fnum:
        indices = list(range(start_frame, end_frame + 1))
    else:
        step = (n - 1) / (fnum - 1) if fnum > 1 else float(n)
        indices = [int(start_frame + i * step) for i in range(fnum)]
        # 去重并保序
        seen, dedup = set(), []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                dedup.append(idx)
        indices = dedup

    return indices


def extract_segment_frames(video_path: str, segments: list, fnum: int = 8):
    """
    从本地视频文件中为每个 segment 提取帧。

    segments : list of dict {start: float, end: float, text: str}
    Returns  : list of list[np.ndarray]  —— 每段一个 RGB 帧列表
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 30.0

    if stream.duration and stream.time_base:
        total_sec = float(stream.duration * stream.time_base)
    else:
        total_sec = max((seg.get("end", 0) for seg in segments), default=30.0)

    total_frames = max(1, int(total_sec * fps))

    # 为每个 segment 计算所需帧索引
    seg_indices_list = []
    all_needed = set()
    for seg in segments:
        indices = get_segment_frame_indices(
            seg.get("start", 0.0), seg.get("end", 0.0), fps, total_frames, fnum
        )
        seg_indices_list.append(indices)
        all_needed.update(indices)

    if not all_needed:
        container.close()
        return [[] for _ in segments]

    all_sorted = sorted(all_needed)
    max_needed = all_sorted[-1]
    index_to_frame = {}
    current_idx = 0

    try:
        for frame in container.decode(video=0):
            if current_idx in all_needed:
                # PyAV decode returns RGB24 frames
                img = frame.to_ndarray(format="rgb24")
                # 立即 resize 到模型输入尺寸，避免原始高分辨率帧积压内存
                if img.shape[0] != 224 or img.shape[1] != 224:
                    img = np.array(
                        av.VideoFrame.from_ndarray(img, format="rgb24")
                           .reformat(width=224, height=224)
                           .to_ndarray(format="rgb24"),
                        dtype=np.uint8,
                    )
                index_to_frame[current_idx] = img
            current_idx += 1
            if current_idx > max_needed:
                break
    finally:
        container.close()

    # 为每段组装帧
    result = []
    for indices in seg_indices_list:
        frames = [index_to_frame[i] for i in indices if i in index_to_frame]
        if not frames and index_to_frame:
            # 如果没有命中（时间戳超出视频时长），用最后一帧兜底
            frames = [list(index_to_frame.values())[-1]]
        result.append(frames)

    return result


# ============================================================================ #
#  生产者：下载视频 + 提取帧（在线程中运行，不涉及 GPU）
# ============================================================================ #

def download_and_extract(row_idx: int, row, fnum: int):
    """
    生产者任务：下载视频 → 提取所有 segment 的帧。
    返回 (row_idx, segs, frames_per_seg)
      - segs          : segment dict 列表
      - frames_per_seg: 与 segs 等长，每项是帧列表（失败时为 None）
    """
    segs_raw = row.get("asr_segments")
    if segs_raw is None:
        return row_idx, [], []

    segs = segs_raw.tolist() if hasattr(segs_raw, "tolist") else list(segs_raw)
    segs = [s for s in segs if isinstance(s, dict)]
    if not segs:
        return row_idx, [], []

    url = row.get("mms_url", "") or ""
    if not url:
        return row_idx, segs, [None] * len(segs)

    video_path = None
    try:
        video_path = download_video(url, TMP_DIR)
        frames_per_seg = extract_segment_frames(video_path, segs, fnum)
    except Exception:
        frames_per_seg = [None] * len(segs)
    finally:
        safe_remove(video_path)

    return row_idx, segs, frames_per_seg


# ============================================================================ #
#  单文件处理（生产者-消费者）
# ============================================================================ #

def process_file(src: str, dst: str, model, fnum: int, threshold: float,
                 row_start: int = 0, row_end: int = None,
                 progress_queue=None, row_indices=None):
    """
    生产者（DL_WORKERS 个线程）：并行下载视频、提取帧
    消费者（GPU 主线程）：积攒到 INFER_BATCH_SIZE 个 segment 后批量推理

    row_indices         : 指定行索引列表（交错分配模式，优先于 row_start/row_end）
    row_start / row_end : 连续行分片范围（row_indices 为 None 时使用）
    progress_queue      : mp.Queue，每完成一行下载 put ('dl',1)，推理 put ('inf',1)
    """
    df = pd.read_parquet(src)
    if row_indices is not None:
        df = df.iloc[row_indices].reset_index(drop=True)
    elif row_end is not None:
        df = df.iloc[row_start:row_end].reset_index(drop=True)
    elif DEBUG_ROWS > 0:
        df = df.head(DEBUG_ROWS)
    n = len(df)

    # row_idx -> scores 列表（长度 = 该行 seg 数）
    row_scores: dict[int, list] = {}
    row_segs:   dict[int, list] = {}

    # 待推理的 segment 批次
    batch_frames: list = []
    batch_texts:  list = []
    batch_meta:   list = []   # (row_idx, seg_idx)

    row_pending: dict = {}  # row_idx -> remaining segment count not yet inferred

    def flush():
        """把已积攒的 segment batch 送 GPU 推理，结果写回 row_scores。"""
        if not batch_frames:
            return
        n_segs = len(batch_frames)
        scores = model.compute_scores(
            batch_frames, batch_texts, fnum=fnum, batch_size=INFER_BATCH_SIZE
        )
        for (r_idx, s_idx), score in zip(batch_meta, scores):
            row_scores[r_idx][s_idx] = score
            row_pending[r_idx] -= 1
            if row_pending[r_idx] <= 0 and progress_queue is not None:
                try:
                    progress_queue.put_nowait(('inf', 1))
                except Exception:
                    pass
        if progress_queue is not None:
            try:
                progress_queue.put_nowait(('seg', n_segs))
            except Exception:
                pass
        batch_frames.clear()
        batch_texts.clear()
        batch_meta.clear()

    rows = [(i, row) for i, (_, row) in enumerate(df.iterrows())]

    with ThreadPoolExecutor(max_workers=DL_WORKERS) as executor:
        # 一次性提交所有下载任务；线程池按 DL_WORKERS 并发执行
        futures = {
            executor.submit(download_and_extract, i, row, fnum): i
            for i, row in rows
        }

        # 哪个下载先完成就先处理，最大化 GPU 利用率
        for future in as_completed(futures):
            try:
                row_idx, segs, frames_per_seg = future.result()
            except Exception as e:
                row_idx = futures[future]
                segs, frames_per_seg = [], []

            if progress_queue is not None:
                try:
                    progress_queue.put_nowait(('dl', 1))
                except Exception:
                    pass

            seg_count = len(segs)
            row_segs[row_idx]   = segs
            row_scores[row_idx] = [None] * seg_count
            row_pending[row_idx] = seg_count

            if seg_count == 0:
                # 没有 segment，无需推理，直接标记完成
                if progress_queue is not None:
                    try:
                        progress_queue.put_nowait(('inf', 1))
                    except Exception:
                        pass
            else:
                # 把该视频的所有 segment 加入当前批次
                for seg_idx, (frames, seg) in enumerate(zip(frames_per_seg, segs)):
                    batch_frames.append(frames)
                    batch_texts.append(seg.get("text", "") or "")
                    batch_meta.append((row_idx, seg_idx))

                # 攒够一批就推理
                if len(batch_frames) >= INFER_BATCH_SIZE:
                    flush()

    # 推理剩余不足一批的 segment
    flush()

    # 按原始行顺序组装输出列
    seg_counts, seg_pass_counts, seg_pass_ratios, seg_scores_col = [], [], [], []
    for i in range(n):
        scores    = row_scores.get(i, [])
        seg_count = len(scores)
        pass_count = sum(1 for s in scores if s is not None and s >= threshold)
        ratio = pass_count / seg_count if seg_count > 0 else float("nan")
        seg_counts.append(seg_count)
        seg_pass_counts.append(pass_count)
        seg_pass_ratios.append(ratio)
        seg_scores_col.append(scores)

    df["seg_count"]      = seg_counts
    df["seg_pass_count"] = seg_pass_counts
    df["seg_pass_ratio"] = seg_pass_ratios
    df["seg_scores"]     = seg_scores_col

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    df.to_parquet(dst, compression="zstd", index=False)
    print(f"  Saved: {dst}  ({n} rows)")


# ============================================================================ #
#  GPU Worker
# ============================================================================ #

def worker(rank: int, gpu_id: int, src: str, dst: str,
           row_start: int, row_end: int, progress_queue=None, row_indices=None):
    """
    子进程：在 gpu_id 上处理 src 文件的 [row_start, row_end) 行，结果写入 dst。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"

    # 每个 GPU 子进程写独立日志，避免被 \r 进度条覆盖
    log_path = os.path.join(TMP_DIR, f"gpu{gpu_id}.log")
    os.makedirs(TMP_DIR, exist_ok=True)
    log_f = open(log_path, "w", buffering=1)

    def log(msg):
        line = f"[GPU {gpu_id}] {msg}"
        print(line, flush=True)
        print(line, file=log_f, flush=True)

    log(f"start  rows=[{row_start}, {row_end})  log={log_path}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from video_clip_model import VideoCLIPFeatureExtractor

    try:
        model = VideoCLIPFeatureExtractor(
            model_path=MODEL_PATH,
            model_base_dir=MODEL_BASE,
            device=device,
        )
        log("model loaded")
    except Exception as e:
        log(f"Model load failed: {e}")
        traceback.print_exc(file=log_f)
        log_f.close()
        return

    try:
        process_file(src, dst, model, FNUM, THRESHOLD, row_start, row_end, progress_queue, row_indices)
        log(f"done  saved={dst}")
    except Exception as e:
        log(f"Failed: {e}")
        traceback.print_exc(file=log_f)
    finally:
        log_f.close()


# ============================================================================ #
#  多 GPU 处理单个文件（行分片 + 合并）
# ============================================================================ #

def process_file_multi_gpu(src: str, dst: str, gpu_ids: list):
    """
    把一个 parquet 文件按行均分给所有 GPU 并行处理，结果合并后写入 dst。
    无论文件数多少，所有 GPU 都能被充分利用。
    """
    # 确定实际要处理的行数
    n_total = pq.read_metadata(src).num_rows
    n_rows  = min(n_total, DEBUG_ROWS) if DEBUG_ROWS > 0 else n_total

    n_gpus  = min(len(gpu_ids), n_rows)
    run_id  = uuid.uuid4().hex[:8]

    # 交错分配：GPU k 取第 k, k+n_gpus, k+2*n_gpus, ... 行
    # 避免慢行集中在某个连续区间导致负载不均
    all_indices = list(range(n_rows))
    gpu_row_indices = [all_indices[rank::n_gpus] for rank in range(n_gpus)]

    # (gpu_id, row_start, row_end, tmp_dst, row_indices)
    chunks = []
    for rank in range(n_gpus):
        tmp_dst = os.path.join(TMP_DIR, f"vc_{run_id}_{rank}.parquet")
        chunks.append((gpu_ids[rank], 0, None, tmp_dst, gpu_row_indices[rank]))

    progress_q = mp.Queue()

    # 并行启动各 GPU 子进程
    processes = []
    for rank, (gpu_id, start, end, tmp_dst, row_idx_list) in enumerate(chunks):
        p = mp.Process(target=worker, args=(rank, gpu_id, src, tmp_dst, start, end, progress_q, row_idx_list))
        p.start()
        processes.append(p)

    # 主进程监控线程：汇总各 GPU 进度，打印统一进度行
    dl_done  = [0]
    inf_done = [0]
    seg_done = [0]
    t0       = time.time()
    stop_ev  = threading.Event()

    def _drain(q):
        while True:
            try:
                tag, val = q.get_nowait()
                if tag == 'dl':
                    dl_done[0] += val
                elif tag == 'inf':
                    inf_done[0] += val
                elif tag == 'seg':
                    seg_done[0] += val
            except Exception:
                break

    def _print_progress(newline=False):
        elapsed = time.time() - t0
        row_rate = inf_done[0] / elapsed if elapsed > 0 else 0.0
        seg_rate = seg_done[0] / elapsed if elapsed > 0 else 0.0
        line = (
            f"\r  dl={dl_done[0]}/{n_rows}  inf={inf_done[0]}/{n_rows}"
            f"  {row_rate:.1f} rows/s  {seg_rate:.1f} segs/s   "
        )
        print(line, end="\n" if newline else "", flush=True)

    def _monitor():
        while not stop_ev.is_set():
            try:
                tag, val = progress_q.get(timeout=0.3)
                if tag == 'dl':
                    dl_done[0] += val
                elif tag == 'inf':
                    inf_done[0] += val
                elif tag == 'seg':
                    seg_done[0] += val
            except Exception:
                pass
            _print_progress()
        _drain(progress_q)
        _print_progress(newline=True)

    mon = threading.Thread(target=_monitor, daemon=True)
    mon.start()

    for p in processes:
        p.join()

    stop_ev.set()
    mon.join()

    # 打印各 GPU 进程退出状态，便于排查异常退出
    for rank, (p, (gpu_id, _, _, _, row_idx_list)) in enumerate(zip(processes, chunks)):
        code = p.exitcode
        status = "ok" if code == 0 else f"ERROR(exitcode={code})"
        print(f"  [GPU {gpu_id}] rows={len(row_idx_list)}  exitcode={code}  {status}", flush=True)

    # 按行顺序流式合并各分片，避免全量加载进内存
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    writer = None
    total_rows = 0
    for _, _, _, tmp_dst, _ in chunks:
        if not os.path.exists(tmp_dst):
            continue
        table = pq.read_table(tmp_dst)
        if writer is None:
            writer = pq.ParquetWriter(dst, table.schema, compression="zstd")
        writer.write_table(table)
        total_rows += len(table)
    if writer:
        writer.close()
    else:
        print(f"[Warning] No output produced for {src}")
        return
    print(f"Saved: {dst}  ({total_rows} rows)")

    for _, _, _, tmp_dst, _ in chunks:
        safe_remove(tmp_dst)


# ============================================================================ #
#  File collection
# ============================================================================ #

def collect_pairs(input_path: str, output_path: str):
    """
    收集 (src, dst) 文件对。
    - input_path 为单个 .parquet 文件时：output_path 若带 .parquet 后缀则视为目标文件路径，
      否则视为目录，目标文件名与源文件名相同。
    - input_path 为目录时：递归收集所有 .parquet，保留相对路径结构到 output_path 目录。
    """
    if os.path.isfile(input_path):
        if output_path.endswith(".parquet"):
            dst = output_path
        else:
            dst = os.path.join(output_path, os.path.basename(input_path))
        return [(input_path, dst)]

    srcs = sorted(
        Path(root) / f
        for root, _, files in os.walk(input_path)
        for f in files
        if f.endswith(".parquet")
    )
    pairs = []
    for src in srcs:
        rel = src.relative_to(input_path)
        dst = Path(output_path) / rel
        pairs.append((str(src), str(dst)))
    return pairs


# ============================================================================ #
#  Main
# ============================================================================ #

def main():
    os.makedirs(TMP_DIR, exist_ok=True)

    all_pairs = collect_pairs(INPUT_DIR, OUTPUT_DIR)
    if RESUME:
        todo = [(s, d) for s, d in all_pairs if not os.path.exists(d)]
    else:
        todo = all_pairs

    print(f"Input : {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Total files: {len(all_pairs)}, todo: {len(todo)}")

    if not todo:
        print("All files already processed.")
        return

    gpu_ids = GPU_IDS if GPU_IDS else [0]

    # 逐文件处理，每个文件均使用全部 GPU 做行分片并行推理
    for i, (src, dst) in enumerate(todo):
        print(f"\n[{i+1}/{len(todo)}] {src}")
        process_file_multi_gpu(src, dst, gpu_ids)

    print("\nAll done.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
