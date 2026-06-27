"""greenaug调参/抠像核对/增强验证三个工具的共享底座: 读LeRobot V2 parquet数据集 + 图像编解码/标注.

tune/matte_check/aug_check都要从数据集列episode, 按相机列名取帧, PNG解码, 给帧标号, 这些与
抠像本身无关的样板集中在这里, 三个入口各自只留独有逻辑 (tune的网页交互, matte_check的红蒙版
对照图, aug_check的换背景验证). 相机列名约定见greenaug.IMAGE_COLUMN_PREFIX.
"""

import glob
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from openpi.training import greenaug


def list_episodes(dataset_root):
    """列数据集所有episode parquet路径 (按文件名排序)."""
    paths = sorted(glob.glob(str(Path(dataset_root) / "data" / "chunk-*" / "*.parquet")))
    return [Path(p) for p in paths]


def available_cameras(episode_path, candidates):
    """从episode的schema里筛出candidates中真实存在的相机列名 (保持candidates顺序)."""
    names = pq.ParquetFile(episode_path).schema_arrow.names
    return [c for c in candidates if greenaug.IMAGE_COLUMN_PREFIX + c in names]


def count_frames(episode_path):
    """episode的帧数 (parquet行数)."""
    return pq.ParquetFile(episode_path).metadata.num_rows


def read_camera_pngs(episode_path, camera):
    """读某相机列的全部帧, 返回内联PNG条目列表 (每项含 ["bytes"])."""
    col = greenaug.IMAGE_COLUMN_PREFIX + camera
    return pq.read_table(episode_path, columns=[col]).column(0).to_pylist()


def decode_rgb(png_bytes):
    """内联PNG字节 -> uint8 RGB (H, W, 3)."""
    arr = np.frombuffer(png_bytes, np.uint8)
    return cv2.cvtColor(cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def frame_indices(num_rows, count):
    """沿时间轴等距取count帧的帧号 (可复现); 不足count则全取."""
    if num_rows <= count:
        return list(range(num_rows))
    return [round(x) for x in np.linspace(0, num_rows - 1, count)]


def label(image, text):
    """左上角标注文字 (黄字), 供contact sheet标帧号."""
    out = image.copy()
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    return out
