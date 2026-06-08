# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import itertools
from glob import glob
from math import ceil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ultralytics.data.utils import exif_size, img2label_paths
from ultralytics.utils import TQDM
from ultralytics.utils.checks import check_requirements


def bbox_iof(polygon1: np.ndarray, bbox2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """计算多边形与边界框之间的前景交并比（IoF）。

    参数：
        polygon1 (np.ndarray)：多边形坐标，形状为 (N, 8)。
        bbox2 (np.ndarray)：边界框坐标，形状为 (M, 4)。
        eps (float, 可选)：防止除零的小值。

    返回：
        (np.ndarray)：IoF 分数，形状为 (N, M)。

    说明：
        多边形格式：[x1, y1, x2, y2, x3, y3, x4, y4]。
        边界框格式：[x_min, y_min, x_max, y_max]。
    """
    check_requirements("shapely>=2.0.0")
    from shapely.geometry import Polygon

    polygon1 = polygon1.reshape(-1, 4, 2)
    lt_point = np.min(polygon1, axis=-2)  # 左上角点
    rb_point = np.max(polygon1, axis=-2)  # 右下角点
    bbox1 = np.concatenate([lt_point, rb_point], axis=-1)

    lt = np.maximum(bbox1[:, None, :2], bbox2[..., :2])
    rb = np.minimum(bbox1[:, None, 2:], bbox2[..., 2:])
    wh = np.clip(rb - lt, 0, np.inf)
    h_overlaps = wh[..., 0] * wh[..., 1]

    left, top, right, bottom = (bbox2[..., i] for i in range(4))
    polygon2 = np.stack([left, top, right, top, right, bottom, left, bottom], axis=-1).reshape(-1, 4, 2)

    sg_polys1 = [Polygon(p) for p in polygon1]
    sg_polys2 = [Polygon(p) for p in polygon2]
    overlaps = np.zeros(h_overlaps.shape)
    for p in zip(*np.nonzero(h_overlaps)):
        overlaps[p] = sg_polys1[p[0]].intersection(sg_polys2[p[-1]]).area
    unions = np.array([p.area for p in sg_polys1], dtype=np.float32)
    unions = unions[..., None]

    unions = np.clip(unions, eps, np.inf)
    outputs = overlaps / unions
    if outputs.ndim == 1:
        outputs = outputs[..., None]
    return outputs


def load_yolo_dota(data_root: str, split: str = "train") -> list[dict[str, Any]]:
    """加载 DOTA 数据集的标注和图片信息。

    参数：
        data_root (str)：数据根目录路径。
        split (str, 可选)：数据集划分，可选 'train' 或 'val'。

    返回：
        (list[dict[str, Any]])：标注字典列表，包含图片信息。

    说明：
        DOTA 数据集的目录结构假设为：
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    assert split in {"train", "val"}, f"Split must be 'train' or 'val', not {split}."
    im_dir = Path(data_root) / "images" / split
    assert im_dir.exists(), f"Can't find {im_dir}, please check your data root."
    im_files = glob(str(Path(data_root) / "images" / split / "*"))
    lb_files = img2label_paths(im_files)
    annos = []
    for im_file, lb_file in zip(im_files, lb_files):
        w, h = exif_size(Image.open(im_file))
        with open(lb_file, encoding="utf-8") as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            lb = np.array(lb, dtype=np.float32)
        annos.append(dict(ori_size=(h, w), label=lb, filepath=im_file))
    return annos


def get_windows(
    im_size: tuple[int, int],
    crop_sizes: tuple[int, ...] = (1024,),
    gaps: tuple[int, ...] = (200,),
    im_rate_thr: float = 0.6,
    eps: float = 0.01,
) -> np.ndarray:
    """获取滑动窗口的坐标，用于图片裁剪。

    参数：
        im_size (tuple[int, int])：原始图片尺寸 (H, W)。
        crop_sizes (tuple[int, ...], 可选)：裁剪窗口尺寸。
        gaps (tuple[int, ...], 可选)：相邻裁剪窗口之间的间隔。
        im_rate_thr (float, 可选)：窗口内有效图片面积占比的阈值。
        eps (float, 可选)：数学运算的 epsilon 值。

    返回：
        (np.ndarray)：窗口坐标数组，形状 (N, 4)，每行为 [x_start, y_start, x_stop, y_stop]。
    """
    h, w = im_size
    windows = []
    for crop_size, gap in zip(crop_sizes, gaps):
        assert crop_size > gap, f"invalid crop_size gap pair [{crop_size} {gap}]"
        step = crop_size - gap

        xn = 1 if w <= crop_size else ceil((w - crop_size) / step + 1)
        xs = [step * i for i in range(xn)]
        if len(xs) > 1 and xs[-1] + crop_size > w:
            xs[-1] = w - crop_size

        yn = 1 if h <= crop_size else ceil((h - crop_size) / step + 1)
        ys = [step * i for i in range(yn)]
        if len(ys) > 1 and ys[-1] + crop_size > h:
            ys[-1] = h - crop_size

        start = np.array(list(itertools.product(xs, ys)), dtype=np.int64)
        stop = start + crop_size
        windows.append(np.concatenate([start, stop], axis=1))
    windows = np.concatenate(windows, axis=0)

    im_in_wins = windows.copy()
    im_in_wins[:, 0::2] = np.clip(im_in_wins[:, 0::2], 0, w)
    im_in_wins[:, 1::2] = np.clip(im_in_wins[:, 1::2], 0, h)
    im_areas = (im_in_wins[:, 2] - im_in_wins[:, 0]) * (im_in_wins[:, 3] - im_in_wins[:, 1])
    win_areas = (windows[:, 2] - windows[:, 0]) * (windows[:, 3] - windows[:, 1])
    im_rates = im_areas / win_areas
    if not (im_rates > im_rate_thr).any():
        max_rate = im_rates.max()
        im_rates[abs(im_rates - max_rate) < eps] = 1
    return windows[im_rates > im_rate_thr]


def get_window_obj(anno: dict[str, Any], windows: np.ndarray, iof_thr: float = 0.7) -> list[np.ndarray]:
    """根据 IoF 阈值获取每个窗口内的目标标注。"""
    h, w = anno["ori_size"]
    label = anno["label"]
    if len(label):
        label[:, 1::2] *= w
        label[:, 2::2] *= h
        iofs = bbox_iof(label[:, 1:], windows)
        # 未归一化且坐标未对齐
        return [(label[iofs[:, i] >= iof_thr]) for i in range(len(windows))]  # window_anns
    else:
        return [np.zeros((0, 9), dtype=np.float32) for _ in range(len(windows))]  # window_anns


def crop_and_save(
    anno: dict[str, Any],
    windows: np.ndarray,
    window_objs: list[np.ndarray],
    im_dir: str,
    lb_dir: str,
    allow_background_images: bool = True,
) -> None:
    """裁剪图片并为每个窗口保存新的标注文件。

    参数：
        anno (dict[str, Any])：标注字典，键包括 'filepath'、'label'、'ori_size'。
        windows (np.ndarray)：窗口坐标数组，形状 (N, 4)。
        window_objs (list[np.ndarray])：每个窗口内的标注列表。
        im_dir (str)：裁剪图片的输出目录路径。
        lb_dir (str)：标注文件的输出目录路径。
        allow_background_images (bool, 可选)：是否保留无标注的背景图片。

    说明：
        DOTA 数据集的目录结构假设为：
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    im = cv2.imread(anno["filepath"])
    name = Path(anno["filepath"]).stem
    for i, window in enumerate(windows):
        x_start, y_start, x_stop, y_stop = window.tolist()
        new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"
        patch_im = im[y_start:y_stop, x_start:x_stop]
        ph, pw = patch_im.shape[:2]

        label = window_objs[i]
        if len(label) or allow_background_images:
            cv2.imwrite(str(Path(im_dir) / f"{new_name}.jpg"), patch_im)
        if len(label):
            label[:, 1::2] -= x_start
            label[:, 2::2] -= y_start
            label[:, 1::2] /= pw
            label[:, 2::2] /= ph

            with open(Path(lb_dir) / f"{new_name}.txt", "w", encoding="utf-8") as f:
                for lb in label:
                    formatted_coords = [f"{coord:.6g}" for coord in lb[1:]]
                    f.write(f"{int(lb[0])} {' '.join(formatted_coords)}\n")


def split_images_and_labels(
    data_root: str,
    save_dir: str,
    split: str = "train",
    crop_sizes: tuple[int, ...] = (1024,),
    gaps: tuple[int, ...] = (200,),
) -> None:
    """对指定数据集划分的图片和标注同时进行切分。

    参数：
        data_root (str)：数据集根目录路径。
        save_dir (str)：切分后数据集的保存目录。
        split (str, 可选)：数据集划分，可选 'train' 或 'val'。
        crop_sizes (tuple[int, ...], 可选)：裁剪尺寸元组。
        gaps (tuple[int, ...], 可选)：裁剪间隔元组。

    说明：
        DOTA 数据集的目录结构假设为：
            - data_root
                - images
                    - split
                - labels
                    - split
        输出目录结构为：
            - save_dir
                - images
                    - split
                - labels
                    - split
    """
    im_dir = Path(save_dir) / "images" / split
    im_dir.mkdir(parents=True, exist_ok=True)
    lb_dir = Path(save_dir) / "labels" / split
    lb_dir.mkdir(parents=True, exist_ok=True)

    annos = load_yolo_dota(data_root, split=split)
    for anno in TQDM(annos, total=len(annos), desc=split):
        windows = get_windows(anno["ori_size"], crop_sizes, gaps)
        window_objs = get_window_obj(anno, windows)
        crop_and_save(anno, windows, window_objs, str(im_dir), str(lb_dir))


def split_trainval(
    data_root: str, save_dir: str, crop_size: int = 1024, gap: int = 200, rates: tuple[float, ...] = (1.0,)
) -> None:
    """使用多种缩放比例切分 DOTA 数据集的训练集和验证集。

    参数：
        data_root (str)：数据集根目录路径。
        save_dir (str)：切分后数据集的保存目录。
        crop_size (int, 可选)：基础裁剪尺寸。
        gap (int, 可选)：基础裁剪间隔。
        rates (tuple[float, ...], 可选)：crop_size 和 gap 的缩放比例。

    说明：
        DOTA 数据集的目录结构假设为：
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
        输出目录结构为：
            - save_dir
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    crop_sizes, gaps = [], []
    for r in rates:
        crop_sizes.append(int(crop_size / r))
        gaps.append(int(gap / r))
    for split in {"train", "val"}:
        split_images_and_labels(data_root, save_dir, split, crop_sizes, gaps)


def split_test(
    data_root: str, save_dir: str, crop_size: int = 1024, gap: int = 200, rates: tuple[float, ...] = (1.0,)
) -> None:
    """切分 DOTA 数据集的测试集，测试集不包含标注文件。

    参数：
        data_root (str)：数据集根目录路径。
        save_dir (str)：切分后数据集的保存目录。
        crop_size (int, 可选)：基础裁剪尺寸。
        gap (int, 可选)：基础裁剪间隔。
        rates (tuple[float, ...], 可选)：crop_size 和 gap 的缩放比例。

    说明：
        DOTA 数据集的目录结构假设为：
            - data_root
                - images
                    - test
        输出目录结构为：
            - save_dir
                - images
                    - test
    """
    crop_sizes, gaps = [], []
    for r in rates:
        crop_sizes.append(int(crop_size / r))
        gaps.append(int(gap / r))
    save_dir = Path(save_dir) / "images" / "test"
    save_dir.mkdir(parents=True, exist_ok=True)

    im_dir = Path(data_root) / "images" / "test"
    assert im_dir.exists(), f"Can't find {im_dir}, please check your data root."
    im_files = glob(str(im_dir / "*"))
    for im_file in TQDM(im_files, total=len(im_files), desc="test"):
        w, h = exif_size(Image.open(im_file))
        windows = get_windows((h, w), crop_sizes=crop_sizes, gaps=gaps)
        im = cv2.imread(im_file)
        name = Path(im_file).stem
        for window in windows:
            x_start, y_start, x_stop, y_stop = window.tolist()
            new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"
            patch_im = im[y_start:y_stop, x_start:x_stop]
            cv2.imwrite(str(save_dir / f"{new_name}.jpg"), patch_im)


if __name__ == "__main__":
    split_trainval(data_root="DOTAv2", save_dir="DOTAv2-split")
    split_test(data_root="DOTAv2", save_dir="DOTAv2-split")
