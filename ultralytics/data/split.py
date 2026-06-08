# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import random
import shutil
from pathlib import Path

from ultralytics.data.utils import IMG_FORMATS, img2label_paths
from ultralytics.utils import DATASETS_DIR, LOGGER, TQDM


def split_classify_dataset(source_dir: str | Path, train_ratio: float = 0.8) -> Path:
    """将分类数据集按比例划分为训练集和验证集目录。

    创建新目录 '{source_dir}_split'，包含 train/val 子目录，保留原始类别结构，
    默认按 80/20 比例划分。

    目录结构：
        划分前：
            caltech/
            ├── class1/
            │   ├── img1.jpg
            │   ├── img2.jpg
            │   └── ...
            ├── class2/
            │   ├── img1.jpg
            │   └── ...
            └── ...

        划分后：
            caltech_split/
            ├── train/
            │   ├── class1/
            │   │   ├── img1.jpg
            │   │   └── ...
            │   ├── class2/
            │   │   ├── img1.jpg
            │   │   └── ...
            │   └── ...
            └── val/
                ├── class1/
                │   ├── img2.jpg
                │   └── ...
                ├── class2/
                │   └── ...
                └── ...

    参数：
        source_dir (str | Path)：分类数据集根目录路径。
        train_ratio (float)：训练集比例，取值范围 0 到 1。

    返回：
        (Path)：创建的划分目录路径。

    示例：
        使用默认 80/20 比例划分数据集
        >>> split_classify_dataset("path/to/caltech")

        使用自定义比例划分
        >>> split_classify_dataset("path/to/caltech", 0.75)
    """
    source_path = Path(source_dir)
    split_path = Path(f"{source_path}_split")
    train_path, val_path = split_path / "train", split_path / "val"

    # 创建目录结构
    split_path.mkdir(exist_ok=True)
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)

    # 遍历每个类别目录
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    total_images = sum(len(list(d.glob("*.*"))) for d in class_dirs)
    stats = f"{len(class_dirs)} classes, {total_images} images"
    LOGGER.info(f"Splitting {source_path} ({stats}) into {train_ratio:.0%} train, {1 - train_ratio:.0%} val...")

    for class_dir in class_dirs:
        # 创建类别子目录
        (train_path / class_dir.name).mkdir(exist_ok=True)
        (val_path / class_dir.name).mkdir(exist_ok=True)

        # 随机打乱并复制文件
        image_files = list(class_dir.glob("*.*"))
        random.shuffle(image_files)
        split_idx = int(len(image_files) * train_ratio)

        for img in image_files[:split_idx]:
            shutil.copy2(img, train_path / class_dir.name / img.name)

        for img in image_files[split_idx:]:
            shutil.copy2(img, val_path / class_dir.name / img.name)

    LOGGER.info(f"Split complete in {split_path} ✅")
    return split_path


def autosplit(
    path: Path = DATASETS_DIR / "coco8/images",
    weights: tuple[float, float, float] = (0.9, 0.1, 0.0),
    annotated_only: bool = False,
) -> None:
    """自动将数据集划分为 train/val/test 三个子集，并将划分结果保存到 autosplit_*.txt 文件中。

    参数：
        path (Path)：图片目录路径。
        weights (tuple[float, float, float])：训练集、验证集、测试集的划分比例。
        annotated_only (bool)：为 True 时只使用有关联 txt 标注文件的图片。

    示例：
        使用默认权重划分图片
        >>> from ultralytics.data.split import autosplit
        >>> autosplit()

        使用自定义权重且仅限已标注图片
        >>> autosplit(path="path/to/images", weights=(0.8, 0.15, 0.05), annotated_only=True)
    """
    path = Path(path)  # 图片目录
    files = sorted(x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS)  # 仅筛选图片文件
    n = len(files)  # 文件总数
    random.seed(0)  # 固定随机种子以保证可复现
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # 将每张图片分配到对应划分

    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]  # 3 个输出文件
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # 删除已有文件

    LOGGER.info(f"Autosplitting images from {path}" + ", using *.txt labeled images only" * annotated_only)
    for i, img in TQDM(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # 检查标签文件是否存在
            with open(path.parent / txt[i], "a", encoding="utf-8") as f:
                f.write(f"./{img.relative_to(path.parent).as_posix()}" + "\n")  # 将图片路径写入 txt 文件


if __name__ == "__main__":
    split_classify_dataset("caltech101")
