# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ultralytics.utils import ASSETS_URL, DATASETS_DIR, LOGGER, NUM_THREADS, TQDM, YAML, clean_url
from ultralytics.utils.checks import check_file
from ultralytics.utils.downloads import download, zip_directory
from ultralytics.utils.files import increment_path


def coco91_to_coco80_class() -> list[int]:
    """将 91 索引的 COCO 类别 ID 转换为 80 索引的 COCO 类别 ID。

    返回：
        (list[int | None]): A list of 91 elements where the index represents the 91-index class ID and the value is the
            corresponding 80-index class ID, or None if there is no mapping.
    """
    return [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        None,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        None,
        24,
        25,
        None,
        None,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        None,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        None,
        60,
        None,
        None,
        61,
        None,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        None,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        None,
    ]


def coco80_to_coco91_class() -> list[int]:
    r"""将 80 索引（val2014）转换为 91 索引（论文中）。

    返回：
        (list[int]): A list of 80 class IDs where each value is the corresponding 91-index class ID.

    示例：
        >>> import numpy as np
        >>> a = np.loadtxt("data/coco.names", dtype="str", delimiter="\n")
        >>> b = np.loadtxt("data/coco_paper.names", dtype="str", delimiter="\n")

        Convert the darknet to COCO format
        >>> x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]

        Convert the COCO to darknet format
        >>> x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]

    References:
        https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    """
    return [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]


def convert_coco(
    labels_dir: str = "../coco/annotations/",
    save_dir: str = "coco_converted/",
    use_segments: bool = False,
    use_keypoints: bool = False,
    cls91to80: bool = True,
    lvis: bool = False,
):
    """将 COCO 数据集标注转换为适合训练 YOLO 模型的 YOLO 标注格式。

    参数：
        labels_dir (str, optional): Path to directory containing COCO dataset annotation files.
        save_dir (str, optional): Path to directory to save results to.
        use_segments (bool, optional): Whether to include segmentation masks in the output.
        use_keypoints (bool, optional): Whether to include keypoint annotations in the output.
        cls91to80 (bool, optional): Whether to map 91 COCO class IDs to the corresponding 80 COCO class IDs.
        lvis (bool, optional): Whether to convert data in lvis dataset way.

    示例：
        >>> from ultralytics.data.converter import convert_coco

        Convert COCO annotations to YOLO format
        >>> convert_coco("coco/annotations/", use_segments=True, use_keypoints=False, cls91to80=False)

        Convert LVIS annotations to YOLO format
        >>> convert_coco("lvis/annotations/", use_segments=True, use_keypoints=False, cls91to80=False, lvis=True)
    """
    # 创建数据集目录
    save_dir = increment_path(save_dir)  # increment if save directory already exists
    for p in save_dir / "labels", save_dir / "images":
        p.mkdir(parents=True, exist_ok=True)  # make dir

    # 转换类别
    coco80 = coco91_to_coco80_class()

    # 导入 json
    for json_file in sorted(Path(labels_dir).resolve().glob("*.json")):
        lname = "" if lvis else json_file.stem.replace("instances_", "")
        fn = Path(save_dir) / "labels" / lname  # folder name
        fn.mkdir(parents=True, exist_ok=True)
        if lvis:
            # 注意：提前创建 train 和 val 文件夹，
            # since LVIS val set contains images from COCO 2017 train in addition to the COCO 2017 val split.
            (fn / "train2017").mkdir(parents=True, exist_ok=True)
            (fn / "val2017").mkdir(parents=True, exist_ok=True)
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        # 创建图像字典
        images = {f"{x['id']:d}": x for x in data["images"]}
        # 创建图像-标注字典
        annotations = defaultdict(list)
        for ann in data["annotations"]:
            annotations[ann["image_id"]].append(ann)

        image_txt = []
        # 写入标签 file
        for img_id, anns in TQDM(annotations.items(), desc=f"Annotations {json_file}"):
            img = images[f"{img_id:d}"]
            h, w = img["height"], img["width"]
            f = str(Path(img["coco_url"]).relative_to("http://images.cocodataset.org")) if lvis else img["file_name"]
            if lvis:
                image_txt.append(str(Path("./images") / f))

            bboxes = []
            segments = []
            keypoints = []
            for ann in anns:
                if ann.get("iscrowd", False):
                    continue
                # COCO 边界框格式为 [左上角 x, 左上角 y, 宽度, 高度]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"] - 1  # class
                box = [cls, *box.tolist()]
                if box not in bboxes:
                    if use_keypoints:
                        if ann.get("keypoints") is None:
                            continue
                        keypoints.append(
                            box + (np.array(ann["keypoints"]).reshape(-1, 3) / np.array([w, h, 1])).reshape(-1).tolist()
                        )
                    bboxes.append(box)
                    if use_segments:
                        seg = ann.get("segmentation")
                        if seg is None or len(seg) == 0:
                            segments.append([])
                        elif len(seg) > 1:
                            s = merge_multi_segment(seg)
                            s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                            segments.append([cls, *s])
                        else:
                            s = [j for i in seg for j in i]  # all segments concatenated
                            s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                            segments.append([cls, *s])

            # 写入
            with open((fn / f).with_suffix(".txt"), "a", encoding="utf-8") as file:
                for i in range(len(bboxes)):
                    if use_keypoints:
                        line = (*(keypoints[i]),)  # cls, box, keypoints
                    else:
                        line = (
                            *(segments[i] if use_segments and len(segments[i]) > 0 else bboxes[i]),
                        )  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")

        if lvis:
            filename = Path(save_dir) / json_file.name.replace("lvis_v1_", "").replace(".json", ".txt")
            with open(filename, "a", encoding="utf-8") as f:
                f.writelines(f"{line}\n" for line in image_txt)

    LOGGER.info(f"{'LVIS' if lvis else 'COCO'} data converted successfully.\nResults saved to {save_dir.resolve()}")


def convert_segment_masks_to_yolo_seg(masks_dir: str, output_dir: str, classes: int):
    """将分割掩码图像数据集转换为 YOLO 分割格式。

    This function takes the directory containing the binary format mask images and converts them into YOLO segmentation
    format. The converted masks are saved in the specified output directory.

    参数：
        masks_dir (str): The path to the directory where all mask images (png, jpg) are stored.
        output_dir (str): The path to the directory where the converted YOLO segmentation masks will be stored.
        classes (int): Total number of classes in the dataset, e.g., 80 for COCO.

    示例：
        >>> from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

        The classes here is the total classes in the dataset, for COCO dataset we have 80 classes
        >>> convert_segment_masks_to_yolo_seg("path/to/masks_directory", "path/to/output/directory", classes=80)

    注意：
        The expected directory structure for the masks is:

            - masks
                ├─ mask_image_01.png or mask_image_01.jpg
                ├─ mask_image_02.png or mask_image_02.jpg
                ├─ mask_image_03.png or mask_image_03.jpg
                └─ mask_image_04.png or mask_image_04.jpg

        After execution, the labels will be organized in the following structure:

            - output_dir
                ├─ mask_yolo_01.txt
                ├─ mask_yolo_02.txt
                ├─ mask_yolo_03.txt
                └─ mask_yolo_04.txt
    """
    pixel_to_class_mapping = {i + 1: i for i in range(classes)}
    for mask_path in Path(masks_dir).iterdir():
        if mask_path.suffix in {".png", ".jpg"}:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取掩码图像
            img_height, img_width = mask.shape  # 获取图像尺寸
            LOGGER.info(f"Processing {mask_path} imgsz = {img_height} x {img_width}")

            unique_values = np.unique(mask)  # 获取表示不同类别的唯一像素值
            yolo_format_data = []

            for value in unique_values:
                if value == 0:
                    continue  # 跳过背景
                class_index = pixel_to_class_mapping.get(value, -1)
                if class_index == -1:
                    LOGGER.warning(f"Unknown class for pixel value {value} in file {mask_path}, skipping.")
                    continue

                # 为当前类别创建二值掩码并查找轮廓
                contours, _ = cv2.findContours(
                    (mask == value).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )  # 查找轮廓

                for contour in contours:
                    if len(contour) >= 3:  # YOLO 要求至少 3 个点才能构成有效分割
                        contour = contour.squeeze()  # 删除单维度条目
                        yolo_format = [class_index]
                        for point in contour:
                            # 归一化坐标
                            yolo_format.append(round(point[0] / img_width, 6))  # 四舍五入到 6 位小数
                            yolo_format.append(round(point[1] / img_height, 6))
                        yolo_format_data.append(yolo_format)
            # Save Ultralytics YOLO format data to file
            output_path = Path(output_dir) / f"{mask_path.stem}.txt"
            with open(output_path, "w", encoding="utf-8") as file:
                for item in yolo_format_data:
                    line = " ".join(map(str, item))
                    file.write(line + "\n")
            LOGGER.info(f"Processed and stored at {output_path} imgsz = {img_height} x {img_width}")


def convert_dota_to_yolo_obb(dota_root_path: str):
    """将 DOTA 数据集标注转换为 YOLO OBB（旋转边界框）格式。

    The function processes images in the 'train' and 'val' folders of the DOTA dataset. For each image, it reads the
    associated label from the original labels directory and writes new labels in YOLO OBB format to a new directory.

    参数：
        dota_root_path (str): The root directory path of the DOTA dataset.

    示例：
        >>> from ultralytics.data.converter import convert_dota_to_yolo_obb
        >>> convert_dota_to_yolo_obb("path/to/DOTA")

    注意：
        The directory structure assumed for the DOTA dataset:

            - DOTA
                ├─ images
                │   ├─ train
                │   └─ val
                └─ labels
                    ├─ train_original
                    └─ val_original

        After execution, the function will organize the labels into:

            - DOTA
                └─ labels
                    ├─ train
                    └─ val
    """
    dota_root_path = Path(dota_root_path)

    # 类别名称到索引的映射
    class_mapping = {
        "plane": 0,
        "ship": 1,
        "storage-tank": 2,
        "baseball-diamond": 3,
        "tennis-court": 4,
        "basketball-court": 5,
        "ground-track-field": 6,
        "harbor": 7,
        "bridge": 8,
        "large-vehicle": 9,
        "small-vehicle": 10,
        "helicopter": 11,
        "roundabout": 12,
        "soccer-ball-field": 13,
        "swimming-pool": 14,
        "container-crane": 15,
        "airport": 16,
        "helipad": 17,
    }

    def convert_label(image_name: str, image_width: int, image_height: int, orig_label_dir: Path, save_dir: Path):
        """将单张图像的 DOTA 标注转换为 YOLO OBB 格式并保存到指定目录。"""
        orig_label_path = orig_label_dir / f"{image_name}.txt"
        save_path = save_dir / f"{image_name}.txt"

        with orig_label_path.open("r") as f, save_path.open("w") as g:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                class_name = parts[8]
                class_idx = class_mapping[class_name]
                coords = [float(p) for p in parts[:8]]
                normalized_coords = [
                    coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(8)
                ]
                formatted_coords = [f"{coord:.6g}" for coord in normalized_coords]
                g.write(f"{class_idx} {' '.join(formatted_coords)}\n")

    for phase in {"train", "val"}:
        image_dir = dota_root_path / "images" / phase
        orig_label_dir = dota_root_path / "labels" / f"{phase}_original"
        save_dir = dota_root_path / "labels" / phase

        save_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(image_dir.iterdir())
        for image_path in TQDM(image_paths, desc=f"Processing {phase} images"):
            if image_path.suffix != ".png":
                continue
            image_name_without_ext = image_path.stem
            img = cv2.imread(str(image_path))
            h, w = img.shape[:2]
            convert_label(image_name_without_ext, w, h, orig_label_dir, save_dir)


def min_index(arr1: np.ndarray, arr2: np.ndarray):
    """在两个二维点数组之间找到距离最短的一对索引。

    参数：
        arr1 (np.ndarray): A NumPy array of shape (N, 2) representing N 2D points.
        arr2 (np.ndarray): A NumPy array of shape (M, 2) representing M 2D points.

    返回：
        (tuple[int, int]): A tuple (idx1, idx2) where idx1 is the index in arr1 and idx2 is the index in arr2 of the
            pair with the shortest distance.
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments: list[list]):
    """通过以每个分割之间的最小距离连接坐标，将多个分割合并为一个列表。
    segment.

    This function connects these coordinates with a thin line to merge all segments into one.

    参数：
        segments (list[list]): Original segmentations in COCO's JSON file. Each element is a list of coordinates, like
            [segmentation1, segmentation2,...].

    返回：
        (list[np.ndarray]): A list of connected segments represented as NumPy arrays.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # 记录每个分割之间距离最小的索引
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # 使用两轮来连接所有分割
    for k in range(2):
        # 正向连接
        if k == 0:
            for i, idx in enumerate(idx_list):
                # 中间分割有两个索引，反转中间分割的索引
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # 处理第一个和最后一个分割
                if i in {0, len(idx_list) - 1}:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def yolo_bbox2segment(im_dir: str | Path, save_dir: str | Path | None = None, sam_model: str = "sam_b.pt", device=None):
    """将现有的目标检测数据集（边界框）转换为 YOLO 格式的分割数据集。

    Generates segmentation data using SAM auto-annotator as needed.

    参数：
        im_dir (str | Path): Path to image directory to convert.
        save_dir (str | Path, optional): Path to save the generated labels, labels will be saved into `labels-segment`
            in the same directory level of `im_dir` if save_dir is None.
        sam_model (str): Segmentation model to use for intermediate segmentation data.
        device (int | str, optional): The specific device to run SAM models.

    注意：
        The input directory structure assumed for dataset:

            - im_dir
                ├─ 001.jpg
                ├─ ...
                └─ NNN.jpg
            - labels
                ├─ 001.txt
                ├─ ...
                └─ NNN.txt
    """
    from ultralytics import SAM
    from ultralytics.data import YOLODataset
    from ultralytics.utils.ops import xywh2xyxy

    # 注意：添加占位符以通过类别索引检查
    dataset = YOLODataset(im_dir, data=dict(names=list(range(1000)), channels=3))
    if len(dataset.labels[0]["segments"]) > 0:  # if it's segment data
        LOGGER.info("Segmentation labels detected, no need to generate new ones!")
        return

    LOGGER.info("Detection labels detected, generating segment labels by SAM model!")
    sam_model = SAM(sam_model)
    for label in TQDM(dataset.labels, total=len(dataset.labels), desc="Generating segment labels"):
        h, w = label["shape"]
        boxes = label["bboxes"]
        if len(boxes) == 0:  # skip empty labels
            continue
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        im = cv2.imread(label["im_file"])
        sam_results = sam_model(im, bboxes=xywh2xyxy(boxes), verbose=False, save=False, device=device)
        label["segments"] = sam_results[0].masks.xyn

    save_dir = Path(save_dir) if save_dir else Path(im_dir).parent / "labels-segment"
    save_dir.mkdir(parents=True, exist_ok=True)
    for label in dataset.labels:
        texts = []
        lb_name = Path(label["im_file"]).with_suffix(".txt").name
        txt_file = save_dir / lb_name
        cls = label["cls"]
        for i, s in enumerate(label["segments"]):
            if len(s) == 0:
                continue
            line = (int(cls[i]), *s.reshape(-1))
            texts.append(("%g " * len(line)).rstrip() % line)
        with open(txt_file, "a", encoding="utf-8") as f:
            f.writelines(text + "\n" for text in texts)
    LOGGER.info(f"Generated segment labels saved in {save_dir}")


def create_synthetic_coco_dataset():
    """基于标签列表中的文件名创建带随机图像的合成 COCO 数据集。

    This function downloads COCO labels, reads image filenames from label list files, creates synthetic images for
    train2017 and val2017 subsets, and organizes them in the COCO dataset structure. It uses multithreading to generate
    images efficiently.

    示例：
        >>> from ultralytics.data.converter import create_synthetic_coco_dataset
        >>> create_synthetic_coco_dataset()

    注意：
        - Requires internet connection to download label files.
        - Generates random RGB images of varying sizes (480x480 to 640x640 pixels).
        - Existing test2017 directory is removed as it's not needed.
        - Reads image filenames from train2017.txt and val2017.txt files.
    """

    def create_synthetic_image(image_file: Path):
        """生成具有随机尺寸和颜色的合成图像，用于数据集增强或测试。"""
        if not image_file.exists():
            size = (random.randint(480, 640), random.randint(480, 640))
            Image.new(
                "RGB",
                size=size,
                color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            ).save(image_file)

    # 下载标签
    dir = DATASETS_DIR / "coco"
    download([f"{ASSETS_URL}/coco2017labels-segments.zip"], dir=dir.parent)

    # 创建合成图像
    shutil.rmtree(dir / "labels" / "test2017", ignore_errors=True)  # 移除不需要的 test2017 目录
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for subset in {"train2017", "val2017"}:
            subset_dir = dir / "images" / subset
            subset_dir.mkdir(parents=True, exist_ok=True)

            # 从标签列表文件读取图像文件名
            label_list_file = dir / f"{subset}.txt"
            if label_list_file.exists():
                with open(label_list_file, encoding="utf-8") as f:
                    image_files = [dir / line.strip() for line in f]

                # 提交所有任务
                futures = [executor.submit(create_synthetic_image, image_file) for image_file in image_files]
                for _ in TQDM(as_completed(futures), total=len(futures), desc=f"Generating images for {subset}"):
                    pass  # 实际工作在后台完成
            else:
                LOGGER.warning(f"Labels file {label_list_file} does not exist. Skipping image creation for {subset}.")

    LOGGER.info("Synthetic COCO dataset created successfully.")


def convert_to_multispectral(path: str | Path, n_channels: int = 10, replace: bool = False, zip: bool = False):
    """通过在波长带之间插值将 RGB 图像转换为多光谱图像。

    This function takes RGB images and interpolates them to create multispectral images with a specified number of
    channels. It can process either a single image or a directory of images.

    参数：
        path (str | Path): Path to an image file or directory containing images to convert.
        n_channels (int): Number of spectral channels to generate in the output image.
        replace (bool): Whether to replace the original image file with the converted one.
        zip (bool): Whether to zip the converted images into a zip file.

    示例：
        Convert a single image
        >>> convert_to_multispectral("path/to/image.jpg", n_channels=10)

        Convert a dataset
        >>> convert_to_multispectral("coco8", n_channels=10)
    """
    from scipy.interpolate import interp1d

    from ultralytics.data.utils import IMG_FORMATS

    path = Path(path)
    if path.is_dir():
        # 处理目录
        im_files = [f for ext in (IMG_FORMATS - {"tif", "tiff"}) for f in path.rglob(f"*.{ext}")]
        for im_path in im_files:
            try:
                convert_to_multispectral(im_path, n_channels)
                if replace:
                    im_path.unlink()
            except Exception as e:
                LOGGER.info(f"Error converting {im_path}: {e}")

        if zip:
            zip_directory(path)
    else:
        # 处理单张图像
        output_path = path.with_suffix(".tiff")
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

        # 一次插值所有像素
        rgb_wavelengths = np.array([650, 510, 475])  # R, G, B wavelengths (nm)
        target_wavelengths = np.linspace(450, 700, n_channels)
        f = interp1d(rgb_wavelengths.T, img, kind="linear", bounds_error=False, fill_value="extrapolate")
        multispectral = f(target_wavelengths)
        cv2.imwritemulti(str(output_path), np.clip(multispectral, 0, 255).astype(np.uint8).transpose(2, 0, 1))
        LOGGER.info(f"Converted {output_path}")


def _infer_ndjson_kpt_shape(image_records: list) -> list:
    """从 NDJSON 姿态标注推断 kpt_shape [num_keypoints, dims]。

    Scans up to 50 pose annotations across image records. Annotation format is [classId, cx, cy, w, h, kp1_x, kp1_y,
    kp1_vis, ...] so keypoint values start at index 5.

    Tries dims=3 first (x, y, visibility) with visibility validation ({0, 1, 2}), then falls back to dims=2 (x, y only)
    when values are unambiguously not divisible by 3.
    """
    kpt_lengths = []
    samples = []  # raw keypoint value slices for visibility checking
    for record in image_records:
        for ann in record.get("annotations", {}).get("pose", []):
            kpt_len = len(ann) - 5  # subtract classId + bbox (4 values)
            if kpt_len > 0:
                kpt_lengths.append(kpt_len)
                samples.append(ann[5:])
            if len(kpt_lengths) >= 50:
                break
        if len(kpt_lengths) >= 50:
            break

    if not kpt_lengths or len(set(kpt_lengths)) != 1:
        raise ValueError("Pose dataset missing required 'kpt_shape'. See https://docs.ultralytics.com/datasets/pose/")

    n = kpt_lengths[0]

    # 尝试 dims=3：需要能被 3 整除，且每第 3 个值（可见性）在 {0, 1, 2} 中
    if n % 3 == 0 and all(v in (0, 1, 2) for s in samples for v in s[2::3]):
        return [n // 3, 3]

    # 尝试 dims=2：仅在不能被 3 整除时（避免将 dims=3 的数据误分类）
    if n % 2 == 0 and n % 3 != 0:
        return [n // 2, 2]

    raise ValueError("Pose dataset missing required 'kpt_shape'. See https://docs.ultralytics.com/datasets/pose/")


async def convert_ndjson_to_yolo(ndjson_path: str | Path, output_path: str | Path | None = None) -> Path:
    """将 NDJSON 数据集格式转换为 Ultralytics YOLO 数据集结构。

    This function converts datasets stored in NDJSON (Newline Delimited JSON) format to the standard YOLO format. For
    detection/segmentation/pose/obb tasks, it creates separate directories for images and labels. For classification
    tasks, it creates the ImageNet-style {split}/{class_name}/ folder structure. It supports parallel processing for
    efficient conversion of large datasets and can download images from URLs.

    The NDJSON format consists of:
    - First line: Dataset metadata with class names, task type, and configuration
    - Subsequent lines: Individual image records with annotations and optional URLs

    参数：
        ndjson_path (str | Path): Path to the input NDJSON file containing dataset information.
        output_path (str | Path | None, optional): Directory where the converted YOLO dataset will be saved. If None,
            uses the DATASETS_DIR directory. Defaults to None.

    返回：
        (Path): Path to the generated data.yaml file (detection) or dataset directory (classification).

    示例：
        Convert a local NDJSON file:
        >>> yaml_path = await convert_ndjson_to_yolo("dataset.ndjson")
        >>> print(f"Dataset converted to: {yaml_path}")

        Convert with custom output directory:
        >>> yaml_path = await convert_ndjson_to_yolo("dataset.ndjson", output_path="./converted_datasets")

        Use with YOLO training
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")
        >>> model.train(data="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-ndjson.ndjson")
    """
    from ultralytics.utils.checks import check_requirements

    check_requirements("aiohttp")
    import aiohttp

    ndjson_path = Path(check_file(ndjson_path))
    output_path = Path(output_path or DATASETS_DIR)
    with open(ndjson_path) as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]
    dataset_record, image_records = lines[0], lines[1:]

    # 对稳定内容和源标识进行哈希。查询字符串被排除，因为签名 URL 在每次导出时都会变化。
    _h = hashlib.sha256()
    for r in lines:
        hash_record = {k: v for k, v in r.items() if k != "url"}
        if r.get("file"):
            hash_record["_source"] = clean_url(r["url"]) if r.get("url") else str(ndjson_path.parent.resolve())
        _h.update(json.dumps(hash_record, sort_keys=True).encode())
    _hash = _h.hexdigest()[:8]

    # 带哈希目录允许相同数据集复用下载，同时防止已变化的数据集发生变异
    # files that another training job may still be reading.
    dataset_dir = output_path / f"{ndjson_path.stem}-{_hash}"
    yaml_path = dataset_dir / "data.yaml"
    if yaml_path.is_file():
        try:
            cached = YAML.load(yaml_path)
            if cached.get("hash") == _hash and all(
                (dataset_dir / cached[split]).is_dir() and (dataset_dir / "labels" / split).is_dir()
                for split in ("train", "val", "test")
                if split in cached
            ):
                return yaml_path
        except Exception:
            pass
    splits = {record["split"] for record in image_records}

    # 检查是否为分类数据集
    is_classification = dataset_record.get("task") == "classify"
    class_names = {int(k): v for k, v in dataset_record.get("class_names", {}).items()}
    inferred_nc = None

    # 下载图像前验证必需字段
    task = dataset_record.get("task", "detect")
    if not is_classification:
        class_ids = {
            int(label[0])
            for record in image_records
            for labels in record.get("annotations", {}).values()
            for label in labels
            if label
        }
        if class_ids or class_names:
            max_class_id = max(class_ids | set(class_names))
            if class_names:
                for i in range(max_class_id + 1):
                    class_names.setdefault(i, f"class{i}")
            else:
                inferred_nc = max_class_id + 1
    if not is_classification:
        if "train" not in splits:
            raise ValueError(f"Dataset missing required 'train' split. Found splits: {sorted(splits)}")
        if "val" not in splits:
            train_records = [r for r in image_records if r.get("split") == "train"]
            if len(train_records) < 2:
                raise ValueError(
                    f"Dataset has only {len(train_records)} image(s) and no 'val' split. "
                    f"Need at least 2 images to auto-split into train/val."
                )
            random.Random(0).shuffle(train_records)  # local RNG to avoid mutating global training seed
            val_count = max(1, len(train_records) // 10)
            for r in train_records[:val_count]:
                r["split"] = "val"
            splits.add("val")
            LOGGER.warning(
                f"WARNING ⚠️ No 'val' split found in dataset. "
                f"Auto-splitting {len(train_records)} images into {len(train_records) - val_count} train, {val_count} val. "
                f"For best results, manually assign validation images in Platform dataset page."
            )
    if task == "pose" and "kpt_shape" not in dataset_record:
        dataset_record["kpt_shape"] = _infer_ndjson_kpt_shape(image_records)

    # 检查数据集是否已存在（允许在划分变更时复用图像）
    _reuse = dataset_dir.exists()
    if _reuse:
        yaml_path.unlink(missing_ok=True)  # 在破坏性操作前使哈希失效（崩溃安全）
        if not is_classification:
            shutil.rmtree(dataset_dir / "labels", ignore_errors=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = None

    if not is_classification:
        # Detection/segmentation/pose/obb: prepare YAML and create base structure
        data_yaml = dict(dataset_record)
        if class_names:
            data_yaml["names"] = class_names
        elif inferred_nc is not None:
            data_yaml["nc"] = inferred_nc
        data_yaml.pop("class_names", None)
        data_yaml.pop("type", None)  # Remove NDJSON-specific fields
        for split in sorted(splits):
            (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
            data_yaml[split] = f"images/{split}"

    async def process_record(session, semaphore, record):
        """使用异步会话处理单张图像记录。"""
        async with semaphore:
            split, original_name = record["split"], record["file"]
            annotations = record.get("annotations", {})

            if is_classification:
                # 分类：将图像放在 {split}/{class_name}/ 文件夹中
                class_ids = annotations.get("classification", [])
                class_id = class_ids[0] if class_ids else 0
                class_name = class_names.get(class_id, str(class_id))
                image_path = dataset_dir / split / class_name / original_name
            else:
                # 检测：写入标签文件并将图像放在 images/{split}/ 中
                image_path = dataset_dir / "images" / split / original_name
                label_path = dataset_dir / "labels" / split / f"{Path(original_name).stem}.txt"
                lines_to_write = []
                for key in annotations:
                    lines_to_write = [" ".join(map(str, item)) for item in annotations[key]]
                    break
                label_path.write_text("\n".join(lines_to_write) + "\n" if lines_to_write else "")

            # 从其他划分目录复用已有图像（避免重新划分时重复下载）或下载
            if not image_path.exists():
                if _reuse:
                    for s in ("train", "val", "test"):
                        if s == split:
                            continue
                        candidate = (
                            (dataset_dir / s / class_name / original_name)
                            if is_classification
                            else (dataset_dir / "images" / s / original_name)
                        )
                        if candidate.exists():
                            image_path.parent.mkdir(parents=True, exist_ok=True)
                            candidate.rename(image_path)
                            break
                if not image_path.exists() and (http_url := record.get("url")):
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    # 带指数退避地重试（3 次尝试：最终尝试前延迟 1 秒、2 秒）
                    for attempt in range(3):
                        error = None
                        try:
                            async with session.get(http_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                                response.raise_for_status()
                                image_path.write_bytes(await response.read())
                            return True
                        except aiohttp.ClientResponseError as e:
                            error = e
                            if e.status not in {408, 429} and e.status < 500:
                                LOGGER.warning(f"Failed to download {http_url}: {e}")
                                return False
                        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                            error = e
                        except Exception as e:  # OSError、磁盘满、权限错误 — 非暂时性，不重试
                            LOGGER.warning(f"Failed to save {http_url}: {e}")
                            return False
                        if attempt < 2:  # 最后一次尝试后不睡眠
                            await asyncio.sleep(2**attempt)  # 1s, 2s backoff
                        else:
                            LOGGER.warning(f"Failed to download {http_url} after 3 attempts: {error}")
                            return False
            return True

    # 使用异步下载处理所有图像（小数据集限制连接数）
    semaphore = asyncio.Semaphore(min(128, len(image_records)))
    async with aiohttp.ClientSession() as session:
        pbar = TQDM(
            total=len(image_records),
            desc=f"Converting {ndjson_path.name} → {dataset_dir} ({len(image_records)} images)",
        )

        async def tracked_process(record):
            result = await process_record(session, semaphore, record)
            pbar.update(1)
            return result

        results = await asyncio.gather(*[tracked_process(record) for record in image_records])
        pbar.close()

    # 验证图像是否成功下载
    success_count = sum(1 for r in results if r)
    if success_count == 0:
        raise RuntimeError(f"Failed to download any images from {ndjson_path}. Check network connection and URLs.")
    if success_count < len(image_records):
        LOGGER.warning(f"Downloaded {success_count}/{len(image_records)} images from {ndjson_path}")

    # 移除数据集中不再存在的孤立图像（防止训练中出现过时的背景图像）
    if _reuse:
        expected_paths = set()
        for r in image_records:
            s, name = r["split"], r["file"]
            if is_classification:
                ann = r.get("annotations", {})
                cids = ann.get("classification", [])
                cid = cids[0] if cids else 0
                expected_paths.add(dataset_dir / s / class_names.get(cid, str(cid)) / name)
            else:
                expected_paths.add(dataset_dir / "images" / s / name)
        img_root = dataset_dir if is_classification else (dataset_dir / "images")
        for p in img_root.rglob("*"):
            if p.is_file() and p not in expected_paths:
                p.unlink()

    if is_classification:
        # 分类：返回数据集目录（check_cls_dataset 需要目录路径）
        return dataset_dir
    else:
        # Detection: write data.yaml with hash for future change detection
        data_yaml["hash"] = _hash
        YAML.save(yaml_path, data_yaml)
        return yaml_path
