# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from ultralytics.data.utils import polygons2masks, polygons2masks_overlap
from ultralytics.utils import LOGGER, IterableSimpleNamespace, colorstr
from ultralytics.utils.checks import check_version
from ultralytics.utils.instance import Instances
from ultralytics.utils.metrics import bbox_ioa
from ultralytics.utils.ops import segment2box, xywh2xyxy, xyxyxyxy2xywhr
from ultralytics.utils.torch_utils import TORCHVISION_0_10, TORCHVISION_0_11, TORCHVISION_0_13

DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)


class BaseTransform:
    """Ultralytics 库中图像变换的基类。

    此类为对图像、目标实例和语义分割掩码应用变换提供统一接口。
    子类应重写 `apply_image`、`apply_instances` 和/或 `apply_semantic` 来实现简单变换，
    或直接重写 `__call__` 来实现需要在图像和标注修改之间共享状态的复杂变换。

    方法：
        get_params：计算图像、实例和语义掩码之间共享的变换参数。
        apply_image：对 labels['img'] 中的图像应用变换。
        apply_instances：对 labels['instances'] 中的目标实例应用变换。
        apply_semantic：对 labels['semantic_mask'] 中的语义掩码应用变换。
        __call__：编排变换流水线。
    """

    def __call__(self, labels):
        """对 labels 字典应用变换。

        参数：
            labels (dict)：包含 'img' 以及可选的 'instances' 和 'semantic_mask' 的字典。

        返回：
            (dict)：变换后的 labels 字典。
        """
        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        labels = self.apply_instances(labels, params)
        labels = self.apply_semantic(labels, params)
        return labels

    def get_params(self, labels):
        """计算并返回变换参数。

        此方法允许在图像、实例和语义掩码变换之间共享随机状态或计算矩阵（如仿射矩阵、翻转决策）。

        参数：
            labels (dict)：输入 labels 字典。

        返回：
            (dict)：传递给 apply_image、apply_instances 和 apply_semantic 的参数。
        """
        return {}

    def apply_image(self, labels, params=None):
        """对图像应用变换。

        参数：
            labels (dict)：包含 'img' 的字典。
            params (dict | None)：来自 get_params 的参数。

        返回：
            (dict)：更新后的 labels 字典。
        """
        return labels

    def apply_instances(self, labels, params=None):
        """对目标实例应用变换。

        参数：
            labels (dict)：包含 'instances' 的字典。
            params (dict | None)：来自 get_params 的参数。

        返回：
            (dict)：更新后的 labels 字典。
        """
        return labels

    def apply_semantic(self, labels, params=None):
        """对语义分割掩码应用变换。

        参数：
            labels (dict)：包含 'semantic_mask' 的字典。
            params (dict | None)：来自 get_params 的参数。

        返回：
            (dict)：更新后的 labels 字典。
        """
        return labels


class Compose:
    """用于组合多个图像变换的类。

    属性：
        transforms (list[Callable])：按顺序应用的变换函数列表。

    方法：
        __call__：对输入数据应用一系列变换。
        append：在现有变换列表末尾追加一个新变换。
        insert：在变换列表的指定索引处插入一个新变换。
        __getitem__：使用索引获取一个或一组变换。
        __setitem__：使用索引设置一个或一组变换。
        tolist：将变换列表转换为标准 Python 列表。

    示例：
        >>> transforms = [RandomFlip(), RandomPerspective(30)]
        >>> compose = Compose(transforms)
        >>> transformed_data = compose(data)
        >>> compose.append(CenterCrop((224, 224)))
        >>> compose.insert(0, RandomFlip())
    """

    def __init__(self, transforms):
        """使用变换列表初始化 Compose 对象。

        参数：
            transforms (list[Callable])：按顺序应用的可调用变换对象列表。
        """
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        """对输入数据应用一系列变换。

        此方法按顺序将 Compose 对象中的每个变换应用于输入数据。

        参数：
            data (Any)：待变换的输入数据，类型取决于列表中的变换。

        返回：
            (Any)：按顺序应用所有变换后的数据。

        示例：
            >>> transforms = [Transform1(), Transform2(), Transform3()]
            >>> compose = Compose(transforms)
            >>> transformed_data = compose(input_data)
        """
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """在现有变换列表末尾追加一个新变换。

        参数：
            transform (BaseTransform)：要添加到组合中的变换。

        示例：
            >>> compose = Compose([RandomFlip(), RandomPerspective()])
            >>> compose.append(RandomHSV())
        """
        self.transforms.append(transform)

    def insert(self, index, transform):
        """在变换列表的指定索引处插入一个新变换。

        参数：
            index (int)：插入新变换的索引位置。
            transform (BaseTransform)：要插入的变换对象。

        示例：
            >>> compose = Compose([Transform1(), Transform2()])
            >>> compose.insert(1, Transform3())
            >>> len(compose.transforms)
            3
        """
        self.transforms.insert(index, transform)

    def __getitem__(self, index: list | int) -> Compose:
        """使用索引获取一个或一组变换。

        参数：
            index (int | list[int])：要获取的变换的索引或索引列表。

        返回：
            (Compose | Any)：如果 index 是列表则返回新 Compose 对象，如果是整数则返回单个变换。

        异常：
            AssertionError：如果 index 类型不是 int 或 list。

        示例：
            >>> transforms = [RandomFlip(), RandomPerspective(10), RandomHSV(0.5, 0.5, 0.5)]
            >>> compose = Compose(transforms)
            >>> single_transform = compose[1]  # 直接返回 RandomPerspective 变换
            >>> multiple_transforms = compose[[0, 1]]  # 返回包含 RandomFlip 和 RandomPerspective 的 Compose 对象
        """
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        return Compose([self.transforms[i] for i in index]) if isinstance(index, list) else self.transforms[index]

    def __setitem__(self, index: list | int, value: list | int) -> None:
        """使用索引设置组合中的一个或多个变换。

        参数：
            index (int | list[int])：要设置变换的索引或索引列表。
            value (Any | list[Any])：在指定索引处设置的变换或变换列表。

        异常：
            AssertionError：如果索引类型无效、值类型与索引类型不匹配或索引超出范围。

        示例：
            >>> compose = Compose([Transform1(), Transform2(), Transform3()])
            >>> compose[1] = NewTransform()  # 替换第二个变换
            >>> compose[[0, 1]] = [NewTransform1(), NewTransform2()]  # 替换前两个变换
        """
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        if isinstance(index, list):
            assert isinstance(value, list), (
                f"The indices should be the same type as values, but got {type(index)} and {type(value)}"
            )
        if isinstance(index, int):
            index, value = [index], [value]
        for i, v in zip(index, value):
            assert i < len(self.transforms), f"list index {i} out of range {len(self.transforms)}."
            self.transforms[i] = v

    def tolist(self):
        """将变换列表转换为标准 Python 列表。

        返回：
            (list)：包含 Compose 实例中所有变换对象的列表。

        示例：
            >>> transforms = [RandomFlip(), RandomPerspective(10), CenterCrop()]
            >>> compose = Compose(transforms)
            >>> transform_list = compose.tolist()
            >>> print(len(transform_list))
            3
        """
        return self.transforms

    def __repr__(self):
        """返回 Compose 对象的字符串表示。

        返回：
            (str)：Compose 对象的字符串表示，包含变换列表。

        示例：
            >>> transforms = [RandomFlip(), RandomPerspective(degrees=10, translate=0.1, scale=0.1)]
            >>> compose = Compose(transforms)
            >>> print(compose)
            Compose([
                RandomFlip(),
                RandomPerspective(degrees=10, translate=0.1, scale=0.1)
            ])
        """
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"


class BaseMixTransform(BaseTransform):
    """Cutmix、MixUp 和 Mosaic 等混合变换的基类。

    此类为在数据集上实现混合变换提供基础，处理基于概率的变换应用并管理多图片和多标签的混合。

    属性：
        dataset (Any)：包含图片和标签的数据集对象。
        pre_transform (Callable | None)：混合前应用的可选变换。
        p (float)：应用混合变换的概率。

    方法：
        __call__：对输入 labels 应用混合变换。
        get_params：准备混合标签并更新文本标签。
        get_indexes：抽象方法，获取待混合图片的索引。
        _update_label_text：更新混合图片的标签文本。

    示例：
        >>> class CustomMixTransform(BaseMixTransform):
        ...     def apply_image(self, labels, params=None):
        ...         # 在此处实现自定义图片混合
        ...         return labels
        ...
        ...     def get_indexes(self):
        ...         return [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
        >>> dataset = YourDataset()
        >>> transform = CustomMixTransform(dataset, p=0.5)
        >>> mixed_labels = transform(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """初始化用于 CutMix、MixUp 和 Mosaic 等混合变换的 BaseMixTransform 对象。

        此类作为图像处理流水线中实现混合变换的基类。

        参数：
            dataset (Any)：包含用于混合的图片和标签的数据集对象。
            pre_transform (Callable | None)：混合前应用的可选变换。
            p (float)：应用混合变换的概率，取值范围 [0.0, 1.0]。
        """
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """对 labels 数据应用预处理变换和 cutmix/mixup/mosaic 变换。

        此方法基于概率因子决定是否应用混合变换。如果应用，则选择额外的图片，
        如指定则应用预处理变换，然后执行混合变换。

        参数：
            labels (dict[str, Any])：包含图片标签数据的字典。

        返回：
            (dict[str, Any])：变换后的 labels 字典，可能包含来自其他图片的混合数据。

        示例：
            >>> transform = BaseMixTransform(dataset, pre_transform=None, p=0.5)
            >>> result = transform({"image": img, "bboxes": boxes, "cls": classes})
        """
        if random.uniform(0, 1) > self.p:
            return labels

        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        labels = self.apply_instances(labels, params)
        labels = self.apply_semantic(labels, params)
        labels.pop("mix_labels", None)
        return labels

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """准备混合标签并更新文本标签。

        参数：
            labels (dict[str, Any])：包含图片标签数据的字典。

        返回：
            (dict[str, Any])：供 apply_image、apply_instances 和 apply_semantic 使用的参数。
        """
        # 获取另外 1 或 3 张图片的索引
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # 获取将用于 Mosaic、CutMix 或 MixUp 的图片信息
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels

        # 更新类别和文本
        self._update_label_text(labels)
        return {"mix_labels": mix_labels}

    def get_indexes(self):
        """获取用于 Mosaic 增强的随机索引。

        返回：
            (int)：来自数据集的随机索引。

        示例：
            >>> transform = BaseMixTransform(dataset)
            >>> index = transform.get_indexes()
            >>> print(index)  # 7
        """
        return random.randint(0, len(self.dataset) - 1)

    @staticmethod
    def _update_label_text(labels: dict[str, Any]) -> dict[str, Any]:
        """更新图像增强中混合标签的标签文本和类别 ID。

        This method processes the 'texts' and 'cls' fields of the input labels dictionary and any mixed labels, creating
        a unified set of text labels and updating class IDs accordingly.

        参数：
            labels (dict[str, Any]): A dictionary containing label information, including 'texts' and 'cls' fields, and
                optionally a 'mix_labels' field with additional label dictionaries.

        返回：
            (dict[str, Any]): The updated labels dictionary with unified text labels and updated class IDs.

        示例：
            >>> labels = {
            ...     "texts": [["cat"], ["dog"]],
            ...     "cls": torch.tensor([[0], [1]]),
            ...     "mix_labels": [{"texts": [["bird"], ["fish"]], "cls": torch.tensor([[0], [1]])}],
            ... }
            >>> updated_labels = BaseMixTransform._update_label_text(labels)
            >>> print(updated_labels["texts"])
            [['cat'], ['dog'], ['bird'], ['fish']]
            >>> print(updated_labels["cls"])
            tensor([[0],
                    [1]])
            >>> print(updated_labels["mix_labels"][0]["cls"])
            tensor([[2],
                    [3]])
        """
        if "texts" not in labels:
            return labels

        mix_texts = [*labels["texts"], *(item for x in labels["mix_labels"] for item in x["texts"])]
        mix_texts = list({tuple(x) for x in mix_texts})
        text2id = {text: i for i, text in enumerate(mix_texts)}

        for label in [labels] + labels["mix_labels"]:
            for i, cls in enumerate(label["cls"].squeeze(-1).tolist()):
                text = label["texts"][int(cls)]
                label["cls"][i] = text2id[tuple(text)]
            label["texts"] = mix_texts
        return labels


class Mosaic(BaseMixTransform):
    """图像数据集的 Mosaic 增强。

    This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image. The
    augmentation is applied to a dataset with a given probability.

    属性：
        dataset: The dataset on which the mosaic augmentation is applied.
        imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
        p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
        n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).
        border (tuple[int, int]): Border size for height and width.

    方法：
        get_indexes: Return a list of random indexes from the dataset.
        get_params: Compute mosaic layout parameters.
        apply_image: Allocate canvas and paste images into mosaic.
        apply_instances: Concatenate and clip instances for mosaic.
        _update_labels: Update labels with padding.
        _cat_labels: Concatenate labels and clips mosaic border instances.

    示例：
        >>> from ultralytics.data.augment import Mosaic
        >>> dataset = YourDataset(...)  # 你的图像数据集
        >>> mosaic_aug = Mosaic(dataset, imgsz=640, p=0.5, n=4)
        >>> augmented_labels = mosaic_aug(original_labels)
    """

    def __init__(self, dataset, imgsz: int = 640, p: float = 1.0, n: int = 4):
        """初始化 Mosaic 增强对象。

        This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image. The
        augmentation is applied to a dataset with a given probability.

        参数：
            dataset (Any): The dataset on which the mosaic augmentation is applied.
            imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
            p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
            n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).
        """
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        assert n in {4, 9}, "grid must be equal to 4 or 9."
        super().__init__(dataset=dataset, p=p)
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height
        self.n = n
        self.buffer_enabled = self.dataset.cache != "ram"

    def get_indexes(self):
        """返回数据集中用于 Mosaic 增强的随机索引列表。

        This method selects random image indexes either from a buffer or from the entire dataset, depending on the
        'buffer_enabled' attribute. It is used to choose images for creating mosaic augmentations.

        返回：
            (list[int]): A list of random image indexes. The length of the list is n-1, where n is the number of images
                used in the mosaic (either 3 or 8, depending on whether n is 4 or 9).

        示例：
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> indexes = mosaic.get_indexes()
            >>> print(len(indexes))  # 输出: 3
        """
        if self.buffer_enabled:  # select images from buffer
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:  # select any images
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """计算 Mosaic 布局参数。

        参数：
            labels (dict[str, Any]): Input labels dictionary.

        返回：
            (dict[str, Any]): Parameters including 'layout' with per-patch geometry.
        """
        params = super().get_params(labels)
        assert labels.get("rect_shape") is None, "rect and mosaic are mutually exclusive."
        assert len(labels.get("mix_labels", [])), "There are no other images for mosaic augment."

        s = self.imgsz
        layout = []
        if self.n == 4:
            yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)
            for i in range(4):
                labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
                img = labels_patch["img"]
                h, w = labels_patch.get("resized_shape", img.shape[:2])
                if i == 0:  # top left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                padw = x1a - x1b
                padh = y1a - y1b
                layout.append(
                    {
                        "labels_patch": labels_patch,
                        "x1a": x1a,
                        "y1a": y1a,
                        "x2a": x2a,
                        "y2a": y2a,
                        "x1b": x1b,
                        "y1b": y1b,
                        "x2b": x2b,
                        "y2b": y2b,
                        "padw": padw,
                        "padh": padh,
                        "img_shape": (h, w),
                    }
                )
        elif self.n == 9:
            hp, wp = -1, -1
            h0, w0 = None, None
            for i in range(9):
                labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
                img = labels_patch["img"]
                h, w = labels_patch.get("resized_shape", img.shape[:2])
                if i == 0:  # center
                    c = s, s, s + w, s + h
                    h0, w0 = h, w
                elif i == 1:  # top
                    c = s, s - h, s + w, s
                elif i == 2:  # top right
                    c = s + wp, s - h, s + wp + w, s
                elif i == 3:  # right
                    c = s + w0, s, s + w0 + w, s + h
                elif i == 4:  # bottom right
                    c = s + w0, s + hp, s + w0 + w, s + hp + h
                elif i == 5:  # bottom
                    c = s + w0 - w, s + h0, s + w0, s + h0 + h
                elif i == 6:  # bottom left
                    c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
                elif i == 7:  # left
                    c = s - w, s + h0 - h, s, s + h0
                elif i == 8:  # top left
                    c = s - w, s + h0 - hp - h, s, s + h0 - hp
                padw, padh = c[:2]
                x1, y1, x2, y2 = (max(x, 0) for x in c)
                layout.append(
                    {
                        "labels_patch": labels_patch,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "padw": padw,
                        "padh": padh,
                        "img_shape": (h, w),
                    }
                )
                hp, wp = h, w
        params["layout"] = layout
        return params

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对图像应用 Mosaic 增强。

        参数：
            labels (dict[str, Any]): Dictionary containing 'img'.
            params (dict | None): Parameters from get_params, including 'layout'.

        返回：
            (dict): Updated labels with mosaic image.
        """
        s = self.imgsz
        layout = params["layout"]
        if self.n == 4:
            img4 = np.full((s * 2, s * 2, labels["img"].shape[2]), 114, dtype=np.uint8)
            for item in layout:
                labels_patch = item["labels_patch"]
                img = labels_patch["img"]
                x1a, y1a, x2a, y2a = item["x1a"], item["y1a"], item["x2a"], item["y2a"]
                x1b, y1b, x2b, y2b = item["x1b"], item["y1b"], item["x2b"], item["y2b"]
                img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            labels["img"] = img4
        elif self.n == 9:
            img9 = np.full((s * 3, s * 3, labels["img"].shape[2]), 114, dtype=np.uint8)
            for item in layout:
                labels_patch = item["labels_patch"]
                img = labels_patch["img"]
                x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
                padw, padh = item["padw"], item["padh"]
                img9[y1:y2, x1:x2] = img[y1 - padh :, x1 - padw :]
            labels["img"] = img9[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对实例应用 Mosaic 增强。

        参数：
            labels (dict[str, Any]): Dictionary containing 'instances' and 'cls'.
            params (dict | None): Parameters from get_params, including 'layout'.

        返回：
            (dict): Updated labels with concatenated instances.
        """
        layout = params["layout"]
        mosaic_labels = []
        for item in layout:
            if self.n == 4:
                padw = item["padw"]
                padh = item["padh"]
            else:  # n == 9
                padw = item["padw"] + self.border[0]
                padh = item["padh"] + self.border[1]
            labels_patch = self._update_labels(item["labels_patch"], padw, padh, item.get("img_shape"))
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        labels.update(final_labels)
        return labels

    @staticmethod
    def _update_labels(labels, padw: int, padh: int, img_shape: tuple[int, int] | None = None) -> dict[str, Any]:
        """使用填充值更新标签坐标。

        This method adjusts the bounding box coordinates of object instances in the labels by adding padding
        values. It also denormalizes the coordinates if they were previously normalized.

        参数：
            labels (dict[str, Any]): A dictionary containing image and instance information.
            padw (int): Padding width to be added to the x-coordinates.
            padh (int): Padding height to be added to the y-coordinates.
            img_shape (tuple[int, int] | None): Optional (h, w) of the original patch image. Needed because apply_image
                may overwrite labels["img"] with the mosaic canvas before apply_instances runs.

        返回：
            (dict): Updated labels dictionary with adjusted instance coordinates.

        示例：
            >>> labels = {"img": np.zeros((100, 100, 3)), "instances": Instances(...)}
            >>> padw, padh = 50, 50
            >>> updated_labels = Mosaic._update_labels(labels, padw, padh)
        """
        nh, nw = img_shape if img_shape is not None else labels["img"].shape[:2]
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(nw, nh)
        labels["instances"].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels: list[dict[str, Any]]) -> dict[str, Any]:
        """拼接并处理 Mosaic 增强的标签。

        This method combines labels from multiple images used in mosaic augmentation, clips instances to the mosaic
        border, and removes zero-area boxes.

        参数：
            mosaic_labels (list[dict[str, Any]]): A list of label dictionaries for each image in the mosaic.

        返回：
            (dict[str, Any]): A dictionary containing concatenated and processed labels for the mosaic image, including:
                - im_file (str): File path of the first image in the mosaic.
                - ori_shape (tuple[int, int]): Original shape of the first image.
                - resized_shape (tuple[int, int]): Shape of the mosaic image (imgsz * 2, imgsz * 2).
                - cls (np.ndarray): Concatenated class labels.
                - instances (Instances): Concatenated instance annotations.
                - mosaic_border (tuple[int, int]): Mosaic border size.
                - texts (list[str], optional): Text labels if present in the original labels.

        示例：
            >>> mosaic = Mosaic(dataset, imgsz=640)
            >>> mosaic_labels = [{"cls": np.array([0, 1]), "instances": Instances(...)} for _ in range(4)]
            >>> result = mosaic._cat_labels(mosaic_labels)
            >>> print(result.keys())
            dict_keys(['im_file', 'ori_shape', 'resized_shape', 'cls', 'instances', 'mosaic_border'])
        """
        if not mosaic_labels:
            return {}
        cls = []
        instances = []
        imgsz = self.imgsz * 2  # mosaic imgsz
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])
        # 最终的标签
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
            "mosaic_border": self.border,
        }
        final_labels["instances"].clip(imgsz, imgsz)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][good]
        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        return final_labels


class MixUp(BaseMixTransform):
    """对图像数据集应用 MixUp 增强。

    This class implements the MixUp augmentation technique as described in the paper [mixup: Beyond Empirical Risk
    Minimization](https://arxiv.org/abs/1710.09412). MixUp combines two images and their labels using a random weight.

    属性：
        dataset (Any): The dataset to which MixUp augmentation will be applied.
        pre_transform (Callable | None): Optional transform to apply before MixUp.
        p (float): Probability of applying MixUp augmentation.

    方法：
        get_params: Compute MixUp parameters including blend ratio.
        apply_image: Blend images using MixUp.
        apply_instances: Concatenate instances for MixUp.

    示例：
        >>> from ultralytics.data.augment import MixUp
        >>> dataset = YourDataset(...)  # 你的图像数据集
        >>> mixup = MixUp(dataset, p=0.5)
        >>> augmented_labels = mixup(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p: float = 0.0) -> None:
        """初始化 MixUp 增强对象。

        MixUp is an image augmentation technique that combines two images by taking a weighted sum of their pixel values
        and labels. This implementation is designed for use with the Ultralytics YOLO framework.

        参数：
            dataset (Any): The dataset to which MixUp augmentation will be applied.
            pre_transform (Callable | None): Optional transform to apply to images before MixUp.
            p (float): Probability of applying MixUp augmentation to an image. Must be in the range [0, 1].
        """
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """计算 MixUp 参数。

        参数：
            labels (dict[str, Any]): Input labels dictionary.

        返回：
            (dict[str, Any]): Parameters including mix ratio 'r'.
        """
        params = super().get_params(labels)
        params["r"] = np.random.beta(32.0, 32.0)
        return params

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """使用 MixUp 混合图片。

        参数：
            labels (dict[str, Any]): Dictionary containing 'img'.
            params (dict | None): Parameters from get_params, including 'r'.

        返回：
            (dict): Updated labels with blended image.
        """
        r = params["r"]
        labels2 = labels["mix_labels"][0]
        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(np.uint8)
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """拼接 MixUp 的实例。

        参数：
            labels (dict[str, Any]): Dictionary containing 'instances' and 'cls'.
            params (dict | None): Parameters from get_params.

        返回：
            (dict): Updated labels with concatenated instances.
        """
        labels2 = labels["mix_labels"][0]
        labels["instances"] = Instances.concatenate([labels["instances"], labels2["instances"]], axis=0)
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"]], 0)
        return labels


class CutMix(BaseMixTransform):
    """对图像数据集应用 CutMix 增强，如论文 https://arxiv.org/abs/1905.04899 所述。

    CutMix combines two images by replacing a random rectangular region of one image with the corresponding region from
    another image, and adjusts the labels proportionally to the area of the mixed region.

    属性：
        dataset (Any): The dataset to which CutMix augmentation will be applied.
        pre_transform (Callable | None): Optional transform to apply before CutMix.
        p (float): Probability of applying CutMix augmentation.
        beta (float): Beta distribution parameter for sampling the mixing ratio.
        num_areas (int): Number of areas to try to cut and mix.

    方法：
        get_params: Compute CutMix parameters including cut area and filtered indexes.
        apply_image: Copy patch from secondary image into primary image.
        apply_instances: Clip and concatenate instances for CutMix.
        _rand_bbox: Generate random bounding box coordinates for the cut region.

    示例：
        >>> from ultralytics.data.augment import CutMix
        >>> dataset = YourDataset(...)  # 你的图像数据集
        >>> cutmix = CutMix(dataset, p=0.5)
        >>> augmented_labels = cutmix(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p: float = 0.0, beta: float = 1.0, num_areas: int = 3) -> None:
        """初始化 CutMix 增强对象。

        参数：
            dataset (Any): The dataset to which CutMix augmentation will be applied.
            pre_transform (Callable | None): Optional transform to apply before CutMix.
            p (float): Probability of applying CutMix augmentation.
            beta (float): Beta distribution parameter for sampling the mixing ratio.
            num_areas (int): Number of areas to try to cut and mix.
        """
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        self.beta = beta
        self.num_areas = num_areas

    def _rand_bbox(self, width: int, height: int) -> tuple[int, int, int, int]:
        """生成用于剪切区域的随机边界框坐标。

        参数：
            width (int): Width of the image.
            height (int): Height of the image.

        返回：
            (tuple[int]): (x1, y1, x2, y2) coordinates of the bounding box.
        """
        # 从 Beta 分布中采样混合比例
        lam = np.random.beta(self.beta, self.beta)

        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)

        # 随机中心点
        cx = np.random.randint(width)
        cy = np.random.randint(height)

        # 边界框坐标
        x1 = np.clip(cx - cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y2 = np.clip(cy + cut_h // 2, 0, height)

        return x1, y1, x2, y2

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """计算 CutMix 参数。

        参数：
            labels (dict[str, Any]): Input labels dictionary.

        返回：
            (dict[str, Any]): Parameters including 'skip', 'area', and 'indexes2'.
        """
        params = super().get_params(labels)
        h, w = labels["img"].shape[:2]

        cut_areas = np.asarray([self._rand_bbox(w, h) for _ in range(self.num_areas)], dtype=np.float32)
        ioa1 = bbox_ioa(cut_areas, labels["instances"].bboxes)  # (self.num_areas, num_boxes)
        idx = np.nonzero(ioa1.sum(axis=1) <= 0)[0]
        if len(idx) == 0:
            params["skip"] = True
            return params

        labels2 = labels["mix_labels"][0]
        area = cut_areas[np.random.choice(idx)]  # randomly select one
        ioa2 = bbox_ioa(area[None], labels2["instances"].bboxes).squeeze(0)
        indexes2 = np.nonzero(ioa2 >= (0.01 if len(labels["instances"].segments) else 0.1))[0]
        if len(indexes2) == 0:
            params["skip"] = True
            return params

        params["area"] = area
        params["indexes2"] = indexes2
        params["w"] = w
        params["h"] = h
        return params

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对图像应用 CutMix。

        参数：
            labels (dict[str, Any]): Dictionary containing 'img'.
            params (dict | None): Parameters from get_params.

        返回：
            (dict): Updated labels with mixed image.
        """
        if params.get("skip"):
            return labels
        x1, y1, x2, y2 = params["area"].astype(np.int32)
        labels2 = labels["mix_labels"][0]
        labels["img"][y1:y2, x1:x2] = labels2["img"][y1:y2, x1:x2]
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对实例应用 CutMix。

        参数：
            labels (dict[str, Any]): Dictionary containing 'instances' and 'cls'.
            params (dict | None): Parameters from get_params.

        返回：
            (dict): Updated labels with mixed instances.
        """
        if params.get("skip"):
            return labels
        labels2 = labels["mix_labels"][0]
        w, h = params["w"], params["h"]
        area = params["area"]
        indexes2 = params["indexes2"]

        instances2 = labels2["instances"][indexes2]
        instances2.convert_bbox("xyxy")
        instances2.denormalize(w, h)

        x1, y1, x2, y2 = area.astype(np.int32)
        instances2.add_padding(-x1, -y1)
        instances2.clip(x2 - x1, y2 - y1)
        instances2.add_padding(x1, y1)

        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"][indexes2]], axis=0)
        labels["instances"] = Instances.concatenate([labels["instances"], instances2], axis=0)
        return labels


class RandomPerspective(BaseTransform):
    """对图像及对应标注实现随机透视和仿射变换。

    This class applies random rotations, translations, scaling, shearing, and perspective transformations to images and
    their associated bounding boxes, segments, and keypoints. It can be used as part of an augmentation pipeline for
    object detection and instance segmentation tasks.

    属性：
        degrees (float): Maximum absolute degree range for random rotations.
        translate (float): Maximum translation as a fraction of the image size.
        scale (float): Scaling factor range, e.g., scale=0.1 means 0.9-1.1.
        shear (float): Maximum shear angle in degrees.
        perspective (float): Perspective distortion factor.
        border (tuple[int, int]): Mosaic border size as (y, x).
        pre_transform (Callable | None): Optional transform to apply before the random perspective.

    方法：
        get_params: Compute affine transformation matrix and related parameters.
        apply_image: Warp the image using the affine matrix.
        apply_instances: Transform bounding boxes, segments, and keypoints.
        apply_semantic: Placeholder for semantic segmentation mask transformation.
        apply_bboxes: Transform bounding boxes using the affine matrix.
        apply_segments: Transform segments and generate new bounding boxes.
        apply_keypoints: Transform keypoints using the affine matrix.
        box_candidates: Filter transformed bounding boxes based on size and aspect ratio.

    示例：
        >>> transform = RandomPerspective(degrees=10, translate=0.1, scale=0.1, shear=10)
        >>> image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        >>> labels = {"img": image, "cls": np.array([0, 1]), "instances": Instances(...)}
        >>> result = transform(labels)
        >>> transformed_image = result["img"]
        >>> transformed_instances = result["instances"]
    """

    def __init__(
        self,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        border: tuple[int, int] = (0, 0),
        pre_transform=None,
    ):
        """使用变换参数初始化 RandomPerspective 对象。

        This class implements random perspective and affine transformations on images and corresponding bounding boxes,
        segments, and keypoints. Transformations include rotation, translation, scaling, and shearing.

        参数：
            degrees (float): Degree range for random rotations.
            translate (float): Fraction of total width and height for random translation.
            scale (float): Scaling factor interval, e.g., a scale factor of 0.5 allows a resize between 50%-150%.
            shear (float): Shear intensity (angle in degrees).
            perspective (float): Perspective distortion factor.
            border (tuple[int, int]): Tuple specifying mosaic border (y, x).
            pre_transform (Callable | None): Function/transform to apply to the image before starting the random
                transformation.
        """
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border  # mosaic border
        self.pre_transform = pre_transform

    def _compute_affine_matrix(self, img: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float]:
        """计算仿射变换矩阵但不应用。

        参数：
            img (np.ndarray): Input image used to determine center and dimensions.
            size (tuple[int, int]): Size of the output image (width, height) used for clipping translation transform.

        返回：
            (M, scale): 3x3 transformation matrix and scale factor.
        """
        # 中心
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # 透视
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # 旋转和缩放
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # 剪切
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # 平移
        T = np.eye(3, dtype=np.float32)

        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * size[0]  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * size[1]  # y translation (pixels)

        # 组合旋转矩阵
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        return M, s

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """计算图像和实例共享的仿射变换参数。

        参数：
            labels (dict[str, Any]): Input labels dictionary containing 'img'.

        返回：
            (dict): Parameters including 'M' (affine matrix), 'scale', 'border', 'orig_shape', and 'size'.
        """
        img = labels["img"]
        border = labels.pop("mosaic_border", self.border)
        size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
        orig_shape = img.shape[:2]
        M, scale = self._compute_affine_matrix(img, size)
        return {"M": M, "scale": scale, "border": border, "orig_shape": orig_shape, "size": size}

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对图像应用仿射变形。

        参数：
            labels (dict[str, Any]): Dictionary containing 'img'.
            params (dict | None): Parameters from get_params, including 'M', 'border', and 'size'.

        返回：
            (dict): Updated labels with warped image and 'resized_shape'.
        """
        img = labels["img"]
        M = params["M"]
        border = params["border"]
        size = params["size"]
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=size, borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=size, borderValue=(114, 114, 114))
            if img.ndim == 2:
                img = img[..., None]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对目标实例应用仿射变换。

        参数：
            labels (dict[str, Any]): Dictionary containing 'instances' and 'cls'.
            params (dict | None): Parameters from get_params, including 'M' and 'scale'.

        返回：
            (dict): Updated labels with transformed and filtered instances.
        """
        cls = labels["cls"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*params["orig_shape"][::-1])

        M = params["M"]
        scale = params["scale"]

        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        keypoints = instances.keypoints
        # 如果有分割掩码，则更新边界框
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M, params["size"])

        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M, params["size"])
        new_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
        # 裁剪
        new_instances.clip(*params["size"])

        # 过滤实例
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        # 使边界框与新边界框保持相同尺度
        i = self.box_candidates(
            box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.01 if len(segments) else 0.10
        )
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        return labels

    def apply_bboxes(self, bboxes: np.ndarray, M: np.ndarray) -> np.ndarray:
        """对边界框应用仿射变换。

        This function applies an affine transformation to a set of bounding boxes using the provided transformation
        matrix.

        参数：
            bboxes (np.ndarray): Bounding boxes in xyxy format with shape (N, 4), where N is the number of bounding
                boxes.
            M (np.ndarray): Affine transformation matrix with shape (3, 3).

        返回：
            (np.ndarray): Transformed bounding boxes in xyxy format with shape (N, 4).

        示例：
            >>> rp = RandomPerspective()
            >>> bboxes = np.array([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=np.float32)
            >>> M = np.eye(3, dtype=np.float32)
            >>> transformed_bboxes = rp.apply_bboxes(bboxes, M)
        """
        n = len(bboxes)
        if n == 0:
            return bboxes

        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # 创建新边界框
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T

    def apply_segments(
        self, segments: np.ndarray, M: np.ndarray, size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """对分割区域应用仿射变换并生成新的边界框。

        This function applies affine transformations to input segments and generates new bounding boxes based on the
        transformed segments. It clips the transformed segments to fit within the new bounding boxes.

        参数：
            segments (np.ndarray): Input segments with shape (N, M, 2), where N is the number of segments and M is the
                number of points in each segment.
            M (np.ndarray): Affine transformation matrix with shape (3, 3).
            size (tuple[int, int]): Size of the output image (width, height) used for clipping the segments.

        返回：
            bboxes (np.ndarray): New bounding boxes with shape (N, 4) in xyxy format.
            segments (np.ndarray): Transformed and clipped segments with shape (N, M, 2).

        示例：
            >>> rp = RandomPerspective()
            >>> segments = np.random.rand(10, 500, 2)  # 10 segments with 500 points each
            >>> M = np.eye(3)  # 恒等变换矩阵
            >>> new_bboxes, new_segments = rp.apply_segments(segments, M)
        """
        n, num = segments.shape[:2]
        if n == 0:
            return [], segments

        xy = np.ones((n * num, 3), dtype=segments.dtype)
        segments = segments.reshape(-1, 2)
        xy[:, :2] = segments
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]
        segments = xy.reshape(n, -1, 2)
        bboxes = np.stack([segment2box(xy, size[0], size[1]) for xy in segments], 0)
        segments[..., 0] = segments[..., 0].clip(bboxes[:, 0:1], bboxes[:, 2:3])
        segments[..., 1] = segments[..., 1].clip(bboxes[:, 1:2], bboxes[:, 3:4])
        return bboxes, segments

    def apply_keypoints(self, keypoints: np.ndarray, M: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """对关键点应用仿射变换。

        This method transforms the input keypoints using the provided affine transformation matrix. It handles
        perspective rescaling if necessary and updates the visibility of keypoints that fall outside the image
        boundaries after transformation.

        参数：
            keypoints (np.ndarray): Array of keypoints with shape (N, K, 3), where N is the number of instances, K is
                the number of keypoints per instance, and 3 represents (x, y, visibility).
            M (np.ndarray): 3x3 affine transformation matrix.
            size (tuple[int, int]): Size of the output image (width, height) used to determine visibility of keypoints.

        返回：
            (np.ndarray): Transformed keypoints array with the same shape as input (N, K, 3).

        示例：
            >>> random_perspective = RandomPerspective()
            >>> keypoints = np.random.rand(5, 17, 3)  # 5 instances, 17 keypoints each
            >>> M = np.eye(3)  # 恒等变换
            >>> transformed_keypoints = random_perspective.apply_keypoints(keypoints, M)
        """
        n, nkpt = keypoints.shape[:2]
        if n == 0:
            return keypoints
        xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
        visible = keypoints[..., 2].reshape(n * nkpt, 1)
        xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]  # perspective rescale or affine
        out_mask = (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0] > size[0]) | (xy[:, 1] > size[1])
        visible[out_mask] = 0
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """对图像及其关联的 labels 应用随机透视和仿射变换。

        This method performs a series of transformations including rotation, translation, scaling, shearing, and
        perspective distortion on the input image and adjusts the corresponding bounding boxes, segments, and keypoints
        accordingly.

        参数：
            labels (dict[str, Any]): A dictionary containing image data and annotations.

        返回：
            (dict[str, Any]): Transformed labels dictionary containing:
                - 'img' (np.ndarray): The transformed image.
                - 'cls' (np.ndarray): Updated class labels.
                - 'instances' (Instances): Updated object instances.
                - 'resized_shape' (tuple[int, int]): New image shape after transformation.

        示例：
            >>> transform = RandomPerspective()
            >>> image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            >>> labels = {
            ...     "img": image,
            ...     "cls": np.array([0, 1, 2]),
            ...     "instances": Instances(bboxes=np.array([[10, 10, 50, 50], [100, 100, 150, 150]])),
            ... }
            >>> result = transform(labels)
            >>> assert result["img"].shape[:2] == result["resized_shape"]

        注意：
            'labels' arg must include:
                - 'img' (np.ndarray): The input image.
                - 'cls' (np.ndarray): Class labels.
                - 'instances' (Instances): Object instances with bounding boxes, segments, and keypoints.
            May include:
                - 'mosaic_border' (tuple[int, int]): Border size for mosaic augmentation.
        """
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        labels.pop("ratio_pad", None)  # do not need ratio pad
        return super().__call__(labels)

    @staticmethod
    def box_candidates(
        box1: np.ndarray,
        box2: np.ndarray,
        wh_thr: int = 2,
        ar_thr: int = 100,
        area_thr: float = 0.1,
        eps: float = 1e-16,
    ) -> np.ndarray:
        """基于尺寸和宽高比条件计算用于进一步处理的候选框。

        This method compares boxes before and after augmentation to determine if they meet specified thresholds for
        width, height, aspect ratio, and area. It's used to filter out boxes that have been overly distorted or reduced
        by the augmentation process.

        参数：
            box1 (np.ndarray): Original boxes before augmentation, shape (4, N) where N is the number of boxes. Format
                is [x1, y1, x2, y2] in absolute coordinates.
            box2 (np.ndarray): Augmented boxes after transformation, shape (4, N). Format is [x1, y1, x2, y2] in
                absolute coordinates.
            wh_thr (int): Width and height threshold in pixels. Boxes smaller than this in either dimension are
                rejected.
            ar_thr (int): Aspect ratio threshold. Boxes with an aspect ratio greater than this value are rejected.
            area_thr (float): Area ratio threshold. Boxes with an area ratio (new/old) less than this value are
                rejected.
            eps (float): Small epsilon value to prevent division by zero.

        返回：
            (np.ndarray): Boolean array of shape (N,) indicating which boxes are candidates. True values correspond to
                boxes that meet all criteria.

        示例：
            >>> random_perspective = RandomPerspective()
            >>> box1 = np.array([[0, 0, 100, 100], [0, 0, 50, 50]]).T
            >>> box2 = np.array([[10, 10, 90, 90], [5, 5, 45, 45]]).T
            >>> candidates = random_perspective.box_candidates(box1, box2)
            >>> print(candidates)
            [True True]
        """
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


class RandomHSV(BaseTransform):
    """随机调整图像的色调（Hue）、饱和度（Saturation）和明度（Value）通道。

    This class applies random HSV augmentation to images within predefined limits set by hgain, sgain, and vgain.

    属性：
        hgain (float): Maximum variation for hue. Range is typically [0, 1].
        sgain (float): Maximum variation for saturation. Range is typically [0, 1].
        vgain (float): Maximum variation for value. Range is typically [0, 1].

    方法：
        apply_image: Apply random HSV augmentation to an image.

    示例：
        >>> import numpy as np
        >>> from ultralytics.data.augment import RandomHSV
        >>> augmenter = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
        >>> image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> labels = {"img": image}
        >>> labels = augmenter(labels)
        >>> augmented_image = labels["img"]
    """

    def __init__(self, hgain: float = 0.5, sgain: float = 0.5, vgain: float = 0.5) -> None:
        """初始化用于随机 HSV（色调、饱和度、明度）增强的 RandomHSV 对象。

        This class applies random adjustments to the HSV channels of an image within specified limits.

        参数：
            hgain (float): Maximum variation for hue. Should be in the range [0, 1].
            sgain (float): Maximum variation for saturation. Should be in the range [0, 1].
            vgain (float): Maximum variation for value. Should be in the range [0, 1].
        """
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def apply_image(self, labels, params: dict[str, Any] | None = None):
        """在预定义范围内对图像应用随机 HSV 增强。

        This method modifies the input image by randomly adjusting its Hue, Saturation, and Value (HSV) channels. The
        adjustments are made within the limits set by hgain, sgain, and vgain during initialization.

        参数：
            labels (dict[str, Any]): A dictionary containing image data and metadata. Must include an 'img' key with the
                image as a numpy array.
            params (dict[str, Any] | None): Unused parameters for API compatibility.

        返回：
            (dict[str, Any]): The labels dictionary with the HSV-augmented image.

        示例：
            >>> hsv_augmenter = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
            >>> labels = {"img": np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)}
            >>> labels = hsv_augmenter.apply_image(labels)
            >>> augmented_img = labels["img"]
        """
        img = labels["img"]
        if img.shape[-1] != 3:  # only apply to RGB images
            return labels
        if self.hgain or self.sgain or self.vgain:
            dtype = img.dtype  # uint8

            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]  # random gains
            x = np.arange(0, 256, dtype=r.dtype)
            # lut_hue = ((x * (r[0] + 1)) % 180).astype(dtype)   # ultralytics<=8.3.78 的原始色调实现
            lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
            lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
            lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
            lut_sat[0] = 0  # prevent pure white changing color, introduced in 8.3.79

            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        return labels


class RandomFlip(BaseTransform):
    """以给定概率对图像应用随机水平或垂直翻转。

    This class performs random image flipping and updates corresponding instance annotations such as bounding boxes and
    keypoints.

    属性：
        p (float): Probability of applying the flip. Must be between 0 and 1.
        direction (str): Direction of flip, either 'horizontal' or 'vertical'.
        flip_idx (array-like): Index mapping for flipping keypoints, if applicable.

    方法：
        __call__: Apply the random flip transformation to an image and its annotations.

    示例：
        >>> transform = RandomFlip(p=0.5, direction="horizontal")
        >>> result = transform({"img": image, "instances": instances})
        >>> flipped_image = result["img"]
        >>> flipped_instances = result["instances"]
    """

    def __init__(self, p: float = 0.5, direction: str = "horizontal", flip_idx: list[int] | None = None) -> None:
        """使用概率和方向初始化 RandomFlip 类。

        This class applies a random horizontal or vertical flip to an image with a given probability. It also updates
        any instances (bounding boxes, keypoints, etc.) accordingly.

        参数：
            p (float): The probability of applying the flip. Must be between 0 and 1.
            direction (str): The direction to apply the flip. Must be 'horizontal' or 'vertical'.
            flip_idx (list[int] | None): Index mapping for flipping keypoints, if any.

        异常：
            AssertionError: If direction is not 'horizontal' or 'vertical', or if p is not between 0 and 1.
        """
        assert direction in {"horizontal", "vertical"}, f"Support direction `horizontal` or `vertical`, got {direction}"
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."

        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """计算随机翻转参数。

        参数：
            labels (dict[str, Any]): Input labels dictionary containing 'img' and 'instances'.

        返回：
            (dict): Parameters including 'flip' (bool), 'h', 'w', 'direction', and 'flip_idx'.
        """
        img = labels["img"]
        instances = labels["instances"]
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w
        return {
            "flip": random.random() < self.p,
            "h": h,
            "w": w,
            "direction": self.direction,
            "flip_idx": self.flip_idx,
        }

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """对图像应用翻转。

        参数：
            labels (dict[str, Any]): Dictionary containing 'img'.
            params (dict): Parameters from get_params.

        返回：
            (dict): Updated labels with flipped (or unchanged) image.
        """
        img = labels["img"]
        if params["flip"]:
            if params["direction"] == "vertical":
                img = np.flipud(img)
            elif params["direction"] == "horizontal":
                img = np.fliplr(img)
        labels["img"] = np.ascontiguousarray(img)
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """对目标实例应用翻转。

        参数：
            labels (dict[str, Any]): Dictionary containing 'instances'.
            params (dict): Parameters from get_params.

        返回：
            (dict): Updated labels with flipped (or unchanged) instances.
        """
        instances = labels.pop("instances")
        instances.convert_bbox(format="xywh")
        if params["flip"]:
            if params["direction"] == "vertical":
                instances.flipud(params["h"])
            elif params["direction"] == "horizontal":
                instances.fliplr(params["w"])
            if params["flip_idx"] is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, params["flip_idx"], :])
        labels["instances"] = instances
        return labels


class LetterBox(BaseTransform):
    """调整图像尺寸并填充，用于检测、实例分割和姿态估计。

    This class resizes and pads images to a specified shape while preserving aspect ratio. It also updates corresponding
    labels and bounding boxes.

    属性：
        new_shape (tuple): Target shape (height, width) for resizing.
        auto (bool): Whether to use minimum rectangle.
        scale_fill (bool): Whether to stretch the image to new_shape.
        scaleup (bool): Whether to allow scaling up. If False, only scale down.
        stride (int): Stride for rounding padding.
        center (bool): Whether to center the image or align to top-left.

    方法：
        __call__: Resize and pad image, update labels and bounding boxes.

    示例：
        >>> transform = LetterBox(new_shape=(640, 640))
        >>> result = transform(labels)
        >>> resized_img = result["img"]
        >>> updated_instances = result["instances"]
    """

    def __init__(
        self,
        new_shape: tuple[int, int] = (640, 640),
        auto: bool = False,
        scale_fill: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
        padding_value: int = 114,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        """初始化用于调整尺寸和填充图像的 LetterBox 对象。

        This class is designed to resize and pad images for object detection, instance segmentation, and pose estimation
        tasks. It supports various resizing modes including auto-sizing, scale-fill, and letterboxing.

        参数：
            new_shape (tuple[int, int]): Target size (height, width) for the resized image.
            auto (bool): If True, use minimum rectangle to resize. If False, use new_shape directly.
            scale_fill (bool): If True, stretch the image to new_shape without padding.
            scaleup (bool): If True, allow scaling up. If False, only scale down.
            center (bool): If True, center the placed image. If False, place image in top-left corner.
            stride (int): Stride of the model (e.g., 32 for YOLOv5).
            padding_value (int): Value for padding the image. Default is 114.
            interpolation (int): Interpolation method for resizing. Default is cv2.INTER_LINEAR.
        """
        self.new_shape = new_shape
        self.auto = auto
        self.scale_fill = scale_fill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # 将图像放在中间或左上角
        self.padding_value = padding_value
        self.interpolation = interpolation

    def __call__(self, labels: dict[str, Any] | None = None, image: np.ndarray = None) -> dict[str, Any] | np.ndarray:
        """调整图像尺寸并填充，用于目标检测、实例分割或姿态估计任务。

        This method applies letterboxing to the input image, which involves resizing the image while maintaining its
        aspect ratio and adding padding to fit the new shape. It also updates any associated labels accordingly.

        参数：
            labels (dict[str, Any] | None): A dictionary containing image data and associated labels, or empty dict if
                None.
            image (np.ndarray | None): The input image as a numpy array. If None, the image is taken from 'labels'.

        返回：
            (dict[str, Any] | np.ndarray): If 'labels' is provided, returns an updated dictionary with the resized and
                padded image, updated labels, and additional metadata. If 'labels' is empty, returns the resized and
                padded image only.

        示例：
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> result = letterbox(labels={"img": np.zeros((480, 640, 3)), "instances": Instances(...)})
            >>> resized_img = result["img"]
            >>> updated_instances = result["instances"]
        """
        if labels is None:
            labels = {}
        return_image_only = len(labels) == 0
        if image is not None:
            labels["img"] = image
        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        if not return_image_only:
            labels = self.apply_instances(labels, params)
        labels = self.apply_semantic(labels, params)
        if return_image_only:
            return labels["img"]
        return labels

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """计算 LetterBox 参数。

        参数：
            labels (dict[str, Any]): Input labels dictionary containing 'img'.

        返回：
            (dict): Parameters including 'orig_shape', 'new_shape', 'ratio', padding, and resize info.
        """
        img = labels["img"]
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 缩放比例（新 / 旧）
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # 计算填充量
        ratio = r, r  # width, height ratios
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scale_fill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        top, bottom = round(dh - 0.1) if self.center else 0, round(dh + 0.1)
        left, right = round(dw - 0.1) if self.center else 0, round(dw + 0.1)

        return {
            "orig_shape": shape,
            "new_shape": new_shape,
            "ratio": ratio,
            "new_unpad": new_unpad,
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
        }

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """调整图像尺寸并填充。

        参数：
            labels (dict[str, Any]): Dictionary containing 'img'.
            params (dict): Parameters from get_params.

        返回：
            (dict): Updated labels with resized and padded image.
        """
        img = labels["img"]
        shape = img.shape[:2]
        new_unpad = params["new_unpad"]

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=self.interpolation)
            if img.ndim == 2:
                img = img[..., None]

        h, w, c = img.shape
        top, bottom = params["top"], params["bottom"]
        left, right = params["left"], params["right"]
        if c == 3:
            img = cv2.copyMakeBorder(
                img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(self.padding_value,) * 3
            )
        else:  # multispectral
            pad_img = np.full((h + top + bottom, w + left + right, c), fill_value=self.padding_value, dtype=img.dtype)
            pad_img[top : top + h, left : left + w] = img
            img = pad_img

        labels["img"] = img
        labels["resized_shape"] = params["new_shape"]
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """在 LetterBox 处理后更新实例坐标。

        参数：
            labels (dict[str, Any]): Dictionary containing 'instances'.
            params (dict): Parameters from get_params.

        返回：
            (dict): Updated labels with transformed instances.
        """
        if "instances" in labels:
            labels = self._update_labels(labels, params["ratio"], params["left"], params["top"], params["orig_shape"])
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (params["left"], params["top"]))  # for evaluation
        return labels

    @staticmethod
    def _update_labels(
        labels: dict[str, Any], ratio: tuple[float, float], padw: float, padh: float, orig_shape: tuple[int, int]
    ) -> dict[str, Any]:
        """在对图像应用 LetterBox 处理后更新标签。

        This method modifies the bounding box coordinates of instances in the labels to account for resizing and padding
        applied during letterboxing.

        参数：
            labels (dict[str, Any]): A dictionary containing image labels and instances.
            ratio (tuple[float, float]): Scaling ratios (width, height) applied to the image.
            padw (float): Padding width added to the image.
            padh (float): Padding height added to the image.
            orig_shape (tuple[int, int]): Original image shape (height, width) before resizing.

        返回：
            (dict[str, Any]): Updated labels dictionary with modified instance coordinates.

        示例：
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> labels = {"instances": Instances(...)}
            >>> ratio = (0.5, 0.5)
            >>> padw, padh = 10, 20
            >>> updated_labels = letterbox._update_labels(labels, ratio, padw, padh, (480, 640))
        """
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*orig_shape[::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


class CopyPaste(BaseMixTransform):
    """CopyPaste 类，用于对图像数据集应用 Copy-Paste 增强。

    This class implements the Copy-Paste augmentation technique as described in the paper "Simple Copy-Paste is a Strong
    Data Augmentation Method for Instance Segmentation" (https://arxiv.org/abs/2012.07177). It combines objects from
    different images to create new training samples.

    属性：
        dataset (Any): The dataset to which Copy-Paste augmentation will be applied.
        pre_transform (Callable | None): Optional transform to apply before Copy-Paste.
        p (float): Probability of applying Copy-Paste augmentation.

    方法：
        get_params: Compute CopyPaste parameters including selected instances and mask.
        apply_image: Draw contours and paste pixels for CopyPaste.
        apply_instances: Concatenate selected instances for CopyPaste.

    示例：
        >>> from ultralytics.data.augment import CopyPaste
        >>> dataset = YourDataset(...)  # 你的图像数据集
        >>> copypaste = CopyPaste(dataset, p=0.5)
        >>> augmented_labels = copypaste(original_labels)
    """

    def __init__(self, dataset=None, pre_transform=None, p: float = 0.5, mode: str = "flip") -> None:
        """使用数据集、预处理变换和应用概率初始化 CopyPaste 对象。"""
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        assert mode in {"flip", "mixup"}, f"Expected `mode` to be `flip` or `mixup`, but got {mode}."
        self.mode = mode

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """对图像及其标签应用 Copy-Paste 增强。"""
        if len(labels["instances"].segments) == 0 or self.p == 0:
            return labels
        if self.mode == "flip":
            params = self.get_params(labels)
            labels = self.apply_image(labels, params)
            labels = self.apply_instances(labels, params)
            labels = self.apply_semantic(labels, params)
            return labels
        return super().__call__(labels)

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """计算 CopyPaste 参数。

        参数：
            labels (dict[str, Any]): Input labels dictionary.

        返回：
            (dict[str, Any]): Parameters including 'instances2', 'selected', and 'im_new'.
        """
        params = {}
        if self.mode == "mixup":
            params = super().get_params(labels)
            labels2 = labels.get("mix_labels", [{}])[0]
        else:
            labels2 = {}

        h, w = labels["img"].shape[:2]
        instances = deepcopy(labels["instances"])
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w, h)

        instances2 = deepcopy(labels2.get("instances")) if labels2 else None
        if instances2 is None:
            instances2 = deepcopy(instances)
            instances2.fliplr(w)

        ioa = bbox_ioa(instances2.bboxes, instances.bboxes)
        indexes = np.nonzero((ioa < 0.30).all(1))[0]
        n = len(indexes)
        sorted_idx = np.argsort(ioa.max(1)[indexes])
        indexes = indexes[sorted_idx]
        selected = indexes[: round(self.p * n)]

        im_new = np.zeros((h, w), np.uint8)

        params["instances"] = instances
        params["instances2"] = instances2
        params["selected"] = selected
        params["im_new"] = im_new
        params["labels2_cls"] = labels2.get("cls")
        params["labels2_img"] = labels2.get("img")
        return params

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对图像应用 CopyPaste。

        参数：
            labels (dict[str, Any]): Dictionary containing 'img'.
            params (dict | None): Parameters from get_params.

        返回：
            (dict): Updated labels with pasted objects.
        """
        im = labels["img"]
        if "mosaic_border" not in labels:
            im = im.copy()

        instances2 = params["instances2"]
        selected = params["selected"]
        im_new = params["im_new"]

        for j in selected:
            cv2.drawContours(im_new, instances2.segments[[j]].astype(np.int32), -1, 1, cv2.FILLED)

        result = params.get("labels2_img")
        if result is None:
            result = cv2.flip(im, 1)
        if result.ndim == 2:
            result = result[..., None]

        i = im_new.astype(bool)
        im[i] = result[i]
        labels["img"] = im
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """对实例应用 CopyPaste。

        参数：
            labels (dict[str, Any]): Dictionary containing 'instances' and 'cls'.
            params (dict | None): Parameters from get_params.

        返回：
            (dict): Updated labels with concatenated instances.
        """
        instances = params["instances"]
        instances2 = params["instances2"]
        selected = params["selected"]
        cls = labels["cls"]
        labels2_cls = params.get("labels2_cls")

        for j in selected:
            cls = np.concatenate((cls, (labels2_cls if labels2_cls is not None else cls)[[j]]), axis=0)
            instances = Instances.concatenate((instances, instances2[[j]]), axis=0)

        labels["cls"] = cls
        labels["instances"] = instances
        return labels


class Albumentations(BaseTransform):
    """用于图像增强的 Albumentations 变换。

    This class applies various image transformations using the Albumentations library. It includes operations such as
    Blur, Median Blur, conversion to grayscale, Contrast Limited Adaptive Histogram Equalization (CLAHE), random changes
    in brightness and contrast, RandomGamma, and image quality reduction through compression.

    属性：
        p (float): Probability of applying the transformations.
        transform (albumentations.Compose): Composed Albumentations transforms.
        contains_spatial (bool): Indicates if the transforms include spatial operations.

    方法：
        __call__: Apply the Albumentations transformations to the input labels.

    示例：
        >>> transform = Albumentations(p=0.5)
        >>> augmented_labels = transform(labels)

    注意：
        - Requires Albumentations version 1.0.3 or higher.
        - Spatial transforms are handled differently to ensure bbox compatibility.
        - Some transforms are applied with very low probability (0.01) by default.
    """

    def __init__(self, p: float = 1.0, transforms: list | None = None) -> None:
        """初始化用于 YOLO 边界框格式参数的 Albumentations 变换对象。

        This class applies various image augmentations using the Albumentations library, including Blur, Median Blur,
        conversion to grayscale, Contrast Limited Adaptive Histogram Equalization, random changes of brightness and
        contrast, RandomGamma, and image quality reduction through compression.

        参数：
            p (float): Probability of applying the augmentations. Must be between 0 and 1.
            transforms (list | None): List of custom Albumentations transforms. If None, uses default transforms.
        """
        self.p = p
        self.transform = None
        prefix = colorstr("albumentations: ")

        try:
            import os

            os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # suppress Albumentations upgrade message
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            # 可用的空间变换列表
            spatial_transforms = {
                "Affine",
                "BBoxSafeRandomCrop",
                "CenterCrop",
                "CoarseDropout",
                "Crop",
                "CropAndPad",
                "CropNonEmptyMaskIfExists",
                "D4",
                "ElasticTransform",
                "Flip",
                "GridDistortion",
                "GridDropout",
                "HorizontalFlip",
                "Lambda",
                "LongestMaxSize",
                "MaskDropout",
                "MixUp",
                "Morphological",
                "NoOp",
                "OpticalDistortion",
                "PadIfNeeded",
                "Perspective",
                "PiecewiseAffine",
                "PixelDropout",
                "RandomCrop",
                "RandomCropFromBorders",
                "RandomGridShuffle",
                "RandomResizedCrop",
                "RandomRotate90",
                "RandomScale",
                "RandomSizedBBoxSafeCrop",
                "RandomSizedCrop",
                "Resize",
                "Rotate",
                "SafeRotate",
                "ShiftScaleRotate",
                "SmallestMaxSize",
                "Transpose",
                "VerticalFlip",
                "XYMasking",
            }  # from https://albumentations.ai/docs/getting_started/transforms_and_targets/#spatial-level-transforms

            # 变换：使用自定义变换（如有），否则使用默认值
            T = (
                [
                    A.Blur(p=0.01),
                    A.MedianBlur(p=0.01),
                    A.ToGray(p=0.01),
                    A.CLAHE(p=0.01),
                    A.RandomBrightnessContrast(p=0.0),
                    A.RandomGamma(p=0.0),
                    A.ImageCompression(quality_range=(75, 100), p=0.0),
                ]
                if transforms is None
                else transforms
            )

            # 组合变换
            self.contains_spatial = any(transform.__class__.__name__ in spatial_transforms for transform in T)
            self.transform = (
                A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
                if self.contains_spatial
                else A.Compose(T)
            )
            if hasattr(self.transform, "set_random_seed"):
                # albumentations>=1.4.21 中确定性变换所需
                self.transform.set_random_seed(torch.initial_seed())
            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """对输入 labels 应用 Albumentations 变换。

        This method applies a series of image augmentations using the Albumentations library. It can perform both
        spatial and non-spatial transformations on the input image and its corresponding labels.

        参数：
            labels (dict[str, Any]): A dictionary containing image data and annotations. Expected keys are:
                - 'img': np.ndarray representing the image
                - 'cls': np.ndarray of class labels
                - 'instances': object containing bounding boxes and other instance information

        返回：
            (dict[str, Any]): The input dictionary with augmented image and updated annotations.

        示例：
            >>> transform = Albumentations(p=0.5)
            >>> labels = {
            ...     "img": np.random.rand(640, 640, 3),
            ...     "cls": np.array([0, 1]),
            ...     "instances": Instances(bboxes=np.array([[0, 0, 1, 1], [0.5, 0.5, 0.8, 0.8]])),
            ... }
            >>> augmented = transform(labels)
            >>> assert augmented["img"].shape == (640, 640, 3)

        注意：
            - The method applies transformations with probability self.p.
            - Spatial transforms update bounding boxes, while non-spatial transforms only modify the image.
            - Requires the Albumentations library to be installed.
        """
        if self.transform is None or random.random() > self.p:
            return labels

        im = labels["img"]
        if im.shape[2] != 3:  # 仅对三通道图像应用 Albumentation
            return labels

        if self.contains_spatial:
            cls = labels["cls"]
            if len(cls):
                labels["instances"].convert_bbox("xywh")
                labels["instances"].normalize(*im.shape[:2][::-1])
                bboxes = labels["instances"].bboxes
                # TODO：添加对分割和关键点的支持
                new = self.transform(image=im, bboxes=bboxes, class_labels=cls)  # transformed
                if len(new["class_labels"]) > 0:  # skip update if no bbox in new im
                    labels["img"] = new["image"]
                    labels["cls"] = np.array(new["class_labels"]).reshape(-1, 1)
                    bboxes = np.array(new["bboxes"], dtype=np.float32)
                labels["instances"].update(bboxes=bboxes)
        else:
            labels["img"] = self.transform(image=labels["img"])["image"]  # transformed

        return labels


class Format(BaseTransform):
    """用于格式化目标检测、实例分割和姿态估计任务中图像标注的类。

    This class standardizes image and instance annotations to be used by the `collate_fn` in PyTorch DataLoader.

    属性：
        bbox_format (str): Format for bounding boxes. Options are 'xywh' or 'xyxy'.
        normalize (bool): Whether to normalize bounding boxes.
        return_mask (bool): Whether to return instance masks for segmentation.
        return_keypoint (bool): Whether to return keypoints for pose estimation.
        return_obb (bool): Whether to return oriented bounding boxes.
        mask_ratio (int): Downsample ratio for masks.
        mask_overlap (bool): Whether to overlap masks.
        batch_idx (bool): Whether to keep batch indexes.
        bgr (float): The probability to return BGR images.

    方法：
        __call__: Format labels dictionary with image, classes, bounding boxes, and optionally masks and keypoints.
        _format_img: Convert image from Numpy array to PyTorch tensor.
        _format_segments: Convert polygon points to bitmap masks.

    示例：
        >>> formatter = Format(bbox_format="xywh", normalize=True, return_mask=True)
        >>> formatted_labels = formatter(labels)
        >>> img = formatted_labels["img"]
        >>> bboxes = formatted_labels["bboxes"]
        >>> masks = formatted_labels["masks"]
    """

    def __init__(
        self,
        bbox_format: str = "xywh",
        normalize: bool = True,
        return_mask: bool = False,
        return_keypoint: bool = False,
        return_obb: bool = False,
        mask_ratio: int = 4,
        mask_overlap: bool = True,
        batch_idx: bool = True,
        bgr: float = 0.0,
    ):
        """使用给定的图像和实例标注格式化参数初始化 Format 类。

        This class standardizes image and instance annotations for object detection, instance segmentation, and pose
        estimation tasks, preparing them for use in PyTorch DataLoader's `collate_fn`.

        参数：
            bbox_format (str): Format for bounding boxes. Options are 'xywh', 'xyxy', etc.
            normalize (bool): Whether to normalize bounding boxes to [0,1].
            return_mask (bool): If True, returns instance masks for segmentation tasks.
            return_keypoint (bool): If True, returns keypoints for pose estimation tasks.
            return_obb (bool): If True, returns oriented bounding boxes.
            mask_ratio (int): Downsample ratio for masks.
            mask_overlap (bool): If True, allows mask overlap.
            batch_idx (bool): If True, keeps batch indexes.
            bgr (float): Probability of returning BGR images instead of RGB.
        """
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.return_mask = return_mask  # set False when training detection only
        self.return_keypoint = return_keypoint
        self.return_obb = return_obb
        self.mask_ratio = mask_ratio
        self.mask_overlap = mask_overlap
        self.batch_idx = batch_idx  # keep the batch indexes
        self.bgr = bgr

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """格式化目标检测、实例分割和姿态估计任务的图像标注。

        This method standardizes the image and instance annotations to be used by the `collate_fn` in PyTorch
        DataLoader. It processes the input labels dictionary, converting annotations to the specified format and
        applying normalization if required.

        参数：
            labels (dict[str, Any]): A dictionary containing image and annotation data with the following keys:
                - 'img': The input image as a numpy array.
                - 'cls': Class labels for instances.
                - 'instances': An Instances object containing bounding boxes, segments, and keypoints.

        返回：
            (dict[str, Any]): A dictionary with formatted data, including:
                - 'img': Formatted image tensor.
                - 'cls': Class labels tensor.
                - 'bboxes': Bounding boxes tensor in the specified format.
                - 'masks': Instance masks tensor (if return_mask is True).
                - 'keypoints': Keypoints tensor (if return_keypoint is True).
                - 'batch_idx': Batch index tensor (if batch_idx is True).

        示例：
            >>> formatter = Format(bbox_format="xywh", normalize=True, return_mask=True)
            >>> labels = {"img": np.random.rand(640, 640, 3), "cls": np.array([0, 1]), "instances": Instances(...)}
            >>> formatted_labels = formatter(labels)
            >>> print(formatted_labels.keys())
        """
        img = labels.pop("img")
        h, w = img.shape[:2]
        cls = labels.pop("cls")
        instances = labels.pop("instances")
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)

        if self.return_mask:
            if nl:
                masks, instances, cls = self._format_segments(instances, cls, w, h)
                masks = torch.from_numpy(masks)
                cls_tensor = torch.from_numpy(cls.squeeze(1))
                if not masks.shape[0] or not cls_tensor.numel():
                    sem_masks = torch.zeros(img.shape[0] // self.mask_ratio, img.shape[1] // self.mask_ratio)
                elif self.mask_overlap:
                    sem_masks = cls_tensor[masks[0].long() - 1]  # (H, W) from (1, H, W) instance indices
                else:
                    # 创建与 mask_overlap=True 一致的语义掩码
                    sem_masks = (masks * cls_tensor[:, None, None]).max(0).values  # (H, W) from (N, H, W) binary
                    overlap = masks.sum(dim=0) > 1  # (H, W)
                    if overlap.any():
                        weights = masks.sum(axis=(1, 2))
                        weighted_masks = masks * weights[:, None, None]  # (N, H, W)
                        weighted_masks[masks == 0] = weights.max() + 1  # handle background
                        smallest_idx = weighted_masks.argmin(dim=0)  # (H, W)
                        sem_masks[overlap] = cls_tensor[smallest_idx[overlap]]
            else:
                masks = torch.zeros(
                    1 if self.mask_overlap else nl, img.shape[0] // self.mask_ratio, img.shape[1] // self.mask_ratio
                )
                sem_masks = torch.zeros(img.shape[0] // self.mask_ratio, img.shape[1] // self.mask_ratio)
            labels["masks"] = masks
            labels["sem_masks"] = sem_masks.float()
        labels["img"] = self._format_img(img)
        labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl, 1)
        labels["bboxes"] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        if self.return_keypoint:
            labels["keypoints"] = (
                torch.empty(0, 3) if instances.keypoints is None else torch.from_numpy(instances.keypoints)
            )
            if self.normalize:
                labels["keypoints"][..., 0] /= w
                labels["keypoints"][..., 1] /= h
        if self.return_obb:
            labels["bboxes"] = (
                xyxyxyxy2xywhr(torch.from_numpy(instances.segments)) if len(instances.segments) else torch.zeros((0, 5))
            )
        # 注意：需要将 obb 归一化为 xywhr 格式以保证宽高一致性
        if self.normalize:
            labels["bboxes"][:, [0, 2]] /= w
            labels["bboxes"][:, [1, 3]] /= h
        # 然后可以使用 collate_fn
        if self.batch_idx:
            labels["batch_idx"] = torch.zeros(nl)
        return labels

    def _format_img(self, img: np.ndarray) -> torch.Tensor:
        """将图像从 Numpy 数组格式化为 YOLO 所用的 PyTorch 张量。

        This function performs the following operations:
        1. Ensures the image has 3 dimensions (adds a channel dimension if needed).
        2. Transposes the image from HWC to CHW format.
        3. Optionally reverses the color channels (e.g., BGR to RGB) based on the bgr probability.
        4. Converts the image to a contiguous array.
        5. Converts the Numpy array to a PyTorch tensor.

        参数：
            img (np.ndarray): Input image as a Numpy array with shape (H, W, C) or (H, W).

        返回：
            (torch.Tensor): Formatted image as a PyTorch tensor with shape (C, H, W).

        示例：
            >>> import numpy as np
            >>> img = np.random.rand(100, 100, 3)
            >>> formatted_img = self._format_img(img)
            >>> print(formatted_img.shape)
            torch.Size([3, 100, 100])
        """
        if len(img.shape) < 3:
            img = img[..., None]
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img[::-1] if random.uniform(0, 1) > self.bgr and img.shape[0] == 3 else img)
        img = torch.from_numpy(img)
        return img

    def _format_segments(
        self, instances: Instances, cls: np.ndarray, w: int, h: int
    ) -> tuple[np.ndarray, Instances, np.ndarray]:
        """将多边形分割转换为位图掩码。

        参数：
            instances (Instances): Object containing segment information.
            cls (np.ndarray): Class labels for each instance.
            w (int): Width of the image.
            h (int): Height of the image.

        返回：
            masks (np.ndarray): Bitmap masks with shape (N, H, W) or (1, H, W) if mask_overlap is True.
            instances (Instances): Updated instances object with sorted segments if mask_overlap is True.
            cls (np.ndarray): Updated class labels, sorted if mask_overlap is True.

        注意：
            - If self.mask_overlap is True, masks are overlapped and sorted by area.
            - If self.mask_overlap is False, each mask is represented separately.
            - Masks are downsampled according to self.mask_ratio.
        """
        segments = instances.segments
        if self.mask_overlap:
            masks, sorted_idx = polygons2masks_overlap((h, w), segments, downsample_ratio=self.mask_ratio)
            masks = masks[None]  # (640, 640) -> (1, 640, 640)
            instances = instances[sorted_idx]
            cls = cls[sorted_idx]
        else:
            masks = polygons2masks((h, w), segments, color=1, downsample_ratio=self.mask_ratio)

        return masks, instances, cls


class LoadVisualPrompt(BaseTransform):
    """从边界框或掩码创建视觉提示，用于模型输入。"""

    def __init__(self, scale_factor: float = 1 / 8) -> None:
        """使用缩放因子初始化 LoadVisualPrompt。

        参数：
            scale_factor (float): Factor to scale the input image dimensions.
        """
        self.scale_factor = scale_factor

    @staticmethod
    def make_mask(boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """从边界框创建二值掩码。

        参数：
            boxes (torch.Tensor): Bounding boxes in xyxy format, shape: (N, 4).
            h (int): Height of the mask.
            w (int): Width of the mask.

        返回：
            (torch.Tensor): Binary masks with shape (N, h, w).
        """
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
        r = torch.arange(w)[None, None, :]  # rows shape(1,1,w)
        c = torch.arange(h)[None, :, None]  # cols shape(1,h,1)

        return (r >= x1) * (r < x2) * (c >= y1) * (c < y2)

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """计算视觉提示参数。

        参数：
            labels (dict[str, Any]): Input labels dictionary.

        返回：
            (dict): Parameters including 'imgsz', 'bboxes', 'masks', and 'cls'.
        """
        imgsz = labels["img"].shape[1:]
        bboxes, masks = None, None
        if "bboxes" in labels:
            bboxes = labels["bboxes"]
            bboxes = xywh2xyxy(bboxes) * torch.tensor(imgsz)[[1, 0, 1, 0]]  # denormalize boxes
        elif "masks" in labels:
            masks = labels["masks"]

        cls = labels["cls"].squeeze(-1).to(torch.int)
        return {"imgsz": imgsz, "bboxes": bboxes, "masks": masks, "cls": cls}

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """创建视觉提示并将其添加到 labels。

        参数：
            labels (dict[str, Any]): Dictionary containing image data and annotations.
            params (dict): Parameters from get_params.

        返回：
            (dict): Updated labels with visual prompts added.
        """
        visuals = self.get_visuals(params["cls"], params["imgsz"], bboxes=params["bboxes"], masks=params["masks"])
        labels["visuals"] = visuals
        return labels

    def get_visuals(
        self,
        category: int | np.ndarray | torch.Tensor,
        shape: tuple[int, int],
        bboxes: np.ndarray | torch.Tensor = None,
        masks: np.ndarray | torch.Tensor = None,
    ) -> torch.Tensor:
        """基于边界框或掩码生成视觉掩码。

        参数：
            category (int | np.ndarray | torch.Tensor): The category labels for the objects.
            shape (tuple[int, int]): The shape of the image (height, width).
            bboxes (np.ndarray | torch.Tensor, optional): Bounding boxes for the objects, xyxy format.
            masks (np.ndarray | torch.Tensor, optional): Masks for the objects.

        返回：
            (torch.Tensor): A tensor containing the visual masks for each category.

        异常：
            ValueError: If neither bboxes nor masks are provided.
        """
        masksz = (int(shape[0] * self.scale_factor), int(shape[1] * self.scale_factor))
        if bboxes is not None:
            if isinstance(bboxes, np.ndarray):
                bboxes = torch.from_numpy(bboxes)
            bboxes *= self.scale_factor
            masks = self.make_mask(bboxes, *masksz).float()
        elif masks is not None:
            if isinstance(masks, np.ndarray):
                masks = torch.from_numpy(masks)  # (N, H, W)
            masks = F.interpolate(masks.unsqueeze(1), masksz, mode="nearest").squeeze(1).float()
        else:
            raise ValueError("LoadVisualPrompt must have bboxes or masks in the label")
        if not isinstance(category, torch.Tensor):
            category = torch.tensor(category, dtype=torch.int)
        cls_unique, inverse_indices = torch.unique(category, sorted=True, return_inverse=True)
        # 注意：RandomLoadText 的 `cls` 索引应该是连续的
        # if len(cls_unique):
        #     assert len(cls_unique) == cls_unique[-1] + 1, (
        #         f"Expected a continuous range of class indices, but got {cls_unique}"
        #     )
        visuals = torch.zeros(cls_unique.shape[0], *masksz)
        for idx, mask in zip(inverse_indices, masks):
            visuals[idx] = torch.logical_or(visuals[idx], mask)
        return visuals


class RandomLoadText(BaseTransform):
    """随机采样正负样本文本并相应更新类别索引。

    This class is responsible for sampling texts from a given set of class texts, including both positive (present in
    the image) and negative (not present in the image) samples. It updates the class indices to reflect the sampled
    texts and can optionally pad the text list to a fixed length.

    属性：
        prompt_format (str): Format string for text prompts.
        neg_samples (tuple[int, int]): Range for randomly sampling negative texts.
        max_samples (int): Maximum number of different text samples in one image.
        padding (bool): Whether to pad texts to max_samples.
        padding_value (list[str]): The text used for padding when padding is True.

    方法：
        __call__: Process the input labels and return updated classes and texts.

    示例：
        >>> loader = RandomLoadText(prompt_format="Object: {}", neg_samples=(5, 10), max_samples=20)
        >>> labels = {"cls": [0, 1, 2], "texts": [["cat"], ["dog"], ["bird"]], "instances": [...]}
        >>> updated_labels = loader(labels)
        >>> print(updated_labels["texts"])
        ['Object: cat', 'Object: dog', 'Object: bird', 'Object: elephant', 'Object: car']
    """

    def __init__(
        self,
        prompt_format: str = "{}",
        neg_samples: tuple[int, int] = (80, 80),
        max_samples: int = 80,
        padding: bool = False,
        padding_value: list[str] = [""],
    ) -> None:
        """初始化 RandomLoadText 类，用于随机采样正负样本文本。

        This class is designed to randomly sample positive texts and negative texts, and update the class indices
        accordingly to the number of samples. It can be used for text-based object detection tasks.

        参数：
            prompt_format (str): Format string for the prompt. The format string should contain a single pair of curly
                braces {} where the text will be inserted.
            neg_samples (tuple[int, int]): A range to randomly sample negative texts. The first integer specifies the
                minimum number of negative samples, and the second integer specifies the maximum.
            max_samples (int): The maximum number of different text samples in one image.
            padding (bool): Whether to pad texts to max_samples. If True, the number of texts will always be equal to
                max_samples.
            padding_value (list[str]): The padding text to use when padding is True.
        """
        self.prompt_format = prompt_format
        self.neg_samples = neg_samples
        self.max_samples = max_samples
        self.padding = padding
        self.padding_value = padding_value

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """计算文本采样参数。

        参数：
            labels (dict[str, Any]): Input labels dictionary containing 'texts', 'cls', and 'instances'.

        返回：
            (dict): Parameters including 'valid_idx', 'new_cls', and 'texts'.
        """
        assert "texts" in labels, "No texts found in labels."
        class_texts = labels["texts"]
        num_classes = len(class_texts)
        cls = np.asarray(labels.pop("cls"), dtype=int)
        pos_labels = np.unique(cls).tolist()

        if len(pos_labels) > self.max_samples:
            pos_labels = random.sample(pos_labels, k=self.max_samples)

        neg_samples = min(min(num_classes, self.max_samples) - len(pos_labels), random.randint(*self.neg_samples))
        neg_labels = [i for i in range(num_classes) if i not in pos_labels]
        neg_labels = random.sample(neg_labels, k=neg_samples)

        sampled_labels = pos_labels + neg_labels
        # 随机打乱
        # random.shuffle(sampled_labels)

        label2ids = {label: i for i, label in enumerate(sampled_labels)}
        valid_idx = np.zeros(len(labels["instances"]), dtype=bool)
        new_cls = []
        for i, label in enumerate(cls.squeeze(-1).tolist()):
            if label not in label2ids:
                continue
            valid_idx[i] = True
            new_cls.append([label2ids[label]])

        # 当有多个提示时随机选择一个
        texts = []
        for label in sampled_labels:
            prompts = class_texts[label]
            assert len(prompts) > 0
            prompt = self.prompt_format.format(prompts[random.randrange(len(prompts))])
            texts.append(prompt)

        if self.padding:
            valid_labels = len(pos_labels) + len(neg_labels)
            num_padding = self.max_samples - valid_labels
            if num_padding > 0:
                texts += random.choices(self.padding_value, k=num_padding)

        assert len(texts) == self.max_samples

        return {"valid_idx": valid_idx, "new_cls": np.array(new_cls), "texts": texts}

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """基于采样文本过滤实例并更新类别标签。

        参数：
            labels (dict[str, Any]): Dictionary containing 'instances' and 'cls'.
            params (dict): Parameters from get_params.

        返回：
            (dict): Updated labels with filtered instances and new class/text entries.
        """
        labels["instances"] = labels["instances"][params["valid_idx"]]
        labels["cls"] = params["new_cls"]
        labels["texts"] = params["texts"]
        return labels


def v8_transforms(dataset, imgsz: int, hyp: IterableSimpleNamespace, stretch: bool = False):
    """应用一系列用于训练的图像变换。

    This function creates a composition of image augmentation techniques to prepare images for YOLO training. It
    includes operations such as mosaic, copy-paste, random perspective, mixup, and various color adjustments.

    参数：
        dataset (Dataset): The dataset object containing image data and annotations.
        imgsz (int): The target image size for resizing.
        hyp (IterableSimpleNamespace): A namespace of hyperparameters controlling various aspects of the
            transformations.
        stretch (bool): If True, applies stretching to the image. If False, uses LetterBox resizing.

    返回：
        (Compose): A composition of image transformations to be applied to the dataset.

    示例：
        >>> from ultralytics.data.dataset import YOLODataset
        >>> from ultralytics.utils import IterableSimpleNamespace
        >>> dataset = YOLODataset(img_path="path/to/images", imgsz=640)
        >>> hyp = IterableSimpleNamespace(mosaic=1.0, copy_paste=0.5, degrees=10.0, translate=0.2, scale=0.9)
        >>> transforms = v8_transforms(dataset, imgsz=640, hyp=hyp)
        >>> augmented_data = transforms(dataset[0])

        >>> # 使用自定义 albumentations
        >>> import albumentations as A
        >>> augmentations = [A.Blur(p=0.01), A.CLAHE(p=0.01)]
        >>> hyp.augmentations = augmentations
        >>> transforms = v8_transforms(dataset, imgsz=640, hyp=hyp)
    """
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
    )

    pre_transform = Compose([mosaic, affine])
    if hyp.copy_paste_mode == "flip":
        pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    else:
        pre_transform.append(
            CopyPaste(
                dataset,
                pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
                p=hyp.copy_paste,
                mode=hyp.copy_paste_mode,
            )
        )
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and (hyp.fliplr > 0.0 or hyp.flipud > 0.0):
            hyp.fliplr = hyp.flipud = 0.0  # both fliplr and flipud require flip_idx
            LOGGER.warning("No 'flip_idx' array defined in data.yaml, disabling 'fliplr' and 'flipud' augmentations.")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")

    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            CutMix(dataset, pre_transform=pre_transform, p=hyp.cutmix),
            Albumentations(p=1.0, transforms=getattr(hyp, "augmentations", None)),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud, flip_idx=flip_idx),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # transforms


# 分类数据增强 -----------------------------------------------------------------------------------------
def classify_transforms(
    size: tuple[int, int] | int = 224,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
    interpolation: str = "BILINEAR",
    crop_fraction: float | None = None,
):
    """创建用于分类任务的图像变换组合。

    This function generates a sequence of torchvision transforms suitable for preprocessing images for classification
    models during evaluation or inference. The transforms include resizing, center cropping, conversion to tensor, and
    normalization.

    参数：
        size (tuple[int, int] | int): The target size for the transformed image. If an int, it defines the shortest
            edge. If a tuple, it defines (height, width).
        mean (tuple[float, float, float]): Mean values for each RGB channel used in normalization.
        std (tuple[float, float, float]): Standard deviation values for each RGB channel used in normalization.
        interpolation (str): Interpolation method of either 'NEAREST', 'BILINEAR' or 'BICUBIC'.
        crop_fraction (float | None): Deprecated, will be removed in a future version.

    返回：
        (torchvision.transforms.Compose): A composition of torchvision transforms.

    示例：
        >>> transforms = classify_transforms(size=224)
        >>> img = Image.open("path/to/image.jpg")
        >>> transformed_img = transforms(img)
    """
    import torchvision.transforms as T  # scope for faster 'import ultralytics'

    scale_size = size if isinstance(size, (tuple, list)) and len(size) == 2 else (size, size)

    if crop_fraction:
        raise DeprecationWarning(
            "'crop_fraction' arg of classify_transforms is deprecated, will be removed in a future version."
        )

    # 保持宽高比，从图像中心裁剪，不添加边框，超出部分丢失
    if scale_size[0] == scale_size[1]:
        # 简单情况：使用 torchvision 内置的 Resize，最短边模式（标量尺寸参数）
        tfl = [T.Resize(scale_size[0], interpolation=getattr(T.InterpolationMode, interpolation))]
    else:
        # 对非正方形目标，将最短边缩放到匹配的目标尺寸
        tfl = [T.Resize(scale_size)]
    tfl += [T.CenterCrop(size), T.ToTensor(), T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))]
    return T.Compose(tfl)


# 分类训练数据增强 --------------------------------------------------------------------------------
def classify_augmentations(
    size: int = 224,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
    scale: tuple[float, float] | None = None,
    ratio: tuple[float, float] | None = None,
    hflip: float = 0.5,
    vflip: float = 0.0,
    auto_augment: str | None = None,
    hsv_h: float = 0.015,  # image HSV-Hue augmentation (fraction)
    hsv_s: float = 0.4,  # image HSV-Saturation augmentation (fraction)
    hsv_v: float = 0.4,  # image HSV-Value augmentation (fraction)
    force_color_jitter: bool = False,
    erasing: float = 0.0,
    interpolation: str = "BILINEAR",
):
    """创建用于分类任务的图像增强变换组合。

    This function generates a set of image transformations suitable for training classification models. It includes
    options for resizing, flipping, color jittering, auto augmentation, and random erasing.

    参数：
        size (int): Target size for the image after transformations.
        mean (tuple[float, float, float]): Mean values for each RGB channel used in normalization.
        std (tuple[float, float, float]): Standard deviation values for each RGB channel used in normalization.
        scale (tuple[float, float] | None): Range of the proportion of the original image area to crop.
        ratio (tuple[float, float] | None): Range of aspect ratio for the cropped area.
        hflip (float): Probability of horizontal flip.
        vflip (float): Probability of vertical flip.
        auto_augment (str | None): Auto augmentation policy. Can be 'randaugment', 'augmix', 'autoaugment' or None.
        hsv_h (float): Image HSV-Hue augmentation factor.
        hsv_s (float): Image HSV-Saturation augmentation factor.
        hsv_v (float): Image HSV-Value augmentation factor.
        force_color_jitter (bool): Whether to apply color jitter even if auto augment is enabled.
        erasing (float): Probability of random erasing.
        interpolation (str): Interpolation method of either 'NEAREST', 'BILINEAR' or 'BICUBIC'.

    返回：
        (torchvision.transforms.Compose): A composition of image augmentation transforms.

    示例：
        >>> transforms = classify_augmentations(size=224, auto_augment="randaugment")
        >>> augmented_image = transforms(original_image)
    """
    # 未安装 Albumentations 时应用的变换
    import torchvision.transforms as T  # scope for faster 'import ultralytics'

    if not isinstance(size, int):
        raise TypeError(f"classify_augmentations() size {size} must be integer, not (list, tuple)")
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
    interpolation = getattr(T.InterpolationMode, interpolation)
    primary_tfl = [T.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=interpolation)]
    if hflip > 0.0:
        primary_tfl.append(T.RandomHorizontalFlip(p=hflip))
    if vflip > 0.0:
        primary_tfl.append(T.RandomVerticalFlip(p=vflip))

    secondary_tfl = []
    disable_color_jitter = False
    if auto_augment:
        assert isinstance(auto_augment, str), f"Provided argument should be string, but got type {type(auto_augment)}"
        # 如果启用了 AutoAugment/RandAugment，通常禁用颜色抖动
        # 这样可以在不破坏旧超参数配置的情况下覆盖
        disable_color_jitter = not force_color_jitter

        if auto_augment == "randaugment":
            if TORCHVISION_0_11:
                secondary_tfl.append(T.RandAugment(interpolation=interpolation))
            else:
                LOGGER.warning('"auto_augment=randaugment" requires torchvision >= 0.11.0. Disabling it.')

        elif auto_augment == "augmix":
            if TORCHVISION_0_13:
                secondary_tfl.append(T.AugMix(interpolation=interpolation))
            else:
                LOGGER.warning('"auto_augment=augmix" requires torchvision >= 0.13.0. Disabling it.')

        elif auto_augment == "autoaugment":
            if TORCHVISION_0_10:
                secondary_tfl.append(T.AutoAugment(interpolation=interpolation))
            else:
                LOGGER.warning('"auto_augment=autoaugment" requires torchvision >= 0.10.0. Disabling it.')

        else:
            raise ValueError(
                f'Invalid auto_augment policy: {auto_augment}. Should be one of "randaugment", '
                f'"augmix", "autoaugment" or None'
            )

    if not disable_color_jitter:
        secondary_tfl.append(T.ColorJitter(brightness=hsv_v, contrast=hsv_v, saturation=hsv_s, hue=hsv_h))

    final_tfl = [
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        T.RandomErasing(p=erasing, inplace=True),
    ]

    return T.Compose(primary_tfl + secondary_tfl + final_tfl)


# 注意：保留此类以保证向后兼容
class ClassifyLetterBox:
    """用于分类任务中调整图像尺寸并填充的类。

    This class is designed to be part of a transformation pipeline, e.g., T.Compose([LetterBox(size), ToTensor()]). It
    resizes and pads images to a specified size while maintaining the original aspect ratio.

    属性：
        h (int): Target height of the image.
        w (int): Target width of the image.
        auto (bool): If True, automatically calculates the short side using stride.
        stride (int): The stride value, used when 'auto' is True.

    方法：
        __call__: Apply the letterbox transformation to an input image.

    示例：
        >>> transform = ClassifyLetterBox(size=(640, 640), auto=False, stride=32)
        >>> img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> result = transform(img)
        >>> print(result.shape)
        (640, 640, 3)
    """

    def __init__(self, size: int | tuple[int, int] = (640, 640), auto: bool = False, stride: int = 32):
        """初始化用于图像预处理的 ClassifyLetterBox 对象。

        This class is designed to be part of a transformation pipeline for image classification tasks. It resizes and
        pads images to a specified size while maintaining the original aspect ratio.

        参数：
            size (int | tuple[int, int]): Target size for the letterboxed image. If an int, a square image of (size,
                size) is created. If a tuple, it should be (height, width).
            auto (bool): If True, automatically calculates the short side based on stride.
            stride (int): The stride value, used when 'auto' is True.
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im: np.ndarray) -> np.ndarray:
        """使用 LetterBox 方法调整图像尺寸并填充。

        This method resizes the input image to fit within the specified dimensions while maintaining its aspect ratio,
        then pads the resized image to match the target size.

        参数：
            im (np.ndarray): Input image as a numpy array with shape (H, W, C).

        返回：
            (np.ndarray): Resized and padded image as a numpy array with shape (hs, ws, 3), where hs and ws are the
                target height and width respectively.

        示例：
            >>> letterbox = ClassifyLetterBox(size=(640, 640))
            >>> image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            >>> resized_image = letterbox(image)
            >>> print(resized_image.shape)
            (640, 640, 3)
        """
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old dimensions
        h, w = round(imh * r), round(imw * r)  # resized image dimensions

        # 计算填充尺寸
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else (self.h, self.w)
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)

        # 创建填充后的图像
        im_out = np.full((hs, ws, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


# 注意：保留此类以保证向后兼容
class CenterCrop:
    """对分类任务的图像应用中心裁剪。

    This class performs center cropping on input images, resizing them to a specified size while maintaining the aspect
    ratio. It is designed to be part of a transformation pipeline, e.g., T.Compose([CenterCrop(size), ToTensor()]).

    属性：
        h (int): Target height of the cropped image.
        w (int): Target width of the cropped image.

    方法：
        __call__: Apply the center crop transformation to an input image.

    示例：
        >>> transform = CenterCrop(640)
        >>> image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        >>> cropped_image = transform(image)
        >>> print(cropped_image.shape)
        (640, 640, 3)
    """

    def __init__(self, size: int | tuple[int, int] = (640, 640)):
        """初始化用于图像预处理的 CenterCrop 对象。

        This class is designed to be part of a transformation pipeline, e.g., T.Compose([CenterCrop(size), ToTensor()]).
        It performs a center crop on input images to a specified size.

        参数：
            size (int | tuple[int, int]): The desired output size of the crop. If size is an int, a square crop (size,
                size) is made. If size is a sequence like (h, w), it is used as the output size.
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im: Image.Image | np.ndarray) -> np.ndarray:
        """对输入图像应用中心裁剪。

        This method crops the largest centered square from the image and resizes it to the specified dimensions.

        参数：
            im (np.ndarray | PIL.Image.Image): The input image as a numpy array of shape (H, W, C) or a PIL Image
                object.

        返回：
            (np.ndarray): The center-cropped and resized image as a numpy array of shape (self.h, self.w, C).

        示例：
            >>> transform = CenterCrop(size=224)
            >>> image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            >>> cropped_image = transform(image)
            >>> assert cropped_image.shape == (224, 224, 3)
        """
        if isinstance(im, Image.Image):  # convert from PIL to numpy array if required
            im = np.asarray(im)
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top : top + m, left : left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


# 注意：保留此类以保证向后兼容
class ToTensor:
    """将图像从 numpy 数组转换为 PyTorch 张量。

    This class is designed to be part of a transformation pipeline, e.g., T.Compose([LetterBox(size), ToTensor()]).

    属性：
        half (bool): If True, converts the image to half precision (float16).

    方法：
        __call__: Apply the tensor conversion to an input image.

    示例：
        >>> transform = ToTensor(half=True)
        >>> img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        >>> tensor_img = transform(img)
        >>> print(tensor_img.shape, tensor_img.dtype)
        torch.Size([3, 640, 640]) torch.float16

    注意：
        The input image is expected to be in BGR format with shape (H, W, C).
        The output tensor will be in BGR format with shape (C, H, W), normalized to [0, 1].
    """

    def __init__(self, half: bool = False):
        """初始化 ToTensor 对象，用于将图像转换为 PyTorch 张量。

        This class is designed to be used as part of a transformation pipeline for image preprocessing in the
        Ultralytics YOLO framework. It converts numpy arrays or PIL Images to PyTorch tensors, with an option for
        half-precision (float16) conversion.

        参数：
            half (bool): If True, converts the tensor to half precision (float16).
        """
        super().__init__()
        self.half = half

    def __call__(self, im: np.ndarray) -> torch.Tensor:
        """将图像从 numpy 数组转换为 PyTorch 张量。

        This method converts the input image from a numpy array to a PyTorch tensor, applying optional half-precision
        conversion and normalization. The image is transposed from HWC to CHW format.

        参数：
            im (np.ndarray): Input image as a numpy array with shape (H, W, C) in BGR order.

        返回：
            (torch.Tensor): The transformed image as a PyTorch tensor in float32 or float16, normalized to [0, 1] with
                shape (C, H, W) in BGR order.

        示例：
            >>> transform = ToTensor(half=True)
            >>> img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            >>> tensor_img = transform(img)
            >>> print(tensor_img.shape, tensor_img.dtype)
            torch.Size([3, 640, 640]) torch.float16
        """
        im = np.ascontiguousarray(im.transpose((2, 0, 1)))  # HWC 转 CHW -> 连续内存
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im
