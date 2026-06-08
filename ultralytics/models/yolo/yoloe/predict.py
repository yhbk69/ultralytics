# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np
import torch

from ultralytics.data.augment import LoadVisualPrompt
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.models.yolo.segment import SegmentationPredictor


class YOLOEVPDetectPredictor(DetectionPredictor):
    """继承 DetectionPredictor 用于 YOLO-EVP (Enhanced Visual Prompting) 预测的类。

    该类为使用视觉提示的 YOLO 模型提供通用功能，包括模型设置、提示处理和
    预处理变换。

    Attributes:
        model (torch.nn.Module): 用于推理的 YOLO 模型。
        device (torch.device): 运行模型的设备（CPU 或 CUDA）。
        prompts (dict | torch.Tensor): 包含类别索引和边界框或掩码的视觉提示。

    Methods:
        setup_model: 初始化 YOLO 模型并设置为评估模式。
        set_prompts: 为模型设置视觉提示。
        pre_transform: 推理前预处理图像和提示。
        inference: 使用视觉提示运行推理。
        get_vpe: 处理源数据以获取视觉提示嵌入。
    """

    def setup_model(self, model, verbose: bool = True):
        """设置用于预测的模型。

        Args:
            model (torch.nn.Module): 要加载或使用的模型。
            verbose (bool, optional): 如果为 True，提供详细日志。
        """
        super().setup_model(model, verbose=verbose)
        self.done_warmup = True

    def set_prompts(self, prompts):
        """为模型设置视觉提示。

        Args:
            prompts (dict): 包含类别索引和边界框或掩码的字典。必须包含带有类别索引的 'cls' 键。
        """
        self.prompts = prompts

    def pre_transform(self, im):
        """推理前预处理图像和提示。

        该方法对输入图像应用 letterbox 并相应地转换视觉提示（边界框或掩码）。

        Args:
            im (list): 输入图像列表。

        Returns:
            (list): 准备好用于模型推理的预处理图像。

        Raises:
            ValueError: 如果提示中既没有提供有效的边界框也没有提供掩码。
        """
        img = super().pre_transform(im)
        bboxes = self.prompts.pop("bboxes", None)
        masks = self.prompts.pop("masks", None)
        category = self.prompts["cls"]
        if len(img) == 1:
            visuals = self._process_single_image(img[0].shape[:2], im[0].shape[:2], category, bboxes, masks)
            prompts = visuals.unsqueeze(0).to(self.device)  # (1, N, H, W)
        else:
            # 注意：目前仅支持边界框作为提示
            assert bboxes is not None, f"Expected bboxes, but got {bboxes}!"
            # 注意：需要 list[np.ndarray]
            assert isinstance(bboxes, list) and all(isinstance(b, np.ndarray) for b in bboxes), (
                f"Expected list[np.ndarray], but got {bboxes}!"
            )
            assert isinstance(category, list) and all(isinstance(b, np.ndarray) for b in category), (
                f"Expected list[np.ndarray], but got {category}!"
            )
            assert len(im) == len(category) == len(bboxes), (
                f"Expected same length for all inputs, but got {len(im)}vs{len(category)}vs{len(bboxes)}!"
            )
            visuals = [
                self._process_single_image(img[i].shape[:2], im[i].shape[:2], category[i], bboxes[i])
                for i in range(len(img))
            ]
            prompts = torch.nn.utils.rnn.pad_sequence(visuals, batch_first=True).to(self.device)  # (B, N, H, W)
        self.prompts = prompts.half() if self.model.fp16 else prompts.float()
        return img

    def _process_single_image(self, dst_shape, src_shape, category, bboxes=None, masks=None):
        """处理单张图像，调整边界框或掩码大小并生成视觉提示。

        Args:
            dst_shape (tuple): 图像的目标形状 (高度, 宽度)。
            src_shape (tuple): 图像的原始形状 (高度, 宽度)。
            category (list | np.ndarray): 视觉提示的类别索引。
            bboxes (list | np.ndarray, optional): 边界框列表，格式为 [x1, y1, x2, y2]。
            masks (np.ndarray, optional): 与图像对应的掩码列表。

        Returns:
            (torch.Tensor): 图像的处理后视觉提示。

        Raises:
            ValueError: 如果既未提供 `bboxes` 也未提供 `masks`。
        """
        if bboxes is not None and len(bboxes):
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.ndim == 1:
                bboxes = bboxes[None, :]
            # 计算缩放因子并调整边界框
            gain = min(dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1])  # 缩放因子 = 旧尺寸 / 新尺寸
            bboxes *= gain
            bboxes[..., 0::2] += round((dst_shape[1] - round(src_shape[1] * gain)) / 2 - 0.1)
            bboxes[..., 1::2] += round((dst_shape[0] - round(src_shape[0] * gain)) / 2 - 0.1)
        elif masks is not None:
            # 调整掩码大小并处理
            resized_masks = super().pre_transform(masks)
            masks = np.stack(resized_masks)  # (N, H, W)
            masks[masks == 114] = 0  # 将填充值重置为 0
        else:
            raise ValueError("Please provide valid bboxes or masks")

        # 使用视觉提示加载器生成视觉提示
        return LoadVisualPrompt().get_visuals(category, dst_shape, bboxes, masks)

    def inference(self, im, *args, **kwargs):
        """使用视觉提示运行推理。

        Args:
            im (torch.Tensor): 输入图像张量。
            *args (Any): 可变长度参数列表。
            **kwargs (Any): 任意关键字参数。

        Returns:
            (torch.Tensor): 模型预测结果。
        """
        return super().inference(im, vpe=self.prompts, *args, **kwargs)

    def get_vpe(self, source):
        """处理源数据以获取视觉提示嵌入 (VPE)。

        Args:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | list | tuple): 要进行预测的
                图像源。接受多种类型，包括文件路径、URL、PIL 图像、numpy 数组和 torch 张量。

        Returns:
            (torch.Tensor): 模型的视觉提示嵌入 (VPE)。
        """
        self.setup_source(source)
        assert len(self.dataset) == 1, "get_vpe only supports one image!"
        for _, im0s, _ in self.dataset:
            im = self.preprocess(im0s)
            return self.model(im, vpe=self.prompts, return_vpe=True)


class YOLOEVPSegPredictor(YOLOEVPDetectPredictor, SegmentationPredictor):
    """用于 YOLO-EVP 分割任务的预测器，结合了检测和分割能力。"""

    pass
