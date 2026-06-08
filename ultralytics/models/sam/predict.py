# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
使用Segment Anything Model（SAM）生成预测。

SAM是一种先进的图像分割模型，提供可提示分割和零样本性能等功能。
此模块包含使用SAM执行分割所需的预测逻辑和辅助工具的实现。
它是Ultralytics框架的组成部分，专为高性能、实时图像分割任务设计。
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops
from ultralytics.utils.metrics import box_iou, mask_iou
from ultralytics.utils.torch_utils import select_device, smart_inference_mode

from .amg import (
    batch_iterator,
    batched_mask_to_box,
    build_all_layer_point_grids,
    calculate_stability_score,
    generate_crop_boxes,
    is_box_near_crop_edge,
    remove_small_regions,
    uncrop_boxes_xyxy,
    uncrop_masks,
)
from .sam3.geometry_encoders import Prompt


class Predictor(BasePredictor):
    """SAM的Predictor类，支持实时图像分割，具有可提示能力。

    此类扩展BasePredictor并实现Segment Anything Model（SAM），用于高级图像分割任务。它支持各种输入提示，
    如点、边界框和掩码，以实现对分割结果的精细控制。

    Attributes:
        args (SimpleNamespace): 预测器的配置参数。
        model (torch.nn.Module): 加载的SAM模型。
        device (torch.device): 模型加载的设备（CPU或GPU）。
        im (torch.Tensor): 预处理后的输入图像。
        features (torch.Tensor): 提取的图像特征。
        prompts (dict[str, Any]): 存储各种提示类型的字典（如边界框、点、掩码）。
        segment_all (bool): 指示是否应执行全图像分割的标志。
        mean (torch.Tensor): 图像归一化的均值。
        std (torch.Tensor): 图像归一化的标准差。

    Methods:
        preprocess: 为模型推理准备输入图像。
        pre_transform: 对输入图像执行初始变换。
        inference: 基于输入提示执行分割推理。
        prompt_inference: 基于提示的分割推理内部函数。
        generate: 为整个图像生成分割掩码。
        setup_model: 初始化SAM模型进行推理。
        get_model: 构建并返回SAM模型。
        postprocess: 后处理模型输出以生成最终结果。
        setup_source: 设置推理的数据源。
        set_image: 设置并预处理单张图像用于推理。
        get_im_features: 使用SAM图像编码器提取图像特征。
        set_prompts: 为后续推理设置提示。
        reset_image: 重置当前图像及其特征。
        remove_small_regions: 从掩码中移除小的不连通区域和孔洞。

    Examples:
        >>> predictor = Predictor()
        >>> predictor.setup_model(model_path="sam_model.pt")
        >>> predictor.set_image("image.jpg")
        >>> bboxes = [[100, 100, 200, 200]]
        >>> results = predictor(bboxes=bboxes)
    """

    stride = 16

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """使用配置、覆盖和回调初始化Predictor。

        设置SAM（Segment Anything Model）的Predictor对象，应用任何配置覆盖或提供的回调。
        初始化SAM的任务特定设置，如将retina_masks设为True以获得最佳结果。

        Args:
            cfg (dict): 包含默认设置的配置字典。
            overrides (dict | None): 覆盖默认配置的字典。
            _callbacks (dict | None): 自定义行为的回调函数字典。
        """
        if overrides is None:
            overrides = {}
        overrides.update(dict(task="segment", mode="predict", batch=1))
        super().__init__(cfg, overrides, _callbacks)
        self.args.retina_masks = True
        self.im = None
        self.features = None
        self.prompts = {}
        self.segment_all = False

    def preprocess(self, im):
        """为模型推理预处理输入图像。

        此方法通过应用变换和归一化来准备输入图像。支持torch.Tensor和np.ndarray列表作为输入格式。
        对于OpenCV加载的图像，输入通常是BGR格式，并在预处理期间转换为RGB。

        Args:
            im (torch.Tensor | list[np.ndarray]): BCHW张量格式的输入图像或HWC NumPy数组列表。
                NumPy数组应为BGR顺序（OpenCV返回的格式），并将转换为RGB。

        Returns:
            (torch.Tensor): 预处理后的图像张量，已归一化并转换为适当的数据类型。

        Examples:
            >>> predictor = Predictor()
            >>> image = torch.rand(1, 3, 640, 640)
            >>> preprocessed_image = predictor.preprocess(image)
        """
        if self.im is not None:
            return self.im
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im)

        im = im.to(self.device)
        if not_tensor:
            im = (im - self.mean) / self.std
        im = im.half() if self.model.fp16 else im.float()
        return im

    def pre_transform(self, im):
        """对输入图像执行初始变换以进行预处理。

        此方法应用如调整大小等变换来准备图像。目前不支持批量推理，因此列表长度应为1。

        Args:
            im (list[np.ndarray]): 包含单张HWC numpy数组格式图像的列表。

        Returns:
            (list[np.ndarray]): 包含变换后图像的列表。

        Raises:
            AssertionError: 如果输入列表包含多于一张图像。

        Examples:
            >>> predictor = Predictor()
            >>> image = np.random.rand(480, 640, 3)  # 单张HWC图像
            >>> transformed = predictor.pre_transform([image])
            >>> print(len(transformed))
            1
        """
        assert len(im) == 1, "SAM model does not currently support batched inference"
        letterbox = LetterBox(self.imgsz, auto=False, center=False)
        return [letterbox(image=x) for x in im]

    def inference(self, im, bboxes=None, points=None, labels=None, masks=None, multimask_output=False, *args, **kwargs):
        """执行基于输入提示的图像分割推理，使用当前加载的图像。

        此方法利用SAM（Segment Anything Model）的架构，包含图像编码器、提示编码器和掩码解码器，
        用于实时和可提示的分割任务。

        Args:
            im (torch.Tensor): 预处理后的输入图像张量，形状为(N, C, H, W)。
            bboxes (np.ndarray | list | None): XYXY格式的边界框，形状为(N, 4)。
            points (np.ndarray | list | None): 指示对象位置的点，形状为(N, 2)，单位为像素。
            labels (np.ndarray | list | None): 点提示的标签，形状为(N,)。1 = 前景，0 = 背景。
            masks (np.ndarray | None): 先前预测的低分辨率掩码，形状为(N, H, W)。对于SAM，H=W=256。
            multimask_output (bool): 是否返回多个掩码的标志。对模糊提示有帮助。
            *args (Any): 额外的位置参数。
            **kwargs (Any): 额外的关键字参数。

        Returns:
            pred_masks (torch.Tensor): 输出掩码，形状为(C, H, W)，其中C是生成的掩码数量。
            pred_scores (torch.Tensor): 长度为C的数组，包含模型为每个掩码预测的质量分数。

        Examples:
            >>> predictor = Predictor()
            >>> predictor.setup_model(model_path="sam_model.pt")
            >>> predictor.set_image("image.jpg")
            >>> results = predictor(bboxes=[[0, 0, 100, 100]])
        """
        # 如果self.prompts中存储了提示则覆盖
        bboxes = self.prompts.pop("bboxes", bboxes)
        points = self.prompts.pop("points", points)
        masks = self.prompts.pop("masks", masks)
        labels = self.prompts.pop("labels", labels)

        if all(i is None for i in [bboxes, points, masks]):
            return self.generate(im, *args, **kwargs)

        return self.prompt_inference(im, bboxes, points, labels, masks, multimask_output)

    def prompt_inference(self, im, bboxes=None, points=None, labels=None, masks=None, multimask_output=False):
        """使用SAM的专门架构基于输入提示执行图像分割推理。

        此内部函数利用Segment Anything Model（SAM）进行基于提示的实时分割。
        它处理各种输入提示，如边界框、点和掩码，以生成分割掩码。

        Args:
            im (torch.Tensor): 预处理后的输入图像张量，形状为(N, C, H, W)。
            bboxes (np.ndarray | list | None): XYXY格式的边界框，形状为(N, 4)。
            points (np.ndarray | list | None): 指示对象位置的点，形状为(N, 2)或(N, num_points, 2)，单位为像素。
            labels (np.ndarray | list | None): 点提示标签，形状为(N)或(N, num_points)。1表示前景，0表示背景。
            masks (np.ndarray | None): 先前预测的低分辨率掩码，形状为(N, H, W)。对于SAM，H=W=256。
            multimask_output (bool): 是否返回多个掩码的标志，有助于处理模糊提示。

        Returns:
            pred_masks (torch.Tensor): 输出掩码，形状为(C, H, W)，其中C是生成的掩码数量。
            pred_scores (torch.Tensor): 模型为每个掩码预测的质量分数，长度为C。

        Examples:
            >>> predictor = Predictor()
            >>> im = torch.rand(1, 3, 1024, 1024)
            >>> bboxes = [[100, 100, 200, 200]]
            >>> masks, scores, logits = predictor.prompt_inference(im, bboxes=bboxes)
        """
        features = self.get_im_features(im) if self.features is None else self.features

        prompts = self._prepare_prompts(im.shape[2:], self.batch[1][0].shape[:2], bboxes, points, labels, masks)
        return self._inference_features(features, *prompts, multimask_output)

    def _inference_features(
        self,
        features,
        bboxes=None,
        points=None,
        labels=None,
        masks=None,
        multimask_output=False,
    ):
        """使用SAM模型对图像特征执行推理。

        Args:
            features (torch.Tensor): 从SAM模型图像编码器提取的图像特征，形状为(B, C, H, W)。
            bboxes (np.ndarray | list[list[float]] | None): XYXY格式的边界框，形状为(N, 4)。
            points (np.ndarray | list[list[float]] | None): 对象位置点，形状为(N, 2)，单位为像素。
            labels (np.ndarray | list[int] | None): 点提示标签，形状为(N,)。1 = 前景，0 = 背景。
            masks (list[np.ndarray] | np.ndarray | None): 对象的掩码，每个掩码为2D数组。
            multimask_output (bool): 是否返回多个掩码的标志，有助于处理模糊提示。

        Returns:
            pred_masks (torch.Tensor): 输出掩码，形状为(C, H, W)，其中C是生成的掩码数量。
            pred_scores (torch.Tensor): 每个掩码的质量分数，长度为C。
        """
        points = (points, labels) if points is not None else None
        # 嵌入提示
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=points, boxes=bboxes, masks=masks)

        # 预测掩码
        pred_masks, pred_scores = self.model.mask_decoder(
            image_embeddings=features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # (N, d, H, W) --> (N*d, H, W), (N, d) --> (N*d, )
        # `d` 可以是 1 或 3，取决于 `multimask_output`。
        return pred_masks.flatten(0, 1), pred_scores.flatten(0, 1)

    def _prepare_prompts(self, dst_shape, src_shape, bboxes=None, points=None, labels=None, masks=None):
        """准备并变换输入提示以基于目标形状进行处理。

        Args:
            dst_shape (tuple[int, int]): 提示的目标形状（高度，宽度）。
            src_shape (tuple[int, int]): 输入图像的源形状（高度，宽度）。
            bboxes (np.ndarray | list | None): XYXY格式的边界框，形状为(N, 4)。
            points (np.ndarray | list | None): 指示对象位置的点，形状为(N, 2)或(N, num_points, 2)，单位为像素。
            labels (np.ndarray | list | None): 点提示标签，形状为(N)或(N, num_points)。1表示前景，0表示背景。
            masks (list[np.ndarray] | np.ndarray | None): 对象的掩码，每个掩码为形状(H, W)的2D数组。

        Returns:
            bboxes (torch.Tensor | None): 变换后的边界框。
            points (torch.Tensor | None): 变换后的点。
            labels (torch.Tensor | None): 变换后的标签。
            masks (torch.Tensor | None): 变换后的掩码。

        Raises:
            AssertionError: 如果传入了标签但点数量与标签数量不匹配。
        """
        r = 1.0 if self.segment_all else min(dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1])
        # 变换输入提示
        if points is not None:
            points = torch.as_tensor(points, dtype=self.torch_dtype, device=self.device)
            points = points[None] if points.ndim == 1 else points
            # 如果用户未传入标签，默认所有标签为正
            if labels is None:
                labels = np.ones(points.shape[:-1])
            labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
            assert points.shape[-2] == labels.shape[-1], (
                f"Number of points {points.shape[-2]} should match number of labels {labels.shape[-1]}."
            )
            points *= r
            if points.ndim == 2:
                # (N, 2) --> (N, 1, 2), (N, ) --> (N, 1)
                points, labels = points[:, None, :], labels[:, None]
        if bboxes is not None:
            bboxes = torch.as_tensor(bboxes, dtype=self.torch_dtype, device=self.device)
            bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
            bboxes *= r
        if masks is not None:
            masks = np.asarray(masks, dtype=np.uint8)
            masks = masks[None] if masks.ndim == 2 else masks
            letterbox = LetterBox(dst_shape, auto=False, center=False, padding_value=0, interpolation=cv2.INTER_NEAREST)
            masks = np.stack([letterbox(image=x).squeeze() for x in masks], axis=0)
            masks = torch.tensor(masks, dtype=self.torch_dtype, device=self.device)
        return bboxes, points, labels, masks

    def generate(
        self,
        im,
        crop_n_layers=0,
        crop_overlap_ratio=512 / 1500,
        crop_downscale_factor=1,
        point_grids=None,
        points_stride=32,
        points_batch_size=64,
        conf_thres=0.88,
        stability_score_thresh=0.95,
        stability_score_offset=0.95,
        crop_nms_thresh=0.7,
    ):
        """使用Segment Anything Model（SAM）执行图像分割。

        此方法利用SAM的先进架构和实时性能，将整张图像分割为组成部分。
        可以选择在图像裁剪上进行操作以实现更精细的分割。

        Args:
            im (torch.Tensor): 预处理后的输入图像张量，形状为(N, C, H, W)。
            crop_n_layers (int): 用于在图像裁剪上生成额外掩码预测的层数。
            crop_overlap_ratio (float): 裁剪之间的重叠比例，在后续层中缩放递减。
            crop_downscale_factor (int): 每层中每边采样点数的缩放因子。
            point_grids (list[np.ndarray] | None): 用于点采样的自定义网格，归一化到[0,1]。
            points_stride (int): 沿图像每边采样的点数。
            points_batch_size (int): 同时处理的点的批量大小。
            conf_thres (float): 基于掩码质量预测进行过滤的置信度阈值[0,1]。
            stability_score_thresh (float): 基于稳定性进行掩码过滤的稳定性阈值[0,1]。
            stability_score_offset (float): 计算稳定性分数的偏移值。
            crop_nms_thresh (float): 用于NMS去除裁剪间重复掩码的IoU阈值。

        Returns:
            pred_masks (torch.Tensor): 分割掩码，形状为(N, H, W)。
            pred_scores (torch.Tensor): 每个掩码的置信度分数，形状为(N,)。
            pred_bboxes (torch.Tensor): 每个掩码的边界框，形状为(N, 4)。

        Examples:
            >>> predictor = Predictor()
            >>> im = torch.rand(1, 3, 1024, 1024)  # 示例输入图像
            >>> masks, scores, boxes = predictor.generate(im)
        """
        import torchvision  # 限制作用域以加快 'import ultralytics' 速度

        self.segment_all = True
        ih, iw = im.shape[2:]
        crop_regions, layer_idxs = generate_crop_boxes((ih, iw), crop_n_layers, crop_overlap_ratio)
        if point_grids is None:
            point_grids = build_all_layer_point_grids(points_stride, crop_n_layers, crop_downscale_factor)
        pred_masks, pred_scores, pred_bboxes, region_areas = [], [], [], []
        for crop_region, layer_idx in zip(crop_regions, layer_idxs):
            x1, y1, x2, y2 = crop_region
            w, h = x2 - x1, y2 - y1
            area = torch.tensor(w * h, device=im.device)
            points_scale = np.array([[w, h]])  # 宽, 高
            # 裁剪图像并插值到输入尺寸
            crop_im = F.interpolate(im[..., y1:y2, x1:x2], (ih, iw), mode="bilinear", align_corners=False)
            # (num_points, 2)
            points_for_image = point_grids[layer_idx] * points_scale
            crop_masks, crop_scores, crop_bboxes = [], [], []
            for (points,) in batch_iterator(points_batch_size, points_for_image):
                pred_mask, pred_score = self.prompt_inference(crop_im, points=points, multimask_output=True)
                # 将预测的掩码插值到输入尺寸
                pred_mask = F.interpolate(pred_mask[None], (h, w), mode="bilinear", align_corners=False)[0]
                idx = pred_score > conf_thres
                pred_mask, pred_score = pred_mask[idx], pred_score[idx]

                stability_score = calculate_stability_score(
                    pred_mask, self.model.mask_threshold, stability_score_offset
                )
                idx = stability_score > stability_score_thresh
                pred_mask, pred_score = pred_mask[idx], pred_score[idx]
                # 布尔类型更节省内存。
                pred_mask = pred_mask > self.model.mask_threshold
                # (N, 4)
                pred_bbox = batched_mask_to_box(pred_mask).float()
                keep_mask = ~is_box_near_crop_edge(pred_bbox, crop_region, [0, 0, iw, ih])
                if not torch.all(keep_mask):
                    pred_bbox, pred_mask, pred_score = pred_bbox[keep_mask], pred_mask[keep_mask], pred_score[keep_mask]

                crop_masks.append(pred_mask)
                crop_bboxes.append(pred_bbox)
                crop_scores.append(pred_score)

            # 在此裁剪区域内执行NMS
            crop_masks = torch.cat(crop_masks)
            crop_bboxes = torch.cat(crop_bboxes)
            crop_scores = torch.cat(crop_scores)
            keep = torchvision.ops.nms(crop_bboxes, crop_scores, self.args.iou)  # 非极大值抑制
            crop_bboxes = uncrop_boxes_xyxy(crop_bboxes[keep], crop_region)
            crop_masks = uncrop_masks(crop_masks[keep], crop_region, ih, iw)
            crop_scores = crop_scores[keep]

            pred_masks.append(crop_masks)
            pred_bboxes.append(crop_bboxes)
            pred_scores.append(crop_scores)
            region_areas.append(area.expand(crop_masks.shape[0]))

        pred_masks = torch.cat(pred_masks)
        pred_bboxes = torch.cat(pred_bboxes)
        pred_scores = torch.cat(pred_scores)
        region_areas = torch.cat(region_areas)

        # 移除裁剪区域之间的重复掩码
        if len(crop_regions) > 1:
            scores = 1 / region_areas
            keep = torchvision.ops.nms(pred_bboxes, scores, crop_nms_thresh)
            pred_masks, pred_bboxes, pred_scores = pred_masks[keep], pred_bboxes[keep], pred_scores[keep]

        return pred_masks, pred_scores, pred_bboxes

    def setup_model(self, model=None, verbose=True):
        """初始化Segment Anything Model（SAM）用于推理。

        此方法通过将SAM模型分配到适当的设备并初始化图像归一化参数及其他Ultralytics兼容性设置来配置SAM模型。

        Args:
            model (torch.nn.Module | None): 预训练的SAM模型。如果为None，则基于配置构建新模型。
            verbose (bool): 如果为True，打印所选设备信息。

        Examples:
            >>> predictor = Predictor()
            >>> predictor.setup_model(model=sam_model, verbose=True)
        """
        device = select_device(self.args.device, verbose=verbose)
        if model is None:
            model = self.get_model()
        # 先将模型移至设备，然后转换数据类型，再设置为eval模式，使评估时的缓存按设备创建。
        model = model.to(device)
        model = model.half() if self.args.half else model.float()
        model.eval()
        self.model = model
        self.device = device
        self.mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device)
        self.std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device)

        # Ultralytics 兼容性设置
        self.model.format = "sam"
        self.model.stride = 32
        self.model.fp16 = self.args.half
        self.done_warmup = True
        self.torch_dtype = torch.float16 if self.model.fp16 else torch.float32

    def get_model(self):
        """检索或构建Segment Anything Model（SAM）用于图像分割任务。"""
        from .build import build_sam  # 延迟导入

        return build_sam(self.args.model)

    def postprocess(self, preds, img, orig_imgs):
        """后处理SAM的推理输出以生成目标检测掩码和边界框。

        此方法将掩码和边框缩放到原始图像尺寸，并对掩码预测应用阈值过滤。
        它利用SAM的高级架构实现实时、可提示的分割任务。

        Args:
            preds (tuple): SAM模型推理的输出，包含：
                - pred_masks (torch.Tensor): 预测的掩码，形状为(N, 1, H, W)。
                - pred_scores (torch.Tensor): 每个掩码的置信度分数，形状为(N, 1)。
                - pred_bboxes (torch.Tensor, optional): 如果segment_all为True时的预测边界框。
            img (torch.Tensor): 处理后的输入图像张量，形状为(C, H, W)。
            orig_imgs (list[np.ndarray] | torch.Tensor): 原始的未处理图像。

        Returns:
            (list[Results]): Results对象列表，包含每张处理后图像的检测掩码、边界框和其他元数据。

        Examples:
            >>> predictor = Predictor()
            >>> preds = predictor.inference(img)
            >>> results = predictor.postprocess(preds, img, orig_imgs)
        """
        # (N, 1, H, W), (N, 1)
        pred_masks, pred_scores = preds[:2]
        pred_bboxes = preds[2] if self.segment_all else None
        names = dict(enumerate(str(i) for i in range(pred_masks.shape[0])))

        if not isinstance(orig_imgs, list):  # 输入图像是torch.Tensor，不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = []
        for masks, orig_img, img_path in zip([pred_masks], orig_imgs, self.batch[0]):
            if masks.shape[0] == 0:
                masks, pred_bboxes = None, torch.zeros((0, 6), device=pred_masks.device)
            else:
                masks = ops.scale_masks(masks[None].float(), orig_img.shape[:2], padding=False)[0]
                masks = masks > self.model.mask_threshold  # 转换为布尔类型
                if pred_bboxes is not None:
                    pred_bboxes = ops.scale_boxes(img.shape[2:], pred_bboxes.float(), orig_img.shape, padding=False)
                else:
                    pred_bboxes = batched_mask_to_box(masks)
                # 注意：SAM模型不返回类别信息。此处的`cls`仅为一致性占位符。
                cls = torch.arange(pred_masks.shape[0], dtype=torch.int32, device=pred_masks.device)
                idx = pred_scores > self.args.conf
                pred_bboxes = torch.cat([pred_bboxes, pred_scores[:, None], cls[:, None]], dim=-1)[idx]
                masks = masks[idx]
            results.append(Results(orig_img, path=img_path, names=names, masks=masks, boxes=pred_bboxes))
        # 重置"全部段"模式。
        self.segment_all = False
        return results

    def set_image(self, image):
        """预处理并设置单张图像用于推理。

        此方法通过以下步骤准备模型对单张图像的推理：设置模型（如果尚未初始化）、配置数据源以及
        预处理图像以进行特征提取。它确保一次只设置一张图像，并提取图像特征供后续使用。

        Args:
            image (str | np.ndarray): 图像文件的路径字符串，或表示由cv2读取的图像（BGR通道顺序）的numpy数组。

        Raises:
            AssertionError: 如果尝试设置超过一张图像。

        Examples:
            >>> predictor = Predictor()
            >>> predictor.set_image("path/to/image.jpg")
            >>> predictor.set_image(cv2.imread("path/to/image.jpg"))

        Notes:
            - 此方法应在对新图像执行推理之前调用。
            - 提取的特征存储在`self.features`属性中供后续使用。
        """
        if self.model is None:
            self.setup_model()
        self.setup_source(image)
        assert len(self.dataset) == 1, "`set_image` only supports setting one image!"
        for batch in self.dataset:
            im = self.preprocess(batch[1])
            self.features = self.get_im_features(im)
            break

    def setup_source(self, source):
        """设置SAM推理的数据源。"""
        if source is None:  # 处理提前调用set_imgsz的情况
            return
        super().setup_source(source, self.stride)
        assert isinstance(self.imgsz, (tuple, list)) and self.imgsz[0] == self.imgsz[1], (
            f"SAM models only support square image size, but got {self.imgsz}."
        )
        self.model.set_imgsz(self.imgsz)

    def get_im_features(self, im):
        """使用SAM模型的图像编码器提取图像特征，用于后续掩码预测。"""
        return self.model.image_encoder(im)

    def set_prompts(self, prompts):
        """设置提示以用于后续推理操作。"""
        self.prompts = prompts

    def reset_image(self):
        """重置当前图像及其特征，清除以便进行后续推理。"""
        self.im = None
        self.features = None

    @staticmethod
    def remove_small_regions(masks, min_area=0, nms_thresh=0.7):
        """从分割掩码中移除小的不连通区域和孔洞。

        此函数对Segment Anything Model（SAM）生成的分割掩码进行后处理。它从输入掩码中
        移除小的不连通区域和孔洞，然后执行非极大值抑制（NMS）以消除任何新产生的重复框。

        Args:
            masks (torch.Tensor): 待处理的分割掩码，形状为(N, H, W)，其中N是掩码数量，H是高度，W是宽度。
            min_area (int): 移除不连通区域和孔洞的最小面积阈值。小于此阈值的区域将被移除。
            nms_thresh (float): NMS算法中用于移除重复框的IoU阈值。

        Returns:
            new_masks (torch.Tensor): 处理后移除了小区域的掩码，形状为(N, H, W)。
            keep (list[int]): NMS后保留掩码的索引，用于过滤对应的框。

        Examples:
            >>> masks = torch.rand(5, 640, 640) > 0.5  # 5个随机二值掩码
            >>> new_masks, keep = remove_small_regions(masks, min_area=100, nms_thresh=0.7)
            >>> print(f"原始掩码: {masks.shape}, 处理后掩码: {new_masks.shape}")
            >>> print(f"保留掩码的索引: {keep}")
        """
        import torchvision  # 限制作用域以加快 'import ultralytics' 速度

        if masks.shape[0] == 0:
            return masks

        # 过滤小的不连通区域和孔洞
        new_masks = []
        scores = []
        for mask in masks:
            mask = mask.cpu().numpy().astype(np.uint8)
            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # 给被修改过的掩码打0分，未修改的打1分，以便NMS优先选择不需要后处理的掩码
            scores.append(float(unchanged))

        # 重新计算框并移除任何新的重复项
        new_masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(new_masks)
        keep = torchvision.ops.nms(boxes.float(), torch.as_tensor(scores), nms_thresh)

        return new_masks[keep].to(device=masks.device, dtype=masks.dtype), keep

    @smart_inference_mode()
    def inference_features(
        self,
        features,
        src_shape,
        dst_shape=None,
        bboxes=None,
        points=None,
        labels=None,
        masks=None,
        multimask_output=False,
    ):
        """对提供的图像特征执行提示预处理和推理，使用SAM模型。

        Args:
            features (torch.Tensor | dict[str, Any]): 从SAM/SAM2模型图像编码器提取的图像特征。
            src_shape (tuple[int, int]): 输入图像的源形状（高度，宽度）。
            dst_shape (tuple[int, int] | None): 提示的目标形状（高度，宽度）。如果为None，默认为(imgsz, imgsz)。
            bboxes (np.ndarray | list[list[float]] | None): xyxy格式的边界框，形状为(N, 4)。
            points (np.ndarray | list[list[float]] | None): 指示对象位置的点，形状为(N, 2)，单位为像素。
            labels (np.ndarray | list[int] | None): 点提示标签，形状为(N,)。
            masks (list[np.ndarray] | np.ndarray | None): 对象的掩码，每个掩码为2D数组。
            multimask_output (bool): 是否返回多个掩码的标志，有助于处理模糊提示。

        Returns:
            pred_masks (torch.Tensor): 输出掩码，形状为(C, H, W)，其中C是生成的掩码数量。
            pred_bboxes (torch.Tensor): 每个掩码的边界框，形状为(N, 6)，其中N是框的数量。
                每个框为xyxy格式，附带有分数和类别的额外列。

        Notes:
            - 输入特征在SAM上为形状(B, C, H, W)的torch.Tensor，在SAM2上为dict[str, Any]。
        """
        dst_shape = dst_shape or (self.args.imgsz, self.args.imgsz)
        prompts = self._prepare_prompts(dst_shape, src_shape, bboxes, points, labels, masks)
        pred_masks, pred_scores = self._inference_features(features, *prompts, multimask_output)
        if pred_masks.shape[0] == 0:
            pred_masks, pred_bboxes = None, torch.zeros((0, 6), device=pred_masks.device)
        else:
            pred_masks = ops.scale_masks(pred_masks[None].float(), src_shape, padding=False)[0]
            pred_masks = pred_masks > self.model.mask_threshold  # 转换为布尔类型
            pred_bboxes = batched_mask_to_box(pred_masks)
            # 注意：SAM模型不返回类别信息。此处的`cls`仅为一致性占位符。
            cls = torch.arange(pred_masks.shape[0], dtype=torch.int32, device=pred_masks.device)
            pred_bboxes = torch.cat([pred_bboxes, pred_scores[:, None], cls[:, None]], dim=-1)
        return pred_masks, pred_bboxes


class SAM2Predictor(Predictor):
    """使用Segment Anything Model 2架构进行高级图像分割的SAM2Predictor类。

    此类扩展基础Predictor类，实现SAM2特有的图像分割任务功能。
    它提供模型初始化、特征提取和基于提示的推理方法。

    Attributes:
        _bb_feat_sizes (list[tuple]): 不同骨干网络层级的特征尺寸。
        model (torch.nn.Module): 加载的SAM2模型。
        device (torch.device): 模型加载的设备（CPU或GPU）。
        features (dict): 缓存图像特征以实现高效推理。
        segment_all (bool): 指示是否应预测所有段的标志。
        prompts (dict[str, Any]): 存储各种类型推理提示的字典。

    Methods:
        get_model: 检索并初始化SAM2模型。
        prompt_inference: 基于各种提示执行图像分割推理。
        set_image: 预处理并设置单张图像用于推理。
        get_im_features: 使用SAM2的图像编码器提取和处理图像特征。

    Examples:
        >>> predictor = SAM2Predictor(cfg)
        >>> predictor.set_image("path/to/image.jpg")
        >>> bboxes = [[100, 100, 200, 200]]
        >>> result = predictor(bboxes=bboxes)[0]
        >>> print(f"预测了{len(result.masks)}个掩码，平均分数{result.boxes.conf.mean():.2f}")
    """

    _bb_feat_sizes = [
        (256, 256),
        (128, 128),
        (64, 64),
    ]
    stride = 16

    def get_model(self):
        """检索并初始化Segment Anything Model 2（SAM2）用于图像分割任务。"""
        from .build import build_sam  # 延迟导入

        return build_sam(self.args.model)

    def _prepare_prompts(self, dst_shape, src_shape, bboxes=None, points=None, labels=None, masks=None):
        """准备并变换输入提示以基于目标形状进行处理。

        Args:
            dst_shape (tuple[int, int]): 提示的目标形状（高度，宽度）。
            src_shape (tuple[int, int]): 输入图像的源形状（高度，宽度）。
            bboxes (np.ndarray | list | None): XYXY格式的边界框，形状为(N, 4)。
            points (np.ndarray | list | None): 指示对象位置的点，形状为(N, 2)或(N, num_points, 2)，单位为像素。
            labels (np.ndarray | list | None): 点提示标签，形状为(N,)或(N, num_points)。1表示前景，0表示背景。
            masks (list | np.ndarray | None): 对象的掩码，每个掩码为2D数组。

        Returns:
            points (torch.Tensor | None): 变换后的点。
            labels (torch.Tensor | None): 变换后的标签。
            masks (torch.Tensor | None): 变换后的掩码。

        Raises:
            AssertionError: 如果传入了标签但点数量与标签数量不匹配。
        """
        bboxes, points, labels, masks = super()._prepare_prompts(dst_shape, src_shape, bboxes, points, labels, masks)
        if bboxes is not None:
            bboxes = bboxes.view(-1, 2, 2)
            bbox_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=bboxes.device).expand(bboxes.shape[0], -1)
            # 注意：将"boxes"和"points"合并为单一的"points"输入
            # （其中boxes被添加到开头）传递给model.sam_prompt_encoder
            if points is not None:
                points = torch.cat([bboxes, points], dim=1)
                labels = torch.cat([bbox_labels, labels], dim=1)
            else:
                points, labels = bboxes, bbox_labels
        return points, labels, masks

    def setup_source(self, source):
        """设置SAM2推理的数据源和图像尺寸。"""
        super().setup_source(source)
        self._bb_feat_sizes = [[int(x / (self.stride * i)) for x in self.imgsz] for i in [1 / 4, 1 / 2, 1]]

    def get_im_features(self, im):
        """从SAM图像编码器提取图像特征用于后续处理。"""
        backbone_out = self.model.forward_image(im)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size) for feat, feat_size in zip(vision_feats, self._bb_feat_sizes)
        ]
        return {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

    def _inference_features(
        self,
        features,
        points=None,
        labels=None,
        masks=None,
        multimask_output=False,
        img_idx=-1,
    ):
        """使用SAM2模型对图像特征执行推理。

        Args:
            features (torch.Tensor | dict[str, Any]): 从SAM2模型图像编码器提取的图像特征，形状为(B, C, H, W)。
                也可以是字典，包含'image_embed'（torch.Tensor，形状为(B, C, H, W)）和
                'high_res_feats'（list[torch.Tensor]）骨干网络的高分辨率特征图。
            points (np.ndarray | list[list[float]] | None): 对象位置点，形状为(N, 2)，单位为像素。
            labels (np.ndarray | list[int] | None): 点提示标签，形状为(N,)。1 = 前景，0 = 背景。
            masks (list[np.ndarray] | np.ndarray | None): 对象的掩码，每个掩码为2D数组。
            multimask_output (bool): 是否返回多个掩码的标志，有助于处理模糊提示。
            img_idx (int): 要处理的图像在批次中的索引。

        Returns:
            pred_masks (torch.Tensor): 输出掩码，形状为(C, H, W)，其中C是生成的掩码数量。
            pred_scores (torch.Tensor): 每个掩码的质量分数，长度为C。
        """
        points = (points, labels) if points is not None else None
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=points,
            boxes=None,
            masks=masks,
        )
        # 预测掩码
        batched_mode = points is not None and points[0].shape[0] > 1  # 多对象预测
        high_res_features = None
        if isinstance(features, dict):
            high_res_features = [feat_level[img_idx].unsqueeze(0) for feat_level in features["high_res_feats"]]
            features = features["image_embed"][[img_idx]]
        pred_masks, pred_scores, _, _ = self.model.sam_mask_decoder(
            image_embeddings=features,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
        # (N, d, H, W) --> (N*d, H, W), (N, d) --> (N*d, )
        # `d` 可以是 1 或 3，取决于 `multimask_output`。
        return pred_masks.flatten(0, 1), pred_scores.flatten(0, 1)


class SAM2VideoPredictor(SAM2Predictor):
    """SAM2VideoPredictor用于处理用户与视频的交互并管理推理状态。

    此类扩展SAM2Predictor的功能，支持视频处理并维护推理操作的状态。
    包含管理非重叠掩码、在非条件输入时清除记忆以及设置预测事件回调的配置。

    Attributes:
        inference_state (dict): 存储当前推理操作状态的字典。
        non_overlap_masks (bool): 指示掩码是否应互不重叠的标志。
        clear_non_cond_mem_around_input (bool): 控制是否清除输入周围非条件记忆的标志。
        clear_non_cond_mem_for_multi_obj (bool): 控制多对象场景下是否清除非条件记忆的标志。
        callbacks (dict): 各种预测生命周期事件的回调字典。

    Methods:
        get_model: 检索并配置启用了二值化的模型。
        inference: 基于给定输入提示执行图像分割推理。
        postprocess: 后处理预测结果，必要时应用非重叠约束。
        add_new_prompts: 为给定对象ID在特定帧上添加新的点或掩码。
        propagate_in_video_preflight: 在追踪之前准备inference_state并整合临时输出。
        init_state: 为预测器初始化推理状态。
        get_im_features: 使用SAM2的图像编码器提取图像特征用于后续分割任务。

    Examples:
        >>> predictor = SAM2VideoPredictor(cfg=DEFAULT_CFG)
        >>> predictor.set_image("path/to/video_frame.jpg")
        >>> bboxes = [[100, 100, 200, 200]]
        >>> results = predictor(bboxes=bboxes)

    Notes:
        `fill_hole_area`属性已定义但当前实现中未使用。
    """

    # fill_hole_area = 8  # 未使用

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """使用配置和可选覆盖初始化预测器。

        此构造函数使用给定配置初始化SAM2VideoPredictor，应用任何指定的覆盖，
        并设置推理状态以及控制预测器行为的某些标志。

        Args:
            cfg (dict): 包含默认设置的配置字典。
            overrides (dict | None): 覆盖默认配置的值字典。
            _callbacks (dict | None): 自定义行为的回调函数字典。
        """
        super().__init__(cfg, overrides, _callbacks)
        self.inference_state = {}
        self.non_overlap_masks = True
        self.clear_non_cond_mem_around_input = False
        self.clear_non_cond_mem_for_multi_obj = False
        self.callbacks["on_predict_start"].append(self.init_state)
        self.clear_non_cond_mem = True  # 是否定期清除非条件记忆

    def get_model(self):
        """检索并配置启用了二值化的模型。

        Notes:
            此方法覆盖基类实现，将binarize标志设置为True。
        """
        model = super().get_model()
        model.set_binarize(True)
        return model

    def inference(self, im, bboxes=None, points=None, labels=None, masks=None):
        """基于给定输入提示执行图像分割推理，使用当前加载的图像。此方法利用SAM（Segment Anything Model）
        的架构，由图像编码器、提示编码器和掩码解码器组成，用于实时和可提示的分割任务。

        Args:
            im (torch.Tensor): 预处理后的输入图像张量，形状为(N, C, H, W)。
            bboxes (np.ndarray | list, optional): XYXY格式的边界框，形状为(N, 4)。
            points (np.ndarray | list, optional): 指示对象位置的点，形状为(N, 2)，单位为像素。
            labels (np.ndarray | list, optional): 点提示的标签，形状为(N,)。1 = 前景，0 = 背景。
            masks (np.ndarray, optional): 先前预测的低分辨率掩码，形状为(N,H,W)。对于SAM，H=W=256。

        Returns:
            pred_masks (torch.Tensor): 输出掩码，形状为CxHxW，其中C是生成的掩码数量。
            pred_scores (torch.Tensor): 长度为C的数组，包含每个掩码的预测质量分数。
        """
        # 如果self.prompts中存储了提示则进行覆盖
        bboxes = self.prompts.pop("bboxes", bboxes)
        points = self.prompts.pop("points", points)
        masks = self.prompts.pop("masks", masks)

        frame = self.dataset.frame
        self.inference_state["im"] = im
        output_dict = self.inference_state["output_dict"]
        if len(output_dict["cond_frame_outputs"]) == 0:  # 初始化提示
            points, labels, masks = self._prepare_prompts(
                im.shape[2:], self.batch[1][0].shape[:2], bboxes, points, labels, masks
            )
            if points is not None:
                for i in range(len(points)):
                    self.add_new_prompts(obj_id=i, points=points[[i]], labels=labels[[i]], frame_idx=frame)
            elif masks is not None:
                for i in range(len(masks)):
                    self.add_new_prompts(obj_id=i, masks=masks[[i]], frame_idx=frame)
        self.propagate_in_video_preflight()

        consolidated_frame_inds = self.inference_state["consolidated_frame_inds"]
        batch_size = len(self.inference_state["obj_idx_to_id"])
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")

        if frame in consolidated_frame_inds["cond_frame_outputs"]:
            storage_key = "cond_frame_outputs"
            current_out = output_dict[storage_key][frame]
            if self.clear_non_cond_mem_around_input and (self.clear_non_cond_mem_for_multi_obj or batch_size <= 1):
                # 清除周围帧的非条件记忆
                self._clear_non_cond_mem_around_input(frame)
        elif frame in consolidated_frame_inds["non_cond_frame_outputs"]:
            storage_key = "non_cond_frame_outputs"
            current_out = output_dict[storage_key][frame]
        else:
            storage_key = "non_cond_frame_outputs"
            current_out = self._run_single_frame_inference(
                output_dict=output_dict,
                frame_idx=frame,
                batch_size=batch_size,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
            output_dict[storage_key][frame] = current_out
            self._prune_non_cond_memory(frame)
        # 创建每对象输出切片，用于追踪后与每个
        # 单独对象进行后续交互。
        self._add_output_per_object(frame, current_out, storage_key)
        self.inference_state["frames_already_tracked"].append(frame)
        pred_masks = current_out["pred_masks"].flatten(0, 1)
        pred_masks = pred_masks[(pred_masks > self.model.mask_threshold).sum((1, 2)) > 0]  # 过滤空白掩码

        return pred_masks, torch.ones(pred_masks.shape[0], dtype=pred_masks.dtype, device=pred_masks.device)

    def postprocess(self, preds, img, orig_imgs):
        """后处理预测结果，必要时应用非重叠约束。

        此方法扩展了后处理功能，如果`non_overlap_masks`标志设置为True，则对预测掩码应用
        非重叠约束。这确保掩码不会重叠，在某些应用中很有用。

        Args:
            preds (tuple[torch.Tensor, torch.Tensor]): 模型预测的掩码和分数。
            img (torch.Tensor): 处理后的图像张量。
            orig_imgs (list[np.ndarray]): 处理前的原始图像。

        Returns:
            (list): 后处理后的预测结果。

        Notes:
            如果`non_overlap_masks`为True，该方法会应用约束以确保掩码互不重叠。
        """
        results = super().postprocess(preds, img, orig_imgs)
        if self.non_overlap_masks:
            for result in results:
                if result.masks is None or len(result.masks) == 0:
                    continue
                result.masks.data = self.model._apply_non_overlapping_constraints(result.masks.data.unsqueeze(0))[0]
        return results

    @smart_inference_mode()
    def add_new_prompts(
        self,
        obj_id,
        points=None,
        labels=None,
        masks=None,
        frame_idx=0,
        inference_state: dict[str, Any] | None = None,
    ):
        """为给定对象ID在特定帧上添加新的点或掩码。

        此方法使用新的提示（点或掩码）更新指定对象和帧索引的推理状态。
        它确保提示要么是点要么是掩码，但不能同时存在，并相应地更新内部状态。
        它还基于提供的提示和现有状态生成新的分割结果。

        Args:
            obj_id (int): 提示关联的对象ID。
            points (torch.Tensor, optional): 兴趣点的坐标。
            labels (torch.Tensor, optional): 对应点的标签。
            masks (torch.Tensor, optional): 对象的二值掩码。
            frame_idx (int, optional): 应用提示的帧索引。
            inference_state (dict[str, Any], optional): 当前推理状态。如果为None，使用实例的推理状态。

        Returns:
            pred_masks (torch.Tensor): 展开的预测掩码。
            pred_scores (torch.Tensor): 指示对象数量的全1张量。

        Raises:
            AssertionError: 如果同时提供了`masks`和`points`，或两者都未提供。

        Notes:
            - 每次调用只能添加一种类型的提示（点或掩码）。
            - 如果该帧是首次被追踪，则视为初始条件帧。
            - 该方法处理输出整合和掩码缩放到原始视频分辨率。
        """
        inference_state = inference_state or self.inference_state
        assert (masks is None) ^ (points is None), "'masks' and 'points' prompts are not compatible with each other."
        obj_idx = self._obj_id_to_idx(obj_id, inference_state)

        point_inputs = None
        pop_key = "point_inputs_per_obj"
        if points is not None:
            point_inputs = {"point_coords": points, "point_labels": labels}
            inference_state["point_inputs_per_obj"][obj_idx][frame_idx] = point_inputs
            pop_key = "mask_inputs_per_obj"
        inference_state["mask_inputs_per_obj"][obj_idx][frame_idx] = masks
        inference_state[pop_key][obj_idx].pop(frame_idx, None)
        # 如果该帧之前未被追踪过，我们将其视为初始条件帧，
        # 意味着输入点应在此帧上生成分割，而不使用其他帧的记忆，
        # 类似于SAM。否则（如果已被追踪），输入点将用于修正已追踪的掩码。
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # 如果是初始条件帧，或者模型将所有接收点击/掩码的帧视为条件帧，
        # 则将该帧添加到条件输出中。
        is_cond = is_init_cond_frame or self.model.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # 获取该对象上任何先前预测的掩码logits，并与新的点击
        # 一起输入SAM掩码解码器。
        prev_sam_mask_logits = None
        # 首先查找临时输出字典，其中包含最近的输出
        # （如果未找到，则查找条件和无条件帧输出）
        if point_inputs is not None:
            prev_out = (
                obj_temp_output_dict[storage_key].get(frame_idx)
                or obj_output_dict["cond_frame_outputs"].get(frame_idx)
                or obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
            )

            if prev_out is not None and prev_out.get("pred_masks") is not None:
                prev_sam_mask_logits = prev_out["pred_masks"].to(
                    device=self.device, non_blocking=self.device.type == "cuda"
                )
                # 限制prev_sam_mask_logits的尺度以避免罕见的数值问题。
                prev_sam_mask_logits.clamp_(-32.0, 32.0)
        current_out = self._run_single_frame_inference(
            output_dict=obj_output_dict,  # 在单个对象的切片上运行
            frame_idx=frame_idx,
            batch_size=1,  # 在单个对象的切片上运行
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=masks,
            reverse=False,
            # 添加点击或掩码时跳过记忆编码器。我们在`propagate_in_video`开始时
            # （用户完成点击后）执行记忆编码器。这允许我们在将所有对象编码到
            # 记忆中之前对其强制执行非重叠约束。
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
            inference_state=inference_state,
        )
        # 将输出添加到输出字典中（用作未来记忆）
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # 将输出掩码缩放到原始视频分辨率
        consolidated_out = self._consolidate_temp_output_across_obj(
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            inference_state=inference_state,
        )
        pred_masks = consolidated_out["pred_masks"].flatten(0, 1)
        return pred_masks.flatten(0, 1), torch.ones(1, dtype=pred_masks.dtype, device=pred_masks.device)

    @smart_inference_mode()
    def propagate_in_video_preflight(self, inference_state: dict[str, Any] | None = None):
        """在追踪之前准备inference_state并整合临时输出。

        此方法标志着追踪开始，不允许在会话重置前添加新对象。
        它整合`temp_output_dict_per_obj`中的临时输出并将其合并到`output_dict`中。
        此外，它清除输入帧周围的非条件记忆，并确保状态与提供的输入一致。

        Args:
            inference_state (dict[str, Any], optional): 当前推理状态。如果为None，使用实例的推理状态。
        """
        inference_state = inference_state or self.inference_state
        # 追踪已开始，在会话重置前不允许添加新对象。
        inference_state["tracking_has_started"] = True
        batch_size = len(inference_state["obj_idx_to_id"])

        # 整合"temp_output_dict_per_obj"中的每对象临时输出并
        # 将其添加到"output_dict"。
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        # "consolidated_frame_inds"包含那些已添加整合临时输出的帧的索引
        # （在本次调用或之前对`propagate_in_video_preflight`的任何调用中）。
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        for is_cond in {False, True}:
            # 分别整合条件和无条件临时输出
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            # 查找所有包含任何对象临时输出的帧
            # （这些应该是刚刚通过`add_new_points`或`add_new_mask`接收到
            # 点击或掩码输入的帧）
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            # 在此帧上跨所有对象整合临时输出
            for frame_idx in temp_frame_inds:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    frame_idx, is_cond=is_cond, run_mem_encoder=True, inference_state=inference_state
                )
                # 将它们合并到"output_dict"中，并创建每对象切片
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(frame_idx, consolidated_out, storage_key, inference_state=inference_state)
                if self.clear_non_cond_mem_around_input and (self.clear_non_cond_mem_for_multi_obj or batch_size <= 1):
                    # 清除周围帧的非条件记忆
                    self._clear_non_cond_mem_around_input(frame_idx)

            # 清除`temp_output_dict_per_obj`中的临时输出
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        # 边界情况：如果输出被添加到"cond_frame_outputs"，则移除"non_cond_frame_outputs"中
        # 同一帧上的任何先前输出
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

        # 确保"consolidated_frame_inds"中的帧索引恰好是那些有点或掩码输入的帧
        # （在正确的工作流程下应该成立）。
        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"] | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames_inds = set()
        for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert all_consolidated_frame_inds == input_frames_inds

    @staticmethod
    def init_state(predictor):
        """为预测器初始化推理状态。

        此函数设置执行视频数据推理所需的初始状态。包括初始化各种字典和有序字典，
        用于存储与追踪过程相关的输入、输出和其他元数据。

        Args:
            predictor (SAM2VideoPredictor): 需要初始化状态的预测器对象。
        """
        if len(predictor.inference_state) > 0:  # 表示已初始化
            return
        assert predictor.dataset is not None
        assert predictor.dataset.mode == "video"
        predictor.inference_state = predictor._init_state(predictor.dataset.frames)

    @staticmethod
    def _init_state(num_frames):
        """初始化推理状态。

        此函数设置执行视频数据推理所需的初始状态。包括初始化各种字典和有序字典，
        用于存储与追踪过程相关的输入、输出和其他元数据。

        Args:
            num_frames (int): 视频中的帧数。
        """
        inference_state = {
            "num_frames": num_frames,  # TODO: 看看是否有机会移除它
            "point_inputs_per_obj": {},  # 每帧的输入点
            "mask_inputs_per_obj": {},  # 每帧的输入掩码
            "constants": {},  # 跨帧不变的值（只需保存一份副本）
            # 客户端对象ID与模型端对象索引之间的映射
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            # 存储模型在每帧上的追踪结果和状态
            "output_dict": {
                "cond_frame_outputs": {},  # 字典，包含{frame_idx: <out>}
                "non_cond_frame_outputs": {},  # 字典，包含{frame_idx: <out>}
            },
            # 每个对象追踪结果的切片（视图），与"output_dict"共享相同的内存
            "output_dict_per_obj": {},
            # 临时存储，当用户与帧交互添加点击或掩码时存放新输出
            # （在传播开始前合并到"output_dict"中）
            "temp_output_dict_per_obj": {},
            # 已持有来自点击或掩码输入的整合输出的帧
            # （我们在追踪过程中直接使用它们的整合输出）
            "consolidated_frame_inds": {
                "cond_frame_outputs": set(),  # 集合，包含帧索引
                "non_cond_frame_outputs": set(),  # 集合，包含帧索引
            },
            # 每个追踪帧的元数据（例如，追踪方向）
            "tracking_has_started": False,
            "frames_already_tracked": [],
        }
        return inference_state

    def get_im_features(self, im, batch=1):
        """使用SAM2的图像编码器提取和处理图像特征以用于后续分割任务。

        Args:
            im (torch.Tensor): 输入图像张量。
            batch (int, optional): 如果存在多个提示，用于扩展特征的批次大小。

        Returns:
            vis_feats (torch.Tensor): 从图像提取的视觉特征。
            vis_pos_embed (torch.Tensor): 视觉特征的位置嵌入。
            feat_sizes (list[tuple]): 包含提取特征大小的列表。

        Notes:
            - 如果`batch`大于1，特征会扩展以适配批次大小。
            - 该方法利用模型的`_prepare_backbone_features`方法来准备骨干网络特征。
        """
        # 检查是否有预计算的骨干网络输出
        backbone_out = getattr(self, "backbone_out", None)
        if backbone_out is None:
            backbone_out = self.model.forward_image(im)
        _, vis_feats, vis_pos_embed, feat_sizes = self.model._prepare_backbone_features(backbone_out, batch=batch)
        return vis_feats, vis_pos_embed, feat_sizes

    def _obj_id_to_idx(self, obj_id, inference_state: dict[str, Any] | None = None):
        """将客户端对象ID映射到模型端对象索引。

        Args:
            obj_id (int): 客户端提供的对象唯一标识符。
            inference_state (dict[str, Any], optional): 当前推理状态。如果为None，使用实例的推理状态。

        Returns:
            (int): 模型端的对象索引。

        Raises:
            RuntimeError: 如果在追踪开始后尝试添加新对象。

        Notes:
            - 该方法更新或检索存储在`inference_state`中的对象ID与索引之间的映射。
            - 它确保新对象只能在追踪开始前添加。
            - 它维护ID与索引之间的双向映射（`obj_id_to_idx`和`obj_idx_to_id`）。
            - 为新对象初始化额外的数据结构以存储输入和输出。
        """
        inference_state = inference_state or self.inference_state
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # 这是一个之前未发送到服务器的新对象ID。我们只允许在追踪
        # *开始之前*添加新对象。
        allow_new_object = not inference_state["tracking_has_started"]
        if allow_new_object:
            # 获取下一个对象槽位
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            # 为此对象设置输入和输出结构
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # 字典，包含{frame_idx: <out>}
                "non_cond_frame_outputs": {},  # 字典，包含{frame_idx: <out>}
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # 字典，包含{frame_idx: <out>}
                "non_cond_frame_outputs": {},  # 字典，包含{frame_idx: <out>}
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}. "
                f"Please call 'reset_state' to restart from scratch."
            )

    def _run_single_frame_inference(
        self,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
        inference_state: dict[str, Any] | None = None,
    ):
        """基于当前输入和先前记忆在单帧上执行追踪。

        Args:
            output_dict (dict): 包含追踪过程输出状态的字典。
            frame_idx (int): 当前帧的索引。
            batch_size (int): 处理帧的批次大小。
            is_init_cond_frame (bool): 指示当前帧是否为初始条件帧。
            point_inputs (dict | None): 输入点及其标签。
            mask_inputs (torch.Tensor | None): 输入的二值掩码。
            reverse (bool): 指示是否以反向顺序执行追踪。
            run_mem_encoder (bool): 指示是否应执行记忆编码器。
            prev_sam_mask_logits (torch.Tensor | None): 当前对象先前的掩码logits。
            inference_state (dict[str, Any], optional): 当前推理状态。如果为None，使用实例的推理状态。

        Returns:
            (dict): 包含追踪步骤输出的字典，包括更新的特征和预测。

        Raises:
            AssertionError: 如果同时提供了`point_inputs`和`mask_inputs`。

        Notes:
            - 该方法假定`point_inputs`和`mask_inputs`互斥。
            - 该方法使用`get_im_features`方法检索图像特征。
            - `maskmem_pos_enc`假定跨帧不变，因此只存储一份副本。
            - `fill_holes_in_mask_scores`函数已被注释，当前因需要CUDA扩展而不支持。
        """
        inference_state = inference_state or self.inference_state
        # 检索正确的图像特征
        current_vision_feats, current_vision_pos_embeds, feat_sizes = self.get_im_features(
            inference_state["im"], batch_size
        )

        # 点和掩码不应同时作为同一帧的输入出现
        assert point_inputs is None or mask_inputs is None
        current_out = self.model.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )

        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            current_out["maskmem_features"] = maskmem_features.to(
                dtype=torch.float16, device=self.device, non_blocking=self.device.type == "cuda"
            )
        # 注意：不支持`fill_holes_in_mask_scores`函数，因为它需要CUDA扩展
        # 可能需要对预测掩码进行孔洞填充
        # if self.fill_hole_area > 0:
        #     pred_masks = current_out["pred_masks"].to(self.device, non_blocking=self.device.type == "cuda")
        #     pred_masks = fill_holes_in_mask_scores(pred_masks, self.fill_hole_area)

        # "maskmem_pos_enc"跨帧相同，因此我们只需存储一份副本
        current_out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(current_out["maskmem_pos_enc"], inference_state)
        return current_out

    def _get_maskmem_pos_enc(self, out_maskmem_pos_enc, inference_state: dict[str, Any] | None = None):
        """跨帧和对象缓存并管理掩码记忆的位置编码。

        此方法通过缓存位置编码（`maskmem_pos_enc`）来优化存储，该编码跨帧和对象不变，
        从而减少推理会话期间存储的冗余信息量。它检查位置编码是否已被缓存；
        如果未缓存，则缓存提供的编码的一个切片。如果批次大小大于1，它扩展缓存的位置编码
        以匹配当前批次大小。

        Args:
            out_maskmem_pos_enc (list[torch.Tensor] | None): 掩码记忆的位置编码。应为张量列表或None。
            inference_state (dict[str, Any], optional): 当前推理状态。如果为None，使用实例的推理状态。

        Returns:
            (list[torch.Tensor]): 掩码记忆的位置编码，缓存或扩展后的版本。

        Notes:
            - 该方法假定`out_maskmem_pos_enc`是张量列表或None。
            - 由于编码跨对象相同，仅缓存单个对象的切片。
            - 该方法检查位置编码是否已在会话常量中缓存。
            - 如果批次大小大于1，缓存编码会被扩展以适配批次大小。
        """
        inference_state = inference_state or self.inference_state
        model_constants = inference_state["constants"]
        # "out_maskmem_pos_enc"应该是张量列表或None
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                # 仅取一个对象的切片，因为它跨对象相同
                maskmem_pos_enc = [x[:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            # 将缓存的maskmem_pos_enc扩展到实际批次大小
            batch_size = out_maskmem_pos_enc[0].shape[0]
            if batch_size > 1:
                out_maskmem_pos_enc = [x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc]
        return out_maskmem_pos_enc

    def _consolidate_temp_output_across_obj(
        self,
        frame_idx,
        is_cond=False,
        run_mem_encoder=False,
        inference_state: dict[str, Any] | None = None,
    ):
        """将每对象临时输出整合为所有对象的单一输出。

        此方法将给定帧上每个对象的临时输出合并为统一输出。
        它从主输出字典中填充任何缺失的对象，如果主输出中不存在则保留占位符。
        可选地，它可以在对对象分数应用非重叠约束后重新运行记忆编码器。

        Args:
            frame_idx (int): 要整合输出的帧索引。
            is_cond (bool, optional): 指示该帧是否被视为条件帧。
            run_mem_encoder (bool, optional): 指定整合输出后是否运行记忆编码器。
            inference_state (dict[str, Any], optional): 当前推理状态。如果为None，使用实例的推理状态。

        Returns:
            (dict): 包含所有对象组合结果的整合输出字典。

        Notes:
            - 该方法用占位符值初始化整合输出，用于缺失的对象。
            - 它在临时和主输出字典中搜索输出。
            - 如果`run_mem_encoder`为True，它应用非重叠约束并重新运行记忆编码器。
            - `maskmem_features`和`maskmem_pos_enc`仅在`run_mem_encoder`为True时填充。
        """
        inference_state = inference_state or self.inference_state
        batch_size = len(inference_state["obj_idx_to_id"])
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # 初始化`consolidated_out`。其"maskmem_features"和"maskmem_pos_enc"
        # 将在对对象分数应用非重叠约束后重新运行记忆编码器时添加。
        # 其"pred_masks"预填充为大的负值（NO_OBJ_SCORE）以表示缺失对象。
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            "pred_masks": torch.full(
                # size=(batch_size, 1, self.imgsz[0] // 4, self.imgsz[1] // 4),
                size=(batch_size, 1, *self._bb_feat_sizes[0]),
                fill_value=-1024.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
            "obj_ptr": torch.full(
                size=(batch_size, self.model.hidden_dim),
                fill_value=-1024.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
            "object_score_logits": torch.full(
                size=(batch_size, 1),
                # object_score_logits的默认值为10.0，即假设对象存在，
                # 因为sigmoid(10)=1，与`MaskDecoder`的`predict_masks`中相同
                fill_value=10.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
        }
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = (
                obj_temp_output_dict[storage_key].get(frame_idx)
                # 如果对象在此帧的"temp_output_dict_per_obj"中不存在，
                # 我们回退并在"output_dict_per_obj"中查找其先前的输出。
                # 我们在"output_dict_per_obj"中查找"cond_frame_outputs"和"non_cond_frame_outputs"
                # 以找到该对象的先前输出。
                or obj_output_dict["cond_frame_outputs"].get(frame_idx)
                or obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
            )
            # 如果对象在"output_dict_per_obj"中也不存在，我们跳过它
            # 并将其掩码分数保留为默认分数（即上面的NO_OBJ_SCORE占位符），
            # 同时将其对象指针设置为虚拟指针。
            if out is None:
                # 对于在此帧上没有任何输入或追踪结果的对象，
                # 填充虚拟对象指针（仅在`run_mem_encoder=True`时执行，
                # 即当我们需要为追踪构建记忆时）。
                if run_mem_encoder:
                    # 用虚拟指针填充对象指针（基于空掩码）
                    consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = self._get_empty_mask_ptr(frame_idx)
                continue
            # 将临时对象输出掩码添加到整合输出掩码
            consolidated_out["pred_masks"][obj_idx : obj_idx + 1] = out["pred_masks"]
            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]

        # 可选地对整合分数应用非重叠约束并重新运行记忆编码器
        if run_mem_encoder:
            high_res_masks = F.interpolate(
                consolidated_out["pred_masks"],
                size=self.imgsz,
                mode="bilinear",
                align_corners=False,
            )
            if self.model.non_overlap_masks_for_mem_enc:
                high_res_masks = self.model._apply_non_overlapping_constraints(high_res_masks)
            consolidated_out["maskmem_features"], consolidated_out["maskmem_pos_enc"] = self._run_memory_encoder(
                batch_size=batch_size,
                high_res_masks=high_res_masks,
                is_mask_from_pts=True,  # 这些帧是用户交互过的
                object_score_logits=consolidated_out["object_score_logits"],
                inference_state=inference_state,
            )

        return consolidated_out

    def _get_empty_mask_ptr(self, frame_idx, inference_state: dict[str, Any] | None = None):
        """基于当前帧的空掩码获取虚拟对象指针。

        Args:
            frame_idx (int): 要为哪个帧生成虚拟对象指针。
            inference_state (dict[str, Any], optional): 当前推理状态。如果为None，使用实例的推理状态。

        Returns:
            (torch.Tensor): 表示从空掩码生成的虚拟对象指针的张量。
        """
        inference_state = inference_state or self.inference_state
        # 检索正确的图像特征
        current_vision_feats, current_vision_pos_embeds, feat_sizes = self.get_im_features(inference_state["im"])

        # 将空掩码和上面的图像特征输入以获取虚拟对象指针
        current_out = self.model.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            # 单个对象的虚拟（空）掩码
            mask_inputs=torch.zeros((1, 1, *self.imgsz), dtype=self.torch_dtype, device=self.device),
            output_dict={},
            num_frames=inference_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
        )
        return current_out["obj_ptr"]

    def _run_memory_encoder(
        self,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts,
        inference_state: dict[str, Any] | None = None,
    ):
        """在掩码上运行记忆编码器。

        通常在应用非重叠约束到对象分数后进行。由于分数发生了变化，
        它们的记忆也需要用记忆编码器重新计算。

        Args:
            batch_size (int): 处理帧的批次大小。
            high_res_masks (torch.Tensor): 要计算记忆的高分辨率掩码。
            object_score_logits (torch.Tensor): 表示对象分数的logits。
            is_mask_from_pts (bool): 指示掩码是否源自点交互。
            inference_state (dict[str, Any], optional): 当前推理状态。如果为None，使用实例的推理状态。

        Returns:
            maskmem_features (torch.Tensor): 编码的掩码特征。
            maskmem_pos_enc (torch.Tensor): 位置编码。
        """
        inference_state = inference_state or self.inference_state
        # 检索正确的图像特征
        current_vision_feats, _, feat_sizes = self.get_im_features(inference_state["im"], batch_size)
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            is_mask_from_pts=is_mask_from_pts,
            object_score_logits=object_score_logits,
        )

        # "maskmem_pos_enc"跨帧相同，因此我们只需存储一份副本
        maskmem_pos_enc = self._get_maskmem_pos_enc(maskmem_pos_enc, inference_state)
        return maskmem_features.to(
            dtype=torch.float16, device=self.device, non_blocking=self.device.type == "cuda"
        ), maskmem_pos_enc

    def _add_output_per_object(
        self, frame_idx, current_out, storage_key, inference_state: dict[str, Any] | None = None
    ):
        """将多对象输出分割为每对象输出切片并将其添加到Output_Dict_Per_Obj。

        生成的切片共享相同的张量存储。

        Args:
            frame_idx (int): 当前帧的索引。
            current_out (dict): 包含多对象输出的当前输出字典。
            storage_key (str): 用于在每对象输出字典中存储输出的键。
            inference_state (dict[str, Any], optional): 当前推理状态。如果为None，使用实例的推理状态。
        """
        inference_state = inference_state or self.inference_state
        maskmem_features = current_out["maskmem_features"]
        assert maskmem_features is None or isinstance(maskmem_features, torch.Tensor)

        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)

        for obj_idx, obj_output_dict in inference_state["output_dict_per_obj"].items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
            }
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]
            obj_output_dict[storage_key][frame_idx] = obj_out

    def _clear_non_cond_mem_around_input(self, frame_idx, inference_state: dict[str, Any] | None = None):
        """移除输入帧周围的非条件记忆。

        当用户提供修正点击时，周围帧的非条件记忆可能仍包含过时的对象外观信息，
        可能会混淆模型。此方法清除交互帧周围的非条件记忆，
        避免同时为模型提供对象的新旧信息。

        Args:
            frame_idx (int): 发生用户交互的当前帧索引。
            inference_state (dict[str, Any], optional): 当前推理状态。如果为None，使用实例的推理状态。
        """
        inference_state = inference_state or self.inference_state
        r = self.model.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.model.num_maskmem
        frame_idx_end = frame_idx + r * self.model.num_maskmem
        for t in range(frame_idx_begin, frame_idx_end + 1):
            inference_state["output_dict"]["non_cond_frame_outputs"].pop(t, None)
            for obj_output_dict in inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(t, None)

    @smart_inference_mode()
    def remove_object(self, inference_state, obj_id, strict=False):
        """从追踪状态中移除一个对象ID。如果strict为True，检查对象ID是否确实存在，
        如果不存在则抛出错误。
        """
        old_obj_idx_to_rm = inference_state["obj_id_to_idx"].get(obj_id, None)
        # 检查要移除的对象ID是否确实存在，并在必要时抛出错误。
        if old_obj_idx_to_rm is None:
            if not strict:
                return inference_state["obj_ids"]
            raise RuntimeError(
                f"Cannot remove object id {obj_id} as it doesn't exist. "
                f"All existing object ids: {inference_state['obj_ids']}."
            )

        # 如果这是唯一剩余的对象ID，我们简单地重置状态。
        if len(inference_state["obj_id_to_idx"]) == 1:
            self.clear_all_points_in_video(inference_state)
            return inference_state["obj_ids"]

        # 移除此对象ID后仍有剩余对象。在这种情况下，
        # 我们需要从推理状态张量中删除该对象的存储。
        # 步骤0：清除该对象ID有点或掩码输入的那些帧上的输入
        # （注意此步骤是必需的，因为它可能将条件帧降级为非条件帧）
        obj_input_frames_inds = set()
        obj_input_frames_inds.update(inference_state["point_inputs_per_obj"][old_obj_idx_to_rm])
        obj_input_frames_inds.update(inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm])
        for frame_idx in obj_input_frames_inds:
            self.clear_all_points_in_frame(inference_state, frame_idx, obj_id)

        # 步骤1：更新对象ID映射（注意必须在步骤0之后执行，
        # 因为步骤0仍需要inference_state中旧的对象ID映射）
        old_obj_ids = inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = old_obj_inds.copy()
        remain_old_obj_inds.remove(old_obj_idx_to_rm)
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        # 构建新映射
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
        inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
        inference_state["obj_ids"] = new_obj_ids

        # 步骤2：对于每对象张量存储，我们在字典键中移动它们的obj_idx。
        # （注意"consolidated_frame_inds"不需要在此步骤中更新，因为它已在步骤0中处理）
        def _map_keys(container):
            new_kvs = []
            for k in old_obj_inds:
                v = container.pop(k)
                if k in old_idx_to_new_idx:
                    new_kvs.append((old_idx_to_new_idx[k], v))
            container.update(new_kvs)

        _map_keys(inference_state["point_inputs_per_obj"])
        _map_keys(inference_state["mask_inputs_per_obj"])
        _map_keys(inference_state["output_dict_per_obj"])
        _map_keys(inference_state["temp_output_dict_per_obj"])

        # 步骤3：对于打包的张量存储，我们索引剩余ID并重建每对象切片。
        def _slice_state(output_dict, storage_key):
            for frame_idx, out in output_dict[storage_key].items():
                out["maskmem_features"] = out["maskmem_features"][remain_old_obj_inds]
                out["maskmem_pos_enc"] = [x[remain_old_obj_inds] for x in out["maskmem_pos_enc"]]
                # "maskmem_pos_enc"跨帧相同，因此我们只需存储一份副本
                out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(out["maskmem_pos_enc"], inference_state)
                out["pred_masks"] = out["pred_masks"][remain_old_obj_inds]
                out["obj_ptr"] = out["obj_ptr"][remain_old_obj_inds]
                out["object_score_logits"] = out["object_score_logits"][remain_old_obj_inds]
                # 同时更新每对象切片
                self._add_output_per_object(frame_idx, out, storage_key, inference_state=inference_state)

        _slice_state(inference_state["output_dict"], "cond_frame_outputs")
        _slice_state(inference_state["output_dict"], "non_cond_frame_outputs")

        return inference_state["obj_ids"]

    @smart_inference_mode()
    def clear_all_points_in_frame(self, inference_state, frame_idx, obj_id):
        """移除特定帧上给定对象的所有输入点或掩码。"""
        obj_idx = self._obj_id_to_idx(obj_id, inference_state)

        # 清除给定帧上的条件信息
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

        # 检查该帧上是否仍有任何剩余输入
        batch_size = len(inference_state["obj_idx_to_id"])
        frame_has_input = False
        for obj_idx2 in range(batch_size):
            if frame_idx in inference_state["point_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break
            if frame_idx in inference_state["mask_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break

        # 如果该帧上没有任何对象的剩余输入，我们进一步清除其条件帧状态
        if not frame_has_input:
            output_dict = inference_state["output_dict"]
            consolidated_frame_inds = inference_state["consolidated_frame_inds"]
            consolidated_frame_inds["cond_frame_outputs"].discard(frame_idx)
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)
            # 移除该帧的条件输出（可能降级为非条件输出）
            out = output_dict["cond_frame_outputs"].pop(frame_idx, None)
            if out is not None:
                # 该帧不再是条件帧，因为它不再接收输入，
                # 因此我们将其输出（如果存在）"降级"为非条件帧输出。
                output_dict["non_cond_frame_outputs"][frame_idx] = out
                inference_state["frames_already_tracked"].pop(frame_idx, None)
            # 类似地，对每个对象的切片输出执行同样操作。
            for obj_idx2 in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx2]
                obj_out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
                if obj_out is not None:
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = obj_out

            # 如果所有条件帧都已被移除，我们也清除追踪输出
            if len(output_dict["cond_frame_outputs"]) == 0:
                self._reset_tracking_results(inference_state)

    @smart_inference_mode()
    def clear_all_points_in_video(self, inference_state):
        """移除视频中所有帧的所有输入点或掩码。"""
        self._reset_tracking_results(inference_state)
        # 移除所有对象ID
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()

    @staticmethod
    def _reset_tracking_results(inference_state):
        """重置视频中所有的追踪输入和结果。"""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()
        inference_state["first_ann_frame_idx"] = None

    def _prune_non_cond_memory(self, frame_idx, inference_state=None):
        """修剪旧的非条件帧以限制内存使用。"""
        if not self.clear_non_cond_mem:
            return
        inference_state = inference_state or self.inference_state

        # 确定窗口大小
        min_frame = frame_idx - self.model.num_maskmem * self.model.memory_temporal_stride_for_eval
        output_dict = inference_state["output_dict"]

        # 修剪全局non_cond_frame_outputs
        for f in [k for k in output_dict["non_cond_frame_outputs"] if k < min_frame]:
            output_dict["non_cond_frame_outputs"].pop(f, None)

        # 修剪每对象non_cond_frame_outputs
        for obj_output_dict in inference_state.get("output_dict_per_obj", {}).values():
            for f in [k for k in obj_output_dict["non_cond_frame_outputs"] if k < min_frame]:
                obj_output_dict["non_cond_frame_outputs"].pop(f, None)


class SAM2DynamicInteractivePredictor(SAM2Predictor):
    """SAM2DynamicInteractivePredictor扩展SAM2Predictor以支持与视频帧或图像序列的动态交互。

    Attributes:
        memory_bank (list): 有序字典：存储每张带有提示的图像状态。
        obj_idx_set (set): 用于跟踪已添加对象索引的集合。
        obj_id_to_idx (OrderedDict): 将对象ID映射到对应的索引。
        obj_idx_to_id (OrderedDict): 将对象索引映射到对应的ID。

    Methods:
        get_model: 检索并配置启用了二值化的模型。
        inference: 对单张图像执行推理，可选地带有提示和对象ID。
        postprocess: 后处理预测结果，必要时应用非重叠约束。
        update_memory: 将imgState追加到memory_bank并为模型更新记忆。
        track_step: 当前图像状态的追踪步骤以预测掩码。
        get_maskmem_enc: 从记忆库获取记忆和位置编码。

    Examples:
            >>> predictor = SAM2DynamicInteractivePredictor(cfg=DEFAULT_CFG)
            >>> predictor(source=support_img1, bboxes=bboxes1, obj_ids=labels1, update_memory=True)
            >>> results1 = predictor(source=query_img1)
            >>> predictor(source=support_img2, bboxes=bboxes2, obj_ids=labels2, update_memory=True)
            >>> results2 = predictor(source=query_img2)
    """

    def __init__(
        self,
        cfg: Any = DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        max_obj_num: int = 3,
        _callbacks: dict | None = None,
    ) -> None:
        """使用配置和可选覆盖初始化预测器。

        此构造函数使用给定配置初始化SAM2DynamicInteractivePredictor，应用任何指定的覆盖。

        Args:
            cfg (Any): 包含默认设置的配置字典。
            overrides (dict[str, Any] | None): 覆盖默认配置的值字典。
            max_obj_num (int): 要追踪的最大对象数。默认为3。设置此值以保持模型的固定特征大小。
            _callbacks (dict | None): 自定义行为的回调函数字典。
        """
        super().__init__(cfg, overrides, _callbacks)
        self.non_overlap_masks = True

        # 初始化记忆库以存储图像状态
        # 注意：可能需要使用字典以获得更好的查询性能
        self.memory_bank = []

        # 初始化对象索引集合和映射
        self.obj_idx_set = set()
        self.obj_id_to_idx = self.obj_idx_to_id = OrderedDict(enumerate(range(max_obj_num)))
        self._max_obj_num = max_obj_num

    @smart_inference_mode()
    def inference(
        self,
        im: torch.Tensor | np.ndarray,
        bboxes: list[list[float]] | None = None,
        masks: torch.Tensor | np.ndarray | None = None,
        points: list[list[float]] | None = None,
        labels: list[int] | None = None,
        obj_ids: list[int] | None = None,
        update_memory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """对单张图像执行推理，可选地带有边界框、掩码、点和对象ID。有两种模式：
        一种是仅对单张图像执行推理而不更新记忆，另一种是使用提供的提示和对象ID更新记忆。
        当update_memory为True时，它使用提供的提示和obj_ids更新记忆。当update_memory为False时，
        它仅对提供的图像执行推理而不更新记忆。

        Args:
            im (torch.Tensor | np.ndarray): 输入图像张量或numpy数组。
            bboxes (list[list[float]] | None): 可选的要更新记忆的边界框列表。
            masks (torch.Tensor | np.ndarray | None): 可选的要更新记忆的掩码。
            points (list[list[float]] | None): 可选的要更新记忆的点列表，每个点为[x, y]。
            labels (list[int] | None): 可选的点提示标签列表（>0为正，0为负）。
            obj_ids (list[int] | None): 可选的与提示对应的对象ID列表。
            update_memory (bool): 指示是否使用新对象更新记忆的标志。

        Returns:
            res_masks (torch.Tensor): 输出掩码，形状为(C, H, W)。
            object_score_logits (torch.Tensor): 每个掩码的质量分数。
        """
        self.get_im_features(im)
        points, labels, masks = self._prepare_prompts(
            dst_shape=self.imgsz,
            src_shape=self.batch[1][0].shape[:2],
            points=points,
            bboxes=bboxes,
            labels=labels,
            masks=masks,
        )

        if update_memory:
            if isinstance(obj_ids, int):
                obj_ids = [obj_ids]
            assert obj_ids is not None, "obj_ids must be provided when update_memory is True"
            assert masks is not None or points is not None, (
                "bboxes, masks, or points must be provided when update_memory is True"
            )
            if points is None:  # 占位符
                points = torch.zeros((len(obj_ids), 0, 2), dtype=self.torch_dtype, device=self.device)
                labels = torch.zeros((len(obj_ids), 0), dtype=torch.int32, device=self.device)
            if masks is not None:
                assert len(masks) == len(obj_ids), "masks and obj_ids must have the same length."
            assert len(points) == len(obj_ids), "points and obj_ids must have the same length."
            self.update_memory(obj_ids, points, labels, masks)

        current_out = self.track_step()
        pred_masks, pred_scores = current_out["pred_masks"], current_out["object_score_logits"]
        # 根据对象索引过滤掩码和logits
        if len(self.obj_idx_set) == 0:
            raise RuntimeError("No objects have been added to the state. Please add objects before inference.")
        idx = list(self.obj_idx_set)  # 类别ID
        pred_masks, pred_scores = pred_masks[idx], pred_scores[idx]
        # 原始分数在[-32,32]范围内，大于0的对象分数表示对象存在，
        # 我们将其映射到[-1,1]范围，并使用激活函数确保对象分数logits非负，
        # 以便将其用作掩码
        pred_scores = torch.clamp_(pred_scores / 32, min=0)
        return pred_masks.flatten(0, 1), pred_scores.flatten(0, 1)

    def get_im_features(self, img: torch.Tensor | np.ndarray) -> None:
        """通过处理输入图像并提取特征来初始化图像状态。

        Args:
            img (torch.Tensor | np.ndarray): 输入图像张量或numpy数组。
        """
        vis_feats, vis_pos_embed, feat_sizes = SAM2VideoPredictor.get_im_features(self, img, batch=self._max_obj_num)
        self.high_res_features = [
            feat.permute(1, 2, 0).view(*feat.shape[1:], *feat_size)
            for feat, feat_size in zip(vis_feats[:-1], feat_sizes[:-1])
        ]

        self.vision_feats = vis_feats
        self.vision_pos_embeds = vis_pos_embed
        self.feat_sizes = feat_sizes

    @smart_inference_mode()
    def update_memory(
        self,
        obj_ids: list[int] | None = None,
        points: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> None:
        """将imgState追加到memory_bank并为模型更新记忆。

        Args:
            obj_ids (list[int]): 与提示对应的对象ID列表。
            points (torch.Tensor | None): 形状为(B, N, 2)的张量，表示N个对象的输入点。
            labels (torch.Tensor | None): 形状为(B, N)的张量，表示输入点的标签。
            masks (torch.Tensor | None): 可选张量，形状为(N, H, W)，表示N个对象的输入掩码。
        """
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            "pred_masks": torch.full(
                size=(self._max_obj_num, 1, self.imgsz[0] // 4, self.imgsz[1] // 4),
                fill_value=-1024.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
            "obj_ptr": torch.full(
                size=(self._max_obj_num, self.model.hidden_dim),
                fill_value=-1024.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
            "object_score_logits": torch.full(
                size=(self._max_obj_num, 1),
                # object_score_logits的默认值为10.0，即假设对象存在，
                # 因为sigmoid(10)=1，与`MaskDecoder`的`predict_masks`中相同
                fill_value=-32,  # 10.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
        }

        for i, obj_id in enumerate(obj_ids):
            assert obj_id < self._max_obj_num
            obj_idx = self._obj_id_to_idx(int(obj_id))
            self.obj_idx_set.add(obj_idx)
            point, label = points[[i]], labels[[i]]
            mask = masks[[i]][None] if masks is not None else None
            # 当前仅支持bbox提示或mask提示，因此断言bbox不为None。
            assert point is not None or mask is not None, "Either bbox, points or mask is required"
            out = self.track_step(obj_idx, point, label, mask)
            if out is not None:
                obj_mask = out["pred_masks"]
                assert obj_mask.shape[-2:] == consolidated_out["pred_masks"].shape[-2:], (
                    f"Expected mask shape {consolidated_out['pred_masks'].shape[-2:]} but got {obj_mask.shape[-2:]} for object {obj_idx}."
                )
                consolidated_out["pred_masks"][obj_idx : obj_idx + 1] = obj_mask
                consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]

                if "object_score_logits" in out:
                    consolidated_out["object_score_logits"][obj_idx : obj_idx + 1] = out["object_score_logits"]

        high_res_masks = F.interpolate(
            consolidated_out["pred_masks"].to(self.device, non_blocking=self.device.type == "cuda"),
            size=self.imgsz,
            mode="bilinear",
            align_corners=False,
        )

        if self.model.non_overlap_masks_for_mem_enc:
            high_res_masks = self.model._apply_non_overlapping_constraints(high_res_masks)
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=self.vision_feats,
            feat_sizes=self.feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=consolidated_out["object_score_logits"],
            is_mask_from_pts=True,
        )
        consolidated_out["maskmem_features"] = maskmem_features
        consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc
        self.memory_bank.append(consolidated_out)

    def _prepare_memory_conditioned_features(self, obj_idx: int | None) -> torch.Tensor:
        """为当前图像状态准备记忆条件化的特征。

        如果提供了``obj_idx``，则为图像中特定的提示对象准备特征。
        如果``obj_idx``为None，则为所有对象准备特征。
        如果没有可用的记忆，则将无记忆嵌入添加到当前视觉特征中。
        否则，使用先前帧的记忆通过transformer注意力机制来条件化当前视觉特征。

        Args:
            obj_idx (int | None): 要为其准备特征的对象索引。

        Returns:
            pix_feat_with_mem (torch.Tensor): 记忆条件化的像素特征。
        """
        if len(self.memory_bank) == 0 or isinstance(obj_idx, int):
            # 对于初始条件帧，不使用任何先前记忆进行编码。
            # 直接添加无记忆嵌入（而不是使用transformer编码器）。
            pix_feat_with_mem = self.vision_feats[-1] + self.model.no_mem_embed
        else:
            # 对于推理帧，使用先前帧的记忆特征
            memory, memory_pos_embed = self.get_maskmem_enc()
            pix_feat_with_mem = self.model.memory_attention(
                curr=self.vision_feats[-1:],
                curr_pos=self.vision_pos_embeds[-1:],
                memory=memory,
                memory_pos=memory_pos_embed,
                num_obj_ptr_tokens=0,  # 对象指针token数
            )
        # 重塑输出 (HW)BC => BCHW
        return pix_feat_with_mem.permute(1, 2, 0).view(
            self._max_obj_num,
            self.model.memory_attention.d_model,
            *self.feat_sizes[-1],
        )

    def get_maskmem_enc(self) -> tuple[torch.Tensor, torch.Tensor]:
        """从记忆中获取记忆和位置编码，用于条件化当前图像特征。"""
        to_cat_memory, to_cat_memory_pos_embed = [], []
        for consolidated_out in self.memory_bank:
            to_cat_memory.append(consolidated_out["maskmem_features"].flatten(2).permute(2, 0, 1))  # (H*W, B, C)
            maskmem_enc = consolidated_out["maskmem_pos_enc"][-1].flatten(2).permute(2, 0, 1)
            maskmem_enc = maskmem_enc + self.model.maskmem_tpos_enc[self.model.num_maskmem - 1]
            to_cat_memory_pos_embed.append(maskmem_enc)

        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
        return memory, memory_pos_embed

    def _obj_id_to_idx(self, obj_id: int) -> int | None:
        """将客户端对象ID映射到模型端对象索引。

        Args:
            obj_id (int): 客户端对象ID。

        Returns:
            (int | None): 模型端对象索引，如果未找到则为None。
        """
        return self.obj_id_to_idx.get(obj_id, None)

    def track_step(
        self,
        obj_idx: int | None = None,
        point: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """当前图像状态的追踪步骤以预测掩码。

        此方法处理图像特征并运行SAM头以预测掩码。如果提供了obj_idx，
        则为图像中特定的提示对象处理特征。如果obj_idx为None，则为图像中的所有对象处理特征。
        该方法支持两种输出方式：不带SAM的基于掩码的输出，以及带记忆条件化特征的完整SAM处理。

        Args:
            obj_idx (int | None): 要为其预测掩码的对象索引。如果为None，则处理所有对象。
            point (torch.Tensor | None): 兴趣点坐标，形状为(N, 2)。
            label (torch.Tensor | None): 对应点的标签，1表示正点击，0表示负点击。
            mask (torch.Tensor | None): 对象的掩码输入，形状为(H, W)。

        Returns:
            current_out (dict[str, Any]): 包含当前输出的字典，包括掩码预测和对象指针。
                键包括'point_inputs', 'mask_inputs', 'pred_masks', 'pred_masks_high_res',
                'obj_ptr', 'object_score_logits'。
        """
        if mask is not None and self.model.use_mask_input_as_output_without_sam:
            # 当use_mask_input_as_output_without_sam=True时，我们直接输出掩码输入
            # （将其视为GT掩码），而不使用SAM提示编码器+掩码解码器。
            pix_feat = self.vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.model.memory_attention.d_model, *self.feat_sizes[-1])
            _, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits = self.model._use_mask_as_output(mask)
        else:
            # 将视觉特征与记忆库中的先前记忆特征融合。
            pix_feat_with_mem = self._prepare_memory_conditioned_features(obj_idx)
            # 如果提供了``obj_idx``（即正在添加提示），仅保留第一个特征图。
            pix_feat_with_mem = pix_feat_with_mem[:1] if obj_idx is not None else pix_feat_with_mem
            _, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits = self.model._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs={"point_coords": point, "point_labels": label} if obj_idx is not None else None,
                mask_inputs=mask,
                multimask_output=False,
                high_res_features=[feat[: pix_feat_with_mem.shape[0]] for feat in self.high_res_features],
            )
        return {
            "pred_masks": low_res_masks,
            "pred_masks_high_res": high_res_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }


class SAM3Predictor(SAM2Predictor):
    """Segment Anything Model 3（SAM3）交互式预测器，用于图像分割任务。"""

    _bb_feat_sizes = [
        (288, 288),
        (144, 144),
        (72, 72),
    ]
    stride = 14

    def setup_model(self, model=None, verbose=True):
        """设置SAM3模型，使用适当的均值和标准差进行预处理。"""
        super().setup_model(model, verbose)
        # 更新均值和标准差
        self.mean = torch.tensor([127.5, 127.5, 127.5]).view(-1, 1, 1).to(self.device)
        self.std = torch.tensor([127.5, 127.5, 127.5]).view(-1, 1, 1).to(self.device)

    def get_model(self):
        """检索并初始化Segment Anything Model 3（SAM3）用于图像分割任务。"""
        from .build_sam3 import build_interactive_sam3  # 延迟导入

        return build_interactive_sam3(self.args.model, compile=self.args.compile)


class SAM3SemanticPredictor(SAM3Predictor):
    """Segment Anything Model 3（SAM3）预测器，用于图像分割任务。"""

    def get_model(self):
        """检索并初始化Segment Anything Model 3（SAM3）用于图像分割任务。"""
        from .build_sam3 import build_sam3_image_model  # 延迟导入

        return build_sam3_image_model(self.args.model, compile=self.args.compile)

    @smart_inference_mode()
    def get_im_features(self, im):
        """使用模型的骨干网络提取图像特征。"""
        return self.model.backbone.forward_image(im)

    def pre_transform(self, im):
        """对输入图像执行初始变换以进行预处理。

        此方法应用如调整大小等变换来准备图像以便进一步预处理。
        目前不支持批量推理，因此列表长度应为1。

        Args:
            im (list[np.ndarray]): 包含单张HWC numpy数组格式图像的列表。

        Returns:
            (list[np.ndarray]): 包含变换后图像的列表。

        Raises:
            AssertionError: 如果输入列表包含多于一张图像。

        Examples:
            >>> predictor = Predictor()
            >>> image = np.random.rand(480, 640, 3)  # 单张HWC图像
            >>> transformed = predictor.pre_transform([image])
            >>> print(len(transformed))
            1
        """
        assert len(im) == 1, "SAM model does not currently support batched inference"
        letterbox = LetterBox(self.imgsz, auto=False, center=False, scale_fill=True)  # SAM3此处硬编码
        return [letterbox(image=x) for x in im]

    def _prepare_geometric_prompts(self, src_shape, bboxes=None, labels=None):
        """通过将边界框和点归一化到目标形状来准备提示。"""
        if bboxes is not None:
            bboxes = torch.as_tensor(bboxes, dtype=self.torch_dtype, device=self.device)
            bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
            # 需要xywh格式作为输入
            bboxes = ops.xyxy2xywh(bboxes)
            bboxes[:, 0::2] /= src_shape[1]
            bboxes[:, 1::2] /= src_shape[0]
            # 如果用户未传入标签，默认所有标签为正
            if labels is None:
                labels = np.ones(bboxes.shape[:-1])
            labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
            assert bboxes.shape[-2] == labels.shape[-1], (
                f"Number of points {bboxes.shape[-2]} should match number of labels {labels.shape[-1]}."
            )
            bboxes = bboxes.view(-1, 1, 4)  # (N, 1, 4)
            labels = labels.view(-1, 1)  # (N, 1)
        return bboxes, labels

    def _inference_features(self, features, bboxes=None, labels=None, text: list[str] | None = None):
        """对提取的特征执行推理，可选地带有边界框和标签。"""
        # 注意：优先级：边界框 > 文本 > 预设类别
        nc = 1 if bboxes is not None else len(text) if text is not None else len(self.model.names)
        geometric_prompt = None
        if bboxes is not None:
            geometric_prompt = self._get_dummy_prompt(nc)
            for i in range(len(bboxes)):
                geometric_prompt.append_boxes(bboxes[[i]], labels[[i]])
            if text is None:
                text = ["visual"]  # 如果未传入文本，边界框需要此`visual`文本提示
        if text is not None and self.model.names != text:
            self.model.set_classes(text=text)
        outputs = self.model.forward_grounding(
            backbone_out=features,
            text_ids=torch.arange(nc, device=self.device, dtype=torch.long),
            geometric_prompt=geometric_prompt,
        )
        return outputs

    def postprocess(self, preds, img, orig_imgs):
        """后处理预测结果，必要时应用非重叠约束。"""
        import torchvision

        pred_boxes = preds["pred_boxes"]  # (nc, num_query, 4)
        pred_logits = preds["pred_logits"]
        pred_masks = preds["pred_masks"]
        pred_scores = pred_logits.sigmoid()
        presence_score = preds["presence_logit_dec"].sigmoid().unsqueeze(1)
        pred_scores = (pred_scores * presence_score).squeeze(-1)
        pred_cls = torch.tensor(
            list(range(pred_scores.shape[0])),
            dtype=pred_scores.dtype,
            device=pred_scores.device,
        )[:, None].expand_as(pred_scores)
        pred_boxes = torch.cat([pred_boxes, pred_scores[..., None], pred_cls[..., None]], dim=-1)

        keep = pred_scores > self.args.conf
        pred_masks, pred_boxes = pred_masks[keep], pred_boxes[keep]
        pred_boxes[:, :4] = ops.xywh2xyxy(pred_boxes[:, :4])

        c = pred_boxes[:, 5:6] * (0 if self.args.agnostic_nms else 7680)  # 类别
        nms_boxes = pred_boxes[:, :4] + c  # 框（按类别偏移）
        keep = torchvision.ops.nms(nms_boxes, pred_boxes[:, 4], self.args.iou)  # 非极大值抑制
        pred_boxes, pred_masks = pred_boxes[keep], pred_masks[keep]

        names = getattr(self.model, "names", [str(i) for i in range(pred_scores.shape[0])])
        if not isinstance(orig_imgs, list):  # 输入图像是torch.Tensor，不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        results = []
        for masks, boxes, orig_img, img_path in zip([pred_masks], [pred_boxes], orig_imgs, self.batch[0]):
            if masks.shape[0] == 0:
                masks, boxes = None, torch.zeros((0, 6), device=pred_masks.device)
            else:
                masks = F.interpolate(masks.float()[None], orig_img.shape[:2], mode="bilinear")[0] > 0.5
                boxes[..., [0, 2]] *= orig_img.shape[1]
                boxes[..., [1, 3]] *= orig_img.shape[0]
            results.append(Results(orig_img, path=img_path, names=names, masks=masks, boxes=boxes))
        return results

    def inference(self, im, bboxes=None, labels=None, text: list[str] | None = None, *args, **kwargs):
        """对单张图像执行推理，可选地带有提示。"""
        bboxes = self.prompts.pop("bboxes", bboxes)
        labels = self.prompts.pop("labels", labels)
        text = self.prompts.pop("text", text)
        features = self.get_im_features(im) if self.features is None else self.features
        prompts = self._prepare_geometric_prompts(self.batch[1][0].shape[:2], bboxes, labels)
        return self._inference_features(features, *prompts, text=text)

    @smart_inference_mode()
    def inference_features(
        self,
        features,
        src_shape,
        bboxes=None,
        labels=None,
        text: list[str] | None = None,
    ):
        """对提供的图像特征执行提示预处理和推理，使用SAM3模型。

        Args:
            features (dict[str, Any]): 从SAM3模型图像编码器提取的图像特征。
            src_shape (tuple[int, int]): 输入图像的源形状（高度，宽度）。
            bboxes (np.ndarray | list[list[float]] | None): xyxy格式的边界框，形状为(N, 4)，单位为像素。
            labels (np.ndarray | list[int] | None): 点提示标签，形状为(N,)。
            text (list[str] | None): 与类别对应的文本提示列表。

        Returns:
            pred_masks (torch.Tensor): 输出掩码，形状为(C, H, W)，其中C是生成的掩码数量。
            pred_bboxes (torch.Tensor): 每个掩码的边界框，形状为(N, 6)，其中N是框的数量。
                每个框为xyxy格式，附带有分数和类别的额外列。

        Notes:
            - 输入特征在SAM上为形状(B, C, H, W)的torch.Tensor，在SAM2上为dict[str, Any]。
        """
        import torchvision

        prompts = self._prepare_geometric_prompts(src_shape[:2], bboxes, labels)
        preds = self._inference_features(features, *prompts, text=text)
        pred_boxes = preds["pred_boxes"]  # (nc, num_query, 4)
        pred_logits = preds["pred_logits"]
        pred_masks = preds["pred_masks"]
        pred_scores = pred_logits.sigmoid()
        presence_score = preds["presence_logit_dec"].sigmoid().unsqueeze(1)
        pred_scores = (pred_scores * presence_score).squeeze(-1)
        pred_cls = torch.tensor(
            list(range(pred_scores.shape[0])),
            dtype=pred_scores.dtype,
            device=pred_scores.device,
        )[:, None].expand_as(pred_scores)
        pred_boxes = torch.cat([pred_boxes, pred_scores[..., None], pred_cls[..., None]], dim=-1)

        keep = pred_scores > self.args.conf
        pred_masks, pred_boxes = pred_masks[keep], pred_boxes[keep]
        pred_boxes[:, :4] = ops.xywh2xyxy(pred_boxes[:, :4])

        c = pred_boxes[:, 5:6] * (0 if self.args.agnostic_nms else 7680)  # 类别
        nms_boxes = pred_boxes[:, :4] + c  # 框（按类别偏移）
        keep = torchvision.ops.nms(nms_boxes, pred_boxes[:, 4], self.args.iou)  # 非极大值抑制
        pred_boxes, pred_masks = pred_boxes[keep], pred_masks[keep]

        if pred_masks.shape[0] == 0:
            pred_masks, pred_boxes = None, torch.zeros((0, 6), device=pred_masks.device)
        else:
            pred_masks = F.interpolate(pred_masks.float()[None], src_shape[:2], mode="bilinear")[0] > 0.5
            pred_boxes[..., 0] *= src_shape[1]
            pred_boxes[..., 1] *= src_shape[0]
            pred_boxes[..., 2] *= src_shape[1]
            pred_boxes[..., 3] *= src_shape[0]
        return pred_masks, pred_boxes

    def reset_prompts(self):
        """重置预测器的提示。"""
        self.prompts = {}
        self.model.text_embeddings = {}

    def _get_dummy_prompt(self, num_prompts=1):
        """获取一个带有零框的虚拟几何提示。"""
        geometric_prompt = Prompt(
            box_embeddings=torch.zeros(0, num_prompts, 4, device=self.device),
            box_mask=torch.zeros(num_prompts, 0, device=self.device, dtype=torch.bool),
        )
        return geometric_prompt


class SAM3VideoPredictor(SAM2VideoPredictor, SAM3Predictor):
    """Segment Anything Model 3（SAM3）视频预测器，用于视频分割任务。"""

    def propagate_in_video(self, inference_state, frame_idx):
        """基于给定输入提示执行图像分割推理，使用当前加载的图像。此方法利用SAM（Segment Anything Model）
        的架构，包括图像编码器、提示编码器和掩码解码器，用于实时和可提示的分割任务。

        Args:
            inference_state (dict): 当前推理状态，包括输入提示和先前的输出。
            frame_idx (int): 视频序列中当前帧的索引。
        """
        frame = frame_idx
        output_dict = inference_state["output_dict"]
        obj_ids = inference_state["obj_ids"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        batch_size = len(inference_state["obj_idx_to_id"])
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")

        if frame in consolidated_frame_inds["cond_frame_outputs"]:
            storage_key = "cond_frame_outputs"
            current_out = output_dict[storage_key][frame]
            if self.clear_non_cond_mem_around_input and (self.clear_non_cond_mem_for_multi_obj or batch_size <= 1):
                # 清除周围帧的非条件记忆
                self._clear_non_cond_mem_around_input(frame)
        elif frame in consolidated_frame_inds["non_cond_frame_outputs"]:
            storage_key = "non_cond_frame_outputs"
            current_out = output_dict[storage_key][frame]
        else:
            storage_key = "non_cond_frame_outputs"
            current_out = self._run_single_frame_inference(
                output_dict=output_dict,
                frame_idx=frame,
                batch_size=batch_size,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
                inference_state=inference_state,
            )
            output_dict[storage_key][frame] = current_out
            self._prune_non_cond_memory(frame, inference_state=inference_state)
        # 创建每对象输出切片，用于追踪后与每个单独对象进行后续交互。
        self._add_output_per_object(frame, current_out, storage_key, inference_state=inference_state)
        inference_state["frames_already_tracked"].append(frame)
        pred_masks = current_out["pred_masks"].flatten(0, 1)
        obj_scores = current_out["object_score_logits"]

        return obj_ids, pred_masks, obj_scores


class SAM3VideoSemanticPredictor(SAM3SemanticPredictor):
    """Segment Anything Model 3（SAM3）视频语义预测器。"""

    HIGH_CONF_THRESH = 0.8
    HIGH_IOU_THRESH = 0.8
    NO_OBJ_LOGIT = -10.0
    NEVER_OCCLUDED = -1
    ALWAYS_OCCLUDED = 100000

    UNCONFIRMED = 1  # 新添加的masklet，尚未被任何检测确认
    CONFIRMED = 2  # 已被至少一个检测确认
    _bb_feat_sizes = [
        (288, 288),
        (144, 144),
        (72, 72),
    ]
    stride = 14

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides=None,
        _callbacks: dict | None = None,
        # 检测输出的概率阈值 -- 仅保留高于此阈值的检测
        # 用于NMS和检测-追踪匹配
        score_threshold_detection=0.5,
        # 检测NMS的IoU阈值
        det_nms_thresh=0.0,
        # 检测-追踪匹配的IoU阈值 -- 当检测与追踪片段的重叠高于此阈值时，视为"已匹配"
        # 通常是一个宽松的阈值，如0.1
        assoc_iou_thresh=0.5,
        # 检测-追踪匹配的IoU阈值，用于判断masklet是否被任何检测"未匹配"
        # 通常是一个更严格的阈值，如0.5
        trk_assoc_iou_thresh=0.5,
        # 检测作为新对象添加的概率阈值
        new_det_thresh=0.0,
        # 热启动参数：我们延迟`hotstart_delay`帧的输出，
        # 1) 基于`hotstart_unmatch_thresh`移除未被任何检测匹配的追踪片段
        # 2) 基于`hotstart_dup_thresh`移除相互重叠的追踪片段
        hotstart_delay=0,
        hotstart_unmatch_thresh=3,
        hotstart_dup_thresh=3,
        init_trk_keep_alive=10,
        max_trk_keep_alive=10,
        min_trk_keep_alive=-4,
        # 基于最近遮挡抑制重叠对象的阈值
        suppress_overlapping_based_on_recent_occlusion_threshold=0.0,
        decrease_trk_keep_alive_for_empty_masklets=True,
        o2o_matching_masklets_enable=False,  # 启用匈牙利匹配来匹配现有masklet
        suppress_det_close_to_boundary=False,
        fill_hole_area=16,
        # 在所有GPU上追踪的对象（masklet）的最大数量（无限制设为-1）
        max_num_objects=-1,
        recondition_every_nth_frame=-1,
        # masklet确认状态（用于抑制未确认的masklet）
        masklet_confirmation_enable=True,
        # 一个masklet在连续被检测和匹配达到`masklet_confirmation_consecutive_det_thresh`次后被确认
        masklet_confirmation_consecutive_det_thresh=3,
        # 边界框启发式参数
        reconstruction_bbox_iou_thresh=0.0,
        reconstruction_bbox_det_score=0.0,
    ):
        """使用配置和可选覆盖初始化SAM3VideoSemanticPredictor。"""
        super().__init__(cfg, overrides, _callbacks)
        self.score_threshold_detection = score_threshold_detection
        self.det_nms_thresh = det_nms_thresh
        self.assoc_iou_thresh = assoc_iou_thresh
        self.trk_assoc_iou_thresh = trk_assoc_iou_thresh
        self.new_det_thresh = new_det_thresh

        # hotstart 参数
        if hotstart_delay > 0:
            assert hotstart_unmatch_thresh <= hotstart_delay
            assert hotstart_dup_thresh <= hotstart_delay
        self.hotstart_delay = hotstart_delay
        self.hotstart_unmatch_thresh = hotstart_unmatch_thresh
        self.hotstart_dup_thresh = hotstart_dup_thresh
        self.init_trk_keep_alive = init_trk_keep_alive
        self.max_trk_keep_alive = max_trk_keep_alive
        self.min_trk_keep_alive = min_trk_keep_alive
        self.suppress_overlapping_based_on_recent_occlusion_threshold = (
            suppress_overlapping_based_on_recent_occlusion_threshold
        )
        self.suppress_det_close_to_boundary = suppress_det_close_to_boundary
        self.decrease_trk_keep_alive_for_empty_masklets = decrease_trk_keep_alive_for_empty_masklets
        self.o2o_matching_masklets_enable = o2o_matching_masklets_enable
        self.fill_hole_area = fill_hole_area
        self._dist_pg_cpu = None  # CPU进程组（首次使用时延迟初始化）

        max_num_objects = 10000  # 无限制
        num_obj_for_compile = 16
        self.max_num_objects = max_num_objects
        self.num_obj_for_compile = num_obj_for_compile
        self.recondition_every_nth_frame = recondition_every_nth_frame
        self.masklet_confirmation_enable = masklet_confirmation_enable
        self.masklet_confirmation_consecutive_det_thresh = masklet_confirmation_consecutive_det_thresh
        self.reconstruction_bbox_iou_thresh = reconstruction_bbox_iou_thresh
        self.reconstruction_bbox_det_score = reconstruction_bbox_det_score

        # 构建SAM3追踪器
        self.tracker = SAM3VideoPredictor(overrides=overrides)

        self.inference_state = {}
        self.callbacks["on_predict_start"].append(self.init_state)

    def setup_model(self, model=None, verbose=True):
        """设置SAM3VideoSemanticPredictor模型。"""
        super().setup_model(model, verbose)
        from .build_sam3 import build_interactive_sam3

        # 初始化不带骨干网络的SAM3追踪器模型（骨干网络在检测器中处理）
        model = build_interactive_sam3(self.args.model, with_backbone=False)
        self.tracker.setup_model(model=model, verbose=False)

    def setup_source(self, source):
        """为SAM3VideoSemanticPredictor模型设置数据源。"""
        super().setup_source(source)
        self.tracker.imgsz = self.imgsz
        self.tracker.model.set_imgsz(self.imgsz)
        self.tracker._bb_feat_sizes = [[int(x / (self.stride * i)) for x in self.imgsz] for i in [1 / 4, 1 / 2, 1]]
        self.interpol_size = self.tracker.model.memory_encoder.mask_downsampler.interpol_size

    @staticmethod
    def init_state(predictor):
        """为预测器初始化推理状态。

        此函数设置执行视频数据推理所需的初始状态。包括初始化各种字典和有序字典，
        用于存储与追踪过程相关的输入、输出和其他元数据。

        Args:
            predictor (SAM3VideoSemanticPredictor): 需要初始化状态的预测器对象。
        """
        if len(predictor.inference_state) > 0:  # 表示已初始化
            return
        assert predictor.dataset is not None
        assert predictor.dataset.mode == "video"
        num_frames = predictor.dataset.frames
        inference_state = {
            "num_frames": num_frames,
            "tracker_inference_states": [],
            "tracker_metadata": {},
            "text_prompt": None,
            "per_frame_geometric_prompt": [None] * num_frames,
        }
        predictor.inference_state = inference_state

    def inference(self, im, bboxes=None, labels=None, text: list[str] | None = None, *args, **kwargs):
        """对视频序列执行推理，可选地带有提示。"""
        frame = self.dataset.frame - 1  # 将帧索引对齐为从0开始
        self.inference_state["im"] = im  # 仅为后续帧传递图像
        if "text_ids" not in self.inference_state:  # 首帧处理
            self.add_prompt(frame_idx=frame, text=text, bboxes=bboxes, labels=labels)
        return self._run_single_frame_inference(frame, reverse=False)

    def postprocess(self, preds, img, orig_imgs):
        """后处理预测结果，必要时应用非重叠约束。"""
        obj_id_to_mask = preds["obj_id_to_mask"]  # 低分辨率掩码
        curr_obj_ids = sorted(obj_id_to_mask.keys())
        if not isinstance(orig_imgs, list):  # 输入图像是torch.Tensor，不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        names = self.model.names if self.model.names != "visual" else {}
        if len(curr_obj_ids) == 0:
            pred_masks, pred_boxes = None, torch.zeros((0, 7), device=self.device)
        else:
            pred_masks = torch.cat([obj_id_to_mask[obj_id] for obj_id in curr_obj_ids], dim=0)
            pred_masks = F.interpolate(pred_masks.float()[None], orig_imgs[0].shape[:2], mode="bilinear")[0] > 0.5
            pred_ids = torch.tensor(curr_obj_ids, dtype=torch.int32, device=pred_masks.device)
            pred_scores = torch.tensor(
                [preds["obj_id_to_score"][obj_id] for obj_id in curr_obj_ids], device=pred_masks.device
            )
            pred_cls = torch.tensor(
                [preds["obj_id_to_cls"][obj_id] for obj_id in curr_obj_ids], device=pred_masks.device
            )
            keep = (pred_scores > self.args.conf) & pred_masks.any(dim=(1, 2))
            pred_masks = pred_masks[keep]
            pred_boxes = batched_mask_to_box(pred_masks)
            pred_boxes = torch.cat(
                [pred_boxes, pred_ids[keep][:, None], pred_scores[keep][..., None], pred_cls[keep][..., None]], dim=-1
            )
            if pred_boxes.shape[0]:
                names = names or dict(enumerate(str(i) for i in range(pred_boxes[:, 6].int().max() + 1)))
            if pred_masks.shape[0] > 1:
                tracker_scores = torch.tensor(
                    [
                        (
                            preds["obj_id_to_tracker_score"][obj_id]
                            if obj_id in preds["obj_id_to_tracker_score"]
                            else 0.0
                        )
                        for obj_id in curr_obj_ids
                    ],
                    device=pred_masks.device,
                )[keep]
                pred_masks = (
                    self._apply_object_wise_non_overlapping_constraints(
                        pred_masks.unsqueeze(1),
                        tracker_scores.unsqueeze(1),
                        background_value=0,
                    ).squeeze(1)
                ) > 0

        results = []
        for masks, boxes, orig_img, img_path in zip([pred_masks], [pred_boxes], orig_imgs, self.batch[0]):
            results.append(Results(orig_img, path=img_path, names=names, masks=masks, boxes=boxes))
        return results

    def _run_single_frame_inference(self, frame_idx, reverse=False, inference_state=None):
        """对单帧执行推理并获取推理结果。"""
        inference_state = inference_state or self.inference_state
        # 准备输入
        tracker_states_local = inference_state["tracker_inference_states"]
        has_text_prompt = inference_state["text_prompt"] is not None
        has_geometric_prompt = inference_state["per_frame_geometric_prompt"][frame_idx] is not None
        # 对当前帧运行推理
        (
            obj_id_to_mask,
            obj_id_to_score,
            obj_id_to_cls,
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            _,
        ) = self._det_track_one_frame(
            frame_idx=frame_idx,
            num_frames=inference_state["num_frames"],
            reverse=reverse,
            im=inference_state["im"],
            text_ids=inference_state["text_ids"],
            geometric_prompt=(
                self._get_dummy_prompt(num_prompts=len(inference_state["text_ids"]))
                if not has_geometric_prompt
                else inference_state["per_frame_geometric_prompt"][frame_idx]
            ),
            tracker_states_local=tracker_states_local,
            tracker_metadata_prev=inference_state["tracker_metadata"],
            allow_new_detections=has_text_prompt or has_geometric_prompt,
        )
        # 更新推理状态
        inference_state["tracker_inference_states"] = tracker_states_local_new
        inference_state["tracker_metadata"] = tracker_metadata_new

        out = {
            "obj_id_to_mask": obj_id_to_mask,
            "obj_id_to_score": obj_id_to_score,  # 首帧检测分数
            "obj_id_to_cls": obj_id_to_cls,  # 首帧检测分数
            "obj_id_to_tracker_score": tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx],
        }
        # removed_obj_ids仅在rank 0上需要以处理热启动延迟缓冲
        metadata = tracker_metadata_new["metadata"]
        removed_obj_ids = metadata["removed_obj_ids"]
        out["removed_obj_ids"] = removed_obj_ids
        out["frame_stats"] = frame_stats
        if self.masklet_confirmation_enable:
            status = metadata["masklet_confirmation"]["status"]
            is_unconfirmed = status == self.UNCONFIRMED
            out["unconfirmed_obj_ids"] = tracker_metadata_new["obj_ids"][is_unconfirmed].tolist()
        else:
            out["unconfirmed_obj_ids"] = []
        return out

    @smart_inference_mode()
    def add_prompt(
        self,
        frame_idx,
        text=None,
        bboxes=None,
        labels=None,
        inference_state=None,
    ):
        """在单帧上添加文本、点或框提示。此方法仅在提示帧上返回推理输出。

        注意文本提示不与特定帧关联（即它们适用于所有帧）。
        但是我们只在`frame_idx`指定的帧上运行推理。
        """
        inference_state = inference_state or self.inference_state
        assert text is not None or bboxes is not None, "at least one type of prompt (text, boxes) must be provided"

        # 1) 处理文本提示
        use_text = text is not None
        text = text if use_text else "visual"
        text_batch = [text] if isinstance(text, str) else text
        inference_state["text_prompt"] = text if use_text else None
        n = len(text_batch)
        text_ids = torch.arange(n, device=self.device, dtype=torch.long)
        inference_state["text_ids"] = text_ids
        if text is not None and self.model.names != text:
            self.model.set_classes(text=text)

        # 2) 处理框提示
        bboxes, labels = self._prepare_geometric_prompts(self.batch[1][0].shape[:2], bboxes, labels)
        assert (bboxes is not None) == (labels is not None)
        geometric_prompt = self._get_dummy_prompt(num_prompts=n)
        if bboxes is not None:
            for i in range(len(bboxes)):
                geometric_prompt.append_boxes(bboxes[[i]], labels[[i]])
        inference_state["per_frame_geometric_prompt"][frame_idx] = geometric_prompt
        out = self._run_single_frame_inference(frame_idx, reverse=False, inference_state=inference_state)
        return frame_idx, out

    def _apply_object_wise_non_overlapping_constraints(self, pred_masks, obj_scores, background_value=-10.0):
        """逐对象应用非重叠约束（即只有一个对象可以占据重叠区域）。"""
        # 用对象分数替换像素分数
        pred_masks_single_score = torch.where(pred_masks > 0, obj_scores[..., None, None], background_value)
        # 基于掩码分数应用像素级非重叠约束
        pixel_level_non_overlapping_masks = self.tracker.model._apply_non_overlapping_constraints(
            pred_masks_single_score
        )
        # 用像素分数替换对象分数。注意，现在只有一个对象可以占据重叠区域
        pred_masks = torch.where(
            pixel_level_non_overlapping_masks > 0,
            pred_masks,
            torch.clamp(pred_masks, max=background_value),
        )
        return pred_masks

    def _det_track_one_frame(
        self,
        im: torch.Tensor,
        text_ids: torch.Tensor,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        geometric_prompt: Prompt,
        tracker_states_local: list[Any],
        tracker_metadata_prev: dict[str, Any],
        allow_new_detections: bool = True,
    ):
        """此函数以SPMD方式处理DenseTracking模型的一步推理。在高层次上，所有GPU
        执行相同的函数调用，就像在单个GPU上一样，但在底层，某些函数调用
        涉及基于分片SAM2状态的分布式计算。

        - `input_batch`包含整个视频的图像和其他输入；它应在各GPU间相同
        - `tracker_states_local`保存此GPU分片中的本地masklet信息
        - `tracker_metadata_prev`管理SAM2对象的元数据，如哪个masklet在哪个GPU上
          它包含全局和本地masklet信息
        """
        # 步骤1：以分布式方式运行骨干网络和检测器 -- 通过Sam3ImageOnVideoMultiGPU实现，
        # 这是一个多GPU模型（分配给`self.detector`），以轮询方式分片帧。
        det_out = self.run_backbone_and_detection(
            im=im,
            text_ids=text_ids,
            geometric_prompt=geometric_prompt,
            allow_new_detections=allow_new_detections,
        )

        # 步骤2：每个GPU传播其本地SAM2状态以获取SAM2预测掩码。
        # 返回的`tracker_low_res_masks_global`包含从所有GPU收集的串联masklet预测
        # （就像在单个GPU上传播一样）。注意此步骤仅运行SAM2传播步骤，
        # 但不为预测的掩码编码新记忆；我们在解决所有启发式规则后在
        # `run_tracker_update_execution_phase`中延迟进行记忆编码。
        if tracker_metadata_prev == {}:
            # 如果masklet元数据未初始化（空字典）则初始化
            tracker_metadata_prev.update(self._initialize_metadata())
        tracker_low_res_masks_global, tracker_obj_scores_global = self.run_tracker_propagation(
            frame_idx=frame_idx,
            tracker_states_local=tracker_states_local,
            tracker_metadata_prev=tracker_metadata_prev,
        )

        # 步骤3：基于检测输出和传播的SAM2预测掩码，我们为SAM2 masklet更新制定计划
        # （即添加和移除哪些对象，如何负载均衡等）。
        # 我们还在这一步全局运行SAM2记忆编码器以解决非重叠约束。
        # **此步骤应包含所有更新所需的启发式规则。** 大多数更新规划将在
        # 主rank（GPU 0）上完成，生成的计划`tracker_update_plan`被广播到
        # 其他GPU（以分布式方式执行）。此步骤还生成新的masklet元数据
        # `tracker_metadata_new`（基于其先前版本`tracker_metadata_prev`）。
        tracker_update_plan, tracker_metadata_new = self.run_tracker_update_planning_phase(
            frame_idx=frame_idx,
            reverse=reverse,
            det_out=det_out,
            tracker_low_res_masks_global=tracker_low_res_masks_global,
            tracker_obj_scores_global=tracker_obj_scores_global,
            tracker_metadata_prev=tracker_metadata_prev,
            tracker_states_local=tracker_states_local,
        )

        # 从更新计划中获取重条件化信息
        reconditioned_obj_ids = tracker_update_plan.get("reconditioned_obj_ids", set())

        # 步骤4：基于`tracker_update_plan`，每个GPU对其本地SAM2推理状态执行更新
        tracker_states_local_new = self.run_tracker_update_execution_phase(
            frame_idx=frame_idx,
            num_frames=num_frames,
            det_out=det_out,
            tracker_states_local=tracker_states_local,
            tracker_update_plan=tracker_update_plan,
        )

        # 步骤5：最后，为当前帧构建输出（只需要在GPU 0上完成，因为只有GPU 0将输出发送到服务器）。
        obj_id_to_mask = self.build_outputs(
            det_out=det_out,
            tracker_low_res_masks_global=tracker_low_res_masks_global,
            tracker_metadata_prev=tracker_metadata_prev,
            tracker_update_plan=tracker_update_plan,
            reconditioned_obj_ids=reconditioned_obj_ids,
        )
        obj_id_to_score = tracker_metadata_new["obj_id_to_score"]
        obj_id_to_cls = tracker_metadata_new["obj_id_to_cls"]
        # 当前帧的一些统计信息作为输出的一部分
        frame_stats = {
            "num_obj_tracked": np.sum(tracker_metadata_new["num_obj"]),
            "num_obj_dropped": tracker_update_plan["num_obj_dropped_due_to_limit"],
        }
        # 将追踪器分数添加到元数据中，除首帧外的帧都应触发
        if tracker_obj_scores_global.shape[0] > 0:
            # 更新前将tracker_obj_scores_global转换为sigmoid分数
            tracker_obj_scores_global = tracker_obj_scores_global.sigmoid().tolist()
            tracker_obj_ids = tracker_metadata_prev["obj_ids"]
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx].update(
                dict(zip(tracker_obj_ids, tracker_obj_scores_global))
            )
        return (
            obj_id_to_mask,  # 字典: obj_id --> 输出掩码
            obj_id_to_score,  # 字典: obj_id --> 输出分数（概率）
            obj_id_to_cls,  # 字典: obj_id --> 输出类别（整数）
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            tracker_obj_scores_global,  # 字典: obj_id --> 追踪器帧级分数
        )

    @staticmethod
    def _suppress_detections_close_to_boundary(boxes, margin=0.025):
        """抑制靠近图像边缘的检测（适用于归一化框）。

        boxes: (N, 4) xyxy格式，归一化到[0,1]
        margin: 图像边缘的比例
        """
        x_min, y_min, x_max, y_max = boxes.unbind(-1)
        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2
        keep = (x_c > margin) & (x_c < 1.0 - margin) & (y_c > margin) & (y_c < 1.0 - margin)

        return keep

    def run_backbone_and_detection(
        self, im: torch.Tensor, text_ids: torch.Tensor, geometric_prompt: Prompt, allow_new_detections: bool
    ):
        """对单帧运行骨干网络和检测。"""
        features = self.get_im_features(im)
        sam3_image_out = self.model.forward_grounding(
            backbone_out=features, text_ids=text_ids, geometric_prompt=geometric_prompt
        )
        det_out = self._extract_detection_outputs(sam3_image_out, allow_new_detections)
        self._cache_backbone_features(sam3_image_out)
        return det_out

    def _extract_detection_outputs(self, sam3_image_out, allow_new_detections):
        """提取并过滤检测输出。"""
        pred_probs = sam3_image_out["pred_logits"].squeeze(-1).sigmoid()
        if not allow_new_detections:
            pred_probs = pred_probs - 1e8

        pred_cls = torch.tensor(
            list(range(pred_probs.shape[0])),
            dtype=pred_probs.dtype,
            device=pred_probs.device,
        )[:, None].expand_as(pred_probs)

        pred_boxes_xyxy = sam3_image_out["pred_boxes_xyxy"]
        pred_masks = sam3_image_out["pred_masks"]

        keep = pred_probs > self.score_threshold_detection
        return {
            "bbox": pred_boxes_xyxy[keep],
            "mask": pred_masks[keep],
            "scores": pred_probs[keep],
            "cls": pred_cls[keep],
        }

    def _cache_backbone_features(self, sam3_image_out):
        """构建并缓存SAM2骨干网络特征。"""
        sam_mask_decoder = self.tracker.model.sam_mask_decoder
        feats = sam3_image_out["backbone_out"]["sam2_backbone_out"]
        tracker_backbone_fpn = [
            sam_mask_decoder.conv_s0(feats["backbone_fpn"][0]),
            sam_mask_decoder.conv_s1(feats["backbone_fpn"][1]),
            feats["backbone_fpn"][2],
        ]
        tracker_backbone_out = {
            "vision_features": tracker_backbone_fpn[-1],
            "vision_pos_enc": feats["vision_pos_enc"],
            "backbone_fpn": tracker_backbone_fpn,
        }
        # 在追踪器中缓存frame_idx的SAM2骨干网络特征
        self.tracker.backbone_out = tracker_backbone_out

    def run_tracker_propagation(
        self, frame_idx: int, tracker_states_local: list[Any], tracker_metadata_prev: dict[str, np.ndarray]
    ):
        """以SPMD方式为单帧运行追踪器传播阶段。"""
        # 步骤1: 传播本地SAM2状态以获取当前帧的预测
        # 此GPU上现有masklet的`low_res_masks_local`
        # - obj_ids_local: list[int] -- 对象ID列表
        # - low_res_masks_local: Tensor -- (num_local_obj, H_mask, W_mask)
        obj_ids_local, low_res_masks_local, obj_scores_local = self._propogate_tracker_one_frame_local_gpu(
            tracker_states_local, frame_idx=frame_idx
        )

        assert np.all(obj_ids_local == tracker_metadata_prev["obj_ids"]), "{} != {}".format(
            obj_ids_local, tracker_metadata_prev["obj_ids"]
        )

        # 步骤2: 将`low_res_masks_local`全收集为`low_res_masks_global`
        # - low_res_masks_global: Tensor -- (num_global_obj, H_mask, W_mask)
        low_res_masks_global = low_res_masks_local
        obj_scores_global = obj_scores_local
        return low_res_masks_global, obj_scores_global

    def _recondition_masklets(
        self,
        frame_idx,
        det_out: dict[str, torch.Tensor],
        trk_id_to_max_iou_high_conf_det: list[int],
        tracker_states_local: list[Any],
        tracker_metadata: dict[str, np.ndarray],
        tracker_obj_scores_global: torch.Tensor,
    ):
        """基于新的高置信度检测对masklet进行重条件化。"""
        # 基于新的检测对masklet进行重条件化
        for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
            new_mask = det_out["mask"][det_idx : det_idx + 1]
            new_mask_binary = (
                F.interpolate(new_mask.unsqueeze(1), size=self.interpol_size, mode="bilinear", align_corners=False) > 0
            )
            HIGH_CONF_THRESH = 0.8
            reconditioned_states_idx = set()
            obj_idx = np.where(tracker_metadata["obj_ids"] == trk_obj_id)[0].item()
            obj_score = tracker_obj_scores_global[obj_idx]
            for state_idx, inference_state in enumerate(tracker_states_local):
                if (
                    trk_obj_id in inference_state["obj_ids"]
                    # 注意：此条件的目的是避免对被遮挡/低质量的掩码进行重条件化。
                    # 不幸的是，由于批处理的原因，这些仍可能被重条件化。我们应考虑移除这些启发式规则。
                    and obj_score > HIGH_CONF_THRESH
                ):
                    LOGGER.debug(
                        f"Adding new mask for track {trk_obj_id} at frame {frame_idx}. Objects {inference_state['obj_ids']} are all reconditioned."
                    )
                    self.tracker.add_new_prompts(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=trk_obj_id,
                        masks=new_mask_binary,
                    )
                    reconditioned_states_idx.add(state_idx)

            for idx in reconditioned_states_idx:
                self.tracker.propagate_in_video_preflight(tracker_states_local[idx])
        return tracker_states_local

    def run_tracker_update_planning_phase(
        self,
        frame_idx: int,
        reverse: bool,
        det_out: dict[str, torch.Tensor],
        tracker_low_res_masks_global: torch.Tensor,
        tracker_obj_scores_global: torch.Tensor,
        tracker_metadata_prev: dict[str, np.ndarray],
        tracker_states_local: list[Any],
    ):
        """以SPMD方式为单帧运行追踪器更新规划阶段。"""
        # 从先前的元数据初始化新元数据（其值将在后续更新）
        tracker_metadata_new = {
            "obj_ids": deepcopy(tracker_metadata_prev["obj_ids"]),
            "num_obj": deepcopy(tracker_metadata_prev["num_obj"]),
            "obj_id_to_score": deepcopy(tracker_metadata_prev["obj_id_to_score"]),
            "obj_id_to_cls": deepcopy(tracker_metadata_prev["obj_id_to_cls"]),
            "obj_id_to_tracker_score_frame_wise": deepcopy(tracker_metadata_prev["obj_id_to_tracker_score_frame_wise"]),
            "obj_id_to_last_occluded": {},  # 将在后续填充
            "max_obj_id": deepcopy(tracker_metadata_prev["max_obj_id"]),
        }

        # 提前初始化reconditioned_obj_ids以避免UnboundLocalError
        reconditioned_obj_ids = set()

        # 步骤1: 在GPU 0上制定更新计划并解决启发式规则
        det_mask_preds: torch.Tensor = det_out["mask"]  # 低分辨率掩码logits
        det_scores_np: np.ndarray = det_out["scores"].float().cpu().numpy()
        det_cls_np: np.ndarray = det_out["cls"].float().cpu().numpy()
        det_bbox_xyxy: torch.Tensor = det_out["bbox"]
        # a) 匹配检测器和追踪器掩码并查找新对象
        (
            new_det_fa_inds,
            unmatched_trk_obj_ids,
            det_to_matched_trk_obj_ids,
            trk_id_to_max_iou_high_conf_det,
            empty_trk_obj_ids,
        ) = self._associate_det_trk(
            det_masks=det_mask_preds,
            det_scores_np=det_scores_np,
            trk_masks=tracker_low_res_masks_global,
            trk_obj_ids=tracker_metadata_prev["obj_ids"],
        )
        if self.suppress_det_close_to_boundary:
            keep = self._suppress_detections_close_to_boundary(det_bbox_xyxy[new_det_fa_inds])
            new_det_fa_inds = new_det_fa_inds[keep.cpu().numpy()]

        # 检查是否已达到可追踪的最大对象数（如达到则丢弃一些检测）
        prev_obj_num = np.sum(tracker_metadata_prev["num_obj"])
        new_det_num = len(new_det_fa_inds)
        num_obj_dropped_due_to_limit = 0
        if prev_obj_num + new_det_num > self.max_num_objects:
            LOGGER.warning(f"hitting {self.max_num_objects=} with {new_det_num=} and {prev_obj_num=}")
            new_det_num_to_keep = self.max_num_objects - prev_obj_num
            num_obj_dropped_due_to_limit = new_det_num - new_det_num_to_keep
            new_det_fa_inds = self._drop_new_det_with_obj_limit(new_det_fa_inds, det_scores_np, new_det_num_to_keep)
            assert len(new_det_fa_inds) == new_det_num_to_keep
            new_det_num = len(new_det_fa_inds)

        # 为新检测分配对象ID并决定将其放置在哪个GPU上
        new_det_obj_ids = tracker_metadata_prev["max_obj_id"] + 1 + np.arange(new_det_num)

        # b) 处理热启动启发式规则以移除对象
        # 此处`metadata`包含存储在GPU 0上（且仅GPU 0可访问）的元数据；
        # 我们不将其广播到其他GPU以节省通信成本，前提是
        # `metadata`不被其他GPU需要
        metadata_new = deepcopy(tracker_metadata_prev["metadata"])
        if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
            obj_ids_newly_removed, metadata_new = self._process_hotstart(
                frame_idx=frame_idx,
                reverse=reverse,
                det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
                new_det_obj_ids=new_det_obj_ids,
                empty_trk_obj_ids=empty_trk_obj_ids,
                unmatched_trk_obj_ids=unmatched_trk_obj_ids,
                metadata=metadata_new,
            )
        else:
            # 如果预热未完成，我们不移除任何对象
            obj_ids_newly_removed = set()
        tracker_metadata_new["metadata"] = metadata_new

        # `tracker_update_plan` should be identical on all GPUs after broadcasting
        tracker_update_plan = {
            "new_det_fa_inds": new_det_fa_inds,  # np.ndarray
            "new_det_obj_ids": new_det_obj_ids,  # np.ndarray
            # "new_det_gpu_ids": new_det_gpu_ids,  # np.ndarray
            "unmatched_trk_obj_ids": unmatched_trk_obj_ids,  # np.ndarray
            "det_to_matched_trk_obj_ids": det_to_matched_trk_obj_ids,  # dict
            "obj_ids_newly_removed": obj_ids_newly_removed,  # set
            "num_obj_dropped_due_to_limit": num_obj_dropped_due_to_limit,  # int
            "trk_id_to_max_iou_high_conf_det": trk_id_to_max_iou_high_conf_det,  # dict
            "reconditioned_obj_ids": reconditioned_obj_ids,  # set
        }

        # 步骤3（可选）：在记忆编码之前基于高置信度检测对masklet进行重条件化
        # 注意：在记忆编码之后运行此步骤可能导致次优结果
        should_recondition_iou = False

        # 基于检测bbox IoU不匹配评估可重条件化的追踪片段
        if self.reconstruction_bbox_iou_thresh > 0 and len(trk_id_to_max_iou_high_conf_det) > 0:
            for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
                det_box = det_out["bbox"][det_idx]
                det_score = det_out["scores"][det_idx]

                try:
                    trk_idx = list(tracker_metadata_prev["obj_ids"]).index(trk_obj_id)
                except ValueError:
                    continue  # 跳过未找到的追踪片段

                tracker_mask = tracker_low_res_masks_global[trk_idx]
                mask_binary = tracker_mask > 0
                mask_area = mask_binary.sum().item()

                if mask_area == 0:
                    continue  # 跳过掩码面积为0的追踪片段

                # 从SAM2掩码获取边界框并转换为归一化坐标
                tracker_box_pixels = batched_mask_to_box(mask_binary.unsqueeze(0)).squeeze(0)
                mask_height, mask_width = tracker_mask.shape[-2:]
                tracker_box_normalized = torch.tensor(
                    [
                        tracker_box_pixels[0] / mask_width,
                        tracker_box_pixels[1] / mask_height,
                        tracker_box_pixels[2] / mask_width,
                        tracker_box_pixels[3] / mask_height,
                    ],
                    device=tracker_box_pixels.device,
                )

                # 计算检测框和SAM2追踪片段边界框之间的IoU
                det_box_batch = det_box.unsqueeze(0)
                tracker_box_batch = tracker_box_normalized.unsqueeze(0)
                iou = box_iou(det_box_batch, tracker_box_batch)[0]

                if iou < self.reconstruction_bbox_iou_thresh and det_score >= self.reconstruction_bbox_det_score:
                    should_recondition_iou = True
                    reconditioned_obj_ids.add(trk_obj_id)

        should_recondition_periodic = (
            self.recondition_every_nth_frame > 0
            and frame_idx % self.recondition_every_nth_frame == 0
            and len(trk_id_to_max_iou_high_conf_det) > 0
        )

        # 如果满足周期性或IoU条件则进行重条件化
        if should_recondition_periodic or should_recondition_iou:
            self._recondition_masklets(
                frame_idx,
                det_out,
                trk_id_to_max_iou_high_conf_det,
                tracker_states_local,
                tracker_metadata_prev,
                tracker_obj_scores_global,
            )

        # 步骤4: 在当前帧的预测掩码上运行SAM2记忆编码器
        # 此操作在所有GPU上执行
        batch_size = tracker_low_res_masks_global.size(0)
        if batch_size > 0:
            if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
                if self.suppress_overlapping_based_on_recent_occlusion_threshold > 0.0:
                    # 注意：tracker_low_res_masks_global会原地更新然后返回
                    tracker_low_res_masks_global = self._suppress_overlapping_based_on_recent_occlusion(
                        frame_idx,
                        tracker_low_res_masks_global,
                        tracker_metadata_prev,
                        tracker_metadata_new,
                        obj_ids_newly_removed,
                        reverse,
                    )

            self._tracker_update_memories(tracker_states_local, frame_idx, low_res_masks=tracker_low_res_masks_global)

        # 步骤4: 基于更新计划更新SAM2元数据
        updated_obj_ids_this_gpu = tracker_metadata_new["obj_ids"]
        if len(new_det_obj_ids) > 0:
            updated_obj_ids_this_gpu = np.concatenate([updated_obj_ids_this_gpu, new_det_obj_ids])
        if len(obj_ids_newly_removed) > 0:
            is_removed = np.isin(updated_obj_ids_this_gpu, list(obj_ids_newly_removed))
            updated_obj_ids_this_gpu = updated_obj_ids_this_gpu[~is_removed]
        tracker_metadata_new["obj_ids"] = updated_obj_ids_this_gpu
        tracker_metadata_new["num_obj"] = len(updated_obj_ids_this_gpu)
        # 更新对象分数和到目前为止分配的最大对象ID
        if len(new_det_obj_ids) > 0:
            tracker_metadata_new["obj_id_to_score"].update(zip(new_det_obj_ids, det_scores_np[new_det_fa_inds]))
            tracker_metadata_new["obj_id_to_cls"].update(zip(new_det_obj_ids, det_cls_np[new_det_fa_inds]))
            # 新对象没有追踪器分数，使用检测分数代替。
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx].update(
                zip(new_det_obj_ids, det_scores_np[new_det_fa_inds])
            )
            tracker_metadata_new["max_obj_id"] = max(tracker_metadata_new["max_obj_id"], np.max(new_det_obj_ids))
        # 对于被移除的对象，我们将它们的分数设置为非常低的值（-1e4），但仍然
        # 保留它们在"obj_id_to_score"中（这样处理输出更容易）
        for obj_id in obj_ids_newly_removed:
            tracker_metadata_new["obj_id_to_score"][obj_id] = -1e4
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx][obj_id] = -1e4
            tracker_metadata_new["obj_id_to_last_occluded"].pop(obj_id, None)
        # 检查"metadata"是否恰好在tracker_metadata_new中
        assert "metadata" in tracker_metadata_new
        if self.masklet_confirmation_enable:
            metadata = self.update_masklet_confirmation_status(
                metadata=tracker_metadata_new["metadata"],
                obj_ids_all_gpu_prev=tracker_metadata_prev["obj_ids"],
                obj_ids_all_gpu_updated=tracker_metadata_new["obj_ids"],
                det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
                new_det_obj_ids=new_det_obj_ids,
            )
            tracker_metadata_new["metadata"] = metadata

        return tracker_update_plan, tracker_metadata_new

    def _suppress_overlapping_based_on_recent_occlusion(
        self,
        frame_idx: int,
        tracker_low_res_masks_global: torch.Tensor,
        tracker_metadata_prev: dict[str, Any],
        tracker_metadata_new: dict[str, Any],
        obj_ids_newly_removed: set[int],
        reverse: bool = False,
    ):
        """基于最近的遮挡信息抑制重叠的掩码。如果对象被热启动移除，
        我们总是抑制它与其他任何对象重叠的情况。

        Args:
            frame_idx (int): 当前帧索引。
            tracker_low_res_masks_global (torch.Tensor): 当前帧的低分辨率掩码。
            tracker_metadata_prev (dict[str, Any]): 前一帧的元数据。
            tracker_metadata_new (dict[str, Any]): 当前帧的元数据。
            obj_ids_newly_removed (set[int]): 已被移除的对象ID。
            reverse (bool): 追踪是否为反向顺序。

        Returns:
            (torch.Tensor): 更新后的低分辨率掩码，其中部分对象已被抑制。
        """
        obj_ids_global = tracker_metadata_prev["obj_ids"]
        binary_tracker_low_res_masks_global = tracker_low_res_masks_global > 0
        batch_size = tracker_low_res_masks_global.size(0)
        if batch_size > 0:
            assert len(obj_ids_global) == batch_size, (
                f"Mismatch in number of objects: {len(obj_ids_global)} vs {batch_size}"
            )
            last_occluded_prev = torch.cat(
                [
                    tracker_metadata_prev["obj_id_to_last_occluded"].get(
                        obj_id,
                        torch.full(
                            (1,),
                            fill_value=(
                                self.NEVER_OCCLUDED if obj_id not in obj_ids_newly_removed else self.ALWAYS_OCCLUDED
                            ),
                            device=binary_tracker_low_res_masks_global.device,
                            dtype=torch.long,
                        ),
                    )
                    for obj_id in obj_ids_global
                ],
                dim=0,
            )
            to_suppress = self._get_objects_to_suppress_based_on_most_recently_occluded(
                binary_tracker_low_res_masks_global,
                last_occluded_prev,
                obj_ids_global,
                frame_idx,
                reverse,
            )

            # 用遮挡信息更新元数据
            is_obj_occluded = ~(binary_tracker_low_res_masks_global.any(dim=(-1, -2)))
            is_obj_occluded_or_suppressed = is_obj_occluded | to_suppress
            last_occluded_new = last_occluded_prev.clone()
            last_occluded_new[is_obj_occluded_or_suppressed] = frame_idx
            # 切分出每个对象的最后遮挡帧
            tracker_metadata_new["obj_id_to_last_occluded"] = {
                obj_id: last_occluded_new[obj_idx : obj_idx + 1] for obj_idx, obj_id in enumerate(obj_ids_global)
            }

            # 在记忆编码前将被抑制的掩码置零
            tracker_low_res_masks_global[to_suppress] = self.NO_OBJ_LOGIT

        return tracker_low_res_masks_global

    def run_tracker_update_execution_phase(
        self,
        frame_idx: int,
        num_frames: int,
        det_out: dict[str, torch.Tensor],
        tracker_states_local: list[Any],
        tracker_update_plan: dict[str, np.ndarray],
    ):
        """以SPMD方式为单帧执行追踪器更新计划。"""
        # 用检测分数初始化追踪分数
        new_det_fa_inds: np.ndarray = tracker_update_plan["new_det_fa_inds"]
        new_det_obj_ids: np.ndarray = tracker_update_plan["new_det_obj_ids"]
        # new_det_gpu_ids: np.ndarray = tracker_update_plan["new_det_gpu_ids"]
        new_det_obj_ids_local: np.ndarray = new_det_obj_ids
        new_det_fa_inds_local: np.ndarray = new_det_fa_inds
        obj_ids_newly_removed: set[int] = tracker_update_plan["obj_ids_newly_removed"]

        # 步骤1: 将检测器的新对象添加到SAM2推理状态中
        if len(new_det_fa_inds_local) > 0:
            new_det_fa_inds_local_t = torch.from_numpy(new_det_fa_inds_local)
            new_det_masks: torch.Tensor = det_out["mask"][new_det_fa_inds_local_t]
            # 用新对象掩码初始化SAM2
            tracker_states_local = self._tracker_add_new_objects(
                frame_idx=frame_idx,
                num_frames=num_frames,
                new_obj_ids=new_det_obj_ids_local,
                new_obj_masks=new_det_masks,
                tracker_states_local=tracker_states_local,
            )

        # 步骤2: 从SAM2推理状态中移除被启发式规则移除的对象
        if len(obj_ids_newly_removed) > 0:
            self._tracker_remove_objects(tracker_states_local, obj_ids_newly_removed)

        return tracker_states_local

    @staticmethod
    def build_outputs(
        det_out: dict[str, torch.Tensor],
        tracker_low_res_masks_global: torch.Tensor,
        tracker_metadata_prev: dict[str, np.ndarray],
        tracker_update_plan: dict[str, np.ndarray],
        reconditioned_obj_ids: set | None = None,
    ):
        """构建当前帧的输出掩码。"""
        new_det_fa_inds: np.ndarray = tracker_update_plan["new_det_fa_inds"]
        new_det_obj_ids: np.ndarray = tracker_update_plan["new_det_obj_ids"]
        obj_id_to_mask = {}  # obj_id --> 输出掩码张量

        # 第1部分: 来自先前SAM2传播的掩码
        existing_masklet_obj_ids = tracker_metadata_prev["obj_ids"]
        existing_masklet_binary = tracker_low_res_masks_global.unsqueeze(1)
        assert len(existing_masklet_obj_ids) == len(existing_masklet_binary)
        for obj_id, mask in zip(existing_masklet_obj_ids, existing_masklet_binary):
            obj_id_to_mask[obj_id] = mask  # (1, H_video, W_video)

        # 第2部分: 来自新检测的掩码
        new_det_fa_inds_t = torch.from_numpy(new_det_fa_inds)
        new_det_low_res_masks = det_out["mask"][new_det_fa_inds_t].unsqueeze(1)
        assert len(new_det_obj_ids) == len(new_det_low_res_masks)
        for obj_id, mask in zip(new_det_obj_ids, new_det_low_res_masks):
            obj_id_to_mask[obj_id] = mask  # (1, H_video, W_video)

        # 第3部分: 使用检测掩码覆盖重条件化对象的掩码
        if reconditioned_obj_ids is not None and len(reconditioned_obj_ids) > 0:
            trk_id_to_max_iou_high_conf_det = tracker_update_plan.get("trk_id_to_max_iou_high_conf_det", {})

            for obj_id in reconditioned_obj_ids:
                det_idx = trk_id_to_max_iou_high_conf_det.get(obj_id)

                if det_idx is not None:
                    obj_id_to_mask[obj_id] = det_out["mask"][det_idx].unsqueeze(0)

        return obj_id_to_mask

    def _get_objects_to_suppress_based_on_most_recently_occluded(
        self,
        binary_low_res_masks: torch.Tensor,
        last_occluded: list[int],
        obj_ids: list[int],
        frame_idx: int | None = None,
        reverse: bool = False,
    ):
        # 抑制最近被遮挡对象的重叠掩码
        assert binary_low_res_masks.dtype == torch.bool, f"Expected boolean tensor, got {binary_low_res_masks.dtype}"
        to_suppress = torch.zeros(
            binary_low_res_masks.size(0),
            device=binary_low_res_masks.device,
            dtype=torch.bool,
        )
        if len(obj_ids) <= 1:
            return to_suppress

        iou = mask_iou(binary_low_res_masks.flatten(1), binary_low_res_masks.flatten(1))  # [N,N]

        # 创建上三角矩阵（i < j）的掩码和IoU阈值
        mask_iou_thresh = iou >= self.suppress_overlapping_based_on_recent_occlusion_threshold
        overlapping_pairs = torch.triu(mask_iou_thresh, diagonal=1)  # [N,N]

        last_occ_expanded_i = last_occluded.unsqueeze(1)  # (N, 1)
        last_occ_expanded_j = last_occluded.unsqueeze(0)  # (1, N)
        # 抑制最近被遮挡的
        cmp_op = torch.gt if not reverse else torch.lt
        suppress_i_mask = (
            overlapping_pairs
            & cmp_op(last_occ_expanded_i, last_occ_expanded_j)  # (last_occ_expanded_i > last_occ_expanded_j)
            & (last_occ_expanded_j > -1)  # j仅在i之前被遮挡时才能抑制i
        )
        suppress_j_mask = (
            overlapping_pairs
            & cmp_op(last_occ_expanded_j, last_occ_expanded_i)
            & (last_occ_expanded_i > -1)  # i仅在j之前被遮挡时才能抑制j
        )
        # 应用抑制
        to_suppress = suppress_i_mask.any(dim=1) | suppress_j_mask.any(dim=0)

        # Log for debugging
        if LOGGER.isEnabledFor(10) and frame_idx is not None:
            suppress_i_mask = suppress_i_mask.cpu().numpy()
            suppress_j_mask = suppress_j_mask.cpu().numpy()
            last_occluded = last_occluded.cpu().numpy()

            # Find all suppression pairs without using torch.where
            batch_size = suppress_i_mask.shape[0]

            # Log i-suppression cases (where i gets suppressed in favor of j)
            for i in range(batch_size):
                for j in range(batch_size):
                    if suppress_i_mask[i, j]:
                        LOGGER.debug(
                            f"{frame_idx=}: Suppressing obj {obj_ids[i]} last occluded {last_occluded[i]} in favor of {obj_ids[j]} last occluded {last_occluded[j]}"
                        )

            # Log j-suppression cases (where j gets suppressed in favor of i)
            for i in range(batch_size):
                for j in range(batch_size):
                    if suppress_j_mask[i, j]:
                        LOGGER.debug(
                            f"{frame_idx=}: Suppressing obj {obj_ids[j]} last occluded {last_occluded[j]} in favor of {obj_ids[i]} last occluded {last_occluded[i]}"
                        )

        return to_suppress

    def _propogate_tracker_one_frame_local_gpu(self, inference_states: list[Any], frame_idx: int):
        """Inference_states: list of inference states, each state corresponds to a different set of objects."""
        obj_ids_local = []
        low_res_masks_list = []
        obj_scores_list = []
        for inference_state in inference_states:
            if len(inference_state["obj_ids"]) == 0:
                continue  # skip propagation on empty inference states

            out_obj_ids, out_low_res_masks, out_obj_scores = self.tracker.propagate_in_video(
                inference_state, frame_idx=frame_idx
            )
            assert isinstance(out_obj_ids, list)
            obj_ids_local.extend(out_obj_ids)
            low_res_masks_list.append(out_low_res_masks.squeeze(1))
            obj_scores_list.append(out_obj_scores.squeeze(1))

        # concatenate the output masklets from all local inference states
        if len(low_res_masks_list) > 0:
            low_res_masks_local = torch.cat(low_res_masks_list, dim=0)
            obj_scores_local = torch.cat(obj_scores_list, dim=0)
            low_res_masks_local = low_res_masks_local.squeeze(1)
        else:
            low_res_masks_local = torch.zeros(0, *self._bb_feat_sizes[0], device=self.device)
            obj_scores_local = torch.zeros(0, device=self.device)

        return obj_ids_local, low_res_masks_local, obj_scores_local

    def _associate_det_trk(
        self,
        det_masks: torch.Tensor,
        det_scores_np: np.ndarray,
        trk_masks: torch.Tensor,
        trk_obj_ids: np.ndarray,
    ):
        """Match detections on the current frame with the existing masklets.

        Args:
            det_masks: (N, H, W) tensor of predicted masks
            det_scores_np: (N,) array of detection scores
            trk_masks: (M, H, W) tensor of track masks
            trk_obj_ids: (M,) array of object IDs corresponding to trk_masks

        Returns:
            new_det_fa_inds: array of new object indices.
            unmatched_trk_obj_ids: array of existing masklet object IDs that are not matched to any detections on this
                frame (for unmatched, we only count masklets with >0 area)
            det_to_matched_trk_obj_ids: dict[int, np.ndarray]: mapping from detector's detection indices to the list of
                matched tracklet object IDs
            empty_trk_obj_ids: array of existing masklet object IDs with zero area in SAM2 prediction
        """
        iou_threshold = self.assoc_iou_thresh
        iou_threshold_trk = self.trk_assoc_iou_thresh
        new_det_thresh = self.new_det_thresh

        assert det_masks.is_floating_point(), "float tensor expected (do not binarize)"
        assert trk_masks.is_floating_point(), "float tensor expected (do not binarize)"
        assert trk_masks.size(0) == len(trk_obj_ids), (
            f"trk_masks and trk_obj_ids should have the same length, {trk_masks.size(0)} vs {len(trk_obj_ids)}"
        )
        if trk_masks.size(0) == 0:
            # all detections are new
            new_det_fa_inds = np.arange(det_masks.size(0))
            unmatched_trk_obj_ids = np.array([], np.int64)
            empty_trk_obj_ids = np.array([], np.int64)
            det_to_matched_trk_obj_ids = {}
            trk_id_to_max_iou_high_conf_det = {}
            return (
                new_det_fa_inds,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                trk_id_to_max_iou_high_conf_det,
                empty_trk_obj_ids,
            )
        elif det_masks.size(0) == 0:
            # all previous tracklets are unmatched if they have a non-zero area
            new_det_fa_inds = np.array([], np.int64)
            trk_is_nonempty = (trk_masks > 0).any(dim=(1, 2)).cpu().numpy()
            unmatched_trk_obj_ids = trk_obj_ids[trk_is_nonempty]
            empty_trk_obj_ids = trk_obj_ids[~trk_is_nonempty]
            det_to_matched_trk_obj_ids = {}
            trk_id_to_max_iou_high_conf_det = {}
            return (
                new_det_fa_inds,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                trk_id_to_max_iou_high_conf_det,
                empty_trk_obj_ids,
            )

        if det_masks.shape[-2:] != trk_masks.shape[-2:]:
            # resize to the smaller size to save GPU memory
            if np.prod(det_masks.shape[-2:]) < np.prod(trk_masks.shape[-2:]):
                trk_masks = F.interpolate(
                    trk_masks.unsqueeze(1),
                    size=det_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            else:
                # resize detections to track size
                det_masks = F.interpolate(
                    det_masks.unsqueeze(1),
                    size=trk_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

        det_masks_binary = det_masks > 0
        trk_masks_binary = trk_masks > 0
        ious = mask_iou(det_masks_binary.flatten(1).float(), trk_masks_binary.flatten(1).float())  # (N, M)

        ious_np = ious.cpu().numpy()
        if self.o2o_matching_masklets_enable:
            from scipy.optimize import linear_sum_assignment

            # Hungarian matching for tracks (one-to-one: each track matches at most one detection)
            cost_matrix = 1 - ious_np  # Hungarian solves for minimum cost
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            trk_is_matched = np.zeros(trk_masks.size(0), dtype=bool)
            for d, t in zip(row_ind, col_ind):
                if ious_np[d, t] >= iou_threshold_trk:
                    trk_is_matched[t] = True
        else:
            trk_is_matched = (ious_np >= iou_threshold_trk).any(axis=0)
        # Non-empty tracks not matched by Hungarian assignment above threshold are unmatched
        trk_is_nonempty = trk_masks_binary.any(dim=(1, 2)).cpu().numpy()
        trk_is_unmatched = np.logical_and(trk_is_nonempty, ~trk_is_matched)
        unmatched_trk_obj_ids = trk_obj_ids[trk_is_unmatched]
        # also record masklets that have zero area in SAM 2 prediction
        empty_trk_obj_ids = trk_obj_ids[~trk_is_nonempty]

        # For detections: allow many tracks to match to the same detection (many-to-one)
        # So, a detection is 'new' if it does not match any track above threshold
        is_new_det = np.logical_and(
            det_scores_np >= new_det_thresh,
            np.logical_not(np.any(ious_np >= iou_threshold, axis=1)),
        )
        new_det_fa_inds = np.nonzero(is_new_det)[0]

        # for each detection, which tracks it matched to (above threshold)
        det_to_matched_trk_obj_ids = {}
        trk_id_to_max_iou_high_conf_det = {}  # trk id --> exactly one detection idx
        det_to_max_iou_trk_idx = np.argmax(ious_np, axis=1)
        det_is_high_conf = (det_scores_np >= self.HIGH_CONF_THRESH) & ~is_new_det
        det_is_high_iou = np.max(ious_np, axis=1) >= self.HIGH_IOU_THRESH
        det_is_high_conf_and_iou = set(np.nonzero(det_is_high_conf & det_is_high_iou)[0])
        for d in range(det_masks.size(0)):
            det_to_matched_trk_obj_ids[d] = trk_obj_ids[ious_np[d, :] >= iou_threshold]
            if d in det_is_high_conf_and_iou:
                trk_obj_id = trk_obj_ids[det_to_max_iou_trk_idx[d]].item()
                trk_id_to_max_iou_high_conf_det[trk_obj_id] = d

        return (
            new_det_fa_inds,
            unmatched_trk_obj_ids,
            det_to_matched_trk_obj_ids,
            trk_id_to_max_iou_high_conf_det,
            empty_trk_obj_ids,
        )

    def _process_hotstart(
        self,
        frame_idx: int,
        reverse: bool,
        det_to_matched_trk_obj_ids: dict[int, np.ndarray],
        new_det_obj_ids: np.ndarray,
        empty_trk_obj_ids: np.ndarray,
        unmatched_trk_obj_ids: np.ndarray,
        metadata: dict[str, Any],
    ):
        """Handle hotstart heuristics to remove unmatched or duplicated objects."""
        # obj_id --> first frame index where the object was detected
        obj_first_frame_idx = metadata["obj_first_frame_idx"]
        # obj_id --> [mismatched frame indices]
        unmatched_frame_inds = metadata["unmatched_frame_inds"]
        trk_keep_alive = metadata["trk_keep_alive"]
        # (first_appear_obj_id, obj_id) --> [overlap frame indices]
        overlap_pair_to_frame_inds = metadata["overlap_pair_to_frame_inds"]
        # removed_obj_ids: object IDs that are suppressed via hot-start
        removed_obj_ids = metadata["removed_obj_ids"]

        obj_ids_newly_removed = set()  # object IDs to be newly removed on this frame
        hotstart_diff = frame_idx - self.hotstart_delay if not reverse else frame_idx + self.hotstart_delay

        # Step 1: log the frame index where each object ID first appears
        for obj_id in new_det_obj_ids:
            if obj_id not in obj_first_frame_idx:
                obj_first_frame_idx[obj_id] = frame_idx
            assert obj_id not in trk_keep_alive
            trk_keep_alive[obj_id] = self.init_trk_keep_alive

        matched_trks = set()
        # We use the det-->tracks list to check for matched objects. Otherwise, we need to compute areas to decide whether they're occluded
        for matched_trks_per_det in det_to_matched_trk_obj_ids.values():
            matched_trks.update(matched_trks_per_det)
        for obj_id in matched_trks:
            # NOTE: To minimize number of configurable params, we use the hotstart_unmatch_thresh to set the max value of trk_keep_alive
            trk_keep_alive[obj_id] = min(self.max_trk_keep_alive, trk_keep_alive[obj_id] + 1)
        for obj_id in unmatched_trk_obj_ids:
            unmatched_frame_inds[obj_id].append(frame_idx)
            # NOTE: To minimize number of configurable params, we use the hotstart_unmatch_thresh to set the min value of trk_keep_alive
            # The max keep alive is 2x the min, means the model prefers to keep the prediction rather than suppress it if it was matched long enough.
            trk_keep_alive[obj_id] = max(self.min_trk_keep_alive, trk_keep_alive[obj_id] - 1)
        if self.decrease_trk_keep_alive_for_empty_masklets:
            for obj_id in empty_trk_obj_ids:
                # NOTE: To minimize number of configurable params, we use the hotstart_unmatch_thresh to set the min value of trk_keep_alive
                trk_keep_alive[obj_id] = max(self.min_trk_keep_alive, trk_keep_alive[obj_id] - 1)

        # Step 2: removed tracks that has not matched with detections for `hotstart_unmatch_thresh` frames with hotstart period
        # a) add unmatched frame indices for each existing object ID
        # note that `unmatched_trk_obj_ids` contains those frames where the SAM2 output mask
        # doesn't match any detection; it excludes those frames where SAM2 gives an empty mask
        # b) remove a masklet if it first appears after `hotstart_diff` and is unmatched for more
        # than `self.hotstart_unmatch_thresh` frames
        for obj_id, frame_indices in unmatched_frame_inds.items():
            if obj_id in removed_obj_ids or obj_id in obj_ids_newly_removed:
                continue  # skip if the object is already removed
            if len(frame_indices) >= self.hotstart_unmatch_thresh:
                is_within_hotstart = (obj_first_frame_idx[obj_id] > hotstart_diff and not reverse) or (
                    obj_first_frame_idx[obj_id] < hotstart_diff and reverse
                )
                if is_within_hotstart:
                    obj_ids_newly_removed.add(obj_id)
                    LOGGER.debug(
                        f"Removing object {obj_id} at frame {frame_idx} "
                        f"since it is unmatched for frames: {frame_indices}"
                    )
            if (
                trk_keep_alive[obj_id] <= 0  # Object has not been matched for too long
                and obj_id not in removed_obj_ids
                and obj_id not in obj_ids_newly_removed
            ):
                LOGGER.debug(f"Removing object {obj_id} at frame {frame_idx}, due to being unmatched")
                # directly removed the object instead of suppressing it
                obj_ids_newly_removed.add(obj_id)

        # Step 3: removed tracks that overlaps with another track for `hotstart_dup_thresh` frames
        # a) find overlaps tracks -- we consider overlap if they match to the same detection
        for _, matched_trk_obj_ids in det_to_matched_trk_obj_ids.items():
            if len(matched_trk_obj_ids) < 2:
                continue  # only count detections that are matched to multiple (>=2) masklets
            # if there are multiple matched track ids, we need to find the one that appeared first;
            # these later appearing ids may be removed since they may be considered as duplicates
            first_appear_obj_id = (
                min(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
                if not reverse
                else max(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
            )
            for obj_id in matched_trk_obj_ids:
                if obj_id != first_appear_obj_id:
                    key = (first_appear_obj_id, obj_id)
                    overlap_pair_to_frame_inds[key].append(frame_idx)

        # b) remove a masklet if it first appears after `hotstart_diff` and it overlaps with another
        # masklet (that appears earlier) for more than `self.hotstart_dup_thresh` frames
        for (first_obj_id, obj_id), frame_indices in overlap_pair_to_frame_inds.items():
            if obj_id in removed_obj_ids or obj_id in obj_ids_newly_removed:
                continue  # skip if the object is already removed
            if (obj_first_frame_idx[obj_id] > hotstart_diff and not reverse) or (
                obj_first_frame_idx[obj_id] < hotstart_diff and reverse
            ):
                if len(frame_indices) >= self.hotstart_dup_thresh:
                    obj_ids_newly_removed.add(obj_id)
                    LOGGER.debug(
                        f"Removing object {obj_id} at frame {frame_idx} "
                        f"since it overlaps with another track {first_obj_id} at frames: {frame_indices}"
                    )

        removed_obj_ids.update(obj_ids_newly_removed)
        return obj_ids_newly_removed, metadata

    def _tracker_update_memories(
        self, tracker_inference_states: list[Any], frame_idx: int, low_res_masks: torch.Tensor
    ):
        """Run Sam2 memory encoder, enforcing non-overlapping constraints globally."""
        if len(tracker_inference_states) == 0:
            return
        # NOTE: inspect this part if we observe OOMs in the demo
        high_res_masks = F.interpolate(
            low_res_masks.unsqueeze(1),
            size=self.interpol_size,
            mode="bilinear",
            align_corners=False,
        )
        # We first apply non-overlapping constraints before memory encoding. This may include some suppression heuristics.
        if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
            high_res_masks = self.tracker.model._suppress_object_pw_area_shrinkage(high_res_masks)
        # Instead of gathering the predicted object scores, we use mask areas as a proxy.
        object_score_logits = torch.where((high_res_masks > 0).any(dim=(-1, -2)), 10.0, -10.0)

        # Run the memory encoder on local slices for each GPU
        start_idx_gpu = 0
        start_idx_state = start_idx_gpu
        for tracker_state in tracker_inference_states:
            num_obj_per_state = len(tracker_state["obj_ids"])
            if num_obj_per_state == 0:
                continue
            # Get the local high-res masks and object score logits for this inference state
            end_idx_state = start_idx_state + num_obj_per_state
            local_high_res_masks = high_res_masks[start_idx_state:end_idx_state]
            local_object_score_logits = object_score_logits[start_idx_state:end_idx_state]
            local_batch_size = local_high_res_masks.size(0)
            # Run Sam2 memory encoder. Note that we do not re-enforce the non-overlapping constraint as it is turned off by default

            encoded_mem = self.tracker._run_memory_encoder(
                local_batch_size,
                local_high_res_masks,
                local_object_score_logits,
                is_mask_from_pts=False,
                inference_state=tracker_state,
            )
            local_maskmem_features, local_maskmem_pos_enc = encoded_mem
            # Store encoded memories in the local inference state
            output_dict = tracker_state["output_dict"]
            for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                if frame_idx not in output_dict[storage_key]:
                    continue
                output_dict[storage_key][frame_idx]["maskmem_features"] = local_maskmem_features
                output_dict[storage_key][frame_idx]["maskmem_pos_enc"] = [pos for pos in local_maskmem_pos_enc]
                # for batched inference state, we also need to add per-object
                # memory slides to support instance interactivity
                self.tracker._add_output_per_object(
                    inference_state=tracker_state,
                    frame_idx=frame_idx,
                    current_out=output_dict[storage_key][frame_idx],
                    storage_key=storage_key,
                )
            start_idx_state += num_obj_per_state

    def _tracker_add_new_objects(
        self,
        frame_idx: int,
        num_frames: int,
        new_obj_ids: list[int],
        new_obj_masks: torch.Tensor,
        tracker_states_local: list[Any],
    ):
        """Add a new object to SAM2 inference states."""
        prev_tracker_state = tracker_states_local[0] if len(tracker_states_local) > 0 else None

        # prepare inference_state
        # batch objects that first appear on the same frame together
        # Clear inference state. Keep the cached image features if available.
        new_tracker_state = self.tracker._init_state(num_frames=num_frames)
        # NOTE: adding image placeholder
        new_tracker_state["im"] = None
        new_tracker_state["backbone_out"] = (
            prev_tracker_state.get("backbone_out", None) if prev_tracker_state is not None else None
        )

        assert len(new_obj_ids) == new_obj_masks.size(0)
        assert new_obj_masks.is_floating_point()
        new_obj_masks = F.interpolate(
            new_obj_masks.unsqueeze(0),
            size=self.interpol_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        new_obj_masks = new_obj_masks > 0

        # add object one by one
        for new_obj_id, new_mask in zip(new_obj_ids, new_obj_masks):
            self.tracker.add_new_prompts(
                inference_state=new_tracker_state,
                frame_idx=frame_idx,
                obj_id=new_obj_id,
                masks=new_mask[None, None],  # add bs, channel
            )
        # NOTE: we skip enforcing the non-overlapping constraint **globally** when adding new objects.
        self.tracker.propagate_in_video_preflight(new_tracker_state)
        tracker_states_local.append(new_tracker_state)
        return tracker_states_local

    def _tracker_remove_objects(self, tracker_states_local: list[Any], obj_ids: list[int]):
        """Remove an object from SAM2 inference states. This would remove the object from all frames in the video."""
        if not obj_ids:
            return
        # Filter out states that become empty after removal
        active_states = []
        for state in tracker_states_local:
            for obj_id in obj_ids:
                # we try to remove `obj_id` on every inference state with `strict=False`
                # it will not do anything if an inference state doesn't contain `obj_id`
                self.tracker.remove_object(state, obj_id, strict=False)

            if len(state["obj_ids"]) > 0:
                active_states.append(state)

        # Update the list in-place
        tracker_states_local[:] = active_states

    def _initialize_metadata(self):
        """Initialize metadata for the masklets."""
        tracker_metadata = {
            "obj_ids": np.array([], np.int32),
            "num_obj": np.zeros(1, np.int32),
            "max_obj_id": -1,
            "obj_id_to_score": {},
            "obj_id_to_cls": {},
            "obj_id_to_tracker_score_frame_wise": defaultdict(dict),
            "obj_id_to_last_occluded": {},
        }
        # "metadata" contains metadata that is only stored on (and accessible to) GPU 0
        # - obj_first_frame_idx: obj_id --> first frame index where the object was detected
        # - unmatched_frame_inds: obj_id --> [mismatched frame indices]
        # - overlap_pair_to_frame_inds: (first_appear_obj_id, obj_id) --> [overlap frame indices]
        # - removed_obj_ids: object IDs that are suppressed via hot-start
        metadata = {
            "obj_first_frame_idx": {},
            "unmatched_frame_inds": defaultdict(list),
            "trk_keep_alive": defaultdict(int),  # This is used only for object suppression not for removal
            "overlap_pair_to_frame_inds": defaultdict(list),
            "removed_obj_ids": set(),
        }
        if self.masklet_confirmation_enable:
            # all the following are np.ndarray with the same shape as `obj_ids_all_gpu`
            metadata["masklet_confirmation"] = {
                # "status" is the confirmation status of each masklet
                "status": np.array([], np.int64),
                # "consecutive_det_num" is the number of consecutive frames where the masklet is
                # detected by the detector (with a matched detection)
                "consecutive_det_num": np.array([], np.int64),
            }
        tracker_metadata["metadata"] = metadata

        return tracker_metadata

    def update_masklet_confirmation_status(
        self,
        metadata: dict[str, Any],
        obj_ids_all_gpu_prev: np.ndarray,
        obj_ids_all_gpu_updated: np.ndarray,
        det_to_matched_trk_obj_ids: dict[int, np.ndarray],
        new_det_obj_ids: np.ndarray,
    ):
        """Update the confirmation status of masklets based on the current frame's detection results."""
        confirmation_data = metadata["masklet_confirmation"]

        # a) first, expand "confirmation_data" to include new masklets added in this frame
        status_prev = confirmation_data["status"]
        consecutive_det_num_prev = confirmation_data["consecutive_det_num"]
        assert status_prev.shape == obj_ids_all_gpu_prev.shape, (
            f"Got {status_prev.shape} vs {obj_ids_all_gpu_prev.shape}"
        )

        obj_id_to_updated_idx = {obj_id: idx for idx, obj_id in enumerate(obj_ids_all_gpu_updated)}
        prev_elem_is_in_updated = np.isin(obj_ids_all_gpu_prev, obj_ids_all_gpu_updated)
        prev_elem_obj_ids_in_updated = obj_ids_all_gpu_prev[prev_elem_is_in_updated]
        prev_elem_inds_in_updated = np.array(
            [obj_id_to_updated_idx[obj_id] for obj_id in prev_elem_obj_ids_in_updated],
            dtype=np.int64,
        )
        # newly added masklets are initialized to "UNCONFIRMED" status
        unconfirmed_val = self.UNCONFIRMED
        status = np.full_like(obj_ids_all_gpu_updated, fill_value=unconfirmed_val)
        status[prev_elem_inds_in_updated] = status_prev[prev_elem_is_in_updated]
        consecutive_det_num = np.zeros_like(obj_ids_all_gpu_updated)
        consecutive_det_num[prev_elem_inds_in_updated] = consecutive_det_num_prev[prev_elem_is_in_updated]

        # b) update the confirmation status of all masklets based on the current frame
        # b.1) update "consecutive_det_num"
        # "is_matched": whether a masklet is matched to a detection on this frame
        is_matched = np.isin(obj_ids_all_gpu_updated, new_det_obj_ids)
        for matched_trk_obj_ids in det_to_matched_trk_obj_ids.values():
            is_matched |= np.isin(obj_ids_all_gpu_updated, matched_trk_obj_ids)
        consecutive_det_num = np.where(is_matched, consecutive_det_num + 1, 0)

        # b.2) update "status"
        change_to_confirmed = consecutive_det_num >= self.masklet_confirmation_consecutive_det_thresh
        status[change_to_confirmed] = self.CONFIRMED

        confirmation_data["status"] = status
        confirmation_data["consecutive_det_num"] = consecutive_det_num
        return metadata

    def _load_checkpoint(self, ckpt_path: str, strict: bool = True):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=strict)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            LOGGER.warning(f"Loaded ckpt with {missing_keys=}, {unexpected_keys=}")
        else:
            LOGGER.info("Loaded ckpt successfully without missing or unexpected keys")

    def _encode_prompt(self, **kwargs):
        return self.model._encode_prompt(**kwargs)

    @staticmethod
    def _drop_new_det_with_obj_limit(new_det_fa_inds, det_scores_np, num_to_keep):
        """Drop a few new detections based on the maximum number of objects. We drop new objects based on their
        detection scores, keeping the high-scoring ones and dropping the low-scoring ones.
        """
        assert 0 <= num_to_keep <= len(new_det_fa_inds)
        if num_to_keep == 0:
            return np.array([], np.int64)  # keep none
        if num_to_keep == len(new_det_fa_inds):
            return new_det_fa_inds  # keep all

        # keep the top-scoring detections
        score_order = np.argsort(det_scores_np[new_det_fa_inds])[::-1]
        new_det_fa_inds = new_det_fa_inds[score_order[:num_to_keep]]
        return new_det_fa_inds
