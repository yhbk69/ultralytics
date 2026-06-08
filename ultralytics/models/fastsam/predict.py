# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
from PIL import Image

from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import scale_masks
from ultralytics.utils.torch_utils import TORCH_1_10

from .utils import adjust_bboxes_to_image_border


class FastSAMPredictor(SegmentationPredictor):
    """专门用于快速 SAM（分割一切模型）分割预测任务的 FastSAMPredictor。

    该类继承自 SegmentationPredictor，专门针对快速 SAM 定制预测流程。它调整了后处理步骤，融合了掩码预测和
    非极大值抑制，并针对单类别分割进行了优化。

    Attributes:
        prompts (dict): 包含分割提示信息的字典（边界框、点、标签、文本）。
        device (torch.device): 模型和张量所在的设备。
        clip (Any, optional): 用于基于文本提示的 CLIP 模型，按需加载。

    Methods:
        postprocess: 对 FastSAM 预测结果应用后处理并处理提示。
        prompt: 基于各种提示类型执行图像分割推理。
        set_prompts: 设置推理期间使用的提示。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """使用配置和回调初始化 FastSAMPredictor。

        初始化一个专门用于快速 SAM（分割一切模型）分割任务的预测器。该预测器继承自 SegmentationPredictor，
        添加了自定义后处理功能，用于掩码预测和非极大值抑制，并针对单类别分割进行了优化。

        Args:
            cfg (dict): 预测器的配置。
            overrides (dict, optional): 配置覆盖项。
            _callbacks (dict, optional): 回调函数字典。
        """
        super().__init__(cfg, overrides, _callbacks)
        self.prompts = {}

    def postprocess(self, preds, img, orig_imgs):
        """对 FastSAM 预测结果应用后处理并处理提示。

        Args:
            preds (list[torch.Tensor]): 模型的原始预测结果。
            img (torch.Tensor): 输入模型的图像张量。
            orig_imgs (list[np.ndarray]): 预处理前的原始图像。

        Returns:
            (list[Results]): 应用提示后的处理后结果。
        """
        bboxes = self.prompts.pop("bboxes", None)
        points = self.prompts.pop("points", None)
        labels = self.prompts.pop("labels", None)
        texts = self.prompts.pop("texts", None)
        results = super().postprocess(preds, img, orig_imgs)
        for result in results:
            full_box = torch.tensor(
                [0, 0, result.orig_shape[1], result.orig_shape[0]], device=result.boxes.data.device, dtype=torch.float32
            )
            boxes = adjust_bboxes_to_image_border(result.boxes.xyxy, result.orig_shape)
            idx = torch.nonzero(box_iou(full_box[None], boxes) > 0.9).flatten()
            if idx.numel() != 0:
                result.boxes.xyxy[idx] = full_box

        return self.prompt(results, bboxes=bboxes, points=points, labels=labels, texts=texts)

    def prompt(self, results, bboxes=None, points=None, labels=None, texts=None):
        """基于边界框、点和文本提示等线索执行图像分割推理。

        Args:
            results (Results | list[Results]): FastSAM 模型在无任何提示时的原始推理结果。
            bboxes (np.ndarray | list, optional): 边界框，形状为 (N, 4)，格式为 XYXY。
            points (np.ndarray | list, optional): 指示物体位置的点，形状为 (N, 2)，单位为像素。
            labels (np.ndarray | list, optional): 点提示的标签，形状为 (N, )。1 = 前景，0 = 背景。
            texts (str | list[str], optional): 文本提示，包含字符串对象的列表。

        Returns:
            (list[Results]): 根据提供的提示过滤和确定的输出结果。
        """
        if bboxes is None and points is None and texts is None:
            return results
        prompt_results = []
        if not isinstance(results, list):
            results = [results]
        for result in results:
            if len(result) == 0:
                prompt_results.append(result)
                continue
            masks = result.masks.data
            if masks.shape[1:] != result.orig_shape:
                masks = (scale_masks(masks[None].float(), result.orig_shape)[0] > 0.5).byte()
            # 边界框提示
            idx = torch.zeros(len(result), dtype=torch.bool, device=self.device)
            if bboxes is not None:
                bboxes = torch.as_tensor(bboxes, dtype=torch.int32, device=self.device)
                bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
                bbox_areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
                mask_areas = torch.stack([masks[:, b[1] : b[3], b[0] : b[2]].sum(dim=(1, 2)) for b in bboxes])
                full_mask_areas = torch.sum(masks, dim=(1, 2))

                union = bbox_areas[:, None] + full_mask_areas - mask_areas
                idx[torch.argmax(mask_areas / union, dim=1)] = True
            if points is not None:
                points = torch.as_tensor(points, dtype=torch.int32, device=self.device)
                points = points[None] if points.ndim == 1 else points
                if labels is None:
                    labels = torch.ones(points.shape[0])
                labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
                assert len(labels) == len(points), (
                    f"Expected `labels` to have the same length as `points`, but got {len(labels)} and {len(points)}."
                )
                point_idx = (
                    torch.ones(len(result), dtype=torch.bool, device=self.device)
                    if labels.sum() == 0  # 所有点均为负样本
                    else torch.zeros(len(result), dtype=torch.bool, device=self.device)
                )
                for point, label in zip(points, labels):
                    point_idx[torch.nonzero(masks[:, point[1], point[0]], as_tuple=True)[0]] = bool(label)
                idx |= point_idx
            if texts is not None:
                if isinstance(texts, str):
                    texts = [texts]
                crop_ims, filter_idx = [], []
                for i, b in enumerate(result.boxes.xyxy.tolist()):
                    x1, y1, x2, y2 = (int(x) for x in b)
                    if (masks[i].sum() if TORCH_1_10 else masks[i].sum(0).sum()) <= 100:  # torch 1.9 错误的变通方案
                        filter_idx.append(i)
                        continue
                    crop = result.orig_img[y1:y2, x1:x2] * masks[i, y1:y2, x1:x2, None].cpu().numpy()
                    crop_ims.append(Image.fromarray(crop[:, :, ::-1]))
                similarity = self._clip_inference(crop_ims, texts)
                text_idx = torch.argmax(similarity, dim=-1)  # (M, )
                if len(filter_idx):
                    # 在过滤前将 text_idx 重新映射回原始索引
                    ori_idxs = [i for i in range(len(result)) if i not in filter_idx]
                    text_idx = torch.tensor(ori_idxs[int(text_idx)], device=self.device)
                idx[text_idx] = True

            prompt_results.append(result[idx])

        return prompt_results

    def _clip_inference(self, images, texts):
        """执行 CLIP 推理，计算图像与文本提示之间的相似度。

        Args:
            images (list[PIL.Image]): 源图像列表，每个应为 RGB 通道顺序的 PIL.Image。
            texts (list[str]): 提示文本列表，每个应为字符串对象。

        Returns:
            (torch.Tensor): 给定图像与文本之间的相似度矩阵，形状为 (M, N)。
        """
        from ultralytics.nn.text_model import CLIP

        if not hasattr(self, "clip"):
            self.clip = CLIP("ViT-B/32", device=self.device)
        images = torch.stack([self.clip.image_preprocess(image).to(self.device) for image in images])
        image_features = self.clip.encode_image(images)
        text_features = self.clip.encode_text(self.clip.tokenize(texts))
        return text_features @ image_features.T  # (M, N)

    def set_prompts(self, prompts):
        """设置推理期间使用的提示。"""
        self.prompts = prompts
