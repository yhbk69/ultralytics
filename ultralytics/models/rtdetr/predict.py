# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class RTDETRPredictor(BasePredictor):
    """RT-DETR（实时检测 Transformer）预测器，继承自 BasePredictor 类用于进行预测。

    该类利用 Vision Transformer 提供实时目标检测，同时保持高准确度。它支持高效的混合编码和基于 IoU 的查询选择
    等关键特性。

    Attributes:
        imgsz (int): 推理的图像尺寸（必须是正方形并按比例填充）。
        args (dict): 预测器的参数覆盖项。
        model (torch.nn.Module): 加载的 RT-DETR 模型。
        batch (list): 当前已处理输入的批次。

    Methods:
        postprocess: 后处理原始模型预测结果以生成边界框和置信度分数。
        pre_transform: 在输入模型进行推理之前对输入图像进行预变换。

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.rtdetr import RTDETRPredictor
        >>> args = dict(model="rtdetr-l.pt", source=ASSETS)
        >>> predictor = RTDETRPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def postprocess(self, preds, img, orig_imgs):
        """后处理模型的原始预测结果以生成边界框和置信度分数。

        该方法根据 `self.args` 中指定的置信度和类别对检测结果进行过滤。它将模型预测结果（已由解码器头部进行
        top-k 选择）转换为包含正确缩放边界框的 Results 对象。

        Args:
            preds (list | tuple): 模型的 [predictions, extra] 列表，其中 predictions 形状为 (bs, num_queries, 6)，
                格式为 [cx, cy, w, h, score, class]。
            img (torch.Tensor): 处理后的输入图像，形状为 (N, 3, H, W)。
            orig_imgs (list | torch.Tensor): 原始未处理的图像。

        Returns:
            (list[Results]): Results 对象列表，包含后处理后的边界框、置信度分数和类别标签。
        """
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        bboxes, scores, labels = preds.split((4, 1, 1), dim=-1)
        if not isinstance(orig_imgs, list):  # 输入图像是 torch.Tensor，不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = []
        for bbox, score, label, orig_img, img_path in zip(bboxes, scores, labels, orig_imgs, self.batch[0]):
            bbox = ops.xywh2xyxy(bbox)
            idx = score.squeeze(-1) > self.args.conf
            if self.args.classes is not None:
                idx = (label == torch.tensor(self.args.classes, device=label.device)).any(1) & idx
            pred = torch.cat([bbox, score, label], dim=-1)[idx][: self.args.max_det]
            oh, ow = orig_img.shape[:2]
            pred[..., [0, 2]] *= ow  # 将 x 坐标缩放至原始宽度
            pred[..., [1, 3]] *= oh  # 将 y 坐标缩放至原始高度
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def pre_transform(self, im):
        """在输入模型进行推理之前对输入图像进行预变换。

        对输入图像进行 letterbox 处理以确保正方形宽高比并按比例填充。

        Args:
            im (list[np.ndarray]): 输入图像，形状为 [(H, W, 3) x N]。

        Returns:
            (list): 经过预变换、准备进行模型推理的图像列表。
        """
        letterbox = LetterBox(self.imgsz, auto=False, scale_fill=True)
        return [letterbox(image=x) for x in im]
