# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops


class DetectionPredictor(BasePredictor):
    """基于检测模型进行预测的类，继承自 BasePredictor。

    该预测器专注于目标检测任务，将模型输出处理为带有边界框和
    类别预测的有意义的检测结果。

    Attributes:
        args (namespace): 预测器的配置参数。
        model (nn.Module): 用于推理的检测模型。
        batch (list): 需要处理的图像批次及元数据。

    Methods:
        postprocess: 将原始模型预测处理为检测结果。
        construct_results: 从处理后的预测构建 Results 对象。
        construct_result: 从单张预测创建单个 Result 对象。
        get_obj_feats: 从特征图中提取目标特征。

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.detect import DetectionPredictor
        >>> args = dict(model="yolo26n.pt", source=ASSETS)
        >>> predictor = DetectionPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """对预测结果进行后处理，返回 Results 对象列表。

        该方法对原始模型预测应用非极大值抑制，并为可视化和
        进一步分析做准备。

        Args:
            preds (torch.Tensor): 模型的原始预测结果。
            img (torch.Tensor): 模型输入格式的处理后输入图像张量。
            orig_imgs (torch.Tensor | list): 预处理前的原始输入图像。
            **kwargs (Any): 额外的关键字参数。

        Returns:
            (list): 包含后处理预测结果的 Results 对象列表。

        Examples:
            >>> predictor = DetectionPredictor(overrides=dict(model="yolo26n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        save_feats = getattr(self, "_feats", None) is not None
        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
        )

        if not isinstance(orig_imgs, list):  # 输入图像是 torch.Tensor，不是列表
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # 将目标特征添加到结果中

        return results

    @staticmethod
    def get_obj_feats(feat_maps, idxs):
        """从特征图中提取目标特征。"""
        import torch

        s = min(x.shape[1] for x in feat_maps)  # 找出最短的向量长度
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
        )  # 将所有向量均值降维到相同长度
        return [feats[idx] if idx.shape[0] else [] for feats, idx in zip(obj_feats, idxs)]  # 针对批次中的每张图像

    def construct_results(self, preds, img, orig_imgs):
        """从模型预测构建 Results 对象列表。

        Args:
            preds (list[torch.Tensor]): 每张图像的预测边界框和分数列表。
            img (torch.Tensor): 用于推理的预处理图像批次。
            orig_imgs (list[np.ndarray]): 预处理前的原始图像列表。

        Returns:
            (list[Results]): 包含每张图像检测信息的 Results 对象列表。
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """从单张图像预测构建单个 Results 对象。

        Args:
            pred (torch.Tensor): 预测的边界框和分数，形状为 (N, 6)，N 为检测数量。
            img (torch.Tensor): 用于推理的预处理图像张量。
            orig_img (np.ndarray): 预处理前的原始图像。
            img_path (str): 原始图像文件的路径。

        Returns:
            (Results): 包含原始图像、图像路径、类别名称和缩放后边界框的 Results 对象。
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
