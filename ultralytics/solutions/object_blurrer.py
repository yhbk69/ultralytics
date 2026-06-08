# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any

import cv2

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import colors


class ObjectBlurrer(BaseSolution):
    """管理实时视频流中检测到的目标模糊处理的类。

    此类扩展了 BaseSolution 类，基于检测到的边界框提供目标模糊功能。
    模糊区域直接更新到输入图像中，可用于隐私保护或其他效果。

    属性:
        blur_ratio (int): 应用于检测目标的模糊强度（值越大模糊越强）。
        iou (float): 目标检测的 IoU（交并比）阈值。
        conf (float): 目标检测的置信度阈值。

    方法:
        process: 对输入图像中的检测目标应用模糊效果。
        extract_tracks: 从检测目标中提取跟踪信息。
        display_output: 显示处理后的输出图像。

    示例:
        >>> blurrer = ObjectBlurrer()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_results = blurrer.process(frame)
        >>> print(f"模糊目标总数: {processed_results.total_tracks}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 ObjectBlurrer 类，用于对视频流或图像中检测到的目标应用模糊效果。

        参数:
            **kwargs (Any): 传递给父类和配置的关键字参数，包括：
                - blur_ratio (float): 模糊效果强度（0.1-1.0，默认=0.5）。
        """
        super().__init__(**kwargs)
        blur_ratio = self.CFG["blur_ratio"]
        if blur_ratio < 0.1:
            LOGGER.warning("模糊比例不能小于 0.1，已更新为默认值 0.5")
            blur_ratio = 0.5
        self.blur_ratio = int(blur_ratio * 100)

    def process(self, im0) -> SolutionResults:
        """对输入图像中的检测目标应用模糊效果。

        此方法提取跟踪信息，对检测目标对应的区域应用模糊处理，并标注边界框。

        参数:
            im0 (np.ndarray): 包含检测目标的输入图像。

        返回:
            (SolutionResults): 包含处理后图像和跟踪目标数的对象。
                - plot_im (np.ndarray): 带有模糊目标的标注输出图像。
                - total_tracks (int): 帧中跟踪到的目标总数。

        示例:
            >>> blurrer = ObjectBlurrer()
            >>> frame = cv2.imread("image.jpg")
            >>> results = blurrer.process(frame)
            >>> print(f"模糊了 {results.total_tracks} 个目标")
        """
        self.extract_tracks(im0)  # 提取跟踪数据
        annotator = SolutionAnnotator(im0, self.line_width)

        # 遍历边界框和类别
        for box, cls, conf in zip(self.boxes, self.clss, self.confs):
            # 裁剪并模糊检测到的目标
            blur_obj = cv2.blur(
                im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])],
                (self.blur_ratio, self.blur_ratio),
            )
            # 在原图中更新模糊区域
            im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = blur_obj
            annotator.box_label(
                box, label=self.adjust_box_label(cls, conf), color=colors(cls, True)
            )  # 标注边界框

        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类函数显示输出

        # 返回 SolutionResults
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
