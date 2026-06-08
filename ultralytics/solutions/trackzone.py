# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any

import cv2
import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class TrackZone(BaseSolution):
    """管理视频流中基于区域的目标跟踪的类。

    此类扩展了 BaseSolution 类，提供在由多边形区域定义的特定区域内跟踪目标的功能。
    区域外的目标将被排除在跟踪之外。

    属性:
        region (np.ndarray): 用于跟踪的多边形区域，表示为点的凸包。
        line_width (int): 绘制边界框和区域边界所用的线条宽度。
        names (list[str]): 模型可检测的类别名称列表。
        boxes (list[np.ndarray]): 跟踪目标的边界框。
        track_ids (list[int]): 每个跟踪目标的唯一标识符。
        clss (list[int]): 跟踪目标的类别索引。

    方法:
        process: 处理视频的每一帧，应用基于区域的跟踪。
        extract_tracks: 从输入帧提取跟踪信息。
        display_output: 显示处理后的输出。

    示例:
        >>> tracker = TrackZone()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = tracker.process(frame)
        >>> cv2.imshow("跟踪帧", results.plot_im)
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 TrackZone 类，用于在视频流中定义的区域内跟踪目标。

        参数:
            **kwargs (Any): 传递给父类的额外关键字参数。
        """
        super().__init__(**kwargs)
        default_region = [(75, 75), (565, 75), (565, 285), (75, 285)]
        self.region = cv2.convexHull(np.array(self.region or default_region, dtype=np.int32))
        self.mask = None

    def process(self, im0: np.ndarray) -> SolutionResults:
        """处理输入帧以在定义区域内跟踪目标。

        此方法初始化标注器，为指定区域创建掩码，仅从掩码区域提取跟踪信息，
        并更新跟踪信息。区域外的目标将被忽略。

        参数:
            im0 (np.ndarray): 待处理的输入图像或帧。

        返回:
            (SolutionResults): 包含处理后图像 `plot_im` 和 `total_tracks`（int），
                表示定义区域内跟踪目标的总数。

        示例:
            >>> tracker = TrackZone()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> results = tracker.process(frame)
        """
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # 初始化标注器

        if self.mask is None:  # 为目标区域创建掩码
            self.mask = np.zeros_like(im0[:, :, 0])
            cv2.fillPoly(self.mask, [self.region], 255)
        masked_frame = cv2.bitwise_and(im0, im0, mask=self.mask)
        self.extract_tracks(masked_frame)

        # 绘制区域边界
        cv2.polylines(im0, [self.region], isClosed=True, color=(255, 255, 255), thickness=self.line_width * 2)

        # 遍历边界框、跟踪 ID、类别索引列表并绘制边界框
        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            annotator.box_label(
                box, label=self.adjust_box_label(cls, conf, track_id=track_id), color=colors(track_id, True)
            )

        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类函数显示输出

        # 返回 SolutionResults
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
