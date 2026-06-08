# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
from typing import Any

import cv2

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class DistanceCalculation(BaseSolution):
    """基于目标轨迹计算实时视频流中两个目标之间距离的类。

    此类扩展了 BaseSolution，使用 YOLO 目标检测和跟踪功能，允许用户在视频流中
    选择两个目标并计算它们之间的距离。

    属性:
        left_mouse_count (int): 左键点击计数器。
        selected_boxes (dict[int, Any]): 以跟踪 ID 为键存储选中的边界框。
        centroids (list[list[int]]): 存储选中边界框质心的列表。

    方法:
        mouse_event_for_distance: 处理鼠标事件以在视频流中选择目标。
        process: 处理视频帧并计算选中目标之间的距离。

    示例:
        >>> distance_calc = DistanceCalculation()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = distance_calc.process(frame)
        >>> cv2.imshow("Distance Calculation", results.plot_im)
        >>> cv2.waitKey(0)
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 DistanceCalculation 类，用于测量视频流中的目标距离。"""
        super().__init__(**kwargs)

        # 鼠标事件信息
        self.left_mouse_count = 0
        self.selected_boxes: dict[int, list[float]] = {}
        self.centroids: list[list[int]] = []  # 存储选中目标的质心坐标

    def mouse_event_for_distance(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """处理鼠标事件以在实时视频流中选择目标进行距离计算。

        参数:
            event (int): 鼠标事件类型（如 cv2.EVENT_MOUSEMOVE、cv2.EVENT_LBUTTONDOWN）。
            x (int): 鼠标的 X 坐标。
            y (int): 鼠标的 Y 坐标。
            flags (int): 与事件关联的标志（如 cv2.EVENT_FLAG_CTRLKEY、cv2.EVENT_FLAG_SHIFTKEY）。
            param (Any): 传递给函数的附加参数。

        示例:
            >>> # 假设 'dc' 是 DistanceCalculation 的实例
            >>> cv2.setMouseCallback("window_name", dc.mouse_event_for_distance)
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_mouse_count += 1
            if self.left_mouse_count <= 2:
                for box, track_id in zip(self.boxes, self.track_ids):
                    if box[0] < x < box[2] and box[1] < y < box[3] and track_id not in self.selected_boxes:
                        self.selected_boxes[track_id] = box

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.selected_boxes = {}
            self.left_mouse_count = 0

    def process(self, im0) -> SolutionResults:
        """处理视频帧并计算两个选中边界框之间的距离。

        此方法从输入帧中提取跟踪数据、标注边界框，并在用户选择了两个目标后
        计算它们之间的距离。

        参数:
            im0 (np.ndarray): 待处理的输入图像帧。

        返回:
            (SolutionResults): 包含处理后图像 `plot_im`、`total_tracks`（跟踪到的目标总数，int）
                和 `pixels_distance`（选中目标之间的像素距离，float）。

        示例:
            >>> import numpy as np
            >>> from ultralytics.solutions import DistanceCalculation
            >>> dc = DistanceCalculation()
            >>> frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> results = dc.process(frame)
            >>> print(f"Distance: {results.pixels_distance:.2f} pixels")
        """
        self.extract_tracks(im0)  # 提取跟踪数据
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # 初始化标注器

        pixels_distance = 0
        # 遍历边界框、跟踪 ID 和类别索引
        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            annotator.box_label(box, color=colors(int(cls), True), label=self.adjust_box_label(cls, conf, track_id))

            # 如果选中框正在被跟踪，更新其位置
            if len(self.selected_boxes) == 2:
                for trk_id in self.selected_boxes.keys():
                    if trk_id == track_id:
                        self.selected_boxes[track_id] = box

        if len(self.selected_boxes) == 2:
            # 计算选中边界框的质心
            self.centroids.extend(
                [[int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)] for box in self.selected_boxes.values()]
            )
            # 计算两个质心之间的欧几里得距离
            pixels_distance = math.sqrt(
                (self.centroids[0][0] - self.centroids[1][0]) ** 2 + (self.centroids[0][1] - self.centroids[1][1]) ** 2
            )
            annotator.plot_distance_and_line(pixels_distance, self.centroids)

        self.centroids = []  # 重置质心为下一帧准备
        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类函数显示输出
        if self.CFG.get("show") and self.env_check:
            cv2.setMouseCallback("Ultralytics Solutions", self.mouse_event_for_distance)

        # 返回包含处理后图像和计算指标的 SolutionResults
        return SolutionResults(plot_im=plot_im, pixels_distance=pixels_distance, total_tracks=len(self.track_ids))
