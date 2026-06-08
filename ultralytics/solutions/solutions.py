# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.solutions.config import SolutionConfig
from ultralytics.utils import ASSETS_URL, LOGGER, ops
from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator


class BaseSolution:
    """Ultralytics 解决方案的基类，提供模型加载、目标跟踪、区域初始化等核心功能。

    此类为各种 Ultralytics 解决方案提供核心功能，包括模型加载、目标跟踪和区域初始化。
    它是实现目标计数、姿态估计和分析等具体计算机视觉解决方案的基础。

    属性:
        LineString: shapely 的线字符串几何类。
        Polygon: shapely 的多边形几何类。
        Point: shapely 的点几何类。
        prep: shapely 的预准备几何函数，用于优化空间操作。
        CFG (dict[str, Any]): 从 YAML 文件加载并用 kwargs 更新的配置字典。
        LOGGER: 解决方案专用的日志记录器实例。
        annotator: 用于在图像上绘制注释的标注器实例。
        tracks: 最新推理得到的 YOLO 跟踪结果。
        track_data: 从跟踪结果中提取的跟踪数据（边界框或 OBB）。
        boxes (list): 跟踪结果的边界框坐标。
        clss (list[int]): 跟踪结果的类别索引。
        track_ids (list[int]): 跟踪结果的跟踪 ID。
        confs (list[float]): 跟踪结果的置信度分数。
        track_line: 当前跟踪线，用于存储跟踪历史记录。
        masks: 跟踪结果的分割掩码。
        r_s: 用于空间操作的区域或线段几何对象。
        frame_no (int): 当前帧编号，仅用于日志记录。
        region (list[tuple[int, int]]): 定义感兴趣区域的坐标元组列表。
        line_width (int): 可视化中使用的线条宽度。
        model (YOLO): 已加载的 YOLO 模型实例。
        names (dict[int, str]): 类别索引到类别名称的映射字典。
        classes (list[int]): 需要跟踪的类别索引列表。
        show_conf (bool): 是否在标注中显示置信度分数。
        show_labels (bool): 是否在标注中显示类别标签。
        device (str): 模型推理设备。
        track_add_args (dict[str, Any]): 跟踪配置的额外参数。
        env_check (bool): 表示环境是否支持图像显示的标志。
        track_history (defaultdict): 存储每个目标跟踪历史的字典。
        profilers (tuple): 用于性能监控的性能分析器实例。

    方法:
        adjust_box_label: 为边界框生成格式化标签。
        extract_tracks: 应用目标跟踪并从输入图像中提取跟踪数据。
        store_tracking_history: 存储指定跟踪 ID 和边界框的目标跟踪历史。
        initialize_region: 根据配置初始化计数区域和线段。
        display_output: 显示处理结果，包括帧或保存的结果。
        process: 处理方法的抽象接口，需由每个 Solution 子类实现。

    示例:
        >>> solution = BaseSolution(model="yolo26n.pt", region=[(0, 0), (100, 0), (100, 100), (0, 100)])
        >>> solution.initialize_region()
        >>> image = cv2.imread("image.jpg")
        >>> solution.extract_tracks(image)
        >>> solution.display_output(image)
    """

    def __init__(self, is_cli: bool = False, **kwargs: Any) -> None:
        """初始化 BaseSolution 类，设置配置和 YOLO 模型。

        参数:
            is_cli (bool): 若为 True，启用 CLI 模式。
            **kwargs (Any): 用于覆盖默认值的额外配置参数。
        """
        self.CFG = vars(SolutionConfig().update(**kwargs))
        self.LOGGER = LOGGER  # 存储日志记录器对象，供多个解决方案类使用

        check_requirements("shapely>=2.0.0")
        from shapely.geometry import LineString, Point, Polygon
        from shapely.prepared import prep

        self.LineString = LineString
        self.Polygon = Polygon
        self.Point = Point
        self.prep = prep
        self.annotator = None  # 初始化标注器
        self.tracks = None
        self.track_data = None
        self.boxes = []
        self.clss = []
        self.track_ids = []
        self.track_line = None
        self.masks = None
        self.r_s = None
        self.frame_no = -1  # 仅用于日志记录

        self.LOGGER.info(f"Ultralytics Solutions: ✅ {self.CFG}")
        self.region = self.CFG["region"]  # 存储区域数据，供其他类使用
        self.line_width = self.CFG["line_width"]

        # 加载模型并存储额外信息（类别、显示置信度、显示标签）
        if self.CFG["model"] is None:
            self.CFG["model"] = "yolo26n.pt"
        self.model = YOLO(self.CFG["model"])
        self.names = self.model.names
        self.classes = self.CFG["classes"]
        self.show_conf = self.CFG["show_conf"]
        self.show_labels = self.CFG["show_labels"]
        self.device = self.CFG["device"]

        self.track_add_args = {  # 跟踪器的高级配置参数
            k: self.CFG[k] for k in {"iou", "conf", "device", "max_det", "half", "tracker"}
        }  # verbose 必须传给 track 方法；在 YOLO 中设为 False 仍然会记录跟踪信息。

        if is_cli and self.CFG["source"] is None:
            d_s = "solutions_ci_demo.mp4" if "-pose" not in self.CFG["model"] else "solution_ci_pose_demo.mp4"
            self.LOGGER.warning(f"source not provided. using default source {ASSETS_URL}/{d_s}")
            from ultralytics.utils.downloads import safe_download

            safe_download(f"{ASSETS_URL}/{d_s}")  # 从 ultralytics 资源下载默认源
            self.CFG["source"] = d_s  # 设置默认源

        # 初始化环境和区域设置
        self.env_check = check_imshow(warn=True)
        self.track_history = defaultdict(list)

        self.profilers = (
            ops.Profile(device=self.device),  # 跟踪耗时分析
            ops.Profile(device=self.device),  # 解决方案耗时分析
        )

    def adjust_box_label(self, cls: int, conf: float, track_id: int | None = None) -> str | None:
        """为边界框生成格式化标签。

        此方法使用类别索引和置信度分数构建边界框标签字符串。如果提供了跟踪 ID，
        则在标签中可选地包含跟踪 ID。标签格式根据 `self.show_conf` 和 `self.show_labels`
        中定义的显示设置自适应调整。

        参数:
            cls (int): 检测目标的类别索引。
            conf (float): 检测的置信度分数。
            track_id (int, 可选): 跟踪目标的唯一标识符。

        返回:
            (str | None): 若 `self.show_labels` 为 True，返回格式化标签字符串；否则返回 None。
        """
        name = ("" if track_id is None else f"{track_id} ") + self.names[cls]
        return (f"{name} {conf:.2f}" if self.show_conf else name) if self.show_labels else None

    def extract_tracks(self, im0: np.ndarray) -> None:
        """应用目标跟踪并从输入图像或帧中提取跟踪数据。

        参数:
            im0 (np.ndarray): 输入图像或帧。

        示例:
            >>> solution = BaseSolution()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> solution.extract_tracks(frame)
        """
        with self.profilers[0]:
            self.tracks = self.model.track(
                source=im0, persist=True, classes=self.classes, verbose=False, **self.track_add_args
            )[0]
        is_obb = self.tracks.obb is not None
        self.track_data = self.tracks.obb if is_obb else self.tracks.boxes  # 提取 OBB 或目标检测的跟踪数据

        if self.track_data and self.track_data.is_track:
            self.boxes = (self.track_data.xyxyxyxy if is_obb else self.track_data.xyxy).cpu()
            self.clss = self.track_data.cls.cpu().tolist()
            self.track_ids = self.track_data.id.int().cpu().tolist()
            self.confs = self.track_data.conf.cpu().tolist()
        else:
            self.LOGGER.warning("No tracks found.")
            self.boxes, self.clss, self.track_ids, self.confs = [], [], [], []

    def store_tracking_history(self, track_id: int, box) -> None:
        """存储目标的跟踪历史。

        此方法通过将边界框中心点追加到轨迹线中来更新给定目标的跟踪历史。
        跟踪历史最多保留 30 个点。

        参数:
            track_id (int): 跟踪目标的唯一标识符。
            box (list[float]): 目标边界框坐标，格式为 [x1, y1, x2, y2]。

        示例:
            >>> solution = BaseSolution()
            >>> solution.store_tracking_history(1, [100, 200, 300, 400])
        """
        # 存储跟踪历史
        self.track_line = self.track_history[track_id]
        self.track_line.append(tuple(box.mean(dim=0)) if box.numel() > 4 else (box[:4:2].mean(), box[1:4:2].mean()))
        if len(self.track_line) > 30:
            self.track_line.pop(0)

    def initialize_region(self) -> None:
        """根据配置设置初始化计数区域和线段。"""
        if self.region is None:
            self.region = [(10, 200), (540, 200), (540, 180), (10, 180)]
        self.r_s = (
            self.Polygon(self.region) if len(self.region) >= 3 else self.LineString(self.region)
        )  # 区域或线段

    def display_output(self, plot_im: np.ndarray) -> None:
        """显示处理结果，包括展示帧、打印计数或保存结果。

        此方法负责可视化目标检测和跟踪过程的输出。它显示带有标注的处理帧，
        并允许用户通过按键交互关闭显示。

        参数:
            plot_im (np.ndarray): 已处理和标注的图像或帧。

        示例:
            >>> solution = BaseSolution()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> solution.display_output(frame)

        注意:
            - 仅当 'show' 配置为 True 且环境支持图像显示时，此方法才会显示输出。
            - 按 'q' 键可关闭显示窗口。
        """
        if self.CFG.get("show") and self.env_check:
            cv2.imshow("Ultralytics Solutions", plot_im)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()  # 关闭当前帧窗口
                return

    def process(self, *args: Any, **kwargs: Any):
        """处理方法，需由每个 Solution 子类实现。"""

    def __call__(self, *args: Any, **kwargs: Any):
        """允许实例像函数一样调用，支持任意参数。"""
        with self.profilers[1]:
            result = self.process(*args, **kwargs)  # 调用子类特定的 process 方法
        track_or_predict = "predict" if type(self).__name__ == "ObjectCropper" else "track"
        track_or_predict_speed = self.profilers[0].dt * 1e3
        solution_speed = (self.profilers[1].dt - self.profilers[0].dt) * 1e3  # 方案耗时 = process - track
        result.speed = {track_or_predict: track_or_predict_speed, "solution": solution_speed}
        if self.CFG["verbose"]:
            self.frame_no += 1
            counts = Counter(self.clss)  # 仅用于日志记录。
            LOGGER.info(
                f"{self.frame_no}: {result.plot_im.shape[0]}x{result.plot_im.shape[1]} {solution_speed:.1f}ms,"
                f" {', '.join([f'{v} {self.names[k]}' for k, v in counts.items()])}\n"
                f"Speed: {track_or_predict_speed:.1f}ms {track_or_predict}, "
                f"{solution_speed:.1f}ms solution per image at shape "
                f"(1, {getattr(self.model, 'channels', 3)}, {result.plot_im.shape[0]}, {result.plot_im.shape[1]})\n"
            )
        return result


class SolutionAnnotator(Annotator):
    """用于可视化和分析计算机视觉任务的专用标注器类。

    此类扩展了基础 Annotator 类，为 Ultralytics 解决方案提供了绘制区域、质心、
    跟踪轨迹和视觉标注的额外方法。它为各种计算机视觉应用（包括目标检测、跟踪、
    姿态估计和分析）提供了全面的可视化能力。

    属性:
        im (np.ndarray): 正在标注的图像。
        line_width (int): 标注线条的粗细。
        font_size (int): 文本标注的字体大小。
        font (str): 文本渲染使用的字体文件路径。
        pil (bool): 是否使用 PIL 进行文本渲染。
        example (str): 用于检测非 ASCII 标签以启用 PIL 渲染的示例文本。

    方法:
        draw_region: 使用指定的点、颜色和粗细绘制区域。
        queue_counts_display: 在指定区域显示队列计数。
        display_analytics: 显示停车场管理的总体统计信息。
        estimate_pose_angle: 计算目标姿态中三个点之间的角度。
        draw_specific_kpts: 在图像上绘制指定的关键点。
        plot_workout_information: 在图像上绘制带标签的文本框。
        plot_angle_and_count_and_stage: 可视化健身监测的角度、步数和阶段。
        plot_distance_and_line: 显示质心之间的距离并用线段连接。
        display_objects_labels: 用目标类别标签标注边界框。
        sweep_annotator: 可视化垂直扫描线和可选标签。
        visioneye: 将目标质心映射并连接到视觉"眼"点。
        adaptive_label: 在边界框中心绘制圆形或矩形背景形状标签。

    示例:
        >>> annotator = SolutionAnnotator(image)
        >>> annotator.draw_region([(0, 0), (100, 100)], color=(0, 255, 0), thickness=5)
        >>> annotator.display_analytics(
        ...     image, text={"Available Spots": 5}, txt_color=(0, 0, 0), bg_color=(255, 255, 255), margin=10
        ... )
    """

    def __init__(
        self,
        im: np.ndarray,
        line_width: int | None = None,
        font_size: int | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        example: str = "abc",
    ):
        """初始化 SolutionAnnotator 类，设置待标注的图像。

        参数:
            im (np.ndarray): 待标注的图像。
            line_width (int, 可选): 在图像上绘制的线条粗细。
            font_size (int, 可选): 文本标注的字体大小。
            font (str): 字体文件路径。
            pil (bool): 是否使用 PIL 渲染文本。
            example (str): 用于检测非 ASCII 标签以启用 PIL 渲染的示例文本。
        """
        super().__init__(im, line_width, font_size, font, pil, example)

    def draw_region(
        self,
        reg_pts: list[tuple[int, int]] | None = None,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 5,
    ):
        """在图像上绘制区域或线段。

        参数:
            reg_pts (list[tuple[int, int]], 可选): 区域点（线段需 2 个点，区域需 4+ 个点）。
            color (tuple[int, int, int]): 区域的 BGR 颜色值（OpenCV 格式）。
            thickness (int): 绘制区域的线条粗细。
        """
        cv2.polylines(self.im, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

        # 在角点上绘制小圆
        for point in reg_pts:
            cv2.circle(self.im, (point[0], point[1]), thickness * 2, color, -1)  # -1 表示填充圆

    def queue_counts_display(
        self,
        label: str,
        points: list[tuple[int, int]] | None = None,
        region_color: tuple[int, int, int] = (255, 255, 255),
        txt_color: tuple[int, int, int] = (0, 0, 0),
    ):
        """在图像上以点为中心显示队列计数，支持自定义字体大小和颜色。

        参数:
            label (str): 队列计数标签。
            points (list[tuple[int, int]], 可选): 用于计算显示文本中心点的区域点。
            region_color (tuple[int, int, int]): BGR 队列区域颜色（OpenCV 格式）。
            txt_color (tuple[int, int, int]): BGR 文本颜色（OpenCV 格式）。
        """
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        center_x = sum(x_values) // len(points)
        center_y = sum(y_values) // len(points)

        text_size = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_width = text_size[0]
        text_height = text_size[1]

        rect_width = text_width + 20
        rect_height = text_height + 20
        rect_top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
        rect_bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)
        cv2.rectangle(self.im, rect_top_left, rect_bottom_right, region_color, -1)

        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2

        # 绘制文本
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),
            0,
            fontScale=self.sf,
            color=txt_color,
            thickness=self.tf,
            lineType=cv2.LINE_AA,
        )

    def display_analytics(
        self,
        im0: np.ndarray,
        text: dict[str, Any],
        txt_color: tuple[int, int, int],
        bg_color: tuple[int, int, int],
        margin: int,
    ):
        """显示解决方案的总体统计信息（如停车场管理和目标计数）。

        参数:
            im0 (np.ndarray): 推理图像。
            text (dict[str, Any]): 标签字典。
            txt_color (tuple[int, int, int]): 文本颜色（BGR，OpenCV 格式）。
            bg_color (tuple[int, int, int]): 背景颜色（BGR，OpenCV 格式）。
            margin (int): 文本与矩形之间的间距，用于更好的显示效果。
        """
        horizontal_gap = int(im0.shape[1] * 0.02)
        vertical_gap = int(im0.shape[0] * 0.01)
        text_y_offset = 0
        for label, value in text.items():
            txt = f"{label}: {value}"
            text_size = cv2.getTextSize(txt, 0, self.sf, self.tf)[0]
            if text_size[0] < 5 or text_size[1] < 5:
                text_size = (5, 5)
            text_x = im0.shape[1] - text_size[0] - margin * 2 - horizontal_gap
            text_y = text_y_offset + text_size[1] + margin * 2 + vertical_gap
            rect_x1 = text_x - margin * 2
            rect_y1 = text_y - text_size[1] - margin * 2
            rect_x2 = text_x + text_size[0] + margin * 2
            rect_y2 = text_y + margin * 2
            cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
            cv2.putText(im0, txt, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)
            text_y_offset = rect_y2

    @staticmethod
    def _point_xy(point: Any) -> tuple[float, float]:
        """将关键点类对象转换为 (x, y) 浮点元组。"""
        if hasattr(point, "detach"):  # torch.Tensor
            point = point.detach()
        if hasattr(point, "cpu"):  # torch.Tensor
            point = point.cpu()
        if hasattr(point, "numpy"):  # torch.Tensor
            point = point.numpy()
        if hasattr(point, "tolist"):  # numpy / torch
            point = point.tolist()
        return float(point[0]), float(point[1])

    @staticmethod
    @lru_cache(maxsize=256)
    def _estimate_pose_angle_cached(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        """计算三个点之间的角度，用于健身监测（带缓存）。"""
        radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
        angle = abs(radians * 180.0 / math.pi)
        return angle if angle <= 180.0 else (360 - angle)

    @staticmethod
    def estimate_pose_angle(a: Any, b: Any, c: Any) -> float:
        """计算三个点之间的角度，用于健身监测。

        参数:
            a (Any): 第一个点的坐标（如列表/元组/NumPy 数组/torch 张量）。
            b (Any): 第二个点（顶点）的坐标。
            c (Any): 第三个点的坐标。

        返回:
            (float): 三个点之间的角度（度）。
        """
        a_xy, b_xy, c_xy = (
            SolutionAnnotator._point_xy(a),
            SolutionAnnotator._point_xy(b),
            SolutionAnnotator._point_xy(c),
        )
        return SolutionAnnotator._estimate_pose_angle_cached(a_xy, b_xy, c_xy)

    def draw_specific_kpts(
        self,
        keypoints: list[list[float]],
        indices: list[int] | None = None,
        radius: int = 2,
        conf_thresh: float = 0.25,
    ) -> np.ndarray:
        """绘制用于健身动作计数的指定关键点。

        参数:
            keypoints (list[list[float]]): 待绘制的关键点数据，每个格式为 [x, y, confidence]。
            indices (list[int], 可选): 待绘制的关键点索引。绘制顺序遵循此列表的顺序。
            radius (int): 关键点半径。
            conf_thresh (float): 关键点的置信度阈值。

        返回:
            (np.ndarray): 绘制了关键点后的图像。

        注意:
            关键点格式: [x, y] 或 [x, y, confidence]。
            直接修改 self.im。
        """
        indices = indices or [2, 5, 7]
        n = len(keypoints)
        points = [
            (int(keypoints[j][0]), int(keypoints[j][1]))
            for j in indices
            if 0 <= j < n and (float(keypoints[j][2]) if len(keypoints[j]) > 2 else 1.0) >= conf_thresh
        ]

        # 在连续点之间绘制线段
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(self.im, start, end, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # 为关键点绘制圆
        for pt in points:
            cv2.circle(self.im, pt, radius, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        return self.im

    def plot_workout_information(
        self,
        display_text: str,
        position: tuple[int, int],
        color: tuple[int, int, int] = (104, 31, 17),
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ) -> int:
        """在图像上绘制健身信息文本框。

        参数:
            display_text (str): 待显示的文本。
            position (tuple[int, int]): 文本在图像上的放置坐标 (x, y)。
            color (tuple[int, int, int]): 文本背景颜色。
            txt_color (tuple[int, int, int]): 文本前景颜色。

        返回:
            (int): 文本的高度。
        """
        (text_width, text_height), _ = cv2.getTextSize(display_text, 0, fontScale=self.sf, thickness=self.tf)

        # 绘制背景矩形
        cv2.rectangle(
            self.im,
            (position[0], position[1] - text_height - 5),
            (position[0] + text_width + 10, position[1] - text_height - 5 + text_height + 10 + self.tf),
            color,
            -1,
        )
        # 绘制文本
        cv2.putText(self.im, display_text, position, 0, self.sf, txt_color, self.tf)

        return text_height

    def plot_angle_and_count_and_stage(
        self,
        angle_text: str,
        count_text: str,
        stage_text: str,
        center_kpt: list[int],
        color: tuple[int, int, int] = (104, 31, 17),
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ):
        """绘制健身监测的姿态角度、计数值和阶段。

        参数:
            angle_text (str): 健身监测的角度值。
            count_text (str): 健身监测的计数值。
            stage_text (str): 健身监测的阶段判定。
            center_kpt (list[int]): 健身监测的质心姿态索引。
            color (tuple[int, int, int]): 文本背景颜色。
            txt_color (tuple[int, int, int]): 文本前景颜色。
        """
        # 格式化文本
        angle_text, count_text, stage_text = f" {angle_text:.2f}", f"Steps : {count_text}", f" {stage_text}"

        # 绘制角度、计数和阶段文本
        angle_height = self.plot_workout_information(
            angle_text, (int(center_kpt[0]), int(center_kpt[1])), color, txt_color
        )
        count_height = self.plot_workout_information(
            count_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + 20), color, txt_color
        )
        self.plot_workout_information(
            stage_text, (int(center_kpt[0]), int(center_kpt[1]) + angle_height + count_height + 40), color, txt_color
        )

    def plot_distance_and_line(
        self,
        pixels_distance: float,
        centroids: list[tuple[int, int]],
        line_color: tuple[int, int, int] = (104, 31, 17),
        centroid_color: tuple[int, int, int] = (255, 0, 255),
    ):
        """在帧上绘制两个质心之间的距离和连线。

        参数:
            pixels_distance (float): 两个边界框质心之间的像素距离。
            centroids (list[tuple[int, int]]): 边界框质心数据。
            line_color (tuple[int, int, int]): 距离连线颜色。
            centroid_color (tuple[int, int, int]): 边界框质心颜色。
        """
        # 获取文本尺寸
        text = f"Pixels Distance: {pixels_distance:.2f}"
        (text_width_m, text_height_m), _ = cv2.getTextSize(text, 0, self.sf, self.tf)

        # 定义带 10 像素边距的角点并绘制矩形
        cv2.rectangle(self.im, (15, 25), (15 + text_width_m + 20, 25 + text_height_m + 20), line_color, -1)

        # 计算带 10 像素边距的文本位置并绘制文本
        text_position = (25, 25 + text_height_m + 10)
        cv2.putText(
            self.im,
            text,
            text_position,
            0,
            self.sf,
            (255, 255, 255),
            self.tf,
            cv2.LINE_AA,
        )

        cv2.line(self.im, centroids[0], centroids[1], line_color, 3)
        cv2.circle(self.im, centroids[0], 6, centroid_color, -1)
        cv2.circle(self.im, centroids[1], 6, centroid_color, -1)

    def display_objects_labels(
        self,
        im0: np.ndarray,
        text: str,
        txt_color: tuple[int, int, int],
        bg_color: tuple[int, int, int],
        x_center: float,
        y_center: float,
        margin: int,
    ):
        """在停车场管理应用中显示边界框标签。

        参数:
            im0 (np.ndarray): 推理图像。
            text (str): 目标/类别名称。
            txt_color (tuple[int, int, int]): 文本前景色。
            bg_color (tuple[int, int, int]): 文本背景色。
            x_center (float): 边界框的 x 坐标中心点。
            y_center (float): 边界框的 y 坐标中心点。
            margin (int): 文本与矩形之间的间距，用于更好的显示效果。
        """
        text_size = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2

        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin
        cv2.rectangle(
            im0,
            (int(rect_x1), int(rect_y1)),
            (int(rect_x2), int(rect_y2)),
            tuple(map(int, bg_color)),  # 确保颜色值为整数
            -1,
        )

        cv2.putText(
            im0,
            text,
            (int(text_x), int(text_y)),
            0,
            self.sf,
            tuple(map(int, txt_color)),  # 确保颜色值为整数
            self.tf,
            lineType=cv2.LINE_AA,
        )

    def sweep_annotator(
        self,
        line_x: int = 0,
        line_y: int = 0,
        label: str | None = None,
        color: tuple[int, int, int] = (221, 0, 186),
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ):
        """绘制扫描标注线和可选标签。

        参数:
            line_x (int): 扫描线的 x 坐标。
            line_y (int): 扫描线的 y 坐标上限。
            label (str, 可选): 在扫描线中心绘制的文本标签。若为 None，则不绘制标签。
            color (tuple[int, int, int]): 线条和标签背景的 BGR 颜色（OpenCV 格式）。
            txt_color (tuple[int, int, int]): 标签文本的 BGR 颜色（OpenCV 格式）。
        """
        # 绘制扫描线
        cv2.line(self.im, (line_x, 0), (line_x, line_y), color, self.tf * 2)

        # 绘制标签（如有提供）
        if label:
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf, self.tf)
            cv2.rectangle(
                self.im,
                (line_x - text_width // 2 - 10, line_y // 2 - text_height // 2 - 10),
                (line_x + text_width // 2 + 10, line_y // 2 + text_height // 2 + 10),
                color,
                -1,
            )
            cv2.putText(
                self.im,
                label,
                (line_x - text_width // 2, line_y // 2 + text_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.sf,
                txt_color,
                self.tf,
            )

    def visioneye(
        self,
        box: list[float],
        center_point: tuple[int, int],
        color: tuple[int, int, int] = (235, 219, 11),
        pin_color: tuple[int, int, int] = (255, 0, 255),
    ):
        """执行精确的人眼视觉映射与绘制。

        参数:
            box (list[float]): 边界框坐标，格式 [x1, y1, x2, y2]。
            center_point (tuple[int, int]): 视点视图的中心点。
            color (tuple[int, int, int]): 目标质心和连线的颜色。
            pin_color (tuple[int, int, int]): 视点标记的颜色。
        """
        center_bbox = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        cv2.circle(self.im, center_point, self.tf * 2, pin_color, -1)
        cv2.circle(self.im, center_bbox, self.tf * 2, color, -1)
        cv2.line(self.im, center_point, center_bbox, color, self.tf)

    def adaptive_label(
        self,
        box: tuple[float, float, float, float],
        label: str = "",
        color: tuple[int, int, int] = (128, 128, 128),
        txt_color: tuple[int, int, int] = (255, 255, 255),
        shape: str = "rect",
        margin: int = 5,
    ):
        """在给定边界框中心绘制带背景矩形或圆形的标签。

        参数:
            box (tuple[float, float, float, float]): 边界框坐标 (x1, y1, x2, y2)。
            label (str): 待显示的文本标签。
            color (tuple[int, int, int]): 矩形的背景颜色 (B, G, R)。
            txt_color (tuple[int, int, int]): 文本颜色 (B, G, R)。
            shape (str): 标签形状。可选: "circle" 或 "rect"。
            margin (int): 文本与矩形边框之间的间距。
        """
        if shape == "circle" and len(label) > 3:
            LOGGER.warning(f"Length of label is {len(label)}, only first 3 letters will be used for circle annotation.")
            label = label[:3]

        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)  # 边界框中心
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.15, self.tf)[0]  # 获取文本尺寸
        text_x, text_y = x_center - text_size[0] // 2, y_center + text_size[1] // 2  # 计算文本左上角坐标

        if shape == "circle":
            cv2.circle(
                self.im,
                (x_center, y_center),
                int(((text_size[0] ** 2 + text_size[1] ** 2) ** 0.5) / 2) + margin,  # 计算半径
                color,
                -1,
            )
        else:
            cv2.rectangle(
                self.im,
                (text_x - margin, text_y - text_size[1] - margin),  # 计算矩形坐标
                (text_x + text_size[0] + margin, text_y + margin),  # 计算矩形坐标
                color,
                -1,
            )

        # 在矩形上方绘制文本
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),  # 计算文本左上角坐标
            cv2.FONT_HERSHEY_SIMPLEX,
            self.sf - 0.15,
            self.get_txt_color(color, txt_color),
            self.tf,
            lineType=cv2.LINE_AA,
        )


class SolutionResults:
    """封装 Ultralytics Solutions 处理结果的类。

    此类用于存储和管理解决方案流水线生成的各种输出，包括计数、角度、
    健身阶段和其他分析数据。它提供了一种结构化方式来访问和处理来自
    不同计算机视觉解决方案（如目标计数、姿态估计和跟踪分析）的结果。

    属性:
        plot_im (np.ndarray): 经过计数、模糊等处理后的图像。
        in_count (int): 视频流中的"进入"总次数。
        out_count (int): 视频流中的"离开"总次数。
        classwise_count (dict[str, int]): 按类别分类的目标计数字典。
        queue_count (int): 队列或等候区域中的目标数量。
        workout_count (int): 健身动作重复次数。
        workout_angle (float): 健身动作中计算的角度。
        workout_stage (str): 健身的当前阶段。
        pixels_distance (float): 两点或目标之间计算的像素距离。
        available_slots (int): 监控区域中的可用车位数量。
        filled_slots (int): 监控区域中的已占用车位数量。
        email_sent (bool): 是否已发送邮件通知的标志。
        total_tracks (int): 跟踪目标的总数。
        region_counts (dict[str, int]): 特定区域内目标的数量。
        speed_dict (dict[str, float]): 跟踪目标的速度信息字典。
        total_crop_objects (int): 使用 ObjectCropper 裁剪目标的总数。
        speed (dict[str, float]): 跟踪和方案处理的性能计时信息。
    """

    def __init__(self, **kwargs):
        """初始化 SolutionResults 对象，支持默认值或用户自定义值。

        参数:
            **kwargs (Any): 用于覆盖默认属性值的可选参数。
        """
        self.plot_im = None
        self.in_count = 0
        self.out_count = 0
        self.classwise_count = {}
        self.queue_count = 0
        self.workout_count = 0
        self.workout_angle = 0.0
        self.workout_stage = None
        self.pixels_distance = 0.0
        self.available_slots = 0
        self.filled_slots = 0
        self.email_sent = False
        self.total_tracks = 0
        self.region_counts = {}
        self.speed_dict = {}  # 用于速度估计
        self.total_crop_objects = 0
        self.speed = {}

        # 用用户自定义值覆盖默认值
        self.__dict__.update(kwargs)

    def __str__(self) -> str:
        """返回 SolutionResults 对象的格式化字符串表示。

        返回:
            (str): 列出非空属性的字符串表示。
        """
        attrs = {
            k: v
            for k, v in self.__dict__.items()
            if k != "plot_im" and v not in [None, {}, 0, 0.0, False]  # 显式排除 `plot_im`
        }
        return ", ".join(f"{k}={v}" for k, v in attrs.items())
