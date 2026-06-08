# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class QueueManager(BaseSolution):
    """基于目标轨迹管理实时视频流中队列计数的类。

    此类扩展了 BaseSolution，提供在视频帧指定区域内跟踪和计数目标的功能。

    属性:
        counts (int): 队列中目标的当前计数。
        rect_color (tuple[int, int, int]): 绘制队列区域矩形的 BGR 颜色元组。
        region_length (int): 定义队列区域的点数。
        track_line (list[tuple[int, int]]): 轨迹线坐标列表。
        track_history (dict[int, list[tuple[int, int]]]): 存储每个目标跟踪历史的字典。

    方法:
        initialize_region: 初始化队列区域。
        process: 处理单帧以进行队列管理。
        extract_tracks: 从当前帧中提取目标轨迹。
        store_tracking_history: 存储目标的跟踪历史。
        display_output: 显示处理后的输出。

    示例:
        >>> cap = cv2.VideoCapture("path/to/video.mp4")
        >>> queue_manager = QueueManager(region=[100, 100, 200, 200, 300, 300])
        >>> while cap.isOpened():
        ...     success, im0 = cap.read()
        ...     if not success:
        ...         break
        ...     results = queue_manager.process(im0)
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 QueueManager，配置视频流中目标跟踪和计数的参数。"""
        super().__init__(**kwargs)
        self.initialize_region()
        self.counts = 0  # 队列计数信息
        self.rect_color = (255, 255, 255)  # 可视化矩形颜色
        self.region_length = len(self.region)  # 存储区域长度以备后续使用

    def process(self, im0) -> SolutionResults:
        """处理单帧视频的队列管理。

        参数:
            im0 (np.ndarray): 待处理的输入图像，通常为视频流中的一帧。

        返回:
            (SolutionResults): 包含处理后图像 `plot_im`、'queue_count'（队列中的目标数，int）
                和 'total_tracks'（跟踪目标总数，int）。

        示例:
            >>> queue_manager = QueueManager()
            >>> frame = cv2.imread("frame.jpg")
            >>> results = queue_manager.process(frame)
        """
        self.counts = 0  # 每帧重置计数
        self.extract_tracks(im0)  # 从当前帧提取轨迹
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # 初始化标注器
        annotator.draw_region(reg_pts=self.region, color=self.rect_color, thickness=self.line_width * 2)  # 绘制区域

        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            # 绘制边界框和计数区域
            annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(track_id, True))
            self.store_tracking_history(track_id, box)  # 存储轨迹历史

            # 缓存频繁访问的属性
            track_history = self.track_history.get(track_id, [])

            # 存储目标的上一帧位置，并检查目标是否在计数区域内
            prev_position = None
            if len(track_history) > 1:
                prev_position = track_history[-2]
            if self.region_length >= 3 and prev_position and self.r_s.contains(self.Point(self.track_line[-1])):
                self.counts += 1

        # 显示队列计数
        annotator.queue_counts_display(
            f"队列计数 : {self.counts}",
            points=self.region,
            region_color=self.rect_color,
            txt_color=(104, 31, 17),
        )
        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类函数显示输出

        # 返回包含处理后数据的 SolutionResults 对象
        return SolutionResults(plot_im=plot_im, queue_count=self.counts, total_tracks=len(self.track_ids))
