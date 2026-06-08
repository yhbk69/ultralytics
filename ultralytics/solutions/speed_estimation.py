# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import deque
from math import sqrt
from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class SpeedEstimator(BaseSolution):
    """基于目标轨迹在实时视频流中估算目标速度的类。

    此类扩展了 BaseSolution 类，提供使用视频流中的跟踪数据估算目标速度的功能。
    速度基于像素位移随时间的变化计算，并使用可配置的米/像素比例因子转换为实际单位。

    属性:
        fps (float): 用于时间计算的视频帧率。
        frame_count (int): 用于跟踪时间信息的全局帧计数器。
        trk_frame_ids (dict): 将跟踪 ID 映射到其首次出现帧索引的字典。
        spd (dict): 锁定后每个目标的最终速度（km/h）。
        trk_hist (dict): 将跟踪 ID 映射到位置历史双端队列的字典。
        locked_ids (set): 速度已最终确定的跟踪 ID 集合。
        max_hist (int): 计算速度前所需的帧历史数量。
        meter_per_pixel (float): 每个像素代表的实际米数，用于场景尺度转换。
        max_speed (int): 允许的最大目标速度，超过此值的将被限制。

    方法:
        process: 处理输入帧，基于跟踪数据估算目标速度。
        store_tracking_history: 存储目标的跟踪历史记录。
        extract_tracks: 从当前帧提取跟踪信息。
        display_output: 显示带有标注的输出。

    示例:
        初始化速度估算器并处理帧
        >>> estimator = SpeedEstimator(meter_per_pixel=0.04, max_speed=120)
        >>> frame = cv2.imread("frame.jpg")
        >>> results = estimator.process(frame)
        >>> cv2.imshow("速度估算", results.plot_im)
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 SpeedEstimator 对象，配置速度估算参数和数据结构。

        参数:
            **kwargs (Any): 传递给父类的额外关键字参数。
        """
        super().__init__(**kwargs)

        self.fps = self.CFG["fps"]  # 用于时间计算的视频帧率
        self.frame_count = 0  # 全局帧计数器
        self.trk_frame_ids = {}  # 跟踪 ID → 首次出现帧索引
        self.spd = {}  # 锁定后每个目标的最终速度（km/h）
        self.trk_hist = {}  # 跟踪 ID → (时间, 位置) 双端队列
        self.locked_ids = set()  # 速度已最终确定的跟踪 ID 集合
        self.max_hist = self.CFG["max_hist"]  # 计算速度前所需的帧历史数量
        self.meter_per_pixel = self.CFG["meter_per_pixel"]  # 场景尺度，取决于相机参数
        self.max_speed = self.CFG["max_speed"]  # 最大速度调整值

    def process(self, im0) -> SolutionResults:
        """处理输入帧，基于跟踪数据估算目标速度。

        参数:
            im0 (np.ndarray): 待处理的输入图像，形状为 (H, W, C)，OpenCV BGR 格式。

        返回:
            (SolutionResults): 包含处理后图像 `plot_im` 和 `total_tracks`（跟踪目标数量）。

        示例:
            处理帧进行速度估算
            >>> estimator = SpeedEstimator()
            >>> image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            >>> results = estimator.process(image)
        """
        self.frame_count += 1
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)

        for box, track_id, _, _ in zip(self.boxes, self.track_ids, self.clss, self.confs):
            self.store_tracking_history(track_id, box)

            if track_id not in self.trk_hist:  # 如果发现新跟踪目标，初始化历史记录
                self.trk_hist[track_id] = deque(maxlen=self.max_hist)
                self.trk_frame_ids[track_id] = self.frame_count

            if track_id not in self.locked_ids:  # 在速度锁定之前持续更新历史记录
                trk_hist = self.trk_hist[track_id]
                trk_hist.append(self.track_line[-1])

                # 收集足够的历史记录后计算并锁定速度
                if len(trk_hist) == self.max_hist:
                    p0, p1 = trk_hist[0], trk_hist[-1]  # 轨迹的第一个和最后一个点
                    dt = (self.frame_count - self.trk_frame_ids[track_id]) / self.fps  # 时间间隔（秒）
                    if dt > 0:
                        dx, dy = p1[0] - p0[0], p1[1] - p0[1]  # 像素位移
                        pixel_distance = sqrt(dx * dx + dy * dy)  # 计算像素距离
                        meters = pixel_distance * self.meter_per_pixel  # 转换为米
                        self.spd[track_id] = int(
                            min((meters / dt) * 3.6, self.max_speed)
                        )  # 转换为 km/h 并存储最终速度
                        self.locked_ids.add(track_id)  # 防止后续更新
                        self.trk_hist.pop(track_id, None)  # 释放内存
                        self.trk_frame_ids.pop(track_id, None)  # 移除帧起始引用

            if track_id in self.spd:
                speed_label = f"{self.spd[track_id]} km/h"
                annotator.box_label(box, label=speed_label, color=colors(track_id, True))  # 绘制边界框

        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类函数显示输出

        # 返回包含处理后图像和跟踪摘要的结果
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
