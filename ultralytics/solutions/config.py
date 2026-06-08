# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2


@dataclass
class SolutionConfig:
    """Ultralytics 视觉 AI 解决方案的集中式配置管理类。

    SolutionConfig 类作为所有 Ultralytics 解决方案模块的统一配置容器：
    https://docs.ultralytics.com/solutions/#solutions。它基于 Python `dataclass` 实现，
    提供清晰、类型安全且易于维护的参数定义。

    属性:
        source (str, 可选): 输入源路径（视频、RTSP 等）。仅用于 Solutions CLI 模式。
        model (str, 可选): 用于推理的 Ultralytics YOLO 模型路径。
        classes (list[int], 可选): 过滤检测结果的类别索引列表。
        show_conf (bool): 是否在视觉输出中显示置信度分数。
        show_labels (bool): 是否在视觉输出中显示类别标签。
        region (list[tuple[int, int]], 可选): 用于目标计数的多边形区域或线段。
        colormap (int, 可选): 用于视觉叠加的 OpenCV 颜色映射常量（如 cv2.COLORMAP_DEEPGREEN）。
        show_in (bool): 是否显示进入区域的物体计数。
        show_out (bool): 是否显示离开区域的物体计数。
        up_angle (float): 姿态健身监测中判定"上"姿态的角度上限阈值。
        down_angle (int): 姿态健身监测中判定"下"姿态的角度下限阈值。
        kpts (list[int]): 需要监测的关键点索引列表，用于姿态分析。
        analytics_type (str): 分析图表类型（"line"折线、"area"面积、"bar"柱状、"pie"饼图等）。
        figsize (tuple[float, float], 可选): matplotlib 图表尺寸（宽度, 高度）。
        blur_ratio (float): 目标模糊比例（0.0 到 1.0），值越大模糊程度越高。
        vision_point (tuple[int, int]): 方向跟踪或透视绘制的参考视点坐标。
        crop_dir (str): 保存裁剪检测图像的目录路径。
        json_file (str, 可选): 包含停车区域数据的 JSON 文件路径。
        line_width (int): 可视化线宽（边界框、关键点、计数字体等）。
        records (int): 触发邮件告警的检测记录数阈值。
        fps (float): 速度估计计算用的帧率（帧/秒）。
        max_hist (int): 速度估计中每个跟踪目标保存的历史位置点/状态的最大数量。
        meter_per_pixel (float): 像素到实际米数的比例尺，用于速度或距离计算。
        max_speed (int): 速度上限（如 km/h），用于可视化告警或约束。
        show (bool): 是否在屏幕上显示可视化输出。
        iou (float): 检测过滤的交并比（IoU）阈值。
        conf (float): 保留检测结果的置信度阈值。
        device (str, 可选): 推理设备（如 'cpu'、'0' 表示 CUDA GPU）。
        max_det (int): 每帧允许的最大检测数量。
        half (bool): 是否使用 FP16 半精度推理（需要支持的 CUDA 设备）。
        tracker (str): 跟踪配置文件 YAML 路径（如 'botsort.yaml'）。
        verbose (bool): 是否启用详细日志输出，用于调试或诊断。
        data (str): 用于相似度搜索的图像目录路径。

    方法:
        update: 使用用户提供的关键字参数更新配置，对无效键值抛出错误。

    示例:
        >>> from ultralytics.solutions.config import SolutionConfig
        >>> cfg = SolutionConfig(model="yolo26n.pt", region=[(0, 0), (100, 0), (100, 100), (0, 100)])
        >>> cfg.update(show=False, conf=0.3)
        >>> print(cfg.model)
    """

    source: str | None = None
    model: str | None = None
    classes: list[int] | None = None
    show_conf: bool = True
    show_labels: bool = True
    region: list[tuple[int, int]] | None = None
    colormap: int | None = cv2.COLORMAP_DEEPGREEN
    show_in: bool = True
    show_out: bool = True
    up_angle: float = 145.0
    down_angle: int = 90
    kpts: list[int] = field(default_factory=lambda: [6, 8, 10])
    analytics_type: str = "line"
    figsize: tuple[int, int] | None = (12.8, 7.2)
    blur_ratio: float = 0.5
    vision_point: tuple[int, int] = (20, 20)
    crop_dir: str = "cropped-detections"
    json_file: str = None
    line_width: int = 2
    records: int = 5
    fps: float = 30.0
    max_hist: int = 5
    meter_per_pixel: float = 0.05
    max_speed: int = 120
    show: bool = False
    iou: float = 0.7
    conf: float = 0.25
    device: str | None = None
    max_det: int = 300
    half: bool = False
    tracker: str = "botsort.yaml"
    verbose: bool = True
    data: str = "images"

    def update(self, **kwargs: Any):
        """使用关键字参数提供的值更新配置参数。"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                url = "https://docs.ultralytics.com/solutions/#solutions-arguments"
                raise ValueError(f"{key} 不是有效的解决方案参数，请参阅 {url}")

        return self
