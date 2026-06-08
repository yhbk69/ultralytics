# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class RegionCounter(BaseSolution):
    """在视频流中用户自定义区域内进行实时目标计数的类。

    此类继承自 `BaseSolution`，提供在视频帧中定义多边形区域、跟踪目标并统计通过每个定义区域的目标数量功能。
    适用于需要在指定区域（如监控区域或分段区域）进行计数的应用场景。

    属性:
        region_template (dict): 用于创建新计数区域的模板，包含默认属性如名称、多边形坐标和显示颜色。
        counting_regions (list): 存储所有已定义区域的列表，每个条目基于 `region_template`，
            包含具体的区域设置如名称、坐标和颜色。
        region_counts (dict): 存储每个命名区域目标计数的字典。

    方法:
        add_region: 添加具有指定属性的新计数区域。
        process: 处理视频帧以统计每个区域中的目标数量。
        initialize_regions: 初始化用于统计每个区域中目标数量的区域。区域可以有多个。

    示例:
        初始化 RegionCounter 并添加计数区域
        >>> counter = RegionCounter()
        >>> counter.add_region("Zone1", [(100, 100), (200, 100), (200, 200), (100, 200)], (255, 0, 0), (255, 255, 255))
        >>> results = counter.process(frame)
        >>> print(f"跟踪目标总数: {results.total_tracks}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 RegionCounter，用于在用户自定义区域内进行实时目标计数。"""
        super().__init__(**kwargs)
        self.region_template = {
            "name": "默认区域",
            "polygon": None,
            "counts": 0,
            "region_color": (255, 255, 255),
            "text_color": (0, 0, 0),
        }
        self.region_counts = {}
        self.counting_regions = []
        self.initialize_regions()

    def add_region(
        self,
        name: str,
        polygon_points: list[tuple],
        region_color: tuple[int, int, int],
        text_color: tuple[int, int, int],
    ) -> dict[str, Any]:
        """基于提供的模板向计数列表添加具有特定属性的新区域。

        参数:
            name (str): 分配给新区域的名称。
            polygon_points (list[tuple]): 定义区域多边形的 (x, y) 坐标列表。
            region_color (tuple[int, int, int]): 区域可视化的 BGR 颜色。
            text_color (tuple[int, int, int]): 区域内文本的 BGR 颜色。

        返回:
            (dict[str, Any]): 区域信息，包括名称、多边形和显示颜色。
        """
        region = self.region_template.copy()
        region.update(
            {
                "name": name,
                "polygon": self.Polygon(polygon_points),
                "region_color": region_color,
                "text_color": text_color,
            }
        )
        self.counting_regions.append(region)
        return region

    def initialize_regions(self):
        """从 `self.region` 初始化区域，仅执行一次。"""
        if self.region is None:
            self.initialize_region()
        if not isinstance(self.region, dict):  # 确保 self.region 已初始化并结构化为字典
            self.region = {"区域#01": self.region}
        for i, (name, pts) in enumerate(self.region.items()):
            region = self.add_region(name, pts, colors(i, True), (255, 255, 255))
            region["prepared_polygon"] = self.prep(region["polygon"])

    def process(self, im0: np.ndarray) -> SolutionResults:
        """处理输入帧以检测并统计每个定义区域内的目标数量。

        参数:
            im0 (np.ndarray): 输入图像帧，目标和区域将在此帧上进行标注。

        返回:
            (SolutionResults): 包含处理后图像 `plot_im`、'total_tracks'（跟踪目标总数，int）
                和 'region_counts'（每个区域的目标计数，dict）。
        """
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)

        for box, cls, track_id, conf in zip(self.boxes, self.clss, self.track_ids, self.confs):
            annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(track_id, True))
            center = self.Point(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
            for region in self.counting_regions:
                if region["prepared_polygon"].contains(center):
                    region["counts"] += 1
                    self.region_counts[region["name"]] = region["counts"]

        # 显示区域计数
        for region in self.counting_regions:
            poly = region["polygon"]
            pts = list(map(tuple, np.array(poly.exterior.coords, dtype=np.int32)))
            (x1, y1), (x2, y2) = [(int(poly.centroid.x), int(poly.centroid.y))] * 2
            annotator.draw_region(pts, region["region_color"], self.line_width * 2)
            annotator.adaptive_label(
                [x1, y1, x2, y2],
                label=str(region["counts"]),
                color=region["region_color"],
                txt_color=region["text_color"],
                margin=self.line_width * 4,
                shape="rect",
            )
            region["counts"] = 0  # 为下一帧重置计数
        plot_im = annotator.result()
        self.display_output(plot_im)

        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids), region_counts=self.region_counts)
