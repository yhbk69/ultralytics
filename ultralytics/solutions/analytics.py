# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from itertools import cycle
from typing import Any

import cv2
import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionResults  # 导入父类
from ultralytics.utils import plt_settings


class Analytics(BaseSolution):
    """用于创建和更新多种可视化分析图表的类。

    此类扩展了 BaseSolution，基于目标检测和跟踪数据提供折线图、柱状图、
    饼图和面积图的生成功能。

    属性:
        type (str): 要生成的图表类型（'line'、'bar'、'pie' 或 'area'）。
        x_label (str): X 轴标签。
        y_label (str): Y 轴标签。
        bg_color (str): 图表背景色。
        fg_color (str): 图表前景色。
        title (str): 图表窗口标题。
        max_points (int): 图表显示的最大数据点数。
        fontsize (int): 文本字体大小。
        color_cycle (cycle): 图表颜色的循环迭代器。
        total_counts (int): 检测到的目标总数（用于折线图）。
        clswise_count (dict[str, int]): 按类别统计的字典。
        fig (Figure): Matplotlib 的 Figure 对象。
        ax (Axes): Matplotlib 的 Axes 对象。
        canvas (FigureCanvasAgg): 用于渲染图表的 Canvas。
        lines (dict): 存储面积图线条对象的字典。
        color_mapping (dict[str, str]): 将类别标签映射到颜色的字典，确保可视化颜色一致。

    方法:
        process: 处理图像数据并更新图表。
        update_graph: 使用新数据点更新图表。

    示例:
        >>> analytics = Analytics(analytics_type="line")
        >>> frame = cv2.imread("image.jpg")
        >>> results = analytics.process(frame, frame_number=1)
        >>> cv2.imshow("Analytics", results.plot_im)
    """

    @plt_settings()
    def __init__(self, **kwargs: Any) -> None:
        """初始化 Analytics 类，支持多种图表类型进行可视化数据展示。"""
        super().__init__(**kwargs)

        import matplotlib.pyplot as plt  # 作用域限定，加快 'import ultralytics' 速度
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.type = self.CFG["analytics_type"]  # 图表类型："line"、"pie"、"bar" 或 "area"
        self.x_label = "Classes" if self.type in {"bar", "pie"} else "Frame#"
        self.y_label = "Total Counts"

        # 预定义数据
        self.bg_color = "#F3F3F3"  # 背景色
        self.fg_color = "#111E68"  # 前景色
        self.title = "Ultralytics Solutions"  # 窗口名称
        self.max_points = 45  # 窗口上绘制的最大数据点数
        self.fontsize = 25  # 显示用的文本字体大小
        figsize = self.CFG["figsize"]  # 输出尺寸，如 (12.8, 7.2) -> 1280x720
        self.color_cycle = cycle(["#DD00BA", "#042AFF", "#FF4447", "#7D24FF", "#BD00FF"])

        self.total_counts = 0  # 存储折线图的总计数值
        self.clswise_count = {}  # 按类别计数的字典
        self.update_every = kwargs.get("update_every", 30)  # 默认每 30 帧更新一次图表
        self.last_plot_im = None  # 上次渲染图表的缓存

        # 确保折线图和面积图初始化
        if self.type in {"line", "area"}:
            self.lines = {}
            self.fig = Figure(facecolor=self.bg_color, figsize=figsize)
            self.canvas = FigureCanvasAgg(self.fig)  # 设置公共坐标轴属性
            self.ax = self.fig.add_subplot(111, facecolor=self.bg_color)
            if self.type == "line":
                (self.line,) = self.ax.plot([], [], color="cyan", linewidth=self.line_width)
        elif self.type in {"bar", "pie"}:
            # 初始化柱状图或饼图
            self.fig, self.ax = plt.subplots(figsize=figsize, facecolor=self.bg_color)
            self.canvas = FigureCanvasAgg(self.fig)  # 设置公共坐标轴属性
            self.ax.set_facecolor(self.bg_color)
            self.color_mapping = {}

            if self.type == "pie":  # 确保饼图为圆形
                self.ax.axis("equal")

    def process(self, im0: np.ndarray, frame_number: int) -> SolutionResults:
        """处理图像数据并运行目标跟踪以更新分析图表。

        参数:
            im0 (np.ndarray): 待处理的输入图像。
            frame_number (int): 用于绘制数据的视频帧号。

        返回:
            (SolutionResults): 包含处理后图像 `plot_im`、'total_tracks'（跟踪到的目标总数，int）
                和 'classwise_count'（按类别计数，dict）。

        异常:
            ValueError: 如果指定了不支持的图表类型。

        示例:
            >>> analytics = Analytics(analytics_type="line")
            >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
            >>> results = analytics.process(frame, frame_number=1)
        """
        self.extract_tracks(im0)  # 提取跟踪数据
        if self.type == "line":
            self.total_counts += len(self.boxes)
            update_required = frame_number % self.update_every == 0 or self.last_plot_im is None
            if update_required:
                self.last_plot_im = self.update_graph(frame_number=frame_number)
            plot_im = self.last_plot_im
            self.total_counts = 0
        elif self.type in {"pie", "bar", "area"}:
            from collections import Counter

            self.clswise_count = Counter(self.names[int(cls)] for cls in self.clss)
            update_required = frame_number % self.update_every == 0 or self.last_plot_im is None
            if update_required:
                self.last_plot_im = self.update_graph(
                    frame_number=frame_number, count_dict=self.clswise_count, plot=self.type
                )
            plot_im = self.last_plot_im
        else:
            raise ValueError(f"不支持的 analytics_type='{self.type}'。支持的类型：line, bar, pie, area。")

        # 返回结果供下游使用
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids), classwise_count=self.clswise_count)

    def update_graph(
        self, frame_number: int, count_dict: dict[str, int] | None = None, plot: str = "line"
    ) -> np.ndarray:
        """使用单个或多个类别的新数据更新图表。

        参数:
            frame_number (int): 当前帧号。
            count_dict (dict[str, int], optional): 以类别名为键、计数值为值的字典，用于多类别图表。
                如果为 None，则更新单条折线图。
            plot (str): 图表类型，可选 'line'、'bar'、'pie' 或 'area'。

        返回:
            (np.ndarray): 包含更新图表的图像。

        示例:
            >>> analytics = Analytics(analytics_type="bar")
            >>> frame_num = 10
            >>> results_dict = {"person": 5, "car": 3}
            >>> updated_image = analytics.update_graph(frame_num, results_dict, plot="bar")
        """
        if count_dict is None:
            # 单条折线更新
            x_data = np.append(self.line.get_xdata(), float(frame_number))
            y_data = np.append(self.line.get_ydata(), float(self.total_counts))

            if len(x_data) > self.max_points:
                x_data, y_data = x_data[-self.max_points :], y_data[-self.max_points :]

            self.line.set_data(x_data, y_data)
            self.line.set_label("Counts")
            self.line.set_color("#7b0068")  # 粉色
            self.line.set_marker("*")
            self.line.set_markersize(self.line_width * 5)
        else:
            labels = list(count_dict.keys())
            counts = list(count_dict.values())
            if plot == "area":
                color_cycle = cycle(["#DD00BA", "#042AFF", "#FF4447", "#7D24FF", "#BD00FF"])
                # 多条线或面积图更新
                x_data = self.ax.lines[0].get_xdata() if self.ax.lines else np.array([])
                y_data_dict = {key: np.array([]) for key in count_dict}
                if self.ax.lines:
                    for line, key in zip(self.ax.lines, count_dict.keys()):
                        y_data_dict[key] = line.get_ydata()

                x_data = np.append(x_data, float(frame_number))
                max_length = len(x_data)
                for key in count_dict:
                    y_data_dict[key] = np.append(y_data_dict[key], float(count_dict[key]))
                    if len(y_data_dict[key]) < max_length:
                        y_data_dict[key] = np.pad(y_data_dict[key], (0, max_length - len(y_data_dict[key])))
                if len(x_data) > self.max_points:
                    x_data = x_data[1:]
                    for key in count_dict:
                        y_data_dict[key] = y_data_dict[key][1:]

                self.ax.clear()
                for key, y_data in y_data_dict.items():
                    color = next(color_cycle)
                    self.ax.fill_between(x_data, y_data, color=color, alpha=0.55)
                    self.ax.plot(
                        x_data,
                        y_data,
                        color=color,
                        linewidth=self.line_width,
                        marker="o",
                        markersize=self.line_width * 5,
                        label=f"{key} Data Points",
                    )
            elif plot == "bar":
                self.ax.clear()  # 清除柱状图数据
                for label in labels:  # 将标签映射到颜色
                    if label not in self.color_mapping:
                        self.color_mapping[label] = next(self.color_cycle)
                colors = [self.color_mapping[label] for label in labels]
                bars = self.ax.bar(labels, counts, color=colors)
                for bar, count in zip(bars, counts):
                    self.ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        str(count),
                        ha="center",
                        va="bottom",
                        color=self.fg_color,
                    )
                # 使用柱状图的标签创建图例
                for bar, label in zip(bars, labels):
                    bar.set_label(label)  # 为每个柱状条分配标签
                self.ax.legend(loc="upper left", fontsize=13, facecolor=self.fg_color, edgecolor=self.fg_color)
            elif plot == "pie":
                total = sum(counts)
                percentages = [size / total * 100 for size in counts]
                self.ax.clear()

                start_angle = 90
                # 创建饼图并生成带百分比的图例标签
                wedges, _ = self.ax.pie(
                    counts, labels=labels, startangle=start_angle, textprops={"color": self.fg_color}, autopct=None
                )
                legend_labels = [f"{label} ({percentage:.1f}%)" for label, percentage in zip(labels, percentages)]

                # 使用楔形块和手动创建的标签来设置图例
                self.ax.legend(wedges, legend_labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                self.fig.subplots_adjust(left=0.1, right=0.75)  # 调整布局以容纳图例

        # 通用图表设置
        self.ax.set_facecolor("#f0f0f0")  # 设置为浅灰色或其他颜色
        self.ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)  # 显示网格线以便更好地观察数据
        self.ax.set_title(self.title, color=self.fg_color, fontsize=self.fontsize)
        self.ax.set_xlabel(self.x_label, color=self.fg_color, fontsize=self.fontsize - 3)
        self.ax.set_ylabel(self.y_label, color=self.fg_color, fontsize=self.fontsize - 3)

        # 添加并格式化图例
        legend = self.ax.legend(loc="upper left", fontsize=13, facecolor=self.bg_color, edgecolor=self.bg_color)
        for text in legend.get_texts():
            text.set_color(self.fg_color)

        # 重绘图表、更新视图、捕获并显示更新后的图表
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
        im0 = np.array(self.canvas.renderer.buffer_rgba())
        im0 = cv2.cvtColor(im0[:, :, :3], cv2.COLOR_RGBA2BGR)
        self.display_output(im0)

        return im0  # 返回图像
