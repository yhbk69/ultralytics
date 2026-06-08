# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any

from ultralytics.engine.results import Results
from ultralytics.solutions.solutions import BaseSolution, SolutionResults


class InstanceSegmentation(BaseSolution):
    """管理图像或视频流中实例分割的类。

    此类扩展了 BaseSolution 类，提供实例分割功能，包括绘制带有边界框和标签
    的分割掩码。

    属性:
        model (str): 用于推理的分割模型。
        line_width (int): 边界框和文本线条的宽度。
        names (dict[int, str]): 将类别索引映射到类别名称的字典。
        clss (list[int]): 检测到的类别索引列表。
        track_ids (list[int]): 检测到的实例跟踪 ID 列表。
        masks (list[np.ndarray]): 检测到的实例分割掩码列表。
        show_conf (bool): 是否显示置信度分数。
        show_labels (bool): 是否显示类别标签。
        show_boxes (bool): 是否显示边界框。

    方法:
        process: 处理输入图像以执行实例分割并标注结果。
        extract_tracks: 从模型预测中提取跟踪数据，包括边界框、类别和掩码。

    示例:
        >>> segmenter = InstanceSegmentation()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = segmenter.process(frame)
        >>> print(f"分割到的实例总数: {results.total_tracks}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 InstanceSegmentation 类，用于检测和标注分割实例。

        参数:
            **kwargs (Any): 传递给 BaseSolution 父类的关键字参数，包括：
                - model (str): 模型名称或路径，默认为 "yolo26n-seg.pt"。
        """
        kwargs["model"] = kwargs.get("model", "yolo26n-seg.pt")
        super().__init__(**kwargs)

        self.show_conf = self.CFG.get("show_conf", True)
        self.show_labels = self.CFG.get("show_labels", True)
        self.show_boxes = self.CFG.get("show_boxes", True)

    def process(self, im0) -> SolutionResults:
        """对输入图像执行实例分割并标注结果。

        参数:
            im0 (np.ndarray): 用于分割的输入图像。

        返回:
            (SolutionResults): 包含标注后图像和跟踪到的实例总数的对象。

        示例:
            >>> segmenter = InstanceSegmentation()
            >>> frame = cv2.imread("image.jpg")
            >>> summary = segmenter.process(frame)
            >>> print(summary)
        """
        self.extract_tracks(im0)  # 提取跟踪数据（边界框、类别和掩码）
        self.masks = getattr(self.tracks, "masks", None)

        # 遍历检测到的类别、跟踪 ID 和分割掩码
        if self.masks is None:
            self.LOGGER.warning("未检测到掩码！请确保使用支持的 Ultralytics 分割模型。")
            plot_im = im0
        else:
            results = Results(im0, path=None, names=self.names, boxes=self.track_data.data, masks=self.masks.data)
            plot_im = results.plot(
                line_width=self.line_width,
                boxes=self.show_boxes,
                conf=self.show_conf,
                labels=self.show_labels,
                color_mode="instance",
            )

        self.display_output(plot_im)  # 使用基类函数显示标注后的输出

        # 返回 SolutionResults
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))
