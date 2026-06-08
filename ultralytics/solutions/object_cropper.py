# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionResults
from ultralytics.utils.plotting import save_one_box


class ObjectCropper(BaseSolution):
    """管理实时视频流或图像中检测到的目标裁剪的类。

    此类扩展了 BaseSolution 类，基于检测到的边界框提供目标裁剪功能。
    裁剪后的图像保存到指定目录以供进一步分析或使用。

    属性:
        crop_dir (str): 存储裁剪目标图像的目录。
        crop_idx (int): 裁剪目标总数的计数器。
        iou (float): 非极大值抑制的 IoU（交并比）阈值。
        conf (float): 用于过滤检测结果的置信度阈值。

    方法:
        process: 从输入图像中裁剪检测到的目标并保存到输出目录。

    示例:
        >>> cropper = ObjectCropper()
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_results = cropper.process(frame)
        >>> print(f"裁剪目标总数: {cropper.crop_idx}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 ObjectCropper 类，用于从检测到的边界框中裁剪目标。

        参数:
            **kwargs (Any): 传递给父类和配置的关键字参数，包括：
                - crop_dir (str): 保存裁剪目标图像的目录路径。
        """
        super().__init__(**kwargs)

        self.crop_dir = self.CFG["crop_dir"]  # 存储裁剪检测结果的目录
        Path(self.crop_dir).mkdir(parents=True, exist_ok=True)
        if self.CFG["show"]:
            self.LOGGER.warning(f"ObjectCropper 不支持 show=True；裁剪结果将保存到 '{self.crop_dir}'。")
            self.CFG["show"] = False
        self.crop_idx = 0  # 初始化裁剪目标总数计数器
        self.iou = self.CFG["iou"]
        self.conf = self.CFG["conf"]

    def process(self, im0) -> SolutionResults:
        """从输入图像中裁剪检测到的目标并将其保存为单独的图像。

        参数:
            im0 (np.ndarray): 包含检测目标的输入图像。

        返回:
            (SolutionResults): 包含裁剪目标总数和处理后图像的 SolutionResults 对象。

        示例:
            >>> cropper = ObjectCropper()
            >>> frame = cv2.imread("image.jpg")
            >>> results = cropper.process(frame)
            >>> print(f"裁剪目标总数: {results.total_crop_objects}")
        """
        with self.profilers[0]:
            results = self.model.predict(
                im0,
                classes=self.classes,
                conf=self.conf,
                iou=self.iou,
                device=self.CFG["device"],
                verbose=False,
            )[0]
            self.clss = results.boxes.cls.tolist()  # 仅用于日志记录

        for box in results.boxes:
            self.crop_idx += 1
            save_one_box(
                box.xyxy,
                im0,
                file=Path(self.crop_dir) / f"crop_{self.crop_idx}.jpg",
                BGR=True,
            )

        # 返回 SolutionResults
        return SolutionResults(plot_im=im0, total_crop_objects=self.crop_idx)
