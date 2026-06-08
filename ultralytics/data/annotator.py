# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

from ultralytics import SAM, YOLO


def auto_annotate(
    data: str | Path,
    det_model: str = "yolo26x.pt",
    sam_model: str = "sam_b.pt",
    device: str = "",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    max_det: int = 300,
    classes: list[int] | None = None,
    output_dir: str | Path | None = None,
) -> None:
    """使用 YOLO 目标检测模型和 SAM 分割模型自动标注图片。

    此函数处理指定目录中的图片：先用 YOLO 模型检测目标，再用 SAM 模型生成分割掩码。
    生成的标注以 YOLO 格式保存为文本文件。

    参数：
        data (str | Path)：包含待标注图片的文件夹路径。
        det_model (str)：预训练 YOLO 检测模型的路径或名称。
        sam_model (str)：预训练 SAM 分割模型的路径或名称。
        device (str)：运行模型的设备（如 'cpu'、'cuda'、'0'）。空字符串表示自动选择。
        conf (float)：检测模型的置信度阈值。
        iou (float)：过滤检测结果中重叠框的 IoU 阈值。
        imgsz (int)：输入图片的缩放尺寸。
        max_det (int)：每张图片的最大检测数量。
        classes (list[int], 可选)：只返回指定类别 ID 的预测结果，过滤其他类别。
        output_dir (str | Path, 可选)：保存标注结果的目录。为 None 时基于输入路径自动创建默认目录。

    示例：
        >>> from ultralytics.data.annotator import auto_annotate
        >>> auto_annotate(data="ultralytics/assets", det_model="yolo26n.pt", sam_model="mobile_sam.pt")
    """
    det_model = YOLO(det_model)
    sam_model = SAM(sam_model)

    data = Path(data)
    if not output_dir:
        output_dir = data.parent / f"{data.stem}_auto_annotate_labels"
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    det_results = det_model(
        data, stream=True, device=device, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, classes=classes
    )

    for result in det_results:
        if class_ids := result.boxes.cls.int().tolist():  # 从检测结果中提取类别 ID
            boxes = result.boxes.xyxy  # 获取边界框输出的 Boxes 对象
            sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=device)
            segments = sam_results[0].masks.xyn

            with open(f"{Path(output_dir) / Path(result.path).stem}.txt", "w", encoding="utf-8") as f:
                for i, s in enumerate(segments):
                    if s.any():
                        segment = map(str, s.reshape(-1).tolist())
                        f.write(f"{class_ids[i]} " + " ".join(segment) + "\n")
