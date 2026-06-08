# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license


def adjust_bboxes_to_image_border(boxes, image_shape, threshold=20):
    """调整边界框，使其在接近图像边框的阈值范围内时贴合图像边框。

    Args:
        boxes (torch.Tensor): 边界框，形状为 (N, 4)，格式为 xyxy。
        image_shape (tuple): 图像尺寸，格式为 (高度, 宽度)。
        threshold (int): 用于判断边界框是否接近边框的像素阈值。

    Returns:
        (torch.Tensor): 调整后的边界框，形状为 (N, 4)。
    """
    # 图像尺寸
    h, w = image_shape

    # 调整靠近图像边框的边界框
    boxes[boxes[:, 0] < threshold, 0] = 0  # x1
    boxes[boxes[:, 1] < threshold, 1] = 0  # y1
    boxes[boxes[:, 2] > w - threshold, 2] = w  # x2
    boxes[boxes[:, 3] > h - threshold, 3] = h  # y2
    return boxes
