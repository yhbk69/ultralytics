import torch
from ultralytics import YOLO

# 加载 YOLO11n 预训练模型
model = YOLO("yolo11n.pt")

# 进行目标检测
# results = model.predict(
#     "ultralytics/assets/bus.jpg",  # 输入图片
#     save=False,                    # 不保存结果图片
#     verbose=False                  # 不打印推理日志
# )
#
# print("=================================预处理===================================")
#
# import cv2
# import torch
# from ultralytics.data.augment import LetterBox
#
# img = cv2.imread("ultralytics/assets/bus.jpg")
#
# print("原图:", img.shape)
#
# img = LetterBox((640,640))(image=img)
#
# print("LetterBox后:", img.shape)
#
# img = img[..., ::-1].copy()
#
# img = img.transpose(2,0,1)
#
# img = torch.from_numpy(img)
#
# img = img.float() / 255
#
# img = img.unsqueeze(0)
#
# print("最终输入:", img.shape)
# print("dtype:", img.dtype)
# print("范围:", img.min(), img.max())
#
# print("=============================================================================")
#
# # 关闭科学计数法显示，方便观察数值
torch.set_printoptions(sci_mode=False)
# print(results)
# boxes = results[0].boxes
#
# for name in dir(boxes):
#     if not name.startswith("_"):
#         print(name)
# print("===========================================================================")
#
# # 输出检测框原始数据
# print(results[0].boxes.data)

"""
输出示例：

tensor([
    [  3.8040, 229.3700, 796.2159, 728.4183, 0.9403, 5.0000],
    [671.0200, 394.8400, 809.8100, 878.7100, 0.8882, 0.0000],
    [221.5200, 405.8000, 344.9700, 857.5400, 0.8771, 0.0000]
])

每一行表示一个检测目标：

[x1, y1, x2, y2, confidence, class_id]

字段含义：

x1 -------- 检测框左上角 x 坐标
y1 -------- 检测框左上角 y 坐标
x2 -------- 检测框右下角 x 坐标
y2 -------- 检测框右下角 y 坐标
confidence - 置信度（模型认为该目标属于某类别的概率）
class_id --- 类别编号

例如：

[3.8040, 229.3700, 796.2159, 728.4183, 0.9403, 5.0000]

表示：

检测框：
    左上角 = (3.8040, 229.3700)
    右下角 = (796.2159, 728.4183)

置信度：
    0.9403（94.03%）

类别：
    5（COCO数据集中的 bus）

COCO类别部分对应关系：

0  -> person
1  -> bicycle
2  -> car
3  -> motorcycle
5  -> bus
7  -> truck

因此上面的示例表示：

检测到一辆 bus，
位置为 (3.8,229.4)-(796.2,728.4)，
置信度为 94.03%。

常用获取方式：

boxes = results[0].boxes

boxes.xyxy   # [N,4] 左上右下坐标
boxes.xywh   # [N,4] 中心点+宽高
boxes.conf   # [N]   置信度
boxes.cls    # [N]   类别ID

例如：

for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    conf = float(box.conf[0])
    cls_id = int(box.cls[0])

    print(
        f"类别={cls_id}, "
        f"置信度={conf:.3f}, "
        f"框=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
    )
"""



x = torch.randn(1,3,640,640)

with torch.no_grad():
    y = model.model(x)

print(type(y))
print(y)