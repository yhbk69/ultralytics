from ultralytics import YOLO
import torch

torch.set_printoptions(
    sci_mode=False,
    threshold=float("inf"),
    linewidth=10000,
    precision=6
)

model = YOLO("yolo11n.pt")

x = torch.randn(1, 3, 640, 640)

with torch.no_grad():
    preds = model.model(x)

# 如果返回 tuple
if isinstance(preds, tuple):
    preds = preds[0]

print("shape =", preds.shape)

# 保存完整张量
# with open("preds.txt", "w", encoding="utf-8") as f:
#     f.write(str(preds))
#
# print("保存完成")
box0 = preds[0, :, 0]
print(box0)
print(box0.shape) # torch.Size([84])
print("box0[0:4]",box0[:4]) # tensor([ 7.691854,  3.676630, 15.393341,  7.234964])

cls_scores = box0[4:]
print(type(cls_scores))

conf, cls = cls_scores.max(0)

print("类别:", cls.item())
print("分数:", conf.item())



import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))