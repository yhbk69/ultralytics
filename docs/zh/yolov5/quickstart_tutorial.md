---
comments: true
description: 使用 Ultralytics YOLOv5 开启您的实时目标检测之旅！本指南涵盖安装、推理和训练，帮助您快速掌握 YOLOv5。
keywords: YOLOv5, 快速入门, 实时目标检测, AI, ML, PyTorch, 推理, 训练, Ultralytics, 机器学习, 深度学习, PyTorch Hub, COCO 数据集
---

# YOLOv5 快速入门 🚀

与 Ultralytics YOLOv5 一起踏入实时[目标检测](https://www.ultralytics.com/glossary/object-detection)的动态领域！本指南旨在为希望掌握 YOLOv5 的 AI 爱好者和专业人士提供一个全面的起点。从初始设置到高级[训练技巧](../modes/train.md)，我们一应俱全。在本指南结束时，您将具备使用最先进的[深度学习](https://www.ultralytics.com/glossary/deep-learning-dl)方法将 YOLOv5 自信地应用到您的项目中的知识。让我们点燃引擎，飞向 YOLOv5！

## 安装

通过克隆 [YOLOv5 仓库](https://github.com/ultralytics/yolov5)并配置环境来做好准备。这将确保所有必需的[依赖项](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)都已安装。请确认您已准备好 [**Python>=3.8.0**](https://www.python.org/) 和 [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)。这些基础工具对于有效运行 YOLOv5 至关重要。

```bash
git clone https://github.com/ultralytics/yolov5 # 克隆仓库
cd yolov5
pip install -r requirements.txt # 安装依赖
```

## 使用 PyTorch Hub 进行推理

体验 YOLOv5 [PyTorch Hub](./tutorials/pytorch_hub_model_loading.md) 推理的简洁性，其中[模型](https://github.com/ultralytics/yolov5/tree/master/models)会从最新的 YOLOv5 [发布版本](https://github.com/ultralytics/yolov5/releases)无缝下载。该方法利用 [PyTorch](https://www.ultralytics.com/glossary/pytorch) 的强大功能，实现轻松的模型加载和执行，使获取预测结果变得简单直接。

```python
import torch

# 模型加载
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # 可以是 'yolov5n' - 'yolov5x6'，或 'custom'

# 对图像进行推理
img = "https://ultralytics.com/images/zidane.jpg"  # 可以是文件、路径、PIL、OpenCV、numpy 或图像列表

# 运行推理
results = model(img)

# 显示结果
results.print()  # 其他选项：.show()、.save()、.crop()、.pandas() 等。请参阅 Predict 模式文档了解更多。
```

## 使用 detect.py 进行推理

利用 `detect.py` 对各种来源进行灵活的[推理](../modes/predict.md)。它会自动从最新的 YOLOv5 [发布版本](https://github.com/ultralytics/yolov5/releases)获取[模型](https://github.com/ultralytics/yolov5/tree/master/models)并轻松保存结果。该脚本非常适合命令行使用以及将 YOLOv5 集成到更大的系统中，支持图像、视频、目录、网络摄像头甚至[直播流](https://en.wikipedia.org/wiki/Streaming_media)等输入。

```bash
python detect.py --weights yolov5s.pt --source 0                              # 网络摄像头
python detect.py --weights yolov5s.pt --source image.jpg                      # 图像
python detect.py --weights yolov5s.pt --source video.mp4                      # 视频
python detect.py --weights yolov5s.pt --source screen                         # 截图
python detect.py --weights yolov5s.pt --source path/                          # 目录
python detect.py --weights yolov5s.pt --source list.txt                       # 图像列表
python detect.py --weights yolov5s.pt --source list.streams                   # 流列表
python detect.py --weights yolov5s.pt --source 'path/*.jpg'                   # glob 模式
python detect.py --weights yolov5s.pt --source 'https://youtu.be/LNwODJXcvt4' # YouTube 视频
python detect.py --weights yolov5s.pt --source 'rtsp://example.com/media.mp4' # RTSP、RTMP、HTTP 流
```

## 训练

按照下面的[训练说明](../modes/train.md)复现 YOLOv5 [COCO 数据集](https://cocodataset.org/#home)基准测试。所需的[模型](https://github.com/ultralytics/yolov5/tree/master/models)和[数据集](../datasets/detect/coco.md)（如 `coco128.yaml` 或完整的 `coco.yaml`）直接从最新的 YOLOv5 [发布版本](https://github.com/ultralytics/yolov5/releases)中获取。在 V100 [GPU](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit) 上训练 YOLOv5n/s/m/l/x 通常分别需要 1/2/4/6/8 天（注意[多 GPU 训练](./tutorials/multi_gpu_training.md)设置会更快）。通过使用尽可能大的 `--batch-size` 来最大化性能，或使用 `--batch-size -1` 来启用 YOLOv5 [AutoBatch](https://github.com/ultralytics/yolov5/pull/5092) 功能，该功能会自动找到最佳的[批次大小](https://www.ultralytics.com/glossary/batch-size)。以下批次大小适用于 V100-16GB GPU。有关模型配置文件（`*.yaml`）的详细信息，请参阅我们的[配置指南](../usage/cfg.md)。

```bash
# 在 COCO128 上训练 YOLOv5n 3 个 epoch
python train.py --data coco128.yaml --epochs 3 --weights yolov5n.pt --batch-size 128

# 在 COCO 上训练 YOLOv5s 300 个 epoch
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5s.yaml --batch-size 64

# 在 COCO 上训练 YOLOv5m 300 个 epoch
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5m.yaml --batch-size 40

# 在 COCO 上训练 YOLOv5l 300 个 epoch
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5l.yaml --batch-size 24

# 在 COCO 上训练 YOLOv5x 300 个 epoch
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5x.yaml --batch-size 16
```

<img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/yolov5-training-curves.avif" alt="YOLOv5 COCO 数据集训练曲线">

总而言之，YOLOv5 不仅是目标检测的最先进工具，更是[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)通过视觉理解改变我们与世界交互方式的见证。随着您通过本指南不断深入并将 YOLOv5 应用到您的项目中，请记住您正处于技术革命的前沿，能够在[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域取得非凡成就。如果您需要进一步的见解或来自同行远见者的支持，欢迎访问我们的 [GitHub 仓库](https://github.com/ultralytics/yolov5)，这是一个充满活力的开发者和研究人员社区。探索更多资源，如用于无需编码即可进行数据集管理和模型训练的 [Ultralytics 平台](https://platform.ultralytics.com)，或查看我们的[解决方案](https://www.ultralytics.com/solutions)页面以了解实际应用和灵感。继续探索，不断创新，享受 YOLOv5 的奇迹。祝检测愉快！🌠🔍
