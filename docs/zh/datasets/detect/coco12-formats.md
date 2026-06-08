---
comments: true
description: 探索 Ultralytics COCO12-Formats 数据集，一个包含全部 12 种支持图像格式（AVIF、BMP、DNG、HEIC、JP2、JPEG、JPG、MPO、PNG、TIF、TIFF、WebP）的测试数据集，用于验证图像加载流程。
keywords: COCO12-Formats, Ultralytics, 数据集, 图像格式, 目标检测, YOLO, AVIF, BMP, DNG, HEIC, JP2, JPEG, PNG, TIFF, WebP, MPO
---

# COCO12-Formats 数据集

## 简介

[Ultralytics](https://www.ultralytics.com/) COCO12-Formats 数据集是一个专门的测试数据集，旨在验证全部 12 种受支持图像格式扩展名的图像加载。它包含 12 张图像（6 张训练，6 张验证），每张以不同格式保存，以确保对图像加载流程进行全面测试。

该数据集对于以下场景非常宝贵：

- **测试图像格式支持**：验证所有支持的格式是否正确加载
- **CI/CD 流程**：自动测试格式兼容性
- **调试**：隔离训练流程中特定格式的问题
- **开发**：验证新格式的添加或更改

## 支持的格式

该数据集为 `ultralytics/data/utils.py` 中定义的 12 种受支持格式扩展名各包含一张图像：

| 格式   | 扩展名    | 描述                       | 训练/验证 |
| ------ | --------- | -------------------------- | --------- |
| AVIF   | `.avif`   | AV1 图像文件格式（现代）   | 训练      |
| BMP    | `.bmp`    | 位图 — 未压缩光栅格式      | 训练      |
| DNG    | `.dng`    | Digital Negative — Adobe RAW 格式 | 训练      |
| HEIC   | `.heic`   | 高效率图像编码             | 训练      |
| JPEG   | `.jpeg`   | 完整扩展名的 JPEG          | 训练      |
| JPG    | `.jpg`    | 短扩展名的 JPEG            | 训练      |
| JP2    | `.jp2`    | JPEG 2000 — 医学/地理空间  | 验证      |
| MPO    | `.mpo`    | Multi-Picture Object（立体图像） | 验证      |
| PNG    | `.png`    | Portable Network Graphics  | 验证      |
| TIF    | `.tif`    | 短扩展名的 TIFF            | 验证      |
| TIFF   | `.tiff`   | Tagged Image File Format   | 验证      |
| WebP   | `.webp`   | 现代网页图像格式           | 验证      |

## 数据集结构

```
coco12-formats/
├── images/
│   ├── train/          # 6 张图像 (avif, bmp, dng, heic, jpeg, jpg)
│   └── val/            # 6 张图像 (jp2, mpo, png, tif, tiff, webp)
├── labels/
│   ├── train/          # 对应的 YOLO 格式标签
│   └── val/
└── coco12-formats.yaml # 数据集配置
```

## 数据集 YAML

COCO12-Formats 数据集使用 YAML 文件配置，定义了数据集路径和类别名称。您可以在 [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco12-formats.yaml)中查看官方的 `coco12-formats.yaml` 文件。

!!! example "ultralytics/cfg/datasets/coco12-formats.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco12-formats.yaml"
    ```

### 依赖要求

某些格式需要额外的依赖：

```bash
pip install pillow pillow-heif pillow-avif-plugin
```

#### AVIF 系统库（可选）

要让 OpenCV 直接读取 AVIF 文件，必须在**构建 OpenCV 之前**安装 `libavif`：

=== "macOS"

    ```bash
    brew install libavif
    ```

=== "Ubuntu/Debian"

    ```bash
    sudo apt install libavif-dev libavif-bin
    ```

=== "从源码构建"

    ```bash
    git clone -b v1.2.1 https://github.com/AOMediaCodec/libavif.git
    cd libavif
    cmake -B build -DAVIF_CODEC_AOM=SYSTEM -DAVIF_BUILD_APPS=ON
    cmake --build build --config Release --parallel
    sudo cmake --install build
    ```

!!! note

    通过 pip 安装的 `opencv-python` 包可能不包含 AVIF 支持，因为它是预构建的。当 OpenCV 缺乏支持时，Ultralytics 使用 Pillow 配合 `pillow-avif-plugin` 作为 AVIF 图像的备用方案。

## 使用方法

要在 COCO12-Formats 数据集上训练 YOLO 模型，请使用以下示例：

!!! example "训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO model
        model = YOLO("yolo26n.pt")

        # Train on COCO12-Formats to test all image formats
        results = model.train(data="coco12-formats.yaml", epochs=1, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Train YOLO on COCO12-Formats
        yolo detect train data=coco12-formats.yaml model=yolo26n.pt epochs=1 imgsz=640
        ```

## 格式特定说明

### AVIF（AV1 图像文件格式）

AVIF 是一种基于 AV1 视频编解码器的现代图像格式，提供出色的压缩效果。需要 `pillow-avif-plugin`：

```bash
pip install pillow-avif-plugin
```

### DNG（Digital Negative）

DNG 是 Adobe 基于 TIFF 的开放 RAW 格式。出于测试目的，该数据集使用带有 `.dng` 扩展名的基于 TIFF 的文件。

### JP2（JPEG 2000）

JPEG 2000 是一种基于小波的图像压缩标准，比传统 JPEG 提供更好的压缩和质量。常用于医学影像（DICOM）、地理空间应用和数字影院。OpenCV 和 Pillow 均原生支持。

### MPO（Multi-Picture Object）

MPO 文件用于立体（3D）图像。数据集将标准 JPEG 数据存储为 `.mpo` 扩展名，用于格式测试。

### HEIC（高效率图像编码）

HEIC 需要 `pillow-heif` 包来正确编码：

```bash
pip install pillow-heif
```

## 用例

### CI/CD 测试

```python
from ultralytics import YOLO


def test_all_image_formats():
    """Test that all image formats load correctly."""
    model = YOLO("yolo26n.pt")
    results = model.train(data="coco12-formats.yaml", epochs=1, imgsz=64)
    assert results is not None
```

### 格式验证

```python
from pathlib import Path

from ultralytics.data.utils import IMG_FORMATS

# Verify all formats are represented
dataset_dir = Path("datasets/coco12-formats/images")
found_formats = {f.suffix[1:].lower() for f in dataset_dir.rglob("*.*")}
assert found_formats == IMG_FORMATS, f"Missing formats: {IMG_FORMATS - found_formats}"
```

## 引用与致谢

如果您在研究中使用 COCO 数据集，请引用：

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{lin2015microsoft,
              title={Microsoft COCO: Common Objects in Context},
              author={Tsung-Yi Lin and Michael Maire and Serge Belongie and Lubomir Bourdev and Ross Girshick and James Hays and Pietro Perona and Deva Ramanan and C. Lawrence Zitnick and Piotr Doll{\'a}r},
              year={2015},
              eprint={1405.0312},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

## 常见问题

### COCO12-Formats 数据集有什么用途？

COCO12-Formats 数据集旨在测试 Ultralytics YOLO 训练流程中的图像格式兼容性。它确保全部 12 种受支持的图像格式（AVIF、BMP、DNG、HEIC、JP2、JPEG、JPG、MPO、PNG、TIF、TIFF、WebP）都能正确加载和处理。

### 为什么要测试多种图像格式？

不同的图像格式具有独特的特性（压缩、位深、色彩空间）。测试所有格式可确保：

- 鲁棒的图像加载代码
- 跨多样化数据集的兼容性
- 早期检测格式特定的错误

### 哪些格式需要特殊依赖？

- **AVIF**：需要 `pillow-avif-plugin`
- **HEIC**：需要 `pillow-heif`