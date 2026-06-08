---
comments: true
description: 学习如何使用 Ultralytics YOLO 对目标检测数据集实现 K 折交叉验证，提升模型的可靠性与鲁棒性。
keywords: Ultralytics, YOLO, K 折交叉验证, 目标检测, sklearn, pandas, PyYAML, 机器学习, 数据集划分
---

# 使用 Ultralytics 进行 K 折交叉验证

## 简介

本指南全面介绍了在 Ultralytics 生态系统中如何对[目标检测](https://www.ultralytics.com/glossary/object-detection)数据集实现 K 折交叉验证。我们将利用 YOLO 检测格式以及 sklearn、pandas 和 PyYAML 等关键 Python 库，引导你完成必要的环境配置、特征向量生成以及 K 折数据集划分的全过程。

<p align="center">
  <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/k-fold-cross-validation-overview.avif" alt="K 折交叉验证数据划分示意图">
</p>

无论你的项目使用的是水果检测数据集还是自定义数据源，本教程旨在帮助你理解并应用 K 折交叉验证，以增强[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)模型的可靠性与鲁棒性。虽然本教程以 `k=5` 折为例，但请记住，最佳折数取决于你的数据集和项目具体情况。

让我们开始吧。

## 环境配置

- 标注文件应采用 [YOLO 检测格式](../datasets/detect/index.md)。

- 本指南假设标注文件已在本地可用。

- 演示使用的是[水果检测](https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection/code)数据集。
    - 该数据集共包含 8479 张图像。
    - 包含 6 个类别标签，各类别实例数量如下表所示。

| 类别标签 | 实例数量 |
| :------- | :------: |
| Apple    |   7049   |
| Grapes   |   7202   |
| Pineapple |  1613   |
| Orange   |  15549   |
| Banana   |   3536   |
| Watermelon |  1976   |

- 需要的 Python 包包括：
    - `ultralytics`
    - `sklearn`
    - `pandas`
    - `pyyaml`

- 本教程使用 `k=5` 折。但你应根据自己的数据集来确定最合适的折数。

1. 为项目创建并激活一个新的 Python 虚拟环境（`venv`）。使用 `pip`（或其他你偏好的包管理器）安装：
    - Ultralytics 库：`pip install -U ultralytics`。也可以克隆官方[仓库](https://github.com/ultralytics/ultralytics)。
    - Scikit-learn、pandas 和 PyYAML：`pip install -U scikit-learn pandas pyyaml`。

2. 确认标注文件符合 [YOLO 检测格式](../datasets/detect/index.md)。
    - 本教程中，所有标注文件位于 `Fruit-Detection/labels` 目录中。

## 为目标检测数据集生成特征向量

1. 首先创建一个新的 `example.py` Python 文件，用于以下步骤。

2. 获取数据集中所有的标注文件。

    ```python
    from pathlib import Path

    dataset_path = Path("./Fruit-detection")  # 替换为你的自定义数据的路径 'path/to/dataset'
    labels = sorted(dataset_path.rglob("*labels/*.txt"))  # 'labels' 目录下的所有数据
    ```

3. 读取数据集 YAML 文件内容，提取类别标签的索引。

    ```python
    import yaml

    yaml_file = "path/to/data.yaml"  # 包含数据目录和 names 字典的 YAML 文件
    with open(yaml_file, encoding="utf8") as y:
        classes = yaml.safe_load(y)["names"]
    cls_idx = sorted(classes.keys())
    ```

4. 初始化一个空的 `pandas` DataFrame。

    ```python
    import pandas as pd

    index = [label.stem for label in labels]  # 使用基础文件名作为 ID（不含扩展名）
    labels_df = pd.DataFrame([], columns=cls_idx, index=index)
    ```

5. 统计每个标注文件中各类别标签的实例数量。

    ```python
    from collections import Counter

    for label in labels:
        lbl_counter = Counter()

        with open(label) as lf:
            lines = lf.readlines()

        for line in lines:
            # YOLO 标注格式中，类别使用每行第一个位置的整数表示
            lbl_counter[int(line.split(" ", 1)[0])] += 1

        labels_df.loc[label.stem] = lbl_counter

    labels_df = labels_df.fillna(0.0)  # 将 `nan` 值替换为 `0.0`
    ```

6. 以下是填充后的 DataFrame 示例：

    ```
                                                           0    1    2    3    4    5
    '0000a16e4b057580_jpg.rf.00ab48988370f64f5ca8ea4...'  0.0  0.0  0.0  0.0  0.0  7.0
    '0000a16e4b057580_jpg.rf.7e6dce029fb67f01eb19aa7...'  0.0  0.0  0.0  0.0  0.0  7.0
    '0000a16e4b057580_jpg.rf.bc4d31cdcbe229dd022957a...'  0.0  0.0  0.0  0.0  0.0  7.0
    '00020ebf74c4881c_jpg.rf.508192a0a97aa6c4a3b6882...'  0.0  0.0  0.0  1.0  0.0  0.0
    '00020ebf74c4881c_jpg.rf.5af192a2254c8ecc4188a25...'  0.0  0.0  0.0  1.0  0.0  0.0
     ...                                                  ...  ...  ...  ...  ...  ...
    'ff4cd45896de38be_jpg.rf.c4b5e967ca10c7ced3b9e97...'  0.0  0.0  0.0  0.0  0.0  2.0
    'ff4cd45896de38be_jpg.rf.ea4c1d37d2884b3e3cbce08...'  0.0  0.0  0.0  0.0  0.0  2.0
    'ff5fd9c3c624b7dc_jpg.rf.bb519feaa36fc4bf630a033...'  1.0  0.0  0.0  0.0  0.0  0.0
    'ff5fd9c3c624b7dc_jpg.rf.f0751c9c3aa4519ea3c9d6a...'  1.0  0.0  0.0  0.0  0.0  0.0
    'fffe28b31f2a70d4_jpg.rf.7ea16bd637ba0711c53b540...'  0.0  6.0  0.0  0.0  0.0  0.0
    ```

行索引对应标注文件，每个标注文件对应数据集中的一张图像；列对应类别标签索引。每行代表一个伪特征向量，记录了数据集中各类别标签的数量。这种数据结构使得[K 折交叉验证](https://www.ultralytics.com/glossary/cross-validation)能够应用于目标检测数据集。

## K 折数据集划分

1. 使用 `sklearn.model_selection` 中的 `KFold` 类生成 `k` 个数据集划分。
    - 重要提示：
        - 设置 `shuffle=True` 可确保划分中类别的随机分布。
        - 通过设置 `random_state=M`（`M` 为选定的整数），可以获得可复现的结果。

    ```python
    import random

    from sklearn.model_selection import KFold

    random.seed(0)  # 确保可复现性
    ksplit = 5
    kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)  # 设置 random_state 以获得可复现结果

    kfolds = list(kf.split(labels_df))
    ```

2. 数据集现已划分为 `k` 折，每折包含 `train` 和 `val` 索引列表。我们将构建一个 DataFrame 以便更清晰地展示这些结果。

    ```python
    folds = [f"split_{n}" for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=index, columns=folds)

    for i, (train, val) in enumerate(kfolds, start=1):
        folds_df[f"split_{i}"].loc[labels_df.iloc[train].index] = "train"
        folds_df[f"split_{i}"].loc[labels_df.iloc[val].index] = "val"
    ```

3. 计算每折中类别标签的分布情况，即 `val` 与 `train` 中各类别数量的比值。

    ```python
    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()

        # 为避免除零错误，在分母上加上一个极小值（1E-7）
        ratio = val_totals / (train_totals + 1e-7)
        fold_lbl_distrb.loc[f"split_{n}"] = ratio
    ```

    理想情况下，所有类别比值在各划分之间以及不同类别之间应大致相近。但这取决于数据集的具体情况。

4. 为每个划分创建目录和数据集 YAML 文件。

    ```python
    import datetime

    supported_extensions = [".jpg", ".jpeg", ".png"]

    # 初始化一个空列表，用于存储图像文件路径
    images = []

    # 遍历支持的扩展名收集图像文件
    for ext in supported_extensions:
        images.extend(sorted((dataset_path / "images").rglob(f"*{ext}")))

    # 创建必要的目录和数据集 YAML 文件
    save_path = Path(dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
    save_path.mkdir(parents=True, exist_ok=True)
    ds_yamls = []

    for split in folds_df.columns:
        # 创建目录
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        # 创建数据集 YAML 文件
        dataset_yaml = split_dir / f"{split}_dataset.yaml"
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": split_dir.as_posix(),
                    "train": "train",
                    "val": "val",
                    "names": classes,
                },
                ds_y,
            )
    ```

5. 最后将图像和标注文件复制到每折对应的目录（'train' 或 'val'）中。
    - **注意：** 此部分代码所需时间取决于数据集大小和系统硬件性能。

    ```python
    import shutil

    from tqdm import tqdm

    for image, label in tqdm(zip(images, labels), total=len(images), desc="正在复制文件"):
        for split, k_split in folds_df.loc[image.stem].items():
            # 目标目录
            img_to_path = save_path / split / k_split / "images"
            lbl_to_path = save_path / split / k_split / "labels"

            # 将图像和标注文件复制到新目录（如文件已存在会抛出 SamefileError）
            shutil.copy(image, img_to_path / image.name)
            shutil.copy(label, lbl_to_path / label.name)
    ```

## 保存记录（可选）

你可以选择将 K 折划分记录和标签分布 DataFrame 保存为 CSV 文件，以供将来参考。

```python
folds_df.to_csv(save_path / "kfold_datasplit.csv")
fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")
```

## 使用 K 折数据划分训练 YOLO

1. 首先加载 YOLO 模型。

    ```python
    from ultralytics import YOLO

    weights_path = "path/to/weights.pt"  # 使用 yolo26n.pt 作为小型模型
    model = YOLO(weights_path, task="detect")
    ```

2. 遍历数据集 YAML 文件执行训练。结果将保存到由 `project` 和 `name` 参数指定的目录中。默认情况下，该目录为 'runs/detect/train#'，其中 # 为整数序号。

    ```python
    results = {}

    # 在此定义额外的参数
    batch = 16
    project = "kfold_demo"
    epochs = 100

    for k, dataset_yaml in enumerate(ds_yamls):
        model = YOLO(weights_path, task="detect")
        results[k] = model.train(
            data=dataset_yaml, epochs=epochs, batch=batch, project=project, name=f"fold_{k + 1}"
        )  # 可添加任意额外的训练参数
    ```

3. 你也可以使用 [Ultralytics data.utils.autosplit](https://docs.ultralytics.com/reference/data/utils) 函数进行自动数据集划分：

    ```python
    from ultralytics.data.split import autosplit

    # 自动将数据集划分为 train/val/test
    autosplit(path="path/to/images", weights=(0.8, 0.2, 0.0), annotated_only=True)
    ```

## 总结

在本指南中，我们探索了使用 K 折交叉验证训练 YOLO 目标检测模型的完整流程。我们学习了如何将数据集划分为 K 个分区，并确保各类别在不同折之间分布均衡。

我们还探索了创建报告 DataFrame 的过程，用于可视化数据划分以及各划分之间的标签分布，这让我们对训练集和验证集的结构有了清晰的认识。

我们还可以选择保存记录以供将来参考，这在大规模项目或排查模型性能问题时尤其有用。

最后，我们通过循环使用每个划分来执行实际的模型训练，并保存训练结果以供进一步分析和比较。

K 折交叉验证是一种充分利用可用数据的稳健方法，有助于确保模型在不同数据子集上的性能可靠且一致。这将产生一个更具泛化能力、更可靠的模型，不易对特定数据模式产生[过拟合](https://www.ultralytics.com/glossary/overfitting)。

请记住，虽然本指南以 YOLO 为例，但这些步骤大多可迁移到其他机器学习模型。理解这些步骤后，你就能在自己的机器学习项目中有效应用交叉验证。

## 常见问题

### 什么是 K 折交叉验证？为什么它在目标检测中很有用？

K 折交叉验证是一种将数据集划分为 'k' 个子集（折）的技术，用于更可靠地评估模型性能。每一折轮流充当训练数据和[验证数据](https://www.ultralytics.com/glossary/validation-data)。在目标检测中，使用 K 折交叉验证有助于确保 Ultralytics YOLO 模型在不同数据划分下的性能稳健且具有泛化能力，从而增强其可靠性。有关配置 K 折交叉验证的详细说明，请参阅[使用 Ultralytics 进行 K 折交叉验证](#简介)。

### 如何使用 Ultralytics YOLO 实现 K 折交叉验证？

要使用 Ultralytics YOLO 实现 K 折交叉验证，需要遵循以下步骤：

1. 确认标注文件符合 [YOLO 检测格式](../datasets/detect/index.md)。
2. 使用 `sklearn`、`pandas` 和 `pyyaml` 等 Python 库。
3. 从数据集中创建特征向量。
4. 使用 `sklearn.model_selection` 中的 `KFold` 划分数据集。
5. 在每个划分上训练 YOLO 模型。

完整指南请参阅文档中的 [K 折数据集划分](#k-折数据集划分) 章节。

### 为什么应该使用 Ultralytics YOLO 进行目标检测？

Ultralytics YOLO 提供最先进、实时的目标检测能力，兼具高[准确率](https://www.ultralytics.com/glossary/accuracy)和高效性。它功能多样，支持检测、分割和分类等多种[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务。此外，它还能无缝集成 [Ultralytics Platform](https://docs.ultralytics.com/platform) 等工具，实现无代码模型训练与部署。更多详情请访问 [Ultralytics YOLO 页面](https://www.ultralytics.com/yolo)。

### 如何确保标注文件符合 Ultralytics YOLO 的正确格式？

标注文件应遵循 YOLO 检测格式。每个标注文件必须列出目标类别及其在图像中的[边界框](https://www.ultralytics.com/glossary/bounding-box)坐标。YOLO 格式确保了数据处理的标准化和流程化。有关正确标注格式的更多信息，请参阅 [YOLO 检测格式指南](../datasets/detect/index.md)。

### 能否对水果检测以外的自定义数据集使用 K 折交叉验证？

可以，只要标注文件符合 YOLO 检测格式，就能对任意自定义数据集使用 K 折交叉验证。只需将数据集路径和类别标签替换为自定义数据集的对应内容即可。这种灵活性确保任何目标检测项目都能受益于 K 折交叉验证的稳健模型评估。实际示例请参阅[生成特征向量](#为目标检测数据集生成特征向量)章节。