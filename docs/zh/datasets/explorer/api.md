---
comments: true
description: 探索 Ultralytics Explorer API，用于通过 SQL 查询、向量相似度搜索和语义搜索进行数据集探索。了解安装和使用技巧。
keywords: Ultralytics, Explorer API, 数据集探索, SQL 查询, 相似度搜索, 语义搜索, Python API, 嵌入, 数据分析
---

# Ultralytics Explorer API

!!! warning "社区提示 ⚠️"

    自 **`ultralytics>=8.3.12`** 起，Ultralytics Explorer 已被移除。要使用 Explorer，请安装 `pip install ultralytics==8.3.11`。[Ultralytics 平台](https://platform.ultralytics.com/)提供了类似（且扩展的）数据集探索功能。

## 简介

<a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/docs/en/datasets/explorer/explorer.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
Explorer API 是一个用于探索数据集的 Python API。它支持使用 SQL 查询、向量相似度搜索和语义搜索来过滤和搜索数据集。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/3VryynorQeo?start=279"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Ultralytics Explorer API 概述
</p>

## 安装

Explorer 依赖一些外部库来实现其功能。这些库会在使用 Explorer 时自动安装。要手动安装这些依赖，请使用以下命令：

```bash
pip install ultralytics[explorer]
```

## 使用方法

```python
from ultralytics import Explorer

# 创建 Explorer 对象
explorer = Explorer(data="coco128.yaml", model="yolo26n.pt")

# 为数据集创建嵌入
explorer.create_embeddings_table()

# 搜索与给定图像相似的图像
df = explorer.get_similar(img="path/to/image.jpg")

# 或者搜索与给定索引相似的图像
df = explorer.get_similar(idx=0)
```

!!! note

    给定数据集和模型对的[嵌入](https://www.ultralytics.com/glossary/embeddings)表仅创建一次并可重复使用。这些在底层使用 [LanceDB](https://lancedb.github.io/lancedb/)，它支持磁盘扩展，因此你可以为 COCO 等大型数据集创建和重用嵌入，而不会耗尽内存。

如果你想强制更新嵌入表，可以向 `create_embeddings_table` 方法传递 `force=True`。

你可以直接访问 LanceDB 表对象进行高级分析。在[使用嵌入表部分](#4-使用嵌入表)了解更多。

## 1. 相似度搜索

相似度搜索是一种查找与给定图像相似图像的技术。它基于相似图像将具有相似嵌入的理念。一旦嵌入表构建完成，你可以通过以下任一方式运行语义搜索：

- 对数据集中的给定索引或索引列表：`exp.get_similar(idx=[1,10], limit=10)`
- 对数据集之外的任何图像或图像列表：`exp.get_similar(img=["path/to/img1", "path/to/img2"], limit=10)`

在多个输入的情况下，使用它们嵌入的聚合。

你会得到一个 pandas DataFrame，包含与输入最相似的 `limit` 个数据点，以及它们在嵌入空间中的距离。你可以使用此数据集进行进一步过滤。

!!! example "语义搜索"

    === "使用图像"

        ```python
        from ultralytics import Explorer

        # 创建 Explorer 对象
        exp = Explorer(data="coco128.yaml", model="yolo26n.pt")
        exp.create_embeddings_table()

        similar = exp.get_similar(img="https://ultralytics.com/images/bus.jpg", limit=10)
        print(similar.head())

        # 使用多个索引搜索
        similar = exp.get_similar(
            img=["https://ultralytics.com/images/bus.jpg", "https://ultralytics.com/images/bus.jpg"],
            limit=10,
        )
        print(similar.head())
        ```

    === "使用数据集索引"

        ```python
        from ultralytics import Explorer

        # 创建 Explorer 对象
        exp = Explorer(data="coco128.yaml", model="yolo26n.pt")
        exp.create_embeddings_table()

        similar = exp.get_similar(idx=1, limit=10)
        print(similar.head())

        # 使用多个索引搜索
        similar = exp.get_similar(idx=[1, 10], limit=10)
        print(similar.head())
        ```

### 绘制相似图像

你还可以使用 `plot_similar` 方法绘制相似图像。该方法接受与 `get_similar` 相同的参数，并以网格形式绘制相似图像。

!!! example "绘制相似图像"

    === "使用图像"

        ```python
        from ultralytics import Explorer

        # 创建 Explorer 对象
        exp = Explorer(data="coco128.yaml", model="yolo26n.pt")
        exp.create_embeddings_table()

        plt = exp.plot_similar(img="https://ultralytics.com/images/bus.jpg", limit=10)
        plt.show()
        ```

    === "使用数据集索引"

        ```python
        from ultralytics import Explorer

        # 创建 Explorer 对象
        exp = Explorer(data="coco128.yaml", model="yolo26n.pt")
        exp.create_embeddings_table()

        plt = exp.plot_similar(idx=1, limit=10)
        plt.show()
        ```

## 2. Ask AI（自然语言查询）

此功能允许你使用自然语言过滤数据集，无需编写 SQL。由 AI 驱动的查询生成器将你的提示转换为查询并返回匹配结果。例如，你可以问："show me 100 images with exactly one person and 2 dogs. There can be other objects too"，它会生成查询并显示这些结果。
注意：此功能使用 LLM，因此结果是概率性的，可能不准确。

!!! example "Ask AI"

    ```python
    from ultralytics.data.explorer import plot_query_result

    from ultralytics import Explorer

    # 创建 Explorer 对象
    exp = Explorer(data="coco128.yaml", model="yolo26n.pt")
    exp.create_embeddings_table()

    df = exp.ask_ai("show me 100 images with exactly one person and 2 dogs. There can be other objects too")
    print(df.head())

    # 绘制结果
    plt = plot_query_result(df)
    plt.show()
    ```

## 3. SQL 查询

你可以使用 `sql_query` 方法在数据集上运行 SQL 查询。该方法接受 SQL 查询作为输入，并返回包含结果的 pandas DataFrame。

!!! example "SQL 查询"

    ```python
    from ultralytics import Explorer

    # 创建 Explorer 对象
    exp = Explorer(data="coco128.yaml", model="yolo26n.pt")
    exp.create_embeddings_table()

    df = exp.sql_query("WHERE labels LIKE '%person%' AND labels LIKE '%dog%'")
    print(df.head())
    ```

### 绘制 SQL 查询结果

你还可以使用 `plot_sql_query` 方法绘制 SQL 查询结果。该方法接受与 `sql_query` 相同的参数，并以网格形式绘制结果。

!!! example "绘制 SQL 查询结果"

    ```python
    from ultralytics import Explorer

    # 创建 Explorer 对象
    exp = Explorer(data="coco128.yaml", model="yolo26n.pt")
    exp.create_embeddings_table()

    # 绘制 SQL 查询
    exp.plot_sql_query("WHERE labels LIKE '%person%' AND labels LIKE '%dog%' LIMIT 10")
    ```

## 4. 使用嵌入表

你也可以直接使用嵌入表。一旦嵌入表创建完成，你可以使用 `Explorer.table` 访问它。

!!! tip

    Explorer 在内部使用 [LanceDB](https://lancedb.github.io/lancedb/) 表。你可以使用 `Explorer.table` 对象直接访问此表并运行原始查询、下推前置和后置过滤器等。

    ```python
    from ultralytics import Explorer

    exp = Explorer()
    exp.create_embeddings_table()
    table = exp.table
    ```

以下是一些你可以对表执行的操作示例：

### 获取原始嵌入

!!! example

    ```python
    from ultralytics import Explorer

    exp = Explorer()
    exp.create_embeddings_table()
    table = exp.table

    embeddings = table.to_pandas()["vector"]
    print(embeddings)
    ```

### 使用前置和后置过滤器进行高级查询

!!! example

    ```python
    from ultralytics import Explorer

    exp = Explorer(model="yolo26n.pt")
    exp.create_embeddings_table()
    table = exp.table

    # 虚拟嵌入
    embedding = [i for i in range(256)]
    rs = table.search(embedding).metric("cosine").where("").limit(10)
    ```

### 创建向量索引

在使用大型数据集时，你还可以创建专用的向量索引以加快查询速度。这通过 LanceDB 表上的 `create_index` 方法实现。

```python
table.create_index(num_partitions=..., num_sub_vectors=...)
```

## 5. 嵌入应用

你可以使用嵌入表进行各种探索性分析。以下是一些示例：

### 相似度索引

Explorer 提供了 `similarity_index` 操作：

- 它尝试估计每个数据点与数据集其余部分的相似程度。
- 通过计算在生成的嵌入空间中有多少图像嵌入距离当前图像小于 `max_dist`，每次考虑 `top_k` 个相似图像。

它返回一个包含以下列的 pandas DataFrame：

- `idx`：数据集中图像的索引
- `im_file`：图像文件的路径
- `count`：数据集中距当前图像小于 `max_dist` 的图像数量
- `sim_im_files`：`count` 个相似图像路径的列表

!!! tip

    对于给定的数据集、模型、`max_dist` 和 `top_k`，相似度索引一旦生成将被重用。如果你的数据集已更改，或你只需要重新生成相似度索引，可以传递 `force=True`。

!!! example "相似度索引"

    ```python
    from ultralytics import Explorer

    exp = Explorer()
    exp.create_embeddings_table()

    sim_idx = exp.similarity_index()
    ```

你可以使用相似度索引构建自定义条件来过滤数据集。例如，你可以使用以下代码过滤出与数据集中任何其他图像都不相似的图像：

```python
import numpy as np

sim_count = np.array(sim_idx["count"])
sim_idx["im_file"][sim_count > 30]
```

### 可视化嵌入空间

你还可以使用你选择的绘图工具可视化嵌入空间。例如，这里有一个使用 Matplotlib 的简单示例：

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 使用 PCA 降维到 3 个分量以进行 3D 可视化
pca = PCA(n_components=3)
reduced_data = pca.fit_transform(embeddings)

# 使用 Matplotlib Axes3D 创建 3D 散点图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# 散点图
ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], alpha=0.5)
ax.set_title("3D Scatter Plot of Reduced 256-Dimensional Data (PCA)")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_zlabel("Component 3")

plt.show()
```

开始使用 Explorer API 创建你自己的 CV 数据集探索报告。如需灵感，请查看 [VOC 探索示例](explorer.md)。

## 基于 Ultralytics Explorer 构建的应用

试用基于 Explorer API 的 [GUI 演示](dashboard.md)

## 常见问题

### Ultralytics Explorer API 用于什么？

Ultralytics Explorer API 专为全面的数据集探索而设计。它允许用户使用 SQL 查询、向量相似度搜索和语义搜索来过滤和搜索数据集。这个强大的 Python API 可以处理大型数据集，使其非常适合使用 Ultralytics 模型的各种[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)任务。

### 如何安装 Ultralytics Explorer API？

要安装 Ultralytics Explorer API 及其依赖，请使用以下命令：

```bash
pip install ultralytics[explorer]
```

这将自动安装 Explorer API 功能所需的所有外部库。有关其他设置详情，请参阅文档的[安装部分](#安装)。

### 如何使用 Ultralytics Explorer API 进行相似度搜索？

你可以使用 Ultralytics Explorer API 通过创建嵌入表并查询相似图像来执行相似度搜索。以下是一个基本示例：

```python
from ultralytics import Explorer

# 创建 Explorer 对象
explorer = Explorer(data="coco128.yaml", model="yolo26n.pt")
explorer.create_embeddings_table()

# 搜索与给定图像相似的图像
similar_images_df = explorer.get_similar(img="path/to/image.jpg")
print(similar_images_df.head())
```

更多详情，请访问[相似度搜索部分](#1-相似度搜索)。

### 在 Ultralytics Explorer 中使用 LanceDB 有什么好处？

LanceDB 在 Ultralytics Explorer 底层使用，提供可扩展的磁盘嵌入表。这确保你可以为 COCO 等大型数据集创建和重用嵌入，而不会耗尽内存。这些表仅创建一次并可重复使用，提高了数据处理的效率。

### Ultralytics Explorer API 中的 Ask AI 功能如何工作？

Ask AI 功能允许用户使用自然语言查询过滤数据集。此功能利用 LLM 在后台将这些查询转换为 SQL 查询。以下是一个示例：

```python
from ultralytics import Explorer

# 创建 Explorer 对象
explorer = Explorer(data="coco128.yaml", model="yolo26n.pt")
explorer.create_embeddings_table()

# 使用自然语言查询
query_result = explorer.ask_ai("show me 100 images with exactly one person and 2 dogs. There can be other objects too")
print(query_result.head())
```

更多示例请查看 [Ask AI 部分](#2-ask-ai自然语言查询)。
