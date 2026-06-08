---
comments: true
description: 深入 Ultralytics Explorer 进行高级数据探索。执行语义搜索、运行 SQL 查询、利用 AI 驱动的自然语言洞察实现无缝数据分析。
keywords: Ultralytics Explorer, 数据探索, 语义搜索, 向量相似度, SQL 查询, AI, 自然语言查询, 机器学习, OpenAI, LLM, Ultralytics 平台
---

# VOC 探索示例

<div align="center">

<a href="https://www.ultralytics.com/events/yolovision" target="_blank"><img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-yolov8-banner.avif" alt="Ultralytics YOLO banner"></a>

</div>

<p align="center">
<a href="https://docs.ultralytics.com/zh">中文</a> |
<a href="https://docs.ultralytics.com/ko">한국어</a> |
<a href="https://docs.ultralytics.com/ja">日本語</a> |
<a href="https://docs.ultralytics.com/ru">Русский</a> |
<a href="https://docs.ultralytics.com/de">Deutsch</a> |
<a href="https://docs.ultralytics.com/fr">Français</a> |
<a href="https://docs.ultralytics.com/es">Español</a> |
<a href="https://docs.ultralytics.com/pt">Português</a> |
<a href="https://docs.ultralytics.com/tr">Türkçe</a> |
<a href="https://docs.ultralytics.com/vi">Tiếng Việt</a> |
<a href="https://docs.ultralytics.com/ar">العربية</a>
</p>

<div align="center">
<br>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://clickpy.clickhouse.com/dashboard/ultralytics"><img src="https://static.pepy.tech/badge/ultralytics" alt="Ultralytics Downloads"></a>
    <a href="https://discord.com/invite/ultralytics"><img alt="Ultralytics Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>
    <a href="https://community.ultralytics.com/"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a>
    <a href="https://www.reddit.com/r/ultralytics/"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run Ultralytics on Gradient"></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Ultralytics In Colab"></a>
    <a href="https://www.kaggle.com/models/ultralytics/yolo26"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open Ultralytics In Kaggle"></a>
    <a href="https://mybinder.org/v2/gh/ultralytics/ultralytics/HEAD?labpath=examples%2Ftutorial.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Open Ultralytics In Binder"></a>
<br>
</div>

欢迎使用 Ultralytics Explorer API 笔记本。本笔记本介绍了用于通过语义搜索、向量搜索和 SQL 查询探索数据集的可用资源。

试试 `yolo explorer`（由 Explorer API 驱动）

安装 `ultralytics` 并在终端中运行 `yolo explorer`，即可在浏览器中运行自定义查询和语义搜索。

!!! warning "社区提示 ⚠️"

    自 **`ultralytics>=8.3.12`** 起，Ultralytics Explorer 已被移除。要使用 Explorer，请安装 `pip install ultralytics==8.3.11`。[Ultralytics 平台](https://platform.ultralytics.com/)提供了类似（且扩展的）数据集探索功能。

## 设置

安装 `ultralytics` 和所需的[依赖](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml)，然后检查软件和硬件。

```bash
!uv pip install ultralytics[explorer] openai
yolo checks
```

## 相似度搜索

利用向量相似度搜索的强大功能，在数据集中查找相似数据点以及它们在嵌入空间中的距离。只需为给定的数据集-模型对创建嵌入表。这只需要一次，之后会自动重用。

```python
exp = Explorer("VOC.yaml", model="yolo26n.pt")
exp.create_embeddings_table()
```

一旦嵌入表构建完成，你可以通过以下任一方式运行语义搜索：

- 对数据集中的给定索引/索引列表，如 `exp.get_similar(idx=[1, 10], limit=10)`
- 对不在数据集中的任何图像/图像列表——`exp.get_similar(img=["path/to/img1", "path/to/img2"], limit=10)`。在多个输入的情况下，使用它们嵌入的聚合。

你会得到一个 pandas DataFrame，包含与输入最相似的 `limit` 个数据点，以及它们在嵌入空间中的距离。你可以使用此数据集进行进一步过滤。

![Ultralytics Explorer 相似度搜索结果](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/similarity-search-table.avif)

```python
# 按索引搜索数据集
similar = exp.get_similar(idx=1, limit=10)
similar.head()
```

你还可以使用 `plot_similar` 工具直接绘制相似样本。

![向量搜索找到的相似图像](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/similarity-search-image-1.avif)

```python
exp.plot_similar(idx=6500, limit=20)
exp.plot_similar(idx=[100, 101], limit=10)  # 也可以传递索引列表或图像列表

exp.plot_similar(img="https://ultralytics.com/images/bus.jpg", limit=10, labels=False)  # 也可以传递外部图像
```

![使用嵌入的相似度搜索可视化](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/similarity-search-image-2.avif)

## Ask AI：使用自然语言搜索或过滤

你可以向 Explorer 对象提示你想要查看的数据点类型，它将尝试返回包含这些结果的 DataFrame。因为它由 LLM 驱动，所以并不总是正确的。在这种情况下，它将返回 `None`。

![Ultralytics Explorer Ask AI 自然语言查询结果](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ask-ai-nlp-table.avif)

```python
df = exp.ask_ai("show me images containing more than 10 objects with at least 2 persons")
df.head(5)
```

要绘制这些结果，你可以使用 `plot_query_result` 工具。示例：

```python
plt = plot_query_result(exp.ask_ai("show me 10 images containing exactly 2 persons"))
Image.fromarray(plt)
```

![Ask AI 查询结果显示匹配图像](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ask-ai-nlp-image-1.avif)

```python
# 绘制
from PIL import Image
from ultralytics.data.explorer import plot_query_result

plt = plot_query_result(exp.ask_ai("show me 10 images containing exactly 2 persons"))
Image.fromarray(plt)
```

## 在数据集上运行 SQL 查询

有时你可能想调查数据集中的某些条目。为此，Explorer 允许你执行 SQL 查询。它接受以下任一格式：

- 以 "WHERE" 开头的查询将自动选择所有列。这可以视为简写查询。
- 你也可以编写完整查询，在其中指定要选择哪些列。

这可用于调查模型性能和特定数据点。例如：

- 假设你的模型在包含人和狗的图像上表现不佳。你可以编写如下查询来选择至少有 2 个人 AND 至少有一只狗的数据点。

你可以结合 SQL 查询和语义搜索来过滤到特定类型的结果。

```python
table = exp.sql_query("WHERE labels LIKE '%person, person%' AND labels LIKE '%dog%' LIMIT 10")
exp.plot_sql_query("WHERE labels LIKE '%person, person%' AND labels LIKE '%dog%' LIMIT 10", labels=True)
```

![Explorer SQL 查询结果表](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/sql-queries-table.avif)

```python
table = exp.sql_query("WHERE labels LIKE '%person, person%' AND labels LIKE '%dog%' LIMIT 10")
print(table)
```

与相似度搜索一样，你也可以使用 `exp.plot_sql_query` 直接绘制 SQL 查询。

![SQL 查询匹配图像可视化](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/sql-query-image-1.avif)

```python
exp.plot_sql_query("WHERE labels LIKE '%person, person%' AND labels LIKE '%dog%' LIMIT 10", labels=True)
```

## 使用嵌入表（高级）

Explorer 在内部使用 [LanceDB](https://lancedb.github.io/lancedb/) 表。你可以使用 `Explorer.table` 对象直接访问此表并运行原始查询、下推前置和后置过滤器等。

```python
table = exp.table
print(table.schema)
```

### 运行原始查询

向量搜索从数据库中查找最近的向量。在推荐系统或搜索引擎中，你可以找到与搜索内容相似的产品。在 LLM 和其他 AI 应用中，每个数据点可以由某些模型生成的嵌入表示，它返回最相关的特征。

在高维向量空间中的搜索，是查找查询向量的 K 近邻（KNN）。

度量 在 LanceDB 中，度量是描述一对向量之间距离的方式。目前，它支持以下度量：

- L2
- Cosine
- Dot Explorer 的相似度搜索默认使用 L2。你可以直接在表上运行查询，或使用 lance 格式构建自定义工具来管理数据集。有关可用 LanceDB 表操作的更多详细信息，请参阅[文档](https://lancedb.github.io/lancedb/)。

![Explorer 原始 SQL 查询结果表](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/raw-queries-table.avif)

```python
dummy_img_embedding = [i for i in range(256)]
table.search(dummy_img_embedding).limit(5).to_pandas()
```

### 与流行数据格式的互转

```python
df = table.to_pandas()
pa_table = table.to_arrow()
```

### 使用嵌入

你可以从 LanceDB 表访问原始嵌入并进行分析。图像嵌入存储在 `vector` 列中。

```python
import numpy as np

embeddings = table.to_pandas()["vector"].tolist()
embeddings = np.array(embeddings)
```

### 散点图

分析嵌入的初步步骤之一是通过降维在 2D 空间中绘制它们。让我们试一个例子。

![Explorer 嵌入散点图可视化](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/scatterplot-sql-queries.avif)

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # pip install scikit-learn

# 使用 PCA 降维到 3 个分量以进行 3D 可视化
pca = PCA(n_components=3)
reduced_data = pca.fit_transform(embeddings)

# 使用 Matplotlib 的 Axes3D 创建 3D 散点图
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

### 相似度索引

这是一个由嵌入表驱动的简单操作示例。Explorer 提供了 `similarity_index` 操作——

- 它尝试估计每个数据点与数据集其余部分的相似程度。
- 通过计算在生成的嵌入空间中有多少图像嵌入距离当前图像小于 `max_dist`，每次考虑 `top_k` 个相似图像。

对于给定的数据集、模型、`max_dist` 和 `top_k`，相似度索引一旦生成将被重用。如果你的数据集已更改，或你只需要重新生成相似度索引，可以传递 `force=True`。与向量和 SQL 搜索类似，这也提供了直接绘制的工具。

```python
sim_idx = exp.similarity_index(max_dist=0.2, top_k=0.01)
exp.plot_similarity_index(max_dist=0.2, top_k=0.01)
```

![数据集相似度索引分析](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/similarity-index.avif)

先看一下图表。

```python
exp.plot_similarity_index(max_dist=0.2, top_k=0.01)
```

现在看一下操作的输出。

```python
sim_idx = exp.similarity_index(max_dist=0.2, top_k=0.01, force=False)

sim_idx
```

让我们创建一个查询，看看哪些数据点的相似度计数大于 30，并绘制与它们相似的图像。

```python
import numpy as np

sim_count = np.array(sim_idx["count"])
sim_idx["im_file"][sim_count > 30]
```

你应该看到类似这样的内容。

![数据集分析的相似度索引可视化](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/similarity-index-image.avif)

```python
exp.plot_similar(idx=[7146, 14035])  # 使用 2 张图像的平均嵌入
```
