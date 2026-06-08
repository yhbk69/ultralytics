---
comments: true
description: 使用 Ultralytics Explorer GUI 解锁高级数据探索。利用语义搜索、运行 SQL 查询并通过 AI 获取自然语言数据洞察。
keywords: Ultralytics Explorer GUI, 语义搜索, 向量相似度, SQL 查询, AI, 自然语言搜索, 数据探索, 机器学习, OpenAI, 大语言模型
---

# Explorer GUI

!!! warning "社区公告 ⚠️"

    自 **`ultralytics>=8.3.12`** 起，Ultralytics Explorer 已被移除。如需使用 Explorer，请安装 `pip install ultralytics==8.3.11`。[Ultralytics Platform](https://platform.ultralytics.com/) 提供了类似（且功能更丰富）的数据集探索功能。

Explorer GUI 基于 [Ultralytics Explorer API](api.md) 构建，允许您运行语义/向量相似度搜索、SQL 查询以及使用由大语言模型驱动的 Ask AI 功能进行自然语言查询。

<p>
    <img width="1709" alt="Ultralytics Explorer GUI 主仪表板界面" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/explorer-dashboard-screenshot-1.avif">
</p>

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/3VryynorQeo?start=306"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>Ultralytics Explorer 仪表板概览
</p>

### 安装

```bash
pip install ultralytics[explorer]
```

!!! note

    Ask AI 功能使用 OpenAI，因此首次运行 GUI 时系统会提示您设置 OpenAI API 密钥。
    使用 `yolo settings openai_api_key="..."` 进行设置。

## 向量语义相似度搜索

[语义搜索](https://www.ultralytics.com/glossary/semantic-search)是一种查找与给定图像相似的图像的技术。它基于相似图像具有相似[嵌入](https://www.ultralytics.com/glossary/embeddings)的理念。在 UI 中，您可以选择一张或多张图像并搜索与之相似的图像。当您想查找与给定图像相似的图像或一组表现不如预期的图像时，这非常有用。

例如，在此 VOC 探索仪表板中，用户选择了几张飞机图像：

<p>
<img width="1710" alt="Explorer 选择飞机图像进行相似度搜索" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/explorer-dashboard-screenshot-2.avif">
</p>

运行相似度搜索后，您将看到相似的结果：

<p>
<img width="1710" alt="Ultralytics Explorer 语义相似度搜索" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/explorer-dashboard-screenshot-3.avif">
</p>

## Ask AI

此功能允许您使用自然语言筛选数据集，无需编写 SQL。AI 驱动的查询生成器将您的提示转换为查询并返回匹配结果。例如，您可以提问："显示 100 张恰好包含一个人和两只狗的图像。也可以有其他目标"，它会生成查询并显示相应结果。以下是提问"显示 10 张恰好包含 5 个人的图像"时的示例输出：

<p>
<img width="1709" alt="Explorer Ask AI 显示包含 5 个人的图像结果" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/explorer-dashboard-screenshot-4.avif">
</p>

注意：此功能使用[大语言模型](https://www.ultralytics.com/glossary/large-language-model-llm)，因此结果是概率性的，可能不完全准确。

## 在 CV 数据集上运行 SQL 查询

您可以在数据集上运行 SQL 查询来进行筛选。即使只提供 WHERE 子句也可以正常工作。例如，以下 WHERE 子句返回包含至少一个人和一只狗的图像：

```sql
WHERE labels LIKE '%person%' AND labels LIKE '%dog%'
```

<p>
<img width="1707" alt="Explorer SQL 查询筛选包含人和狗的图像" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/explorer-dashboard-screenshot-5.avif">
</p>

此演示使用 Explorer API 构建，您可以使用它来创建自己的探索性 notebook 或脚本，获取数据集的深度洞察。开始使用请查看 [Explorer API 文档](api.md)。

## 常见问题

### 什么是 Ultralytics Explorer GUI？如何安装？

Ultralytics Explorer GUI 是一个强大的界面，利用 [Ultralytics Explorer API](api.md) 解锁高级数据探索功能。它允许您运行语义/向量相似度搜索、SQL 查询以及使用由[大语言模型](https://www.ultralytics.com/glossary/large-language-model-llm)驱动的 Ask AI 功能进行自然语言查询。

安装 Explorer GUI 使用以下 pip 命令：

```bash
pip install ultralytics[explorer]
```

注意：使用 Ask AI 功能需要设置 OpenAI API 密钥：`yolo settings openai_api_key="..."`。

### Ultralytics Explorer GUI 中的语义搜索功能如何工作？

Ultralytics Explorer GUI 中的语义搜索功能允许您根据嵌入向量查找与给定图像相似的图像。此技术对于识别和探索具有视觉相似性的图像非常有用。要使用此功能，在 UI 中选择一张或多张图像并执行相似图像搜索。结果将显示与所选图像密切相似的图像，有助于高效的数据集探索和[异常检测](https://www.ultralytics.com/glossary/anomaly-detection)。

了解更多关于语义搜索和其他功能，请访问[功能概览](#向量语义相似度搜索)部分。

### 可以在 Ultralytics Explorer GUI 中使用自然语言筛选数据集吗？

可以，借助由大语言模型驱动的 Ask AI 功能，您可以使用自然语言查询来筛选数据集，无需精通 SQL。例如，您可以提问"显示 100 张恰好包含一个人和两只狗的图像。也可以有其他目标"，AI 将在底层生成相应的查询以返回所需结果。

### 如何在 Ultralytics Explorer GUI 中对数据集运行 SQL 查询？

Ultralytics Explorer GUI 允许您直接在数据集上运行 SQL 查询，以高效筛选和管理数据。要运行查询，导航到 GUI 中的 SQL 查询部分并编写查询。例如，要显示至少包含一个人和一只狗的图像，可以使用：

```sql
WHERE labels LIKE '%person%' AND labels LIKE '%dog%'
```

您也可以只提供 WHERE 子句，使查询过程更加灵活。

更多详情请参阅 [SQL 查询部分](#在-cv-数据集上运行-sql-查询)。

### 使用 Ultralytics Explorer GUI 进行数据探索有什么好处？

Ultralytics Explorer GUI 通过语义搜索、SQL 查询和通过 Ask AI 功能的自然语言交互增强了数据探索。这些功能允许用户：

- 高效查找视觉相似的图像。
- 使用复杂的 SQL 查询筛选数据集。
- 利用 AI 进行自然语言搜索，无需高级 SQL 专业知识。

这些功能使其成为开发者、研究人员和数据科学家获取数据集深度洞察的通用工具。

在 [Explorer GUI 文档](#explorer-gui)中探索更多这些功能。
