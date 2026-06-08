---
description: 了解 Ultralytics 如何收集和使用匿名数据来增强 YOLO Python 包，同时将用户隐私和控制权放在首位。
keywords: Ultralytics, 数据收集, YOLO, Python 包, Google Analytics, Sentry, 隐私, 匿名数据, 用户控制, 崩溃报告
---

# Ultralytics Python 包数据收集

## 概述

[Ultralytics](https://www.ultralytics.com/) 致力于持续提升用户体验和我们 Python 包的功能，包括我们开发的先进 YOLO 模型。我们通过收集匿名使用统计数据和崩溃报告来发现改进机会，并确保软件的可靠性。本文档透明地说明了我们收集哪些数据、用途何在，以及您对此数据收集拥有的选择权。

## 匿名化 Google Analytics

[Google Analytics](https://developers.google.com/analytics) 是 Google 提供的一项网站分析服务，用于跟踪和报告网站流量。它使我们能够收集有关 Python 包使用情况的数据，这对于在设计决策和功能方面做出明智选择至关重要。

### 我们收集的内容

- **使用指标**：这些指标帮助我们了解包的调用频率和使用方式、哪些功能最受欢迎，以及常用的命令行参数。
- **系统信息**：我们收集有关您计算环境的通用非身份识别信息，以确保我们的包在各种系统上都能良好运行。
- **性能数据**：了解模型在训练、验证和推理过程中的性能，有助于我们发现优化机会。

有关 Google Analytics 和[数据隐私](https://www.ultralytics.com/glossary/data-privacy)的更多信息，请访问 [Google Analytics 隐私](https://support.google.com/analytics/answer/6004245)。

### 我们如何使用这些数据

- **功能改进**：来自使用指标的数据指导我们提升用户满意度和界面设计。
- **优化**：性能数据帮助我们针对不同的硬件和软件配置对模型进行微调，以获得更好的效率和速度。
- **趋势分析**：通过研究使用趋势，我们可以预测并响应社区不断变化的需求。

### 隐私考量

我们采取多项措施来确保您托付给我们的数据的隐私和安全：

- **匿名化**：我们将 Google Analytics 配置为对收集的数据进行匿名化处理，这意味着不会收集任何个人身份信息（PII）。您可以放心使用我们的服务，您的个人详细信息始终保密。
- **聚合**：数据仅以聚合形式进行分析。这种做法确保我们可以观察整体模式，而不会暴露任何单个用户的活动。
- **不收集图像数据**：Ultralytics 不会收集、处理或查看任何训练或推理图像。

## Sentry 崩溃报告

[Sentry](https://sentry.io/welcome/) 是一款以开发者为中心的错误追踪软件，可帮助实时识别、诊断和解决问题，确保应用程序的健壮性和可靠性。在我们的包中，它通过崩溃报告提供关键洞察，为软件的稳定性和持续改进做出了重要贡献。

!!! note

    通过 Sentry 进行的崩溃报告仅在您的系统上已预装 `sentry-sdk` Python 包时才会激活。该包不包含在 `ultralytics` 的依赖项中，也不会由 Ultralytics 自动安装。

### 我们收集的内容

如果您的系统上已预装 `sentry-sdk` Python 包，崩溃事件可能会发送以下信息：

- **崩溃日志**：关于崩溃时应用程序状态的详细报告，这对我们的调试工作至关重要。
- **错误消息**：我们记录包运行过程中产生的错误消息，以便快速理解和解决潜在问题。

要了解更多关于 Sentry 如何处理数据的信息，请访问 [Sentry 隐私政策](https://sentry.io/privacy/)。

### 我们如何使用这些数据

- **调试**：分析崩溃日志和错误消息使我们能够迅速识别并修复软件错误。
- **稳定性指标**：通过持续监控崩溃情况，我们旨在提高包的稳定性和可靠性。

### 隐私考量

- **敏感信息**：我们确保崩溃日志中不包含任何个人身份或敏感用户数据，以保护您信息的机密性。
- **受控收集**：我们的崩溃报告机制经过精心校准，仅收集故障排查所必需的信息，同时尊重用户隐私。

通过详细说明用于数据收集的工具，并附上各自隐私页面的 URL 链接，我们为用户提供了全面的实践视图，强调透明度和对用户隐私的尊重。

## 禁用数据收集

我们坚信应该让用户完全控制自己的数据。默认情况下，我们的包配置为收集分析和崩溃报告，以帮助改善所有用户的体验。然而，我们尊重部分用户可能希望退出此数据收集的选择。

要退出分析和崩溃报告的发送，只需在 YOLO 设置中设置 `sync=False`。这将确保不会有任何数据从您的机器传输到我们的分析工具。

### 查看设置

要了解您设置的当前配置，可以直接查看：

!!! example "查看设置"

    === "Python"

        您可以使用 Python 查看设置。首先从 `ultralytics` 模块导入 `settings` 对象。使用以下命令打印和返回设置：
        ```python
        from ultralytics import settings

        # 查看所有设置
        print(settings)

        # 返回分析和崩溃报告设置
        value = settings["sync"]
        ```

    === "CLI"

        或者，命令行界面允许您通过简单的命令检查设置：
        ```bash
        yolo settings
        ```

### 修改设置

Ultralytics 允许用户轻松修改设置。更改可以通过以下方式进行：

!!! example "更新设置"

    === "Python"

        在 Python 环境中，调用 `settings` 对象的 `update` 方法来更改设置：
        ```python
        from ultralytics import settings

        # 禁用分析和崩溃报告
        settings.update({"sync": False})

        # 将设置重置为默认值
        settings.reset()
        ```

    === "CLI"

        如果您更喜欢使用命令行界面，以下命令将允许您修改设置：
        ```bash
        # 禁用分析和崩溃报告
        yolo settings sync=False

        # 将设置重置为默认值
        yolo settings reset
        ```

`sync=False` 设置将阻止任何数据发送到 Google Analytics 或 Sentry。您使用 Ultralytics 包的所有会话都会遵循这一设置，并保存到磁盘以供后续会话使用。

## 隐私承诺

Ultralytics 认真对待用户隐私。我们按照以下原则设计数据收集实践：

- **透明**：我们对收集哪些数据及其使用方式保持公开。
- **控制**：我们给予用户对其数据的完全控制权。
- **安全**：我们采用行业标准的安全措施来保护我们收集的数据。

## 问题或疑虑

如果您对我们的数据收集实践有任何问题或疑虑，请通过我们的[联系表单](https://www.ultralytics.com/contact)或 [support@ultralytics.com](mailto:support@ultralytics.com) 与我们联系。我们致力于确保用户在使用我们的包时，在隐私方面感到知情和放心。

## 常见问题

### Ultralytics 如何确保其收集数据的隐私？

Ultralytics 通过几项关键措施将用户隐私放在首位。首先，所有通过 Google Analytics 和 Sentry 收集的数据均经过匿名化处理，确保不会收集任何个人身份信息（PII）。其次，数据仅以聚合形式分析，使我们能够观察整体模式而不会识别单个用户的活动。最后，我们不收集任何训练或推理图像，进一步保护用户数据。这些措施与我们关于透明度和隐私的承诺一致。更多详情，请参见我们的[隐私考量](#隐私考量)部分。

### Ultralytics 通过 Google Analytics 收集哪些类型的数据？

Ultralytics 使用 Google Analytics 收集三类主要数据：

- **使用指标**：包括 YOLO Python 包的使用频率和方式、受欢迎的功能以及常用的命令行参数。
- **系统信息**：关于运行该包的计算环境的通用非身份识别信息。
- **性能数据**：与模型在训练、验证和推理过程中性能相关的指标。

这些数据帮助我们提升用户体验并优化软件性能。详情请参见[匿名化 Google Analytics](#匿名化-google-analytics) 部分。

### 如何在 Ultralytics YOLO 包中禁用数据收集？

要退出数据收集，只需在 YOLO 设置中设置 `sync=False`。此操作将停止任何分析或崩溃报告的传输。您可以通过 Python 或 CLI 方法禁用数据收集：

!!! example "更新设置"

    === "Python"

        ```python
        from ultralytics import settings

        # 禁用分析和崩溃报告
        settings.update({"sync": False})

        # 将设置重置为默认值
        settings.reset()
        ```

    === "CLI"

        ```bash
        # 禁用分析和崩溃报告
        yolo settings sync=False

        # 将设置重置为默认值
        yolo settings reset
        ```

有关修改设置的更多详情，请参见[修改设置](#修改设置)部分。

### Ultralytics YOLO 中的 Sentry 崩溃报告如何工作？

如果 `sentry-sdk` 包已预装，Sentry 会在每次发生崩溃事件时收集详细的崩溃日志和错误消息。这些数据帮助我们快速诊断和解决问题，提高 YOLO Python 包的健壮性和可靠性。收集的崩溃日志会经过清理，去除任何个人身份信息以保护用户隐私。更多信息请参见 [Sentry 崩溃报告](#sentry-崩溃报告)部分。

### 我可以查看 Ultralytics YOLO 中当前的数据收集设置吗？

是的，您可以轻松查看当前设置，以了解数据收集偏好的配置。使用以下方法查看这些设置：

!!! example "查看设置"

    === "Python"

        ```python
        from ultralytics import settings

        # 查看所有设置
        print(settings)

        # 返回分析和崩溃报告设置
        value = settings["sync"]
        ```

    === "CLI"

        ```bash
        yolo settings
        ```

更多详情，请参见[查看设置](#查看设置)部分。
