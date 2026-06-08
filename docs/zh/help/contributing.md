---
comments: true
description: 了解如何为 Ultralytics YOLO 开源仓库做出贡献。遵循有关拉取请求、行为准则和错误报告的指南。
keywords: Ultralytics, YOLO, 开源, 贡献, 拉取请求, 行为准则, 错误报告, GitHub, CLA, Google风格文档字符串, AGPL-3.0
---

# 为 Ultralytics 开源项目做贡献

欢迎！我们很高兴您考虑为 [Ultralytics](https://www.ultralytics.com/) [开源](https://github.com/ultralytics) 项目做出贡献。您的参与不仅有助于提高我们仓库的质量，还能造福整个 [计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv) 社区。本指南提供了清晰的指导方针和最佳实践，帮助您快速上手。

[![Ultralytics 开源贡献者](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/yMR7BgwHQ3g"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何为 Ultralytics 仓库做贡献 | Ultralytics 模型、数据集和文档 🚀
</p>

## 🤝 行为准则

为确保为每个人营造一个热情和包容的环境，所有贡献者都必须遵守我们的 [行为准则](https://docs.ultralytics.com/help/code-of-conduct)。**尊重**、**友善** 和 **专业精神** 是我们社区的核心价值观。

## 🚀 通过拉取请求做贡献

我们非常感谢您通过 [拉取请求 (PR)](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) 所做的贡献。为了让审核过程尽可能顺畅，请按照以下步骤操作：

1. **[Fork 仓库](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)：** 首先将相关的 Ultralytics 仓库（例如 [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)）Fork 到您的 GitHub 账户中。
2. **[创建分支](https://docs.github.com/en/desktop/making-changes-in-a-branch/managing-branches-in-github-desktop)：** 在您 Fork 的仓库中创建一个新分支，并使用清晰、描述性的名称来反映您的更改（例如 `fix-issue-123`、`add-feature-xyz`）。
3. **进行更改：** 实现您的改进或修复。确保您的代码符合项目的风格指南，并且不会引入新的错误或警告。
4. **测试您的更改：** 在提交之前，请在本地测试您的更改，确认它们按预期工作，并且不会导致 [回归问题](https://en.wikipedia.org/wiki/Software_regression)。如果您引入了新功能，请添加相应的测试。
5. **[提交更改](https://docs.github.com/en/desktop/making-changes-in-a-branch/committing-and-reviewing-changes-to-your-project-in-github-desktop)：** 使用简洁且具有描述性的提交信息来提交您的更改。如果您的更改解决了某个特定的 Issue，请在提交信息中包含 Issue 编号（例如 `Fix #123: 修正了计算错误。`）。
6. **[创建拉取请求](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)：** 从您的分支向原始 Ultralytics 仓库的 `main` 分支提交一个拉取请求。提供一个清晰的标题和详细的描述，说明您更改的目的和范围。

### 📝 CLA 签署

在我们能够合并您的拉取请求之前，您必须签署我们的 [贡献者许可协议 (CLA)](https://docs.ultralytics.com/help/CLA)。这份法律协议确保您的贡献得到适当授权，允许项目继续根据 [AGPL-3.0 许可证](https://www.ultralytics.com/legal/agpl-3-0-software-license) 发布。

提交拉取请求后，CLA 机器人将指导您完成签署过程。若要签署 CLA，只需在您的 PR 中添加一条评论，内容如下：

```
I have read the CLA Document and I sign the CLA
```

### ✍️ Google 风格文档字符串

在添加新的函数或类时，请包含 [Google 风格文档字符串](https://google.github.io/styleguide/pyguide.html)，以提供清晰、标准化的文档。请始终将输入和输出的 `types` 用括号括起来（例如 `(bool)`、`(np.ndarray)`）。

!!! example "文档字符串示例"

    === "Google 风格"

        此示例说明了标准的 Google 风格文档字符串格式。请注意它如何清晰地分隔函数描述、参数、返回值和示例，以实现最大的可读性。

        ```python
        def example_function(arg1, arg2=4):
            """演示 Google 风格文档字符串的示例函数。

            Args:
                arg1 (int): 第一个参数。
                arg2 (int): 第二个参数。

            Returns:
                (bool): 如果参数相等则返回 True，否则返回 False。

            Examples:
                >>> example_function(4, 4)  # True
                >>> example_function(1, 2)  # False
            """
            return arg1 == arg2
        ```

    === "Google 风格（命名返回值）"

        此示例演示了如何记录命名的返回值变量。使用命名返回值可以使您的代码更加自文档化且更易于理解，特别是对于复杂函数而言。

        ```python
        def example_function(arg1, arg2=4):
            """演示 Google 风格文档字符串的示例函数。

            Args:
                arg1 (int): 第一个参数。
                arg2 (int): 第二个参数。

            Returns:
                equals (bool): 如果参数相等则返回 True，否则返回 False。

            Examples:
                >>> example_function(4, 4)  # True
            """
            equals = arg1 == arg2
            return equals
        ```

    === "Google 风格（多返回值）"

        此示例展示了如何记录返回多个值的函数。为了清晰起见，每个返回值都应单独记录其类型和描述。

        ```python
        def example_function(arg1, arg2=4):
            """演示 Google 风格文档字符串的示例函数。

            Args:
                arg1 (int): 第一个参数。
                arg2 (int): 第二个参数。

            Returns:
                equals (bool): 如果参数相等则返回 True，否则返回 False。
                added (int): 两个输入参数的和。

            Examples:
                >>> equals, added = example_function(2, 2)  # True, 4
            """
            equals = arg1 == arg2
            added = arg1 + arg2
            return equals, added
        ```

        注意：即使 Python 以元组的形式返回多个值（例如 `return masks, scores`），为了清晰性和更好的工具集成，应始终单独记录每个值。在记录返回多个值的函数时：

        ✅ 正确 - 单独记录每个返回值：
        ```
        Returns:
           (np.ndarray): 预测的掩码，形状为 HxWxN。
           (list): 每个实例的置信度分数。
        ```

        ❌ 错误 - 不要以包含嵌套元素的元组形式记录：
        ```
        Returns:
           (tuple): 包含以下内容的元组：
               - (np.ndarray): 预测的掩码，形状为 HxWxN。
               - (list): 每个实例的置信度分数。
        ```

    === "Google 风格（带类型提示）"

        此示例将 Google 风格文档字符串与 Python 类型提示结合在一起。使用类型提示时，您可以省略文档字符串参数部分中的类型信息，因为它已经在函数签名中指定了。

        ```python
        def example_function(arg1: int, arg2: int = 4) -> bool:
            """演示 Google 风格文档字符串的示例函数。

            Args:
                arg1: 第一个参数。
                arg2: 第二个参数。

            Returns:
                如果参数相等则返回 True，否则返回 False。

            Examples:
                >>> example_function(1, 1)  # True
            """
            return arg1 == arg2
        ```

    === "单行文档字符串"

        对于较小或较简单的函数，单行文档字符串可能就足够了。这些应该是简明但完整的句子，以大写字母开头并以句号结尾。

        ```python
        def example_small_function(arg1: int, arg2: int = 4) -> bool:
            """带有单行文档字符串的示例函数。"""
            return arg1 == arg2
        ```

### ✅ GitHub Actions CI 测试

所有拉取请求在合并之前都必须通过 [GitHub Actions](https://github.com/features/actions) [持续集成](https://docs.ultralytics.com/help/CI) (CI) 测试。这些测试包括代码检查、单元测试和其他检查，以确保您的更改符合项目的质量标准。请查看 CI 输出并解决出现的任何问题。

## ✨ 代码贡献最佳实践

在为 Ultralytics 项目贡献代码时，请牢记以下最佳实践：

- **避免代码重复：** 尽可能复用现有代码，并尽量减少不必要的参数。
- **进行更小、更集中的更改：** 专注于有针对性的修改，而不是大规模的变更。
- **尽可能简化：** 寻找机会简化代码或删除不必要的部分。
- **考虑兼容性：** 在进行更改之前，考虑它们是否会破坏使用 Ultralytics 的现有代码。
- **使用一致的格式：** 像 [Ruff Formatter](https://github.com/astral-sh/ruff) 这样的工具可以帮助保持风格的一致性。
- **添加适当的测试：** 为新功能添加 [测试](https://docs.ultralytics.com/guides/model-testing)，以确保它们按预期工作。

## 👀 审核拉取请求

审核拉取请求是另一种有价值的贡献方式。在审核 PR 时：

- **检查单元测试：** 确认 PR 包含针对新功能或更改的测试。
- **审核文档更新：** 确保 [文档](https://docs.ultralytics.com/) 已更新以反映更改。
- **评估性能影响：** 考虑更改可能对 [性能](https://docs.ultralytics.com/guides/yolo-performance-metrics) 产生的影响。
- **验证 CI 测试：** 确认所有 [持续集成测试](https://docs.ultralytics.com/help/CI) 均已通过。
- **提供建设性反馈：** 就任何问题或疑虑提供具体、清晰的反馈。
- **认可付出的努力：** 认可作者的付出，以保持积极的协作氛围。

## 🐞 报告错误

我们高度重视错误报告，因为它们有助于我们提高项目的质量和可靠性。通过 [GitHub Issues](https://github.com/ultralytics/ultralytics/issues) 报告错误时：

- **检查现有的 Issues：** 首先搜索一下，看看该错误是否已经被报告过。
- **提供一个 [最小可复现示例](https://docs.ultralytics.com/help/minimum-reproducible-example/)：** 创建一个简短、自包含的代码片段，能够持续复现该问题。这对于高效调试至关重要。
- **描述环境：** 说明您的操作系统、Python 版本、相关库的版本（例如 [`torch`](https://pytorch.org/)、[`ultralytics`](https://github.com/ultralytics/ultralytics)）以及硬件（[CPU](https://en.wikipedia.org/wiki/Central_processing_unit)/[GPU](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit)）。
- **解释预期行为与实际行为：** 清楚地说明您期望发生什么以及实际发生了什么。包括任何错误信息或回溯信息。

## 📜 许可证

Ultralytics 对其仓库使用 [GNU Affero 通用公共许可证 v3.0 (AGPL-3.0)](https://www.ultralytics.com/legal/agpl-3-0-software-license)。此许可证促进了软件开发中的 [开放性](https://en.wikipedia.org/wiki/Openness)、[透明度](https://www.ultralytics.com/glossary/transparency-in-ai) 和 [协作改进](https://en.wikipedia.org/wiki/Collaborative_software)。它确保所有用户都有使用、修改和分享软件的自由，从而培养一个强大的协作与创新社区。

我们鼓励所有贡献者熟悉 [AGPL-3.0 许可证](https://opensource.org/license/agpl-3-0) 的条款，以便有效且合乎道德地为 Ultralytics 开源社区做出贡献。

## 🌍 在 AGPL-3.0 下开源您的 YOLO 项目

在您的项目中使用 Ultralytics YOLO 模型或代码？[AGPL-3.0 许可证](https://opensource.org/license/agpl-3-0) 要求您的整个衍生作品也必须在 AGPL-3.0 下开源。这确保了基于开源基础构建的修改和更大项目保持开放。

### 为什么遵守 AGPL-3.0 很重要

- **保持软件开放：** 确保改进和衍生作品造福社区。
- **法律要求：** 使用 AGPL-3.0 许可的代码会将您的项目绑定到其条款。
- **促进协作：** 鼓励分享和透明度。

如果您不想将您的项目开源，请考虑获取 [企业许可证](https://www.ultralytics.com/license)。

### 如何遵守 AGPL-3.0

遵守意味着在 AGPL-3.0 许可证下公开发布您项目的 **完整的对应源代码**。

1. **选择您的起点：**
    - **Fork Ultralytics YOLO：** 如果您是紧密地在其基础上构建，请直接 Fork [Ultralytics YOLO 仓库](https://github.com/ultralytics/ultralytics)。
    - **使用 Ultralytics 模板：** 从 [Ultralytics 模板仓库](https://github.com/ultralytics/template) 开始，以获取一个集成了 YOLO 的干净、模块化的设置。

2. **为您的项目添加许可证：**
    - 添加一个包含 [AGPL-3.0 许可证](https://opensource.org/license/agpl-3-0) 全文的 `LICENSE` 文件。
    - 在每个源文件的顶部添加一个声明，指明许可证。

3. **发布您的源代码：**
    - 公开您 **整个项目的源代码**（例如在 GitHub 上）。这包括：
        - 包含 YOLO 模型或代码的完整大型应用程序或系统。
        - 对原始 Ultralytics YOLO 代码所做的任何修改。
        - 用于训练、验证和推理的脚本。
        - 如果进行了修改或微调，则需要包含 [模型权重](https://www.ultralytics.com/glossary/model-weights)。
        - [配置文件](https://docs.ultralytics.com/usage/cfg)、环境设置（`requirements.txt`、[`Dockerfiles`](https://docs.docker.com/reference/dockerfile/)）。
        - 如果属于 [Web 应用程序](https://en.wikipedia.org/wiki/Web_application) 的一部分，则需要包含后端和前端代码。
        - 您修改过的任何 [第三方库](<https://en.wikipedia.org/wiki/Library_(computing)#Third-party>)。
        - 如果运行/重新训练需要且可重新分发，则需要提供 [训练数据](https://www.ultralytics.com/glossary/training-data)。

4. **清晰记录：**
    - 更新您的 `README.md`，声明该项目根据 AGPL-3.0 获得许可。
    - 包含关于如何从源代码设置、构建和运行您的项目的清晰说明。
    - 适当地注明 Ultralytics YOLO，并链接回 [原始仓库](https://github.com/ultralytics/ultralytics)。示例：
        ```markdown
        本项目使用了根据 AGPL-3.0 许可的 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 代码。
        ```

### 示例仓库结构

请参考 [Ultralytics 模板仓库](https://github.com/ultralytics/template) 以获取实际的示例结构：

```
my-yolo-project/
│
├── LICENSE               # AGPL-3.0 许可证全文
├── README.md             # 项目描述、设置、用法、许可证信息和署名
├── pyproject.toml        # 依赖项（或 requirements.txt）
├── scripts/              # 训练/推理脚本
│   └── train.py
├── src/                  # 您项目的源代码
│   ├── __init__.py
│   ├── data_loader.py
│   └── model_wrapper.py  # 与 YOLO 交互的代码
├── tests/                # 单元/集成测试
├── configs/              # YAML/JSON 配置文件
├── docker/               # Dockerfiles（如果使用）
│   └── Dockerfile
└── .github/              # GitHub 专用文件（例如 CI 的工作流）
    └── workflows/
        └── ci.yml
```

通过遵循这些指南，您可以确保遵守 AGPL-3.0，从而支持使像 Ultralytics YOLO 这样强大工具成为可能的开源生态系统。

## 结论

感谢您有兴趣为 [Ultralytics](https://www.ultralytics.com/) [开源](https://github.com/ultralytics) YOLO 项目做出贡献。您的参与对于塑造我们软件的未来以及构建一个充满活力的创新与协作社区至关重要。无论您是增强代码、报告错误还是提出新功能建议，您的贡献都无比宝贵。

我们很高兴看到您的想法变为现实，并感谢您为推动 [目标检测](https://www.ultralytics.com/glossary/object-detection) 技术发展所做的承诺。让我们一起在这段激动人心的开源旅程中继续成长与创新。

## 常见问题

### 我为什么要为 Ultralytics YOLO 开源仓库做贡献？

为 Ultralytics YOLO 开源仓库做贡献可以改进软件，使其对整个社区来说更健壮、功能更丰富。贡献可以包括代码增强、错误修复、文档改进以及新功能的实现。此外，贡献还可以让您与该领域的其他熟练开发者和专家合作，从而提升您自己的技能和声誉。有关如何开始的详细信息，请参阅 [通过拉取请求做贡献](#通过拉取请求做贡献) 部分。

### 我如何为 Ultralytics YOLO 签署贡献者许可协议 (CLA)？

要签署贡献者许可协议 (CLA)，请在提交拉取请求后按照 CLA 机器人提供的说明进行操作。此过程确保您的贡献在 AGPL-3.0 许可证下得到适当授权，从而维护开源项目的法律完整性。在您的拉取请求中添加一条评论，内容如下：

```
I have read the CLA Document and I sign the CLA
```

有关更多信息，请参阅 [CLA 签署](#cla-签署) 部分。

### 什么是 Google 风格文档字符串，为什么 Ultralytics YOLO 的贡献要求使用它们？

Google 风格文档字符串为函数和类提供了清晰、简洁的文档，从而提高了代码的可读性和可维护性。这些文档字符串以特定的格式规则概述了函数的目的、参数和返回值。在为 Ultralytics YOLO 做贡献时，遵循 Google 风格文档字符串可以确保您添加的内容有良好的文档记录且易于理解。有关示例和指南，请访问 [Google 风格文档字符串](#google-风格文档字符串) 部分。

### 我如何确保我的更改通过 GitHub Actions CI 测试？

在您的拉取请求可以被合并之前，它必须通过所有 GitHub Actions 持续集成 (CI) 测试。这些测试包括代码检查、单元测试和其他检查，以确保代码符合项目的质量标准。请查看 CI 输出并修复任何问题。有关 CI 流程和故障排除提示的详细信息，请参阅 [GitHub Actions CI 测试](#github-actions-ci-测试) 部分。

### 如何在 Ultralytics YOLO 仓库中报告错误？

要报告错误，请在您的错误报告中提供一个清晰简洁的 [最小可复现示例](https://docs.ultralytics.com/help/minimum-reproducible-example/)。这有助于开发者快速识别和修复问题。确保您的示例是最小的，但足以复现问题。有关报告错误的更详细步骤，请参阅 [报告错误](#报告错误) 部分。

### 如果我在自己的项目中使用 Ultralytics YOLO，AGPL-3.0 许可证意味着什么？

如果您在项目中使用 Ultralytics YOLO 代码或模型（根据 AGPL-3.0 许可），AGPL-3.0 许可证要求您的整个项目（衍生作品）也必须在 AGPL-3.0 下获得许可，并且其完整的源代码必须公开发布。这确保了软件的开放性在其所有衍生作品中得以保留。如果您不能满足这些要求，则需要获取 [企业许可证](https://www.ultralytics.com/license)。有关详细信息，请参阅 [开源您的项目](#在-agpl-30-下开源您的-yolo-项目) 部分。
