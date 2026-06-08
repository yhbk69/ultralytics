---
comments: true
description: 探索 Ultralytics 帮助中心，涵盖指南、常见问题、CI 流程和政策，助力您的 YOLO 模型使用体验与贡献。
keywords: Ultralytics, YOLO, 帮助中心, 文档, 指南, 常见问题, 贡献, CI, MRE, CLA, 行为准则, 安全政策, 隐私政策
---

# 帮助

欢迎来到 Ultralytics 帮助页面。本页面汇集了实用指南、政策与常见问题解答，帮助您更好地使用 Ultralytics YOLO 模型与代码仓库。

- [常见问题解答 (FAQ)](FAQ.md)：查找 Ultralytics YOLO 用户与贡献者社区遇到的常见问题的答案。
- [贡献指南](contributing.md)：了解贡献的相关流程，包括如何提交拉取请求、报告错误等。
- [持续集成 (CI) 指南](CI.md)：深入了解我们所采用的 CI 流程，附有每个 Ultralytics 仓库的状态报告。
- [贡献者许可协议 (CLA)](CLA.md)：查阅 CLA，了解为 Ultralytics 项目做贡献所涉及的权利与责任。
- [最小可复现示例 (MRE) 指南](minimum-reproducible-example.md)：了解创建 MRE 的流程，这对于及时有效地解决错误报告至关重要。
- [行为准则](code-of-conduct.md)：我们的社区准则旨在为所有协作者营造一个相互尊重、开放包容的氛围。
- [环境、健康与安全 (EHS) 政策](environmental-health-safety.md)：深入了解我们对可持续发展及所有利益相关方福祉的承诺。
- [安全政策](security.md)：熟悉我们的安全协议以及报告漏洞的流程。
- [隐私政策](privacy.md)：阅读我们的隐私政策，了解我们如何在所有服务与运营中保护您的数据并尊重您的隐私。

我们鼓励您查阅这些资源，以获得顺畅高效的使用体验。如需额外支持，请通过 [GitHub Issues](https://github.com/ultralytics/ultralytics/issues) 或 [Ultralytics 社区](https://community.ultralytics.com/) 联系我们。

## 常见问题

### 什么是 Ultralytics YOLO，它对我的[机器学习](https://www.ultralytics.com/glossary/machine-learning-ml)项目有什么帮助？

Ultralytics YOLO（You Only Look Once）是一种先进的实时[目标检测](https://www.ultralytics.com/glossary/object-detection)模型。其最新版本 YOLO26 提供更快、更轻量、端到端免 NMS 的推理能力，针对边缘和低功耗设备进行了优化，非常适合从实时视频分析到高级机器学习研究等广泛应用场景。YOLO 在图像和视频中检测物体的高效性使其成为企业和研究人员在项目中集成强大[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)能力的首选方案。

有关 YOLO26 的更多详情，请访问 [YOLO26 文档](../models/yolo26.md)。

### 如何为 Ultralytics YOLO 仓库做贡献？

为 Ultralytics YOLO 仓库做贡献非常简单。首先请查阅[贡献指南](contributing.md)，了解提交拉取请求、报告错误等的相关流程。您还需要签署[贡献者许可协议 (CLA)](CLA.md)，以确保您的贡献获得法律认可。若要有效地报告错误，请参考[最小可复现示例 (MRE) 指南](minimum-reproducible-example.md)。

### 为什么应该使用 Ultralytics 平台进行机器学习项目？

Ultralytics 平台为管理您的机器学习项目提供了无缝、无代码的解决方案。它让您能够轻松地生成、训练和部署像 YOLO26 这样的 AI 模型。独特的特性包括云端训练、实时跟踪和直观的数据集管理。Ultralytics 平台简化了从数据处理到[模型部署](https://www.ultralytics.com/glossary/model-deployment)的整个工作流程，是初学者和高级用户不可或缺的工具。

要开始使用，请访问 [Ultralytics 平台快速入门](../platform/quickstart.md)。

### Ultralytics 中的持续集成 (CI) 是什么，它如何确保高质量代码？

Ultralytics 中的持续集成 (CI) 涉及一系列自动化流程，用于确保代码库的完整性和质量。我们的 CI 设置包括 Docker 部署、断链检查、[CodeQL 分析](https://github.com/github/codeql)和 PyPI 发布。这些流程通过对新提交的代码自动运行测试和检查，有助于维护仓库的稳定性和安全性。

更多信息请参阅[持续集成 (CI) 指南](CI.md)。

### Ultralytics 如何处理[数据隐私](https://www.ultralytics.com/glossary/data-privacy)？

Ultralytics 非常重视数据隐私。我们的[隐私政策](privacy.md)概述了我们如何收集和使用匿名化数据来改进 YOLO 包，同时优先考虑用户隐私和控制权。我们遵守严格的数据保护法规，确保您的信息安全无虞。

更多信息请查阅我们的[隐私政策](privacy.md)。
