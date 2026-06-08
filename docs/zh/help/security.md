---
description: 了解 Ultralytics 用于保护用户数据和系统的安全措施与工具。了解我们如何通过 Snyk、CodeQL、Dependabot 等工具应对漏洞。
keywords: Ultralytics 安全策略, Snyk 扫描, CodeQL 扫描, Dependabot 警报, 密钥扫描, 漏洞报告, GitHub 安全, 开源安全
---

# Ultralytics 安全策略

在 [Ultralytics](https://www.ultralytics.com/)，用户数据和系统的安全至关重要。为确保我们[开源项目](https://github.com/ultralytics)的安全，我们实施了多项措施来检测和防范安全漏洞。

## Snyk 扫描

我们使用 [Snyk](https://security.snyk.io/package/pip/ultralytics) 对 Ultralytics 仓库进行全面的安全扫描。Snyk 强大的扫描能力不仅限于依赖项检查，还会检查我们的代码和 Dockerfile 中的各种漏洞。通过主动识别和解决这些问题，我们为用户提供了更高水平的安全性和可靠性。

[![ultralytics](https://img.shields.io/badge/Snyk_security-monitored-8A2BE2)](https://security.snyk.io/package/pip/ultralytics)

## GitHub CodeQL 扫描

我们的安全策略包括 GitHub 的 [CodeQL](https://docs.github.com/en/code-security/code-scanning/introduction-to-code-scanning/about-code-scanning-with-codeql) 扫描。CodeQL 深入分析我们的代码库，通过分析代码的语义结构来识别 SQL 注入和 XSS 等复杂漏洞。这种高级分析可确保尽早发现并解决潜在的安全风险。

[![CodeQL](https://github.com/ultralytics/ultralytics/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/ultralytics/ultralytics/actions/workflows/github-code-scanning/codeql)

## GitHub Dependabot 警报

[Dependabot](https://docs.github.com/en/code-security/dependabot) 已集成到我们的工作流中，用于监控依赖项中的已知漏洞。当某个依赖项中发现漏洞时，Dependabot 会向我们发出警报，以便迅速采取补救措施。

## GitHub 密钥扫描警报

我们使用 GitHub [密钥扫描](https://docs.github.com/en/code-security/secret-scanning/managing-alerts-from-secret-scanning) 警报来检测意外推送到仓库中的敏感数据，例如凭据和私钥。这种早期检测机制有助于防止潜在的安全漏洞和数据泄露。

## 私有漏洞报告

我们启用了私有漏洞报告功能，允许用户谨慎地报告潜在的安全问题。这种方式有助于负责任的披露，确保漏洞得到安全高效的处理。

如果您怀疑或发现我们任何仓库中存在安全漏洞，请立即告知我们。您可以通过我们的[联系表单](https://www.ultralytics.com/contact)或发送邮件至 [security@ultralytics.com](mailto:security@ultralytics.com) 直接联系我们。我们的安全团队将尽快调查并回复。

感谢您帮助我们确保所有 Ultralytics 开源项目对每个人都安全可靠。

## 常见问题

### Ultralytics 采取了哪些安全措施来保护用户数据？

Ultralytics 采用全面的安全策略来保护用户数据和系统。主要措施包括：

- **Snyk 扫描**：进行安全扫描以检测代码和 Dockerfile 中的漏洞。
- **GitHub CodeQL**：分析代码语义以检测 SQL 注入等复杂漏洞。
- **Dependabot 警报**：监控依赖项中的已知漏洞并发送警报以便迅速修复。
- **密钥扫描**：检测代码仓库中的凭据或私钥等敏感数据，防止数据泄露。
- **私有漏洞报告**：为用户提供安全渠道，以便谨慎报告潜在的安全问题。

这些工具确保主动识别和解决安全问题，增强整体系统安全性。更多详情请参阅上述章节，或联系安全团队咨询任何问题。

### Ultralytics 如何使用 Snyk 进行安全扫描？

Ultralytics 使用 [Snyk](https://security.snyk.io/package/pip/ultralytics) 对其仓库进行全面的安全扫描。Snyk 不仅限于基本的依赖项检查，还会检查代码和 Dockerfile 中的各种漏洞。通过主动识别和解决潜在的安全问题，Snyk 有助于确保 Ultralytics 的开源项目保持安全可靠。

要查看 Snyk 徽章并了解其部署详情，请参阅 [Snyk 扫描章节](#snyk-扫描)。

### 什么是 CodeQL？它如何增强 Ultralytics 的安全性？

[CodeQL](https://docs.github.com/en/code-security/code-scanning/introduction-to-code-scanning/about-code-scanning-with-codeql) 是一种通过 GitHub 集成到 Ultralytics 工作流中的安全分析工具。它深入分析代码库，识别 SQL 注入和跨站脚本（XSS）等复杂漏洞。CodeQL 分析代码的语义结构，提供高级别的安全保障，确保尽早发现并缓解潜在风险。

有关 CodeQL 使用方式的更多信息，请访问 [GitHub CodeQL 扫描章节](#github-codeql-扫描)。

### Dependabot 如何帮助维护 Ultralytics 的代码安全？

[Dependabot](https://docs.github.com/en/code-security/dependabot) 是一种自动化工具，用于监控和管理依赖项中的已知漏洞。当 Dependabot 在 Ultralytics 项目依赖项中检测到漏洞时，它会发送警报，使团队能够快速解决和缓解问题。这确保了依赖项保持安全和最新，最大限度地降低潜在的安全风险。

更多详情请参阅 [GitHub Dependabot 警报章节](#github-dependabot-警报)。

### Ultralytics 如何处理私有漏洞报告？

Ultralytics 鼓励用户通过私有渠道报告潜在的安全问题。用户可以通过[联系表单](https://www.ultralytics.com/contact)或发送邮件至 [security@ultralytics.com](mailto:security@ultralytics.com) 谨慎地报告漏洞。这确保了负责任的披露，使安全团队能够安全高效地调查和处理漏洞。

有关私有漏洞报告的更多信息，请参阅[私有漏洞报告章节](#私有漏洞报告)。
