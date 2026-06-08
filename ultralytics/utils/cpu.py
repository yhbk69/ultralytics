# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import platform
import re
import subprocess
import sys
from pathlib import Path


class CPUInfo:
    """提供跨平台的 CPU 品牌和型号信息。

    查询特定平台的源以获取人类可读的 CPU 描述，并将其规范化为在 macOS、Linux 和 Windows 上
    一致的呈现形式。如果特定平台的探测失败，则使用通用的平台标识符，以确保始终返回稳定的字符串。

    方法:
        name: 使用特定平台的源返回规范化的 CPU 名称，具有健壮的回退机制。
        _clean: 规范化并美化常见的厂商品牌字符串和频率模式。
        __str__: 在字符串上下文中返回规范化的 CPU 名称。

    示例:
        >>> CPUInfo.name()
        'Apple M4 Pro'
        >>> str(CPUInfo())
        'Intel Core i7-9750H 2.60GHz'
    """

    @staticmethod
    def name() -> str:
        """从特定平台的源返回规范化的 CPU 型号字符串。"""
        try:
            if sys.platform == "darwin":
                # 查询 macOS sysctl 获取 CPU 品牌字符串
                s = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
                ).stdout.strip()
                if s:
                    return CPUInfo._clean(s)
            elif sys.platform.startswith("linux"):
                # 解析 /proc/cpuinfo 获取第一个 "model name" 条目
                p = Path("/proc/cpuinfo")
                if p.exists():
                    for line in p.read_text(errors="ignore").splitlines():
                        if "model name" in line:
                            return CPUInfo._clean(line.split(":", 1)[1])
            elif sys.platform.startswith("win"):
                try:
                    import winreg as wr

                    with wr.OpenKey(wr.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                        val, _ = wr.QueryValueEx(k, "ProcessorNameString")
                        if val:
                            return CPUInfo._clean(val)
                except Exception:
                    # Windows 注册表访问失败时回退到通用平台回退机制
                    pass
            # 通用平台回退机制
            s = platform.processor() or getattr(platform.uname(), "processor", "") or platform.machine()
            return CPUInfo._clean(s or "Unknown CPU")
        except Exception:
            # 确保即使发生意外故障也始终返回字符串
            s = platform.processor() or platform.machine() or ""
            return CPUInfo._clean(s or "Unknown CPU")

    @staticmethod
    def _clean(s: str) -> str:
        """规范化并美化原始 CPU 描述字符串。"""
        s = re.sub(r"\s+", " ", s.strip())
        s = s.replace("(TM)", "").replace("(tm)", "").replace("(R)", "").replace("(r)", "").strip()
        if m := re.search(r"(Intel.*?i\d[\w-]*) CPU @ ([\d.]+GHz)", s, re.I):
            return f"{m.group(1)} {m.group(2)}"
        if m := re.search(r"(AMD.*?Ryzen.*?[\w-]*) CPU @ ([\d.]+GHz)", s, re.I):
            return f"{m.group(1)} {m.group(2)}"
        return s

    def __str__(self) -> str:
        """返回规范化的 CPU 名称。"""
        return self.name()


if __name__ == "__main__":
    print(CPUInfo.name())
