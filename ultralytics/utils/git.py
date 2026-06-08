# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from functools import cached_property
from pathlib import Path


class GitRepo:
    """表示本地 Git 仓库，暴露分支、提交和远程元数据。

    该类通过从给定路径向上搜索 .git 条目来发现仓库根目录，解析实际的 .git 目录（包括工作树），
    并直接从磁盘文件读取 Git 元数据。它不调用 git 二进制文件，因此可在受限环境中工作。
    所有元数据属性都是延迟解析和缓存的；构造新实例以刷新状态。

    属性:
        root (Path | None): 包含 .git 条目的仓库根目录；如果不在仓库中则为 None。
        gitdir (Path | None): 解析后的 .git 目录路径；处理工作树；未解析则为 None。
        head (str | None): HEAD 的原始内容；分离 HEAD 时为 SHA，分支头时为 "ref: <refname>"。
        is_repo (bool): 提供的路径是否位于 Git 仓库内。
        branch (str | None): HEAD 指向分支时的当前分支名；分离 HEAD 或非仓库时为 None。
        commit (str | None): HEAD 的当前提交 SHA；无法确定时为 None。
        origin (str | None): 从 gitdir/config 读取的 "origin" 远程 URL；未设置或不可用时为 None。

    示例:
        从当前工作目录初始化并读取元数据
        >>> from pathlib import Path
        >>> repo = GitRepo(Path.cwd())
        >>> repo.is_repo
        True
        >>> repo.branch, repo.commit[:7], repo.origin
        ('main', '1a2b3c4', 'https://example.com/owner/repo.git')

    注意:
        - 通过读取文件解析元数据：HEAD、packed-refs 和 config；不使用子进程调用。
        - 首次访问时使用 cached_property 缓存属性；重新创建对象以反映仓库变更。
    """

    def __init__(self, path: Path = Path(__file__).resolve()):
        """通过从起始路径发现仓库根目录来初始化 Git 仓库上下文。

        参数:
            path (Path, 可选): 用作定位仓库根目录起始点的文件或目录路径。
        """
        self.root = self._find_root(path)
        self.gitdir = self._gitdir(self.root) if self.root else None

    @staticmethod
    def _find_root(p: Path) -> Path | None:
        """返回仓库根目录或 None。"""
        return next((d for d in [p, *list(p.parents)] if (d / ".git").exists()), None)

    @staticmethod
    def _gitdir(root: Path) -> Path | None:
        """解析实际的 .git 目录（处理工作树）。"""
        g = root / ".git"
        if g.is_dir():
            return g
        if g.is_file():
            t = g.read_text(errors="ignore").strip()
            if t.startswith("gitdir:"):
                return (root / t.split(":", 1)[1].strip()).resolve()
        return None

    @staticmethod
    def _read(p: Path | None) -> str | None:
        """如果文件存在则读取并去除首尾空白。"""
        return p.read_text(errors="ignore").strip() if p and p.exists() else None

    @cached_property
    def head(self) -> str | None:
        """HEAD 文件内容。"""
        return self._read(self.gitdir / "HEAD" if self.gitdir else None)

    def _ref_commit(self, ref: str) -> str | None:
        """获取引用对应的提交（处理 packed-refs）。"""
        rf = self.gitdir / ref
        if s := self._read(rf):
            return s
        pf = self.gitdir / "packed-refs"
        b = pf.read_bytes().splitlines() if pf.exists() else []
        tgt = ref.encode()
        for line in b:
            if line[:1] in (b"#", b"^") or b" " not in line:
                continue
            sha, name = line.split(b" ", 1)
            if name.strip() == tgt:
                return sha.decode()
        return None

    @property
    def is_repo(self) -> bool:
        """如果在 git 仓库内则为 True。"""
        return self.gitdir is not None

    @cached_property
    def branch(self) -> str | None:
        """当前分支名或 None。"""
        if not self.is_repo or not self.head or not self.head.startswith("ref: "):
            return None
        ref = self.head[5:].strip()
        return ref[len("refs/heads/") :] if ref.startswith("refs/heads/") else ref

    @cached_property
    def commit(self) -> str | None:
        """当前提交 SHA 或 None。"""
        if not self.is_repo or not self.head:
            return None
        return self._ref_commit(self.head[5:].strip()) if self.head.startswith("ref: ") else self.head

    @cached_property
    def origin(self) -> str | None:
        """Origin URL 或 None。"""
        if not self.is_repo:
            return None
        cfg = self.gitdir / "config"
        remote, url = None, None
        for s in (self._read(cfg) or "").splitlines():
            t = s.strip()
            if t.startswith("[") and t.endswith("]"):
                remote = t.lower()
            elif t.lower().startswith("url =") and remote == '[remote "origin"]':
                url = t.split("=", 1)[1].strip()
                break
        return url


if __name__ == "__main__":
    import time

    g = GitRepo()
    if g.is_repo:
        t0 = time.perf_counter()
        print(f"repo={g.root}\nbranch={g.branch}\ncommit={g.commit}\norigin={g.origin}")
        dt = (time.perf_counter() - t0) * 1000
        print(f"\n⏱️ Profiling: total {dt:.3f} ms")
