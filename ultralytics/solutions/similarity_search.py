# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import LOGGER, TORCH_VERSION
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.torch_utils import TORCH_2_4, select_device

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 在某些系统上避免 OpenMP 冲突


class VisualAISearch:
    """基于 OpenCLIP 生成高质量图像和文本嵌入，并利用 FAISS 进行快速相似性检索的语义图像搜索系统。

    此类在共享语义空间中对齐图像和文本嵌入，使用户能够使用自然语言查询在高精度和高速度下搜索大量图像集合。

    属性:
        data (str): 包含图像的目录。
        device (str): 计算设备，例如 'cpu' 或 'cuda'。
        faiss_index (str): FAISS 索引文件的路径。
        data_path_npy (str): 存储图像路径的 numpy 文件路径。
        data_dir (Path): 数据目录的 Path 对象。
        model: 已加载的 CLIP 模型。
        index: 用于相似性搜索的 FAISS 索引。
        image_paths (list[str]): 图像文件路径列表。

    方法:
        extract_image_feature: 从图像中提取 CLIP 嵌入。
        extract_text_feature: 从文本中提取 CLIP 嵌入。
        load_or_build_index: 加载已有 FAISS 索引或构建新索引。
        search: 执行语义搜索以查找相似图像。

    示例:
        初始化并搜索图像
        >>> searcher = VisualAISearch(data="path/to/images", device="cuda")
        >>> results = searcher.search("a cat sitting on a chair", k=10)
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 VisualAISearch 类，加载 FAISS 索引和 CLIP 模型。"""
        assert TORCH_2_4, f"VisualAISearch 需要 torch>=2.4（发现 torch=={TORCH_VERSION}）"
        from ultralytics.nn.text_model import build_text_model

        check_requirements("faiss-cpu")

        self.faiss = __import__("faiss")
        self.faiss_index = "faiss.index"
        self.data_path_npy = "paths.npy"
        self.data_dir = Path(kwargs.get("data", "images"))
        self.device = select_device(kwargs.get("device", "cpu"))

        if not self.data_dir.exists():
            from ultralytics.utils import ASSETS_URL

            LOGGER.warning(f"未找到 {self.data_dir}。正在从 {ASSETS_URL}/images.zip 下载 images.zip")
            from ultralytics.utils.downloads import safe_download

            safe_download(url=f"{ASSETS_URL}/images.zip", unzip=True, retry=3)
            self.data_dir = Path("images")

        self.model = build_text_model("clip:ViT-B/32", device=self.device)

        self.index = None
        self.image_paths = []

        self.load_or_build_index()

    def extract_image_feature(self, path: Path) -> np.ndarray:
        """从给定图像路径提取 CLIP 图像嵌入。"""
        return self.model.encode_image(Image.open(path)).detach().cpu().numpy()

    def extract_text_feature(self, text: str) -> np.ndarray:
        """从给定文本查询中提取 CLIP 文本嵌入。"""
        return self.model.encode_text(self.model.tokenize([text])).detach().cpu().numpy()

    def load_or_build_index(self) -> None:
        """加载已有 FAISS 索引或从图像特征构建新索引。

        检查磁盘上是否已存在 FAISS 索引和图像路径。如果找到则直接加载。否则，
        从数据目录中的所有图像提取特征构建新索引，对特征进行归一化，并保存索引和图像路径供后续使用。
        """
        # 检查 FAISS 索引和对应的图像路径是否已存在
        if Path(self.faiss_index).exists() and Path(self.data_path_npy).exists():
            LOGGER.info("正在加载已有 FAISS 索引...")
            self.index = self.faiss.read_index(self.faiss_index)  # 从磁盘加载 FAISS 索引
            self.image_paths = np.load(self.data_path_npy)  # 加载已保存的图像路径列表
            return  # 索引成功加载后退出函数

        # 如果索引不存在，从零开始构建
        LOGGER.info("正在从图像构建 FAISS 索引...")
        vectors = []  # 用于存储图像特征向量的列表

        # 遍历数据目录中的所有图像文件
        for file in self.data_dir.iterdir():
            # 跳过非有效图像格式的文件
            if file.suffix.lower().lstrip(".") not in IMG_FORMATS:
                continue
            try:
                # 提取图像的特征向量并添加到列表中
                vectors.append(self.extract_image_feature(file))
                self.image_paths.append(file.name)  # 存储对应的图像名称
            except Exception as e:
                LOGGER.warning(f"跳过 {file.name}: {e}")

        # 如果没有成功创建任何向量，则抛出错误
        if not vectors:
            raise RuntimeError("无法生成任何图像嵌入。")

        vectors = np.vstack(vectors).astype("float32")  # 将所有向量堆叠为 NumPy 数组并转换为 float32
        self.faiss.normalize_L2(vectors)  # 将向量归一化为单位长度以用于余弦相似度计算

        self.index = self.faiss.IndexFlatIP(vectors.shape[1])  # 使用内积创建新的 FAISS 索引
        self.index.add(vectors)  # 将归一化后的向量添加到 FAISS 索引
        self.faiss.write_index(self.index, self.faiss_index)  # 将新构建的 FAISS 索引保存到磁盘
        np.save(self.data_path_npy, np.array(self.image_paths))  # 将图像路径列表保存到磁盘

        LOGGER.info(f"已索引 {len(self.image_paths)} 张图像。")

    def search(self, query: str, k: int = 30, similarity_thresh: float = 0.1) -> list[str]:
        """返回与给定查询在语义上最相似的 top-k 张图像。

        参数:
            query (str): 用于搜索的自然语言文本查询。
            k (int, optional): 返回结果的最大数量。
            similarity_thresh (float, optional): 过滤结果的最小相似度阈值。

        返回:
            (list[str]): 按相似度分数排名的图像文件名列表。

        示例:
            搜索匹配查询的图像
            >>> searcher = VisualAISearch(data="images")
            >>> results = searcher.search("red car", k=5, similarity_thresh=0.2)
        """
        text_feat = self.extract_text_feature(query).astype("float32")
        self.faiss.normalize_L2(text_feat)

        D, index = self.index.search(text_feat, k)
        results = [
            (self.image_paths[i], float(D[0][idx])) for idx, i in enumerate(index[0]) if D[0][idx] >= similarity_thresh
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        LOGGER.info("\n排序结果:")
        for name, score in results:
            LOGGER.info(f"  - {name} | 相似度: {score:.4f}")

        return [r[0] for r in results]

    def __call__(self, query: str) -> list[str]:
        """搜索函数的直接调用接口。"""
        return self.search(query)


class SearchApp:
    """基于 Flask 的语义图像搜索 Web 界面，支持自然语言查询。

    此类提供简洁、响应式的前端界面，使用户能够输入自然语言查询并即时查看
    从索引数据库中检索到的最相关图像。

    属性:
        render_template: Flask 模板渲染函数。
        request: Flask 请求对象。
        searcher (VisualAISearch): VisualAISearch 类的实例。
        app (Flask): Flask 应用实例。

    方法:
        index: 处理用户查询并显示搜索结果。
        run: 启动 Flask Web 应用。

    示例:
        启动搜索应用
        >>> app = SearchApp(data="path/to/images", device="cuda")
        >>> app.run(debug=True)
    """

    def __init__(self, data: str = "images", device: str | None = None) -> None:
        """使用 VisualAISearch 后端初始化 SearchApp。

        参数:
            data (str, optional): 包含要索引和搜索的图像的目录路径。
            device (str, optional): 运行推理的设备（例如 'cpu'、'cuda'）。
        """
        check_requirements("flask>=3.0.1")
        from flask import Flask, render_template, request

        self.render_template = render_template
        self.request = request
        self.searcher = VisualAISearch(data=data, device=device)
        self.app = Flask(
            __name__,
            template_folder="templates",
            static_folder=Path(data).resolve(),  # 用于提供图像的绝对路径
            static_url_path="/images",  # 图像的 URL 前缀
        )
        self.app.add_url_rule("/", view_func=self.index, methods=["GET", "POST"])

    def index(self) -> str:
        """处理用户查询并在 Web 界面中显示搜索结果。"""
        results = []
        if self.request.method == "POST":
            query = self.request.form.get("query", "").strip()
            results = self.searcher(query)
        return self.render_template("similarity-search.html", results=results)

    def run(self, debug: bool = False) -> None:
        """启动 Flask Web 应用服务器。"""
        self.app.run(debug=debug)
