# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
本模块提供 Ultralytics YOLO 模型（包括目标检测、实例分割、图像分类、姿态估计和多目标跟踪）的超参数调优功能。

超参数调优是系统性地搜索能产生最佳模型性能的最优超参数集合的过程。
这在像 YOLO 这样的深度学习模型中尤其重要，因为超参数的微小变化可能导致模型准确性和效率的显著差异。

Examples:
    在 COCO8 上以 imgsz=640 和 epochs=10 对 YOLO26n 进行 300 次调优迭代。
    >>> from ultralytics import YOLO
    >>> model = YOLO("yolo26n.pt")
    >>> model.tune(data="coco8.yaml", epochs=10, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)
"""

from __future__ import annotations

import gc
import json
import random
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ultralytics.cfg import CFG_INT_KEYS, get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG, LOGGER, YAML, callbacks, colorstr, remove_colorstr
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.patches import torch_load
from ultralytics.utils.plotting import plot_tune_results


class Tuner:
    """YOLO 模型的超参数调优类。

    该类通过在给定迭代次数内根据搜索空间对超参数进行变异并重新训练模型来评估其性能，
    从而进化 YOLO 模型的超参数。支持本地 NDJSON 存储和用于多机超参数优化的分布式 MongoDB Atlas 协调。

    Attributes:
        space (dict[str, tuple]): 包含用于变异的边界和缩放因子的超参数搜索空间。
        tune_dir (Path): 保存进化日志和结果的目录。
        tune_file (Path): 保存进化日志的 NDJSON 文件路径。
        args (SimpleNamespace): 调优过程的配置参数。
        callbacks (dict): 调优期间执行的回调函数。
        prefix (str): 日志消息的前缀字符串。
        mongodb (MongoClient): 可选的用于分布式调优的 MongoDB 客户端。
        collection (Collection): 存储调优结果的 MongoDB 集合。

    Methods:
        _mutate: 基于边界和缩放因子对超参数进行变异。
        __call__: 在多次迭代中执行超参数进化。

    Examples:
        在 COCO8 上以 imgsz=640 和 epochs=10 对 YOLO26n 进行 300 次调优迭代。
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")
        >>> model.tune(
        >>>     data="coco8.yaml",
        >>>     epochs=10,
        >>>     iterations=300,
        >>>     plots=False,
        >>>     save=False,
        >>>     val=False
        >>> )

        使用分布式 MongoDB Atlas 跨多机协调调优：
        >>> model.tune(
        >>>     data="coco8.yaml",
        >>>     epochs=10,
        >>>     iterations=300,
        >>>     mongodb_uri="mongodb+srv://user:pass@cluster.mongodb.net/",
        >>>     mongodb_db="ultralytics",
        >>>     mongodb_collection="tune_results"
        >>> )

        使用自定义搜索空间调优：
        >>> model.tune(space={"lr0": (1e-5, 1e-2), "momentum": (0.7, 0.98)})
    """

    def __init__(self, args=DEFAULT_CFG, _callbacks: dict | None = None):
        """使用配置初始化 Tuner。

        Args:
            args (dict): 超参数进化的配置。
            _callbacks (dict | None, optional): 调优期间执行的回调函数。
        """
        self.space = args.pop("space", None) or {  # key: (min, max, gain(可选))
            # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
            "lr0": (1e-5, 1e-2),  # 初始学习率 (即 SGD=1E-2, Adam=1E-3)
            "lrf": (0.01, 1.0),  # 最终 OneCycleLR 学习率 (lr0 * lrf)
            "momentum": (0.7, 0.98, 0.3),  # SGD momentum/Adam beta1
            "weight_decay": (0.0, 0.001),  # 优化器权重衰减 5e-4
            "warmup_epochs": (0.0, 5.0),  # 预热轮数 (可为小数)
            "warmup_momentum": (0.0, 0.95),  # 预热初始动量
            "box": (1.0, 20.0),  # box 损失增益
            "cls": (0.1, 4.0),  # cls 损失增益 (按像素缩放)
            "cls_pw": (0.0, 1.0),  # cls 幂权重
            "dfl": (0.4, 12.0),  # dfl 损失增益
            "hsv_h": (0.0, 0.1),  # 图像 HSV-色调增强 (比例)
            "hsv_s": (0.0, 0.9),  # 图像 HSV-饱和度增强 (比例)
            "hsv_v": (0.0, 0.9),  # 图像 HSV-明度增强 (比例)
            "degrees": (0.0, 45.0),  # 图像旋转 (+/- 度)
            "translate": (0.0, 0.9),  # 图像平移 (+/- 比例)
            "scale": (0.0, 0.95),  # 图像缩放 (+/- 增益)
            "shear": (0.0, 10.0),  # 图像剪切 (+/- 度)
            "perspective": (0.0, 0.001),  # 图像透视 (+/- 比例), 范围 0-0.001
            "flipud": (0.0, 1.0),  # 图像上下翻转 (概率)
            "fliplr": (0.0, 1.0),  # 图像左右翻转 (概率)
            "bgr": (0.0, 1.0),  # 图像通道 BGR (概率)
            "mosaic": (0.0, 1.0),  # 图像马赛克 (概率)
            "mixup": (0.0, 1.0),  # 图像 mixup (概率)
            "cutmix": (0.0, 1.0),  # 图像 cutmix (概率)
            "copy_paste": (0.0, 1.0),  # 分割复制粘贴 (概率)
            "close_mosaic": (0.0, 10.0),  # 关闭数据加载器马赛克 (轮数)
        }
        mongodb_uri = args.pop("mongodb_uri", None)
        mongodb_db = args.pop("mongodb_db", "ultralytics")
        mongodb_collection = args.pop("mongodb_collection", "tuner_results")

        self.args = get_cfg(overrides=args)
        self.args.exist_ok = self.args.resume  # 使用相同的 tune_dir 恢复
        self.tune_dir = get_save_dir(self.args, name=self.args.name or "tune")
        self.args.name, self.args.exist_ok, self.args.resume = (None, False, False)  # 重置以避免影响训练
        self.tune_file = self.tune_dir / "tune_results.ndjson"
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.prefix = colorstr("Tuner: ")
        callbacks.add_integration_callbacks(self)

        # MongoDB Atlas 支持 (可选)
        self.mongodb = None
        if mongodb_uri:
            self._init_mongodb(mongodb_uri, mongodb_db, mongodb_collection)

        LOGGER.info(
            f"{self.prefix}Initialized Tuner instance with 'tune_dir={self.tune_dir}'\n"
            f"{self.prefix}💡 Learn about tuning at https://docs.ultralytics.com/guides/hyperparameter-tuning"
        )

    def _connect(self, uri: str = "", max_retries: int = 3):
        """创建 MongoDB 客户端，连接失败时使用指数退避重试。

        Args:
            uri (str): 包含凭据和集群信息的 MongoDB 连接字符串。
            max_retries (int): 放弃前的最大连接尝试次数。

        Returns:
            (MongoClient): 已连接的 MongoDB 客户端实例。
        """
        check_requirements("pymongo")

        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

        for attempt in range(max_retries):
            try:
                client = MongoClient(
                    uri,
                    serverSelectionTimeoutMS=30000,
                    connectTimeoutMS=20000,
                    socketTimeoutMS=40000,
                    retryWrites=True,
                    retryReads=True,
                    maxPoolSize=30,
                    minPoolSize=3,
                    maxIdleTimeMS=60000,
                )
                client.admin.command("ping")  # 测试连接
                LOGGER.info(f"{self.prefix}Connected to MongoDB Atlas (attempt {attempt + 1})")
                return client
            except (ConnectionFailure, ServerSelectionTimeoutError):
                if attempt == max_retries - 1:
                    raise
                wait_time = 2**attempt
                LOGGER.warning(
                    f"{self.prefix}MongoDB connection failed (attempt {attempt + 1}), retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

    def _init_mongodb(self, mongodb_uri="", mongodb_db="", mongodb_collection=""):
        """初始化 MongoDB 连接用于分布式调优。

        连接到 MongoDB Atlas 以进行跨多台机器的分布式超参数优化。每个工作节点将结果保存到共享集合中，
        并从所有工作节点的进化结果中读取最新的最佳超参数。

        Args:
            mongodb_uri (str): MongoDB 连接字符串。
            mongodb_db (str, optional): 数据库名称。
            mongodb_collection (str, optional): 集合名称。

        Notes:
            - 创建 fitness 索引以快速查询最佳结果
            - 如果连接失败，回退到本地 NDJSON 模式
            - 使用连接池和重试逻辑以保障生产环境的可靠性
        """
        self.mongodb = self._connect(mongodb_uri)
        self.collection = self.mongodb[mongodb_db][mongodb_collection]
        self.collection.create_index([("fitness", -1)], background=True)
        LOGGER.info(f"{self.prefix}Using MongoDB Atlas for distributed tuning")

    def _get_mongodb_results(self, n: int = 5) -> list:
        """从 MongoDB 获取按 fitness 排序的前 N 条结果。

        Args:
            n (int): 要检索的顶部结果数量。

        Returns:
            (list[dict]): 包含 fitness 分数和超参数的结果文档列表。
        """
        try:
            return list(self.collection.find().sort("fitness", -1).limit(n))
        except Exception:
            return []

    @staticmethod
    def _json_default(x):
        """将类张量值转换为 JSON 可序列化格式。"""
        return x.item() if hasattr(x, "item") else str(x)

    def _result_record(
        self,
        iteration: int,
        fitness: float,
        hyperparameters: dict[str, float],
        datasets: dict[str, dict],
        save_dirs: dict[str, str] | None = None,
    ) -> dict:
        """构建一条本地调优结果记录。"""
        result = {
            "iteration": iteration,
            "fitness": round(fitness, 5),
            "hyperparameters": hyperparameters,
            "datasets": datasets,
        }
        if save_dirs:
            result["save_dirs"] = save_dirs
        return result

    def _save_to_mongodb(
        self,
        fitness: float,
        hyperparameters: dict[str, float],
        metrics: dict,
        datasets: dict[str, dict],
        iteration: int,
    ):
        """将结果保存到 MongoDB，并进行适当的类型转换。

        Args:
            fitness (float): 使用这些超参数获得的 fitness 分数。
            hyperparameters (dict[str, float]): 超参数值字典。
            metrics (dict): 完整的训练指标字典（mAP、精确率、召回率、损失等）。
            datasets (dict[str, dict]): 每个数据集在本次迭代中的指标。
            iteration (int): 当前迭代编号。
        """
        try:
            self.collection.insert_one(
                {
                    "fitness": fitness,
                    "hyperparameters": {k: (v.item() if hasattr(v, "item") else v) for k, v in hyperparameters.items()},
                    "metrics": metrics,
                    "datasets": datasets,
                    "timestamp": datetime.now(),
                    "iteration": iteration,
                }
            )
        except Exception as e:
            LOGGER.warning(f"{self.prefix}MongoDB save failed: {e}")

    def _sync_mongodb_to_file(self):
        """将 MongoDB 结果同步到本地 NDJSON 调优日志。

        从 MongoDB 下载所有结果并按时间顺序写入本地 NDJSON 文件。
        这使得在使用分布式调优时，恢复、变异和绘图都基于同一本地数据源。
        """
        try:
            all_results = list(self.collection.find().sort("iteration", 1))
            if not all_results:
                return

            with open(self.tune_file, "w", encoding="utf-8") as f:
                for result in all_results:
                    f.write(
                        json.dumps(
                            self._result_record(
                                result["iteration"],
                                result["fitness"] or 0.0,
                                result.get("hyperparameters", {}),
                                result.get("datasets", {}),
                                result.get("save_dirs"),
                            ),
                            default=self._json_default,
                        )
                        + "\n"
                    )

        except Exception as e:
            LOGGER.warning(f"{self.prefix}MongoDB to NDJSON sync failed: {e}")

    def _load_local_results(self) -> list[dict]:
        """从 NDJSON 日志加载本地调优结果。"""
        if not self.tune_file.exists():
            return []
        with open(self.tune_file, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _local_results_to_array(self, results: list[dict], n: int | None = None) -> np.ndarray | None:
        """将本地 NDJSON 记录转换为 fitness-加-超参数 的 numpy 数组。"""
        if not results:
            return None
        x = np.array(
            [
                [r.get("fitness", 0.0)]
                + [r.get("hyperparameters", {}).get(k, getattr(self.args, k)) for k in self.space]
                for r in results
            ],
            dtype=float,
        )
        if n is None:
            return x
        order = np.argsort(-x[:, 0])
        return x[order][:n]

    def _save_local_result(self, result: dict):
        """向本地 NDJSON 日志追加一条调优结果。"""
        with open(self.tune_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, default=self._json_default) + "\n")

    @staticmethod
    def _best_metrics(result: dict) -> dict | None:
        """汇总最佳结果指标用于日志记录。"""
        datasets = result.get("datasets", {})
        if len(datasets) == 1:
            return next(iter(datasets.values()))
        if len(datasets) > 1:
            return {k: round(v.get("fitness") or 0.0, 5) for k, v in datasets.items()}
        return None

    @staticmethod
    def _has_training_metrics(result: dict, require_all: bool = False) -> bool:
        """返回调优结果是否包含训练指标。"""
        datasets = result.get("datasets", {})
        return bool(datasets) and (all(datasets.values()) if require_all else any(datasets.values()))

    @classmethod
    def _best_result_index(cls, results: list[dict], fitness: np.ndarray) -> int:
        """返回最佳结果索引，优先选择包含训练指标的行。"""
        valid = [i for i, result in enumerate(results) if cls._has_training_metrics(result)]
        return valid[int(fitness[valid].argmax())] if valid else int(fitness.argmax())

    @staticmethod
    def _dataset_names(data: list) -> list[str]:
        """为日志和每次运行的目录创建稳定的唯一数据集名称。"""
        stems = [Path(str(d)).stem for d in data]
        totals, seen = Counter(stems), Counter()
        names = []
        for stem in stems:
            seen[stem] += 1
            names.append(f"{stem}-{seen[stem]}" if totals[stem] > 1 else stem)
        return names

    @staticmethod
    def _crossover(x: np.ndarray, alpha: float = 0.2, k: int = 9) -> np.ndarray:
        """从最多前 k 个父代中进行 BLX-α 交叉 (x[:,0]=fitness, 其余=基因)。"""
        k = min(k, len(x))
        # fitness 权重 (偏移到 >0)；如果退化则回退为均匀分布
        weights = x[:, 0] - x[:, 0].min() + 1e-6
        if not np.isfinite(weights).all() or weights.sum() == 0:
            weights = np.ones_like(weights)
        idxs = random.choices(range(len(x)), weights=weights, k=k)
        parents_mat = np.stack([x[i][1:] for i in idxs], 0)  # (k, ng) 去除 fitness
        lo, hi = parents_mat.min(0), parents_mat.max(0)
        span = hi - lo
        # 当 span 为零时给一个小值以避免没有变异
        span = np.where(span == 0, np.random.uniform(0.01, 0.1, span.shape), span)
        return np.random.uniform(lo - alpha * span, hi + alpha * span)

    def _mutate(
        self,
        n: int = 9,
        mutation: float = 0.5,
        sigma: float = 0.2,
    ) -> dict[str, float]:
        """基于 `self.space` 中指定的边界和缩放因子对超参数进行变异。

        Args:
            n (int): 要考虑的顶级父代数量。
            mutation (float): 在任何给定迭代中参数变异的概率。
            sigma (float): 高斯随机数生成器的标准差。

        Returns:
            (dict[str, float]): 包含变异后超参数的字典。
        """
        x = None

        # 如果可用，先尝试 MongoDB
        if self.mongodb:
            if results := self._get_mongodb_results(n):
                # MongoDB 已按 fitness DESC 排序，因此 results[0] 是最佳的
                x = np.array(
                    [
                        [r["fitness"]] + [r["hyperparameters"].get(k, self.args.get(k)) for k in self.space.keys()]
                        for r in results
                    ]
                )
            elif self.collection.name in self.collection.database.list_collection_names():  # Tuner 在其他地方启动
                x = np.array([[0.0] + [getattr(self.args, k) for k in self.space.keys()]])

        # 如果 MongoDB 不可用或为空，回退到本地 NDJSON
        if x is None:
            x = self._local_results_to_array(self._load_local_results(), n=n)

        # 如果有数据则变异，否则使用默认值
        if x is not None:
            rng = np.random.default_rng()
            ng = len(self.space)

            # 交叉
            genes = self._crossover(x)

            # 变异
            gains = np.array([v[2] if len(v) == 3 else 1.0 for v in self.space.values()])  # gains 0-1
            factors = np.ones(ng)
            while np.all(factors == 1):  # 变异直到发生变化 (防止重复)
                mask = rng.random(ng) < mutation
                step = rng.standard_normal(ng) * (sigma * gains)
                factors = np.where(mask, np.exp(step), 1.0).clip(0.25, 4.0)
            hyp = {k: float(genes[i] * factors[i]) for i, k in enumerate(self.space.keys())}
        else:
            hyp = {k: getattr(self.args, k) for k in self.space.keys()}

        # 约束到界限内
        for k, bounds in self.space.items():
            hyp[k] = round(min(max(hyp[k], bounds[0]), bounds[1]), 5)

        # 更新类型
        if "close_mosaic" in hyp:
            hyp["close_mosaic"] = round(hyp["close_mosaic"])
        if "epochs" in hyp:
            hyp["epochs"] = round(hyp["epochs"])

        return hyp

    def __call__(self, iterations: int = 10, cleanup: bool = True):
        """当 Tuner 实例被调用时执行超参数进化过程。

        此方法遍历指定数量的迭代，执行以下步骤：
        1. 将 MongoDB 结果同步到本地 NDJSON（如果使用分布式模式）
        2. 使用最佳先前结果或默认值对超参数进行变异
        3. 使用变异后的超参数训练 YOLO 模型
        4. 将 fitness 分数和超参数记录到 MongoDB 和/或 NDJSON
        5. 跟踪所有迭代中的最佳配置

        Args:
            iterations (int): 运行进化的代数数量。
            cleanup (bool): 是否在调优期间删除迭代权重以减少存储空间。
        """
        t0 = time.time()
        self.tune_dir.mkdir(parents=True, exist_ok=True)
        (self.tune_dir / "weights").mkdir(parents=True, exist_ok=True)
        best_save_dirs = {}
        n_successful = 0  # 本次调用中有真实训练指标的迭代（不包括已恢复/MongoDB 行）

        # 启动时将 MongoDB 同步到本地 NDJSON 以实现正确的恢复逻辑
        if self.mongodb:
            self._sync_mongodb_to_file()

        start = 0
        if self.tune_file.exists():
            start = len(self._load_local_results())
            LOGGER.info(f"{self.prefix}Resuming tuning run {self.tune_dir} from iteration {start + 1}...")
        for i in range(start, iterations):
            # 在前 300 次迭代中线性衰减 sigma 从 0.2 → 0.1
            frac = min(i / 300.0, 1.0)
            sigma_i = 0.2 - 0.1 * frac

            # 变异超参数
            mutated_hyp = self._mutate(sigma=sigma_i)
            LOGGER.info(f"{self.prefix}Starting iteration {i + 1}/{iterations} with hyperparameters: {mutated_hyp}")

            train_args = {**vars(self.args), **mutated_hyp}
            data = train_args.pop("data")
            if not isinstance(data, (list, tuple)):
                data = [data]
            dataset_names = self._dataset_names(data)
            save_dir = (
                [get_save_dir(get_cfg(train_args))]
                if len(data) == 1
                else [get_save_dir(get_cfg(train_args), name=name) for name in dataset_names]
            )
            weights_dir = [s / "weights" for s in save_dir]
            metrics = {}
            all_fitness = []
            dataset_metrics = {}
            for j, (d, dataset) in enumerate(zip(data, dataset_names)):
                metrics_i = {}
                try:
                    train_args["data"] = d
                    train_args["save_dir"] = str(save_dir[j])  # 传递 save_dir 给子进程以确保使用相同路径
                    # 使用变异后的超参数训练 YOLO 模型（在子进程中运行以避免数据加载器挂起）
                    launch = [
                        __import__("sys").executable,
                        "-m",
                        "ultralytics.cfg.__init__",
                    ]  # 解决 yolo 找不到的问题
                    cmd = [*launch, "train", *(f"{k}={v}" for k, v in train_args.items())]
                    subprocess.run(cmd, check=True)
                    ckpt_file = weights_dir[j] / ("best.pt" if (weights_dir[j] / "best.pt").exists() else "last.pt")
                    metrics_i = torch_load(ckpt_file)["train_metrics"]
                    metrics = metrics_i

                    # 清理
                    time.sleep(1)
                    gc.collect()
                    torch.cuda.empty_cache()

                except Exception as e:
                    LOGGER.error(f"training failure for hyperparameter tuning iteration {i + 1}\n{e}")

                # 保存结果 - MongoDB 优先
                dataset_metrics[dataset] = metrics_i
                all_fitness.append(metrics_i.get("fitness") or 0.0)
            fitness = sum(all_fitness) / len(all_fitness)
            result = self._result_record(
                i + 1,
                fitness,
                mutated_hyp,
                dataset_metrics,
                {dataset: str(s) for dataset, s in zip(dataset_names, save_dir)},
            )
            if self._has_training_metrics(result, require_all=True):
                n_successful += 1
            stop_after_iteration = False
            if self.mongodb:
                self._save_to_mongodb(fitness, mutated_hyp, metrics, dataset_metrics, i + 1)
                self._sync_mongodb_to_file()
                total_mongo_iterations = self.collection.count_documents({})
                if total_mongo_iterations >= iterations:
                    stop_after_iteration = True
            else:
                self._save_local_result(result)

            # 获取最佳结果
            results = self._load_local_results()
            x = self._local_results_to_array(results)
            fitness = x[:, 0]  # 第一列
            best_idx = self._best_result_index(results, fitness)
            best_result = results[best_idx]
            n_attempted = (i + 1) - start  # 本次调用中尝试的迭代数
            current_best_save_dirs = best_result.get("save_dirs", {})
            best_is_current = best_idx == i
            if best_is_current:
                if cleanup:
                    for s in best_save_dirs.values():
                        if s not in current_best_save_dirs.values():
                            shutil.rmtree(s, ignore_errors=True)
                for dataset, weight_dir in zip(dataset_names, weights_dir):
                    best_weights_dir = (
                        self.tune_dir / "weights" if len(data) == 1 else self.tune_dir / "weights" / dataset
                    )
                    best_weights_dir.mkdir(parents=True, exist_ok=True)
                    for ckpt in weight_dir.glob("*.pt"):
                        shutil.copy2(ckpt, best_weights_dir)
                best_save_dirs = current_best_save_dirs
            elif cleanup:
                for s in save_dir:
                    shutil.rmtree(s, ignore_errors=True)  # 删除迭代目录以减少存储空间
                best_save_dirs = current_best_save_dirs

            # 绘制调优结果
            plot_tune_results(str(self.tune_file))

            # 保存并打印调优结果
            if n_successful == n_attempted:
                status = "complete ✅"
            elif n_successful == 0:
                status = "complete (all failed) ❌"
            else:
                status = f"complete ({n_successful}/{n_attempted} succeeded) ⚠️"
            has_valid_best = self._has_training_metrics(best_result)
            header_lines = [
                f"{self.prefix}{i + 1}/{iterations} iterations {status} ({time.time() - t0:.2f}s)",
                f"{self.prefix}Results saved to {colorstr('bold', self.tune_dir)}",
            ]
            if has_valid_best:
                header_lines.extend(
                    [
                        f"{self.prefix}Best fitness={fitness[best_idx]} observed at iteration {best_idx + 1}",
                        f"{self.prefix}Best fitness metrics are {self._best_metrics(best_result)}",
                        f"{self.prefix}Best fitness model is "
                        f"{self.tune_dir / 'weights' if len(best_result.get('datasets', {})) == 1 else 'not saved for multi-dataset tuning'}",
                    ]
                )
            header = "\n".join(header_lines)
            LOGGER.info("\n" + header)
            if not has_valid_best:
                LOGGER.error(
                    f"{self.prefix}No iterations produced training metrics; skipping best_hyperparameters.yaml"
                )
            else:
                data = {
                    k: int(v) if k in CFG_INT_KEYS else float(v) for k, v in zip(self.space.keys(), x[best_idx, 1:])
                }
                YAML.save(
                    self.tune_dir / "best_hyperparameters.yaml",
                    data=data,
                    header=remove_colorstr(header.replace(self.prefix, "# ")) + "\n",
                )
                YAML.print(self.tune_dir / "best_hyperparameters.yaml")
            if stop_after_iteration:
                LOGGER.info(
                    f"{self.prefix}Target iterations ({iterations}) reached in MongoDB ({total_mongo_iterations}). Stopping."
                )
                break
