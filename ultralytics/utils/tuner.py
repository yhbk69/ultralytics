# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import numpy as np

from ultralytics.cfg import TASK2DATA, TASK2METRIC, get_cfg, get_save_dir
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER, NUM_THREADS, checks, colorstr

RAY_SEARCH_ALG_REQUIREMENTS = {
    "random": None,
    "ax": "ax-platform",
    "bayesopt": "bayesian-optimization==1.4.3",
    "bohb": ["hpbandster", "ConfigSpace"],
    "hebo": "HEBO>=0.2.0",
    "hyperopt": "hyperopt",
    "nevergrad": "nevergrad",
    "optuna": "optuna",
    "zoopt": "zoopt",
}


def _sanitize_tune_value(value: dict):
    """将 NumPy 支持的 Tune 值转换为原生 Python 类型，用于 YAML 序列化。

    参数:
        value (dict): 要转换的值。可以是字典、列表、元组、NumPy 标量或 NumPy 数组。

    返回:
        转换后的值，NumPy 类型替换为原生 Python 类型。
    """
    if isinstance(value, dict):
        return {k: _sanitize_tune_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_tune_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_tune_value(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _get_ray_search_alg_kind(search_alg):
    """返回已知搜索器对象的标准化 Ray Tune 搜索算法类型。

    参数:
        search_alg (str | ray.tune.search.Searcher): 要识别的搜索算法。可以是 None、字符串或 Ray Tune 搜索器对象。

    返回:
        str | None: 标准化的搜索算法名称，如果未识别则返回 None。
    """
    if search_alg is None:
        return None
    if isinstance(search_alg, str):
        normalized = search_alg.strip().lower()
        return normalized or None

    cls = search_alg.__class__
    module, name = cls.__module__, cls.__name__
    if name == "AxSearch" and module.startswith("ray.tune.search.ax"):
        return "ax"
    if name == "TuneBOHB" and module.startswith("ray.tune.search.bohb"):
        return "bohb"
    if name == "ZOOptSearch" and module.startswith("ray.tune.search.zoopt"):
        return "zoopt"
    return None


def _validate_ax_search_space(space):
    """验证 Tune 搜索空间是否可被 Ax 使用。

    参数:
        space (dict): 要验证的超参数搜索空间。

    返回:
        list: 转换后的 Ax 参数。

    异常:
        ImportError: 如果未安装所需的 'ax-platform' 包。
    """
    checks.check_requirements(RAY_SEARCH_ALG_REQUIREMENTS["ax"])

    from ray.tune.search.ax.ax_search import AxSearch

    return AxSearch.convert_search_space(space)


def _create_ax_search(space, task):
    """创建带有已初始化实验的 Ax 搜索器。

    参数:
        space (dict): 超参数搜索空间。
        task (str): 任务类型（如 'detect'、'segment'、'classify'）。

    返回:
        AxSearch (ray.tune.search.Searcher): 配置的 Ax 搜索算法。

    异常:
        ImportError: 如果未安装所需的 Ax 包。
    """
    parameters = _validate_ax_search_space(space)

    from ax.service.ax_client import AxClient
    from ax.service.utils.instantiation import ObjectiveProperties
    from ray.tune.search.ax.ax_search import AxSearch

    ax_client = AxClient()
    ax_client.create_experiment(
        parameters=parameters,
        objectives={TASK2METRIC[task]: ObjectiveProperties(minimize=False)},
    )
    return AxSearch(ax_client=ax_client)


def _convert_bohb_search_space(space):
    """将 Tune 搜索空间转换为 BOHB 兼容的 ConfigSpace 和仅固定参数的 Tune param_space。

    参数:
        space (dict): 超参数搜索空间。

    返回:
        (tuple): 包含 ConfigSpace 对象和固定参数字典的元组。

    异常:
        ValueError: 如果搜索空间包含网格搜索参数或不支持的采样器。
        ImportError: 如果未安装所需的 BOHB 包。
    """
    checks.check_requirements(RAY_SEARCH_ALG_REQUIREMENTS["bohb"])

    import ConfigSpace
    from ray.tune.search.sample import Categorical, Float, Integer, LogUniform, Quantized, Uniform
    from ray.tune.search.variant_generator import parse_spec_vars
    from ray.tune.utils import flatten_dict

    resolved_space = flatten_dict(space, prevent_delimiter=True)
    resolved_vars, domain_vars, grid_vars = parse_spec_vars(resolved_space)
    if grid_vars:
        raise ValueError("Grid search parameters cannot be automatically converted to a TuneBOHB search space.")

    cs = ConfigSpace.ConfigurationSpace()
    for path, domain in domain_vars:
        par = "/".join(str(p) for p in path)
        sampler = domain.get_sampler()
        if isinstance(sampler, Quantized):
            raise ValueError("TuneBOHB does not support quantized search spaces with the current ConfigSpace version.")

        if isinstance(domain, Float) and isinstance(sampler, (Uniform, LogUniform)):
            cs.add(
                ConfigSpace.UniformFloatHyperparameter(
                    par, lower=domain.lower, upper=domain.upper, log=isinstance(sampler, LogUniform)
                )
            )
        elif isinstance(domain, Integer) and isinstance(sampler, (Uniform, LogUniform)):
            upper = domain.upper - 1  # Tune 整数搜索空间上界是排他的
            cs.add(
                ConfigSpace.UniformIntegerHyperparameter(
                    par, lower=domain.lower, upper=upper, log=isinstance(sampler, LogUniform)
                )
            )
        elif isinstance(domain, Categorical) and isinstance(sampler, Uniform):
            cs.add(ConfigSpace.CategoricalHyperparameter(par, choices=domain.categories))
        else:
            raise ValueError(
                f"TuneBOHB does not support parameters of type {type(domain).__name__} "
                f"with sampler type {type(domain.sampler).__name__}."
            )

    fixed_param_space = {"/".join(str(p) for p in path): value for path, value in resolved_vars}
    return cs, fixed_param_space


def _create_bohb_search(space, task):
    """创建与当前 ConfigSpace 版本兼容的 BOHB 搜索器。

    参数:
        space (dict): 超参数搜索空间。
        task (str): 任务类型（如 'detect'、'segment'、'classify'）。

    返回:
        (tuple): 包含 TuneBOHB 搜索器和固定参数空间字典的元组。

    异常:
        ImportError: 如果未安装所需的 BOHB 包。
    """
    cs, fixed_param_space = _convert_bohb_search_space(space)

    from ray.tune.search.bohb.bohb_search import TuneBOHB

    return TuneBOHB(space=cs, metric=TASK2METRIC[task], mode="max"), fixed_param_space


def _create_nevergrad_search(task):
    """创建带有默认优化器的 Nevergrad 搜索器。

    参数:
        task (str): 任务类型（如 'detect'、'segment'、'classify'）。

    返回:
        (NevergradSearch): 配置的 Nevergrad 搜索算法。

    异常:
        ImportError: 如果未安装 'nevergrad' 包。
    """
    checks.check_requirements(RAY_SEARCH_ALG_REQUIREMENTS["nevergrad"])

    import nevergrad as ng
    from ray.tune.search.nevergrad import NevergradSearch

    return NevergradSearch(optimizer=ng.optimizers.OnePlusOne, metric=TASK2METRIC[task], mode="max")


def _convert_zoopt_search_space(space):
    """将 Tune 搜索空间转换为 ZOOpt 兼容的维度和仅固定参数的 Tune param_space。

    参数:
        space (dict): 超参数搜索空间。

    返回:
        (tuple): 包含 ZOOpt 维度字典和固定参数空间字典的元组。

    异常:
        ImportError: 如果未安装 'zoopt' 包。
    """
    checks.check_requirements(RAY_SEARCH_ALG_REQUIREMENTS["zoopt"])

    from ray.tune.search.variant_generator import parse_spec_vars
    from ray.tune.search.zoopt import ZOOptSearch
    from ray.tune.utils import flatten_dict

    resolved_space = flatten_dict(space, prevent_delimiter=True)
    resolved_vars, _, _ = parse_spec_vars(resolved_space)
    fixed_param_space = {"/".join(str(p) for p in path): value for path, value in resolved_vars}
    dim_dict = ZOOptSearch.convert_search_space(space)
    return dim_dict, fixed_param_space


def _create_zoopt_search(space, task, iterations):
    """创建带有所需预算和转换搜索空间的 ZOOpt 搜索器。

    参数:
        space (dict): 超参数搜索空间。
        task (str): 任务类型（如 'detect'、'segment'、'classify'）。
        iterations (int): ZOOpt 的最大试验数（预算）。

    返回:
        (tuple): 包含 ZOOptSearch 搜索器和固定参数空间字典的元组。

    异常:
        ImportError: 如果未安装 'zoopt' 包。
    """
    dim_dict, fixed_param_space = _convert_zoopt_search_space(space)

    from ray.tune.search.zoopt import ZOOptSearch

    return ZOOptSearch(
        algo="asracos", budget=iterations, dim_dict=dim_dict, metric=TASK2METRIC[task], mode="max"
    ), fixed_param_space


def _resolve_ray_search_alg(search_alg, task, space, iterations):
    """解析搜索算法并为已知的 Ray Tune 搜索器标准化 Tune param_space。

    参数:
        search_alg (str | object | None): 要使用的搜索算法。可以是字符串名称、预实例化的 Ray Tune 搜索器对象，或 None 表示默认行为。
        task (str): 任务类型（如 'detect'、'segment'、'classify'）。
        space (dict): 超参数搜索空间。
        iterations (int): 要运行的最大试验数。

    返回:
        (tuple): 包含 (resolved_search_alg, tuner_param_space, resolved_search_alg_kind) 的元组。
            - resolved_search_alg: 配置的搜索器或 None。
            - tuner_param_space: 标准化的参数空间。
            - resolved_search_alg_kind: 标准化的算法名称或 None。

    异常:
        ValueError: 如果提供了不支持的 search_alg 字符串。
        ModuleNotFoundError: 如果所选算法的依赖未安装。
    """
    if search_alg is None:
        return None, space, None

    normalized = _get_ray_search_alg_kind(search_alg)
    if isinstance(search_alg, str):
        if not normalized:
            return None, space, None
        if normalized not in RAY_SEARCH_ALG_REQUIREMENTS:
            supported = ", ".join(sorted(RAY_SEARCH_ALG_REQUIREMENTS))
            raise ValueError(f"Unsupported Ray Tune search_alg '{search_alg}'. Supported values: {supported}.")
        if normalized == "random":
            return None, space, normalized

    try:
        if normalized == "ax":
            if isinstance(search_alg, str):
                return _create_ax_search(space, task), {}, normalized
            _validate_ax_search_space(space)
            return search_alg, {}, normalized
        if normalized == "bohb":
            if isinstance(search_alg, str):
                resolved_search_alg, tuner_param_space = _create_bohb_search(space, task)
            else:
                _, tuner_param_space = _convert_bohb_search_space(space)
                resolved_search_alg = search_alg
            return resolved_search_alg, tuner_param_space, normalized
        if normalized == "nevergrad":
            return _create_nevergrad_search(task), space, normalized
        if normalized == "zoopt":
            if isinstance(search_alg, str):
                resolved_search_alg, tuner_param_space = _create_zoopt_search(space, task, iterations)
            else:
                _, tuner_param_space = _convert_zoopt_search_space(space)
                resolved_search_alg = search_alg
            return resolved_search_alg, tuner_param_space, normalized
        if not isinstance(search_alg, str):
            return search_alg, space, None

        requirements = RAY_SEARCH_ALG_REQUIREMENTS[normalized]
        if requirements:
            checks.check_requirements(requirements)

        from ray.tune.search import create_searcher

        return create_searcher(normalized, metric=TASK2METRIC[task], mode="max"), space, normalized
    except (ImportError, ModuleNotFoundError) as e:
        raise ModuleNotFoundError(
            f"Ray Tune search_alg '{search_alg}' requires additional dependencies. Original error: {e}"
        ) from e


def run_ray_tune(
    model,
    space: dict | None = None,
    grace_period: int = 10,
    gpu_per_trial: int | None = None,
    iterations: int = 10,
    search_alg=None,
    **train_args,
):
    """使用 Ray Tune 运行超参数调优。

    参数:
        model (YOLO): 要运行调优的模型。
        space (dict, 可选): 超参数搜索空间。如果未提供，使用默认空间。
        grace_period (int, 可选): ASHA 调度器的宽限期（epoch 数）。
        gpu_per_trial (int, 可选): 每次试验分配的 GPU 数量。
        iterations (int, 可选): 要运行的最大试验数。
        search_alg (str | ray.tune.search.Searcher | ray.tune.search.SearchAlgorithm, 可选): 要使用的搜索算法。
            字符串解析为支持的 Ray Tune 搜索器。预实例化的对象会被重用，
            已知有特殊 Tune param_space 要求的搜索器会自动标准化。
        **train_args (Any): 传递给 `train()` 方法的额外参数。

    返回:
        (ray.tune.ResultGrid): 包含超参数搜索结果的 ResultGrid。

    示例:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")  # 加载 YOLO26n 模型

        开始对 YOLO26n 在 COCO8 数据集上进行超参数调优
        >>> result_grid = model.tune(data="coco8.yaml", use_ray=True)
    """
    LOGGER.info("💡 Learn about RayTune at https://docs.ultralytics.com/integrations/ray-tune")
    try:
        checks.check_requirements("ray[tune]")

        import ray
        from ray import tune
        from ray.tune import RunConfig
        from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB
    except ImportError:
        raise ModuleNotFoundError('Ray Tune required but not found. To install run: pip install "ray[tune]"')

    try:
        import wandb

        assert hasattr(wandb, "__version__")
    except (ImportError, AssertionError):
        wandb = False

    checks.check_version(ray.__version__, ">=2.0.0", "ray")
    default_space = {
        # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
        "lr0": tune.uniform(1e-5, 1e-2),  # 初始学习率（如 SGD=1E-2, Adam=1E-3）
        "lrf": tune.uniform(0.01, 1.0),  # 最终 OneCycleLR 学习率 (lr0 * lrf)
        "momentum": tune.uniform(0.7, 0.98),  # SGD 动量/Adam beta1
        "weight_decay": tune.uniform(0.0, 0.001),  # 优化器权重衰减
        "warmup_epochs": tune.uniform(0.0, 5.0),  # 预热 epoch（可以是小数）
        "warmup_momentum": tune.uniform(0.0, 0.95),  # 预热初始动量
        "box": tune.uniform(1.0, 20.0),  # 框损失增益
        "cls": tune.uniform(0.1, 4.0),  # 分类损失增益（按像素缩放）
        "cls_pw": tune.uniform(0.0, 1.0),  # 分类幂权重（按像素缩放）
        "dfl": tune.uniform(0.4, 12.0),  # dfl 损失增益
        "hsv_h": tune.uniform(0.0, 0.1),  # 图像 HSV-色相增强（比例）
        "hsv_s": tune.uniform(0.0, 0.9),  # 图像 HSV-饱和度增强（比例）
        "hsv_v": tune.uniform(0.0, 0.9),  # 图像 HSV-明度增强（比例）
        "degrees": tune.uniform(0.0, 45.0),  # 图像旋转（+/- 度）
        "translate": tune.uniform(0.0, 0.9),  # 图像平移（+/- 比例）
        "scale": tune.uniform(0.0, 0.95),  # 图像缩放（+/- 增益）
        "shear": tune.uniform(0.0, 10.0),  # 图像剪切（+/- 度）
        "perspective": tune.uniform(0.0, 0.001),  # 图像透视（+/- 比例），范围 0-0.001
        "flipud": tune.uniform(0.0, 1.0),  # 图像上下翻转（概率）
        "fliplr": tune.uniform(0.0, 1.0),  # 图像左右翻转（概率）
        "bgr": tune.uniform(0.0, 1.0),  # 交换 RGB↔BGR 通道（概率）
        "mosaic": tune.uniform(0.0, 1.0),  # 图像马赛克（概率）
        "mixup": tune.uniform(0.0, 1.0),  # 图像混合（概率）
        "cutmix": tune.uniform(0.0, 1.0),  # 图像 cutmix（概率）
        "copy_paste": tune.uniform(0.0, 1.0),  # 分割复制粘贴（概率）
        "close_mosaic": tune.randint(0, 11),  # 关闭数据加载器马赛克（epoch）
    }

    # 将模型放入 ray store
    task = model.task
    model_in_store = ray.put(model)
    base_name = train_args.get("name", "tune")

    def _tune(config):
        """使用指定超参数训练 YOLO 模型并返回结果。"""
        model_to_train = ray.get(model_in_store)  # 从 ray store 获取模型进行调优
        model_to_train.trainer = None
        model_to_train.reset_callbacks()
        config = _sanitize_tune_value(dict(config))
        config.update(train_args)

        # 设置试验特定名称用于 W&B 日志记录
        try:
            trial_id = tune.get_trial_id()  # 获取当前试验 ID（如 "2c2fc_00000"）"2c2fc_00000")
            trial_suffix = trial_id.split("_")[-1] if "_" in trial_id else trial_id
            config["name"] = f"{base_name}_{trial_suffix}"
        except Exception:
            # 不在 Ray Tune 上下文中或获取试验 ID 出错，使用基本名称
            config["name"] = base_name

        results = model_to_train.train(**config)
        return results.results_dict

    # 获取搜索空间
    if not space and not train_args.get("resume"):
        space = default_space
        LOGGER.warning("Search space not provided, using default search space.")

    # 获取数据集
    data = train_args.get("data", TASK2DATA[task])
    space["data"] = data
    if "data" not in train_args:
        LOGGER.warning(f'Data not provided, using default "data={data}".')

    resolved_search_alg, tuner_param_space, resolved_search_alg_kind = _resolve_ray_search_alg(
        search_alg, task, space, iterations
    )

    # 定义带有分配资源的可训练函数
    trainable_with_resources = tune.with_resources(_tune, {"cpu": NUM_THREADS, "gpu": gpu_per_trial or 0})

    # 定义超参数搜索的调度器
    max_t = train_args.get("epochs") or DEFAULT_CFG_DICT["epochs"] or 100
    scheduler = ASHAScheduler(
        time_attr="epoch",
        metric=TASK2METRIC[task],
        mode="max",
        max_t=max_t,
        grace_period=min(grace_period, max_t),
        reduction_factor=3,
    )
    if resolved_search_alg_kind == "bohb":
        scheduler = HyperBandForBOHB(
            time_attr="epoch",
            metric=TASK2METRIC[task],
            mode="max",
            max_t=max_t,
            reduction_factor=3,
        )

    # 创建 Ray Tune 超参数搜索调优器
    tune_dir = get_save_dir(
        get_cfg(
            DEFAULT_CFG,
            {**train_args, **{"exist_ok": train_args.pop("resume", False)}},  # 使用相同 tune_dir 恢复
        ),
        name=train_args.pop("name", "tune"),  # runs/{task}/{tune_dir}
    )  # 必须是绝对目录
    tune_dir.mkdir(parents=True, exist_ok=True)
    if tune.Tuner.can_restore(tune_dir):
        LOGGER.info(f"{colorstr('Tuner: ')} Resuming tuning run {tune_dir}...")
        tuner = tune.Tuner.restore(str(tune_dir), trainable=trainable_with_resources, resume_errored=True)
    else:
        tuner = tune.Tuner(
            trainable_with_resources,
            param_space=tuner_param_space,
            tune_config=tune.TuneConfig(
                search_alg=resolved_search_alg,
                scheduler=scheduler,
                num_samples=iterations,
                trial_name_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
                trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
            ),
            run_config=RunConfig(storage_path=tune_dir.parent, name=tune_dir.name),
        )

    # 运行超参数搜索
    tuner.fit()

    # 获取超参数搜索结果
    results = tuner.get_results()

    # 关闭 Ray 以清理工作进程
    ray.shutdown()

    return results
