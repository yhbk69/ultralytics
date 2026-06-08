# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
在数据集上训练模型。

Usage:
    $ yolo mode=train model=yolo26n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

from __future__ import annotations

import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics import __version__
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset, convert_ndjson_to_yolo_if_needed
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.optim import MuSGD
from ultralytics.utils import (
    DEFAULT_CFG,
    GIT,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    attempt_compile,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
    unwrap_model,
)


class BaseTrainer:
    """用于创建训练器的基础类。

    该类为 YOLO 模型训练提供基础功能，处理训练循环、验证、检查点保存及各种训练工具。
    支持单 GPU 和多 GPU 分布式训练。

    Attributes:
        args (SimpleNamespace): 训练器配置。
        validator (BaseValidator): 验证器实例。
        model (nn.Module): 模型实例。
        callbacks (defaultdict): 回调函数字典。
        save_dir (Path): 结果保存目录。
        wdir (Path): 权重保存目录。
        last (Path): 最新检查点路径。
        best (Path): 最佳检查点路径。
        save_period (int): 每 x 个 epoch 保存一次检查点（小于 1 时禁用）。
        batch_size (int): 训练批次大小。
        epochs (int): 训练的 epoch 数。
        start_epoch (int): 训练的起始 epoch。
        device (torch.device): 训练使用的设备。
        amp (bool): 是否启用 AMP（自动混合精度）。
        scaler (torch.amp.GradScaler): AMP 的梯度缩放器。
        data (dict): 包含路径和元数据的数据集字典。
        ema (ModelEMA): 模型的 EMA（指数移动平均）。
        resume (bool): 是否从检查点恢复训练。
        lf (callable): 学习率调度函数。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器。
        best_fitness (float): 达到的最佳适应度值。
        fitness (float): 当前适应度值。
        loss (torch.Tensor): 当前损失值。
        tloss (torch.Tensor): 损失项的运行均值。
        loss_names (list): 损失名称列表。
        csv (Path): 结果 CSV 文件路径。
        metrics (dict): 指标字典。
        plots (dict): 绘图字典。

    Methods:
        train: 执行训练过程。
        validate: 在验证集上运行验证。
        save_model: 保存模型训练检查点。
        get_dataset: 获取训练和验证数据集。
        setup_model: 加载、创建或下载模型。
        build_optimizer: 为模型构建优化器。

    Examples:
        初始化训练器并开始训练
        >>> trainer = BaseTrainer(cfg="config.yaml")
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """初始化 BaseTrainer 类。

        Args:
            cfg (str | dict | SimpleNamespace, optional): 配置文件路径或配置对象。
            overrides (dict, optional): 配置覆盖项。
            _callbacks (dict, optional): 回调函数字典。
        """
        self.hub_session = overrides.pop("session", None)  # HUB
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device)
        # 更新 "-1" 设备，使训练后验证不会重复搜索
        self.args.device = os.getenv("CUDA_VISIBLE_DEVICES") if "cuda" in str(self.device) else str(self.device)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # 目录
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # 为日志器更新名称
        self.wdir = self.save_dir / "weights"  # 权重目录
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # 创建目录
            self.args.save_dir = str(self.save_dir)
            # 保存运行参数，将增强序列化为 repr 以兼容恢复功能
            args_dict = vars(self.args).copy()
            if args_dict.get("augmentations") is not None:
                # 将 Albumentations 变换序列化为 repr 字符串以兼容检查点
                args_dict["augmentations"] = [repr(t) for t in args_dict["augmentations"]]
            YAML.save(self.save_dir / "args.yaml", args_dict)  # 保存运行参数
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # 检查点路径
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100  # 防止用户在定时训练时意外传入 epochs=None
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # 设备
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # CPU 训练更快，因为时间主要花在推理而非数据加载上

        # 回调 - 提前初始化，以便 on_pretrain_routine_start 可以捕获原始的 args.data
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

        if isinstance(self.args.device, str) and len(self.args.device):  # 即 device='0' 或 device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # 即 device=[0, 1, 2, 3]（CLI 多 GPU 为列表）
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:  # 即 device='cpu' 或 'mps'
            world_size = 0
        elif torch.cuda.is_available():  # 即 device=None 或 device='' 或 device=数字
            world_size = 1  # 默认使用设备 0
        else:  # 即 device=None 或 device=''
            world_size = 0

        self.ddp = world_size > 1 and "LOCAL_RANK" not in os.environ
        self.world_size = world_size
        # 在 get_dataset() 之前运行 on_pretrain_routine_start，以捕获原始的 args.data（例如 ul:// URI）
        if RANK in {-1, 0} and not self.ddp:
            callbacks.add_integration_callbacks(self)
            self.run_callbacks("on_pretrain_routine_start")

        # 模型和数据集
        self.model = check_model_file_from_stem(self.args.model)  # 添加后缀，即 yolo26n -> yolo26n.pt
        with torch_distributed_zero_first(LOCAL_RANK):  # 避免多次自动下载数据集
            self.data = self.get_dataset()

        self.ema = None

        # 优化工具初始化
        self.lf = None
        self.scheduler = None

        # Epoch 级别指标
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        if self.csv.exists() and not self.args.resume:
            self.csv.unlink()
        self.plot_idx = [0, 1, 2]
        self.nan_recovery_attempts = 0

    def add_callback(self, event: str, callback):
        """将给定的回调追加到事件的回调列表中。"""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """用给定的回调覆盖指定事件的现有回调。"""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """运行与特定事件关联的所有现有回调。"""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """执行训练过程，多 GPU 使用 DDP 子进程，单 GPU 直接训练。"""
        # 如果是 DDP 训练则运行子进程，否则正常训练
        if self.ddp:
            # 参数检查
            if self.args.rect:
                LOGGER.warning("'rect=True' 与多 GPU 训练不兼容，将设置 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                raise ValueError(
                    "AutoBatch 的 batch<1 不支持多 GPU 训练，"
                    f"请指定一个有效的 GPU 数量的整数倍批次大小 {self.world_size}，即 batch={self.world_size * 8}。"
                )

            # 命令
            cmd, file = None, None
            try:
                cmd, file = generate_ddp_command(self)
                LOGGER.info(f"{colorstr('DDP:')} 调试命令 {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                if file is not None:
                    ddp_cleanup(self, str(file))

        else:
            self._do_train()

    def _setup_scheduler(self):
        """初始化训练学习率调度器。"""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # 余弦 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # 线性
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self):
        """初始化并设置分布式数据并行的训练参数。"""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # 设置以强制超时
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 小时
            rank=RANK,
            world_size=self.world_size,
        )

    def _build_train_pipeline(self):
        """构建当前批次大小的数据加载器、优化器和调度器。"""
        batch_size = self.batch_size // max(self.world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
        )
        # 注意：训练 DOTA 数据集时，双倍批次大小可能会在超过 2000 个目标的图像上导致 OOM
        self.test_loader = self.get_dataloader(
            self.data.get("val") or self.data.get("test"),
            batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
            rank=LOCAL_RANK,
            mode="val",
        )
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # 优化前累积损失
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # 缩放权重衰减
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        self._setup_scheduler()

    def _setup_train(self):
        """在训练循环之前配置模型、优化器、数据加载器和训练工具。"""
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # 编译模型
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

        # 冻结层
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # 始终冻结这些层
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN 转为 0（因训练结果不稳定而注释掉）
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # 只有浮点 Tensor 才能需要梯度
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True
        if not any(v.requires_grad for v in self.model.parameters()):
            raise RuntimeError(
                f"'freeze={self.args.freeze}' 冻结了整个模型，没有剩余的可训练参数。"
                f"请减少 'freeze' 或传入特定层索引的列表。"
            )

        # 检查 AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True 或 False
        if self.amp and RANK in {-1, 0}:  # 单 GPU 和 DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # 备份回调，因为 check_amp() 会重置它们
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # 恢复回调
        if RANK > -1 and self.world_size > 1:  # DDP
            dist.broadcast(self.amp.int(), src=0)  # 从 rank 0 广播到所有其他 rank；gloo 不支持布尔广播
        self.amp = bool(self.amp)  # 转为布尔值
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        # 检查 imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # 网格大小（最大步幅）
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # 用于多尺度训练

        if self.world_size > 1:
            # static_graph=True 允许在单次前向传播中多次使用同一参数（例如 torch.compile 下的
            # o2m+o2o 姿态损失分支中的 flow_model）
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[RANK],
                static_graph=bool(self.args.compile),
            )

        # 批次大小
        if self.batch_size < 1 and RANK == -1:  # 仅单 GPU，估算最佳批次大小
            self.args.batch = self.batch_size = self.auto_batch()

        self._build_train_pipeline()
        self.validator = self.get_validator()
        self.ema = ModelEMA(self.model)
        self.set_class_weights()  # 数据加载器就绪后计算类别权重
        if RANK in {-1, 0}:
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            if self.args.plots:
                self.plot_training_labels()

        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # 不要移动
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self):
        """执行完整的训练循环，包括设置、epoch 迭代、验证和最终评估。"""
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)  # 批次数量
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # 预热迭代次数
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # 清零恢复后的梯度以确保训练开始时的稳定性
        self._oom_retries = 0  # 第一个 epoch 的 OOM 自动缩减计数器
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # 抑制 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # 更新数据加载器属性（可选）
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # 预热
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x 插值
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for x in self.optimizer.param_groups:
                        # 偏置学习率从 0.1 降到 lr0，其他所有学习率从 0.0 升到 lr0
                        x["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                x["initial_lr"] * self.lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # 前向传播
                try:
                    with autocast(self.amp):
                        batch = self.preprocess_batch(batch)
                        if self.args.compile:
                            # 解耦推理和损失计算以提升编译性能
                            preds = self.model(batch["img"])
                            loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                        else:
                            loss, self.loss_items = self.model(batch)
                        self.loss = loss.sum()
                        if RANK != -1:
                            self.loss *= self.world_size
                        self.tloss = (
                            self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)
                        )

                    # 反向传播
                    self.scaler.scale(self.loss).backward()
                except torch.cuda.OutOfMemoryError:
                    if epoch > self.start_epoch or self._oom_retries >= 3 or RANK != -1:
                        raise  # 仅在单 GPU 的第一个 epoch 自动缩减，最多重试 3 次
                    self._oom_retries += 1
                    old_batch = self.batch_size
                    self.args.batch = self.batch_size = max(self.batch_size // 2, 1)
                    LOGGER.warning(
                        f"CUDA out of memory with batch={old_batch}. "
                        f"Reducing to batch={self.batch_size} and retrying ({self._oom_retries}/3)."
                    )
                    batch = loss = preds = None
                    self.loss = self.loss_items = self.tloss = None
                    self._clear_memory()
                    self._build_train_pipeline()  # 重建数据加载器、优化器、调度器
                    self.scheduler.last_epoch = self.start_epoch - 1
                    nb = len(self.train_loader)
                    nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
                    last_opt_step = -1
                    self.optimizer.zero_grad()
                    break  # 以缩减后的批次大小重启 epoch 循环
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # 定时停止
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # 如果是 DDP 训练
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # 将 'stop' 广播到所有 rank
                            self.stop = broadcast_list[0]
                        if self.stop:  # 训练时间已超限
                            break

                # 日志
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU 内存使用量
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # 损失
                            batch["cls"].shape[0],  # 批次大小，即 8
                            batch["img"].shape[-1],  # imgsz，即 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")
                if self.stop:
                    break  # 允许批次之间外部停止（例如平台取消）
            else:
                # for/else: 此块仅在 for 循环正常完成时运行（没有 OOM 重试）
                self._oom_retries = 0  # 第一个 epoch 成功后重置 OOM 计数器

            if self._oom_retries and not self.stop:
                continue  # OOM 恢复中断了 for 循环，以缩减后的批次大小重新开始

            if hasattr(unwrap_model(self.model).criterion, "update"):
                unwrap_model(self.model).criterion.update()

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # 用于日志器

            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # 验证
            final_epoch = epoch + 1 >= self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(None if self.device.type == "mps" else 0.5)  # 防止显存峰值
                self.metrics, self.fitness = self.validate()

            # NaN 恢复
            if self._handle_nan_recovery(epoch):
                continue

            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # 保存模型
                if (self.args.save or final_epoch) and self.save_model():
                    self.run_callbacks("on_model_save")

            # 调度器
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # 不要移动
                self.stop |= epoch >= self.epochs  # 如果超过 epoch 数则停止
            self.run_callbacks("on_fit_epoch_end")
            # 内存利用率超过 50% 时清理；MPS 因内存泄漏始终清理 https://github.com/ultralytics/ultralytics/issues/22621
            self._clear_memory(None if self.device.type == "mps" else 0.5)

            # 早停
            if RANK != -1:  # 如果是 DDP 训练
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # 将 'stop' 广播到所有 rank
                self.stop = broadcast_list[0]
            if self.stop:
                break  # 必须中断所有 DDP rank
            epoch += 1

        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        # 使用 best.pt 进行最终验证
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def auto_batch(self, max_num_obj=0, dataset_size=0):
        """根据模型和设备内存约束计算最佳批次大小。"""
        max_imgsz = int(self.args.imgsz * (1 + self.args.multi_scale))  # 无需对齐步幅
        return check_train_batch_size(
            model=self.model,
            imgsz=max_imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
            dataset_size=dataset_size,
        )  # 返回批次大小

    def _get_memory(self, fraction=False):
        """获取加速器内存使用量（GB）或占总内存的比例。"""
        memory, total = 0, 0
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
            if fraction:
                return __import__("psutil").virtual_memory().percent / 100
        elif self.device.type != "cpu":
            memory = torch.cuda.memory_reserved()
            if fraction:
                total = torch.cuda.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)

    def _clear_memory(self, threshold: float | None = None):
        """通过调用垃圾回收器和清空缓存来清理加速器内存。"""
        if threshold:
            assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."
            if self._get_memory(fraction=True) <= threshold:
                return
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def read_results_csv(self):
        """使用 polars 将 results.csv 读取到字典中。"""
        import polars as pl  # 限制作用域以加快 'import ultralytics' 速度

        try:
            return pl.read_csv(self.csv, infer_schema_length=None).to_dict(as_series=False)
        except Exception:
            return {}

    def _model_train(self):
        """将模型设置为训练模式。"""
        self.model.train()
        # 冻结 BN 统计
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def save_model(self):
        """保存模型训练检查点及附加元数据。"""
        import io

        ema = deepcopy(unwrap_model(self.ema.ema)).half()
        if not all(torch.isfinite(v).all() for v in ema.state_dict().values() if isinstance(v, torch.Tensor)):
            LOGGER.warning(f"Skipping checkpoint save at epoch {self.epoch}: EMA contains NaN/Inf")
            return False

        # 将检查点序列化到字节缓冲区一次（比重复调用 torch.save() 更快）
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # 恢复和最终检查点从 EMA 派生
                "ema": ema,
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),  # 保存为字典
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "git": {
                    "root": str(GIT.root),
                    "branch": GIT.branch,
                    "commit": GIT.commit,
                    "origin": GIT.origin,
                },
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # 获取序列化内容以保存

        # 保存检查点
        self.wdir.mkdir(parents=True, exist_ok=True)  # 确保权重目录存在
        self.last.write_bytes(serialized_ckpt)  # 保存 last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # 保存 best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # 保存 epoch，即 'epoch3.pt'
        return True

    def get_dataset(self):
        """从数据字典中获取训练和验证数据集。

        Returns:
            (dict): 包含训练/验证/测试数据集和类别名称的字典。
        """
        try:
            self.args.data = convert_ndjson_to_yolo_if_needed(self.args.data)

            # 任务特定的数据集检查
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # 用于验证 'yolo train data=url.zip' 用法
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
        return data

    def setup_model(self):
        """加载、创建或下载任意任务的模型。

        Returns:
            (dict | None): 用于恢复训练的检查点，如果没有加载检查点则为 None。
        """
        if isinstance(self.model, torch.nn.Module):  # 如果模型已经提前加载，无需设置
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = load_checkpoint(self.model)
            cfg = weights.yaml
        if isinstance(self.args.pretrained, (str, Path)):
            weights, _ = load_checkpoint(self.args.pretrained)
        elif self.args.pretrained is False and not self.resume:
            weights = None
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK in {-1, 0})  # 调用 Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """执行单步训练优化器，包含梯度裁剪和 EMA 更新。"""
        self.scaler.unscale_(self.optimizer)  # 反缩放梯度
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """允许根据任务类型对模型输入和真实标签进行自定义预处理。"""
        return batch

    def validate(self):
        """使用 self.validator 在验证集上运行验证。

        Returns:
            (tuple): 包含以下内容的元组：
                - metrics (dict | None): 验证指标字典，如果跳过验证则为 None。
                - fitness (float | None): 验证的适应度分数，如果跳过验证则为 None。
        """
        if self.ema and self.world_size > 1:
            # 将 EMA 缓冲区从 rank 0 同步到所有 rank
            for buffer in self.ema.ema.buffers():
                dist.broadcast(buffer, src=0)
        metrics = self.validator(self)
        if metrics is None:
            return None, None
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # 如果未找到则使用损失作为适应度度量
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """获取模型，加载 cfg 文件时抛出 NotImplementedError。"""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """抛出 NotImplementedError（必须由子类实现）。"""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """抛出 NotImplementedError（子类必须返回 `torch.utils.data.DataLoader`）。"""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """构建数据集。"""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """返回带有标签的训练损失项的损失字典，如果 loss_items 为 None 则返回损失名称列表。

        Notes:
            分类任务不需要此方法，但分割和检测任务需要。
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """在训练前设置或更新模型参数。"""
        self.model.names = self.data["names"]

    def set_class_weights(self):
        """计算并设置类别权重以处理类别不平衡。在子类中重写。"""
        pass

    def build_targets(self, preds, targets):
        """构建 YOLO 模型训练的目标张量。"""
        pass

    def progress_string(self):
        """返回描述训练进度的字符串。"""
        return ""

    # TODO: 可能需要将以下函数放入回调中
    def plot_training_samples(self, batch, ni):
        """在 YOLO 训练期间绘制训练样本。"""
        pass

    def plot_training_labels(self):
        """绘制 YOLO 模型的训练标签。"""
        pass

    def save_metrics(self, metrics):
        """将训练指标保存到 CSV 文件。"""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # 列数
        t = time.time() - self.train_time_start
        self.csv.parent.mkdir(parents=True, exist_ok=True)  # 确保父目录存在
        s = "" if self.csv.exists() else ("%s," * n % ("epoch", "time", *keys)).rstrip(",") + "\n"
        with open(self.csv, "a", encoding="utf-8") as f:
            f.write(s + ("%.6g," * n % (self.epoch + 1, t, *vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """从 CSV 文件绘制指标图。"""
        plot_results(file=self.csv, on_plot=self.on_plot)  # 保存 results.png

    def on_plot(self, name, data=None):
        """注册绘图（例如供回调使用）。"""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """对 YOLO 模型执行最终评估和验证。"""
        model = self.best if self.best.exists() else None
        with torch_distributed_zero_first(LOCAL_RANK):  # 仅在 GPU 0 上执行 strip；其他 GPU 等待
            if RANK in {-1, 0}:
                ckpt = strip_optimizer(self.last) if self.last.exists() else {}
                if model:
                    # 从 last.pt 更新 best.pt 的 train_metrics
                    strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
        if model:
            LOGGER.info(f"\nValidating {model}...")
            self.validator.args.plots = self.args.plots
            self.validator.args.compile = False  # 禁用最终验证编译，因为太慢
            self.metrics = self.validator(model=model)
            self.metrics.pop("fitness", None)
            self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """检查恢复检查点是否存在并相应更新参数。"""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())
                ckpt_args = load_checkpoint(last)[0].args
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)  # 恢复模型
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                    "augmentations",
                    "save_period",
                    "workers",
                    "cache",
                    "patience",
                    "time",
                    "freeze",
                    "val",
                    "plots",
                ):  # 允许在恢复时更新参数以减少内存或更新设备
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

                # 处理恢复时的增强参数：检查用户是否提供了自定义增强
                if ckpt_args.get("augmentations") is not None:
                    # 增强在检查点中保存为 repr，但无法自动恢复
                    LOGGER.warning(
                        "Custom Albumentations transforms were used in the original training run but are not "
                        "being restored. To preserve custom augmentations when resuming, you need to pass the "
                        "'augmentations' parameter again to get expected results. Example: \n"
                        f"model.train(resume=True, augmentations={ckpt_args['augmentations']})"
                    )

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def _load_checkpoint_state(self, ckpt):
        """从检查点加载优化器、缩放器、EMA 和 best_fitness。"""
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        if self.ema and ckpt.get("ema"):
            self.ema = ModelEMA(self.model)  # 使用 EMA 验证会创建无法更新的推理张量
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            self.ema.updates = ckpt["updates"]
        self.best_fitness = ckpt.get("best_fitness", 0.0)

    def _handle_nan_recovery(self, epoch):
        """检测并从 NaN/Inf 损失和适应度崩溃中恢复，通过加载最近的检查点。"""
        loss_nan = self.loss is not None and not self.loss.isfinite()
        fitness_nan = self.fitness is not None and not np.isfinite(self.fitness)
        fitness_collapse = self.best_fitness and self.best_fitness > 0 and self.fitness == 0
        corrupted = RANK in {-1, 0} and loss_nan and (fitness_nan or fitness_collapse)
        reason = "Loss NaN/Inf" if loss_nan else "Fitness NaN/Inf" if fitness_nan else "Fitness collapse"
        if RANK != -1:  # DDP: 广播到所有 rank
            broadcast_list = [corrupted if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)
            corrupted = broadcast_list[0]
        if not corrupted:
            return False
        if epoch == self.start_epoch:
            LOGGER.warning(f"{reason} detected but can not recover from last.pt...")
            return False  # 第一个 epoch 无法恢复，让训练继续
        if not self.last.exists():
            raise RuntimeError(f"{reason} detected but no valid last.pt is available for recovery")
        self.nan_recovery_attempts += 1
        if self.nan_recovery_attempts > 3:
            raise RuntimeError(f"Training failed: NaN persisted for {self.nan_recovery_attempts} epochs")
        LOGGER.warning(f"{reason} detected (attempt {self.nan_recovery_attempts}/3), recovering from last.pt...")
        self._model_train()  # 加载检查点前将模型设为训练模式以避免推理张量错误
        _, ckpt = load_checkpoint(self.last)
        ema_state = ckpt["ema"].float().state_dict()
        if not all(torch.isfinite(v).all() for v in ema_state.values() if isinstance(v, torch.Tensor)):
            raise RuntimeError(f"Checkpoint {self.last} is corrupted with NaN/Inf weights")
        unwrap_model(self.model).load_state_dict(ema_state)  # 将 EMA 权重加载到模型中
        self._load_checkpoint_state(ckpt)  # 加载优化器/缩放器/EMA/best_fitness
        del ckpt, ema_state
        self.scheduler.last_epoch = epoch - 1
        return True

    def resume_training(self, ckpt):
        """从给定检查点恢复 YOLO 训练。"""
        if ckpt is None or not self.resume:
            return
        start_epoch = ckpt.get("epoch", -1) + 1
        assert 0 < start_epoch < self.epochs, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # 微调额外的 epoch
        self._load_checkpoint_state(ckpt)
        if getattr(unwrap_model(self.model), "end2end", False):
            # 初始化损失并恢复 o2o 和 o2m 参数
            unwrap_model(self.model).criterion = unwrap_model(self.model).init_criterion()
            unwrap_model(self.model).criterion.updates = start_epoch - 1
            unwrap_model(self.model).criterion.update()
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """更新数据加载器以停止使用 mosaic 增强。"""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """为给定模型构建优化器。

        Args:
            model (torch.nn.Module): 需要构建优化器的模型。
            name (str, optional): 使用的优化器名称。如果为 'auto'，则根据迭代次数自动选择。
            lr (float, optional): 优化器的学习率。
            momentum (float, optional): 优化器的动量因子。
            decay (float, optional): 优化器的权重衰减。
            iterations (float, optional): 迭代次数，当 name 为 'auto' 时决定优化器选择。

        Returns:
            (torch.optim.Optimizer): 构建的优化器。
        """
        g = [{}, {}, {}, {}]  # 优化器参数组
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # 归一化层，即 BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.data.get("nc", 10)  # 类别数
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 拟合方程，精确到 6 位小数
            name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # Adam 不超过 0.01

        use_muon = name == "MuSGD"
        for module_name, module in unwrap_model(model).named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if param.ndim >= 2 and use_muon:
                    g[3][fullname] = param  # muon 参数
                elif "bias" in fullname:  # 偏置（无衰减）
                    g[2][fullname] = param
                elif isinstance(module, bn) or "logit_scale" in fullname:  # 权重（无衰减）
                    # ContrastiveHead 和 BNContrastiveHead 通过 'logit_scale' 包含在此处
                    g[1][fullname] = param
                else:  # 权重（有衰减）
                    g[0][fullname] = param
        if not use_muon:
            g = [x.values() for x in g[:3]]  # 转换为参数列表

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "MuSGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optim_args = dict(lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optim_args = dict(lr=lr, momentum=momentum)
        elif name == "SGD" or name == "MuSGD":
            optim_args = dict(lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        num_params = [len(g[0]), len(g[1]), len(g[2])]  # 参数组数量
        g[2] = {"params": g[2], **optim_args, "param_group": "bias"}
        g[0] = {"params": g[0], **optim_args, "weight_decay": decay, "param_group": "weight"}
        g[1] = {"params": g[1], **optim_args, "weight_decay": 0.0, "param_group": "bn"}
        muon, sgd = (0.2, 1.0)
        if use_muon:
            num_params[0] = len(g[3])  # 更新参数数量
            g[3] = {"params": g[3], **optim_args, "weight_decay": decay, "use_muon": True, "param_group": "muon"}
            import re

            # MuSGD 微调时为某些参数使用更高的学习率
            pattern = re.compile(r"(?=.*23)(?=.*cv3)|proto\.semseg")
            g_ = []  # 新参数组
            for x in g:
                p = x.pop("params")
                p1 = [v for k, v in p.items() if pattern.search(k)]
                p2 = [v for k, v in p.items() if not pattern.search(k)]
                g_.extend([{"params": p1, **x, "lr": lr * 3}, {"params": p2, **x}])
            g = g_
        optimizer = getattr(optim, name, partial(MuSGD, muon=muon, sgd=sgd))(params=g)

        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{num_params[1]} weight(decay=0.0), {num_params[0]} weight(decay={decay}), {num_params[2]} bias(decay=0.0)"
        )
        return optimizer
