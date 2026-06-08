# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from typing import TYPE_CHECKING

from . import USER_CONFIG_DIR
from .torch_utils import TORCH_1_9

if TYPE_CHECKING:
    from ultralytics.engine.trainer import BaseTrainer


def find_free_network_port() -> int:
    """查找本地主机上的空闲端口。

    在单节点训练中非常有用，当我们不想连接到真正的主节点但必须设置
    `MASTER_PORT` 环境变量时。

    返回:
        (int): 可用的网络端口号。
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # 端口号


def generate_ddp_file(trainer: BaseTrainer) -> str:
    """为多GPU训练生成 DDP（分布式数据并行）文件。

    该函数创建一个临时的 Python 文件，用于在多个 GPU 上进行分布式训练。
    该文件包含在分布式环境中初始化训练器所需的配置。

    参数:
        trainer (ultralytics.engine.trainer.BaseTrainer): 包含训练配置和参数的训练器。
            必须具有 args 属性且为类实例。

    返回:
        (str): 生成的临时 DDP 文件的路径。

    注意:
        生成的文件保存在 USER_CONFIG_DIR/DDP 目录中，包含:
        - 训练器类导入
        - 来自训练器参数的配置覆盖
        - 模型路径配置
        - 训练初始化代码
    """
    module, name = f"{trainer.__class__.__module__}.{trainer.__class__.__name__}".rsplit(".", 1)

    # 将增强序列化为 JSON 安全的字典，以避免 DDP 子进程中的 NameError
    overrides = vars(trainer.args).copy()
    if overrides.get("augmentations") is not None:
        import albumentations as A

        overrides["augmentations"] = [A.to_dict(t) for t in overrides["augmentations"]]

    content = f"""
# Ultralytics 多GPU训练临时文件（使用后应自动删除）
from pathlib import Path, PosixPath  # 用于将模型参数以 Path 而非 str 存储
overrides = {overrides}

if __name__ == "__main__":
    from {module} import {name}
    from ultralytics.utils import DEFAULT_CFG_DICT

    # 将增强从字典反序列化为 Albumentations 变换对象
    if overrides.get("augmentations") is not None:
        import albumentations as A
        overrides["augmentations"] = [A.from_dict(t) for t in overrides["augmentations"]]

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # 处理额外的 'save_dir' 键
    trainer = {name}(cfg=cfg, overrides=overrides)
    trainer.args.model = "{getattr(trainer.hub_session, "model_url", trainer.args.model)}"
    results = trainer.train()
"""
    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(trainer)}.py",
        mode="w+",
        encoding="utf-8",
        dir=USER_CONFIG_DIR / "DDP",
        delete=False,
    ) as file:
        file.write(content)
    return file.name


def generate_ddp_command(trainer: BaseTrainer) -> tuple[list[str], str]:
    """生成分布式训练命令。

    参数:
        trainer (ultralytics.engine.trainer.BaseTrainer): 包含分布式训练配置的训练器。

    返回:
        cmd (list[str]): 用于执行分布式训练的命令。
        file (str): 为 DDP 训练创建的临时文件路径。
    """
    import __main__  # noqa 本地导入以避免 https://github.com/Lightning-AI/pytorch-lightning/issues/15218

    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # 删除 save_dir
    file = generate_ddp_file(trainer)
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"
    port = find_free_network_port()
    cmd = [
        sys.executable,
        "-m",
        dist_cmd,
        "--nproc_per_node",
        f"{trainer.world_size}",
        "--master_port",
        f"{port}",
        file,
    ]
    return cmd, file


def ddp_cleanup(trainer: BaseTrainer, file: str) -> None:
    """删除在分布式数据并行（DDP）训练期间创建的临时文件。

    该函数检查提供的文件名中是否包含训练器的 ID，表明它是为 DDP 训练创建的临时文件，
    如果是则删除该文件。

    参数:
        trainer (ultralytics.engine.trainer.BaseTrainer): 用于分布式训练的训练器。
        file (str): 可能需要删除的文件路径。

    示例:
        >>> trainer = YOLOTrainer()
        >>> file = "/tmp/ddp_temp_123456789.py"
        >>> ddp_cleanup(trainer, file)
    """
    if f"{id(trainer)}.py" in file:  # 如果文件名中包含临时文件后缀
        os.remove(file)
