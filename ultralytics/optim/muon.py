"""
Muon 优化器模块。

本模块实现了 Muon 优化算法及其与 SGD 的混合变体 MuSGD。
Muon 通过牛顿-舒尔茨迭代对更新进行正交化处理，适用于神经网络训练中
2D 及以上维度的参数张量。

主要组件:
    - zeropower_via_newtonschulz5: 牛顿-舒尔茨迭代正交化核心函数
    - muon_update: 带动量与正交化的 Muon 更新计算
    - MuSGD: Muon + SGD 混合优化器
    - Muon: 纯 Muon 优化器（非分布式场景）
"""

from __future__ import annotations

import torch
from torch import optim


def zeropower_via_newtonschulz5(G: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """使用牛顿-舒尔茨迭代计算矩阵 G 的零次幂 / 正交化结果。

    本函数通过五次牛顿-舒尔茨迭代来近似计算输入矩阵 G 的正交化。迭代系数经过优化，
    以最大化零点的收敛斜率，得到的结果近似于 SVD 分解中 G = USV^T 的 UV^T 部分，
    但收敛条件有所放松，在优化场景中经验效果良好。

    Args:
        G (torch.Tensor): 待正交化的输入二维张量/矩阵。
        eps (float, optional): 加在范数上的小量，用于数值稳定性。默认: 1e-7。

    Returns:
        (torch.Tensor): 与输入 G 形状相同的正交化矩阵。

    Examples:
        >>> G = torch.randn(128, 64)
        >>> G_ortho = zeropower_via_newtonschulz5(G)
        >>> print(G_ortho.shape)
        torch.Size([128, 64])

    Notes:
        - 使用 bfloat16 精度进行计算。
        - 执行恰好 5 步牛顿-舒尔茨迭代，系数固定不变。
        - 当行数大于列数时自动转置以提高效率。
        - 输出近似于 US'V^T，其中 S' 的对角线元素在 ~Uniform(0.5, 1.5) 范围内。
        - 并不产生精确的 UV^T，但在神经网络优化中经验效果良好。
    """
    assert len(G.shape) == 2
    X = G.bfloat16()
    X /= X.norm() + eps  # 确保最大奇异值 <= 1
    if G.size(0) > G.size(1):
        X = X.T
    # 固定执行 5 步牛顿-舒尔茨迭代，每步使用相同的优化系数
    for a, b, c in [
        # 原始参数 (a, b, c)，经过优化以最大化收敛速度
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
    ]:
        # 单步牛顿-舒尔茨迭代: X_{k+1} = a*X_k + b*X_k*X_k^T*X_k + c*X_k*X_k^T*X_k*X_k^T*X_k
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def muon_update(grad: torch.Tensor, momentum: torch.Tensor, beta: float = 0.95, nesterov: bool = True) -> torch.Tensor:
    """计算带动量和正交化的 Muon 优化器更新量。

    本函数对梯度施加动量，可选使用 Nesterov 加速，然后通过牛顿-舒尔茨迭代对更新量
    进行正交化。对于卷积滤波器（4D 张量），会在正交化前进行形状重塑，并在最终更新时
    根据参数维度进行缩放。

    Args:
        grad (torch.Tensor): 待更新的梯度张量，可以是 2D 或 4D（用于卷积滤波器）。
        momentum (torch.Tensor): 动量缓冲区张量，通过 lerp 原地修改。
        beta (float, optional): 指数移动平均的动量系数。默认: 0.95。
        nesterov (bool, optional): 是否使用 Nesterov 动量加速。默认: True。

    Returns:
        (torch.Tensor): 与输入 grad 形状相同的正交化更新张量。对于 4D 输入，
            返回重塑回原始维度的结果。

    Examples:
        >>> grad = torch.randn(64, 128)
        >>> momentum = torch.zeros_like(grad)
        >>> update = muon_update(grad, momentum, beta=0.95, nesterov=True)
        >>> print(update.shape)
        torch.Size([64, 128])

    Notes:
        - 动量缓冲区原地更新: momentum = beta * momentum + (1-beta) * grad。
        - 使用 Nesterov: update = beta * momentum + (1-beta) * grad。
        - 不使用 Nesterov: update = momentum。
        - 4D 张量（卷积滤波器）重塑为 2D：(out_channels, in_channels*height*width) 以进行正交化。
        - 最终更新量乘以 sqrt(max(1, dim[-2] / dim[-1])) 以根据参数维度进行缩放。
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # 处理卷积滤波器的情况：将 4D 张量重塑为 2D 以便正交化
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update)
    # 根据参数维度缩放更新量，防止宽矩阵产生过大的更新
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class MuSGD(optim.Optimizer):
    """混合优化器，结合 Muon 和 SGD 更新用于神经网络训练。

    本优化器实现了 Muon（基于动量和牛顿-舒尔茨迭代正交化）与标准带动量 SGD 的组合。
    允许不同参数组使用混合 Muon+SGD 方式或纯 SGD 方式，为不同类型的网络层提供灵活的
    优化策略。

    Args:
        params (Iterable): 待优化的参数或定义参数组的字典。
        muon (float, optional): 混合模式下 Muon 更新的权重因子。默认: 0.5。
        sgd (float, optional): 混合模式下 SGD 更新的权重因子。默认: 0.5。

    Attributes:
        muon (float): 施加于 Muon 学习率的缩放因子。
        sgd (float): 混合模式下施加于 SGD 学习率的缩放因子。

    Examples:
        >>> param_groups = [
        ...     {
        ...         "params": model.conv_params,
        ...         "lr": 0.02,
        ...         "use_muon": True,
        ...         "momentum": 0.95,
        ...         "nesterov": True,
        ...         "weight_decay": 0.01,
        ...     },
        ...     {
        ...         "params": model.other_params,
        ...         "lr": 0.01,
        ...         "use_muon": False,
        ...         "momentum": 0.9,
        ...         "nesterov": False,
        ...         "weight_decay": 0,
        ...     },
        ... ]
        >>> optimizer = MuSGD(param_groups, muon=0.5, sgd=0.5)
        >>> loss = model(data)
        >>> loss.backward()
        >>> optimizer.step()

    Notes:
        - 'use_muon': True 的参数组会同时接收 Muon 和 SGD 更新。
        - 'use_muon': False 的参数组仅接收 SGD 更新。
        - Muon 更新使用正交化，对 2D 及以上维度的参数张量效果最佳。
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        use_muon: bool = False,
        muon: float = 0.5,
        sgd: float = 0.5,
    ):
        """初始化 MuSGD 优化器，具备混合 Muon 和 SGD 的能力。

        Args:
            params (Iterable): 待优化的参数或定义参数组的字典。
            lr (float): 学习率。
            momentum (float): SGD 的动量因子。
            weight_decay (float): 权重衰减（L2 惩罚）。
            nesterov (bool): 是否使用 Nesterov 动量。
            use_muon (bool): 是否启用 Muon 更新。
            muon (float): Muon 分量的缩放因子，控制 Muon 更新的贡献比例。
            sgd (float): SGD 分量的缩放因子，控制 SGD 更新的贡献比例。
        """
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            use_muon=use_muon,
        )
        super().__init__(params, defaults)
        self.muon = muon
        self.sgd = sgd

    @torch.no_grad()
    def step(self, closure=None):
        """执行单步优化。

        根据每个参数组中的 'use_muon' 标志，应用混合 Muon+SGD 更新或纯 SGD 更新。
        对于启用 Muon 的参数组，参数会同时接收正交化的 Muon 更新和标准的 SGD 动量更新，
        两者按各自的缩放因子加权合并。

        Args:
            closure (Callable, optional): 重新评估模型并返回损失的闭包函数。默认: None。

        Returns:
            (torch.Tensor | None): 如果提供了 closure 则返回损失值，否则返回 None。

        Notes:
            - 梯度为 None 的参数会被跳过。
            - Muon 更新使用牛顿-舒尔茨正交化，对 2D 及以上维度的张量效果最佳。
            - 混合模式下权重衰减仅施加于 SGD 分量，Muon 分量不受影响。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Muon + SGD 混合更新分支
            if group["use_muon"]:
                # 以分布式方式生成权重更新
                for p in group["params"]:
                    lr = group["lr"]
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["momentum_buffer_SGD"] = torch.zeros_like(p)

                    # 计算 Muon 正交化更新
                    update = muon_update(
                        grad, state["momentum_buffer"], beta=group["momentum"], nesterov=group["nesterov"]
                    )
                    # 施加 Muon 更新，缩放因子为 muon
                    p.add_(update.reshape(p.shape), alpha=-(lr * self.muon))

                    # 计算 SGD 标准动量更新
                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])
                    state["momentum_buffer_SGD"].mul_(group["momentum"]).add_(grad)
                    sgd_update = (
                        grad.add(state["momentum_buffer_SGD"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer_SGD"]
                    )
                    # 施加 SGD 更新，缩放因子为 sgd
                    p.add_(sgd_update, alpha=-(lr * self.sgd))
            else:  # 纯 SGD 分支
                for p in group["params"]:
                    lr = group["lr"]
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    state["momentum_buffer"].mul_(group["momentum"]).add_(grad)
                    update = (
                        grad.add(state["momentum_buffer"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer"]
                    )
                    p.add_(update, alpha=-lr)
        return loss


class Muon(optim.Optimizer):
    """Muon 优化器，用于非分布式训练场景。

    本优化器实现了 Muon 算法，结合了基于动量的更新和牛顿-舒尔茨迭代正交化。
    在参数更新前以乘法方式施加权重衰减，适用于需要正交化更新的神经网络层
    （如全连接层、卷积层等 2D 及以上维度的参数）。

    Args:
        params (iterable): 待优化的参数或定义参数组的字典。
        lr (float, optional): 学习率。默认: 0.02。
        weight_decay (float, optional): 权重衰减（L2 惩罚）系数。默认: 0。
        momentum (float, optional): 指数移动平均的动量系数。默认: 0.95。

    Attributes:
        param_groups (list): 参数组列表，包含各自的优化设置。
        state (dict): 存储每个参数优化器状态的字典，如动量缓冲区。

    Examples:
        >>> model = YourModel()
        >>> optimizer = Muon(model.parameters(), lr=0.02, weight_decay=0.01, momentum=0.95)
        >>> loss = model(data)
        >>> loss.backward()
        >>> optimizer.step()

    Notes:
        - 专为非分布式训练环境设计。
        - 对所有参数使用带正交化的 Muon 更新。
        - 权重衰减在参数更新前以乘法方式施加。
        - 梯度为 None 的参数会被赋予零梯度以保证同步。
    """

    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0, momentum: float = 0.95):
        """初始化 Muon 优化器，使用基于正交化的更新方式。

        Args:
            params (Iterable): 待优化的参数或定义参数组的字典。
            lr (float): 学习率，用于控制每次参数更新的步长。
            weight_decay (float): 以乘法方式施加的权重衰减因子。
            momentum (float): 梯度累积的动量因子，越接近 1 表示历史梯度影响越大。
        """
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """执行单步优化。

        对所有参数施加 Muon 更新，包含动量累积和正交化处理。
        权重衰减在参数更新前以乘法方式（p *= 1 - lr * weight_decay）施加，
        然后减去按学习率缩放的正交化 Muon 更新量。

        Args:
            closure (Callable[[], torch.Tensor] | None, optional): 重新评估模型并返回损失的闭包函数。
                默认: None。

        Returns:
            (torch.Tensor | None): 如果提供了 closure 则返回损失值，否则返回 None。

        Examples:
            >>> optimizer = Muon(model.parameters())
            >>> loss = model(inputs)
            >>> loss.backward()
            >>> optimizer.step()

        Notes:
            - 梯度为 None 的参数会被赋予零梯度以保证分布式训练中的同步。
            - 权重衰减施加方式: p *= (1 - lr * weight_decay)，即乘法衰减而非加法。
            - Muon 更新使用牛顿-舒尔茨正交化，对 2D 及以上维度的张量效果最佳。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # 不跳过该参数，而是赋予零梯度以强制分布式同步
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                # 计算带动量与正交化的 Muon 更新量
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                # 乘法式权重衰减: p = p * (1 - lr * weight_decay)
                p.mul_(1 - group["lr"] * group["weight_decay"])
                # 施加正交化更新: p = p - lr * update
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss