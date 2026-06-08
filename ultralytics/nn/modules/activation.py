# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""激活函数模块。"""

import torch
import torch.nn as nn


class AGLU(nn.Module):
    """来自 AGLU 论文的统一激活函数模块。

    实现了带可学习参数 lambda 和 kappa 的参数化激活函数，
    基于 AGLU（自适应门控线性单元）方法。

    Attributes:
        act (nn.Softplus): 使用负 beta 的 Softplus 激活函数。
        lambd (nn.Parameter): 可学习的 lambda 参数，使用均匀分布初始化。
        kappa (nn.Parameter): 可学习的 kappa 参数，使用均匀分布初始化。

    Methods:
        forward: 计算统一激活函数的前向传播结果。

    Examples:
        >>> import torch
        >>> m = AGLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
        >>> print(output.shape)
        torch.Size([2])

    References:
        https://github.com/kostas1515/AGLU
    """

    def __init__(self, device=None, dtype=None) -> None:
        """初始化统一激活函数，创建可学习参数。"""
        super().__init__()
        self.act = nn.Softplus(beta=-1.0)
        self.lambd = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # lambda 参数
        self.kappa = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # kappa 参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用自适应门控线性单元（AGLU）激活函数。

        使用可学习参数 lambda 和 kappa 实现 AGLU 激活，通过自适应地融合
        线性和非线性分量对输入进行变换。

        Args:
            x (torch.Tensor): 待激活的输入张量。

        Returns:
            (torch.Tensor): 经过 AGLU 激活后的输出张量，形状与输入相同。
        """
        lam = torch.clamp(self.lambd, min=0.0001)  # 对 lambda 进行截断，避免除零错误
        return torch.exp((1 / lam) * self.act((self.kappa * x) - torch.log(lam)))
