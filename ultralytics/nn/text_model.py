# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from ultralytics.utils import checks
from ultralytics.utils.torch_utils import smart_inference_mode

try:
    import clip
except ImportError:
    checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
    import clip


class TextModel(nn.Module):
    """文本编码模型的抽象基类。

    定义了视觉-语言任务中文本编码模型的接口规范。
    子类必须实现 tokenize 和 encode_text 方法，以提供文本分词和编码功能。

    Methods:
        tokenize: 将输入文本转换为模型可处理的 token。
        encode_text: 将 token 编码为归一化的特征向量。
    """

    def __init__(self):
        """初始化 TextModel 基类。"""
        super().__init__()

    @abstractmethod
    def tokenize(self, texts):
        """将输入文本转换为 token，供模型处理。"""
        pass

    @abstractmethod
    def encode_text(self, texts, dtype):
        """将 token 编码为归一化特征向量。"""
        pass


class CLIP(TextModel):
    """实现 OpenAI CLIP（对比语言-图像预训练）文本编码器。

    基于 OpenAI 的 CLIP 模型提供文本编码器，可将文本转换为与图像特征对齐的
    共享嵌入空间中的特征向量。

    Attributes:
        model (clip.model.CLIP): 已加载的 CLIP 模型。
        image_preprocess (callable): 图像预处理变换函数。
        device (torch.device): 模型所在设备。

    Methods:
        tokenize: 将输入文本转换为 CLIP token。
        encode_text: 将 token 编码为归一化特征向量。

    Examples:
        >>> import torch
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> clip_model = CLIP(size="ViT-B/32", device=device)
        >>> tokens = clip_model.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> text_features = clip_model.encode_text(tokens)
        >>> print(text_features.shape)
    """

    def __init__(self, size: str, device: torch.device) -> None:
        """初始化 CLIP 文本编码器。

        实现 TextModel 接口，使用 OpenAI CLIP 模型进行文本编码。
        加载指定大小的预训练 CLIP 模型，并准备进行文本编码任务。

        Args:
            size (str): 模型大小标识符（如 'ViT-B/32'）。
            device (torch.device): 加载模型的目标设备。
        """
        super().__init__()
        self.model, self.image_preprocess = clip.load(size, device=device)
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts: str | list[str], truncate: bool = True) -> torch.Tensor:
        """将输入文本转换为 CLIP token。

        Args:
            texts (str | list[str]): 待分词的文本或文本列表。
            truncate (bool, optional): 是否截断超过 CLIP 上下文长度的文本，默认为 True
                以避免过长输入引发 RuntimeError，同时允许显式关闭此行为。

        Returns:
            (torch.Tensor): 分词后的 token 张量，形状为 (batch_size, context_length)，
                可直接传入模型。

        Examples:
            >>> model = CLIP("ViT-B/32", device="cpu")
            >>> tokens = model.tokenize("a photo of a cat")
            >>> print(tokens.shape)  # torch.Size([1, 77])
            >>> strict_tokens = model.tokenize("a photo of a cat", truncate=False)  # 严格长度检查
            >>> print(strict_tokens.shape)  # 内容与 tokens 相同，因提示词不足 77 个 token
        """
        return clip.tokenize(texts, truncate=truncate).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """将 token 编码为归一化特征向量。

        通过 CLIP 模型处理 token 输入生成特征向量，再进行 L2 归一化。
        归一化后的向量可用于文本-图像相似度计算。

        Args:
            texts (torch.Tensor): token 输入张量，通常由 tokenize() 方法创建。
            dtype (torch.dtype, optional): 输出特征向量的数据类型。

        Returns:
            (torch.Tensor): L2 范数为 1 的归一化文本特征向量。

        Examples:
            >>> clip_model = CLIP("ViT-B/32", device="cuda")
            >>> tokens = clip_model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = clip_model.encode_text(tokens)
            >>> features.shape
            torch.Size([2, 512])
        """
        txt_feats = self.model.encode_text(texts).to(dtype)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        return txt_feats

    @smart_inference_mode()
    def encode_image(self, image: Image.Image | torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """将图像编码为归一化特征向量。

        通过 CLIP 模型处理图像输入生成特征向量，再进行 L2 归一化。
        归一化后的向量可用于文本-图像相似度计算。

        Args:
            image (PIL.Image | torch.Tensor): PIL 图像或预处理后的张量。
                若传入 PIL 图像，将使用模型的图像预处理函数自动转换。
            dtype (torch.dtype, optional): 输出特征向量的数据类型。

        Returns:
            (torch.Tensor): L2 范数为 1 的归一化图像特征向量。

        Examples:
            >>> from ultralytics.nn.text_model import CLIP
            >>> from PIL import Image
            >>> clip_model = CLIP("ViT-B/32", device="cuda")
            >>> image = Image.open("path/to/image.jpg")
            >>> image_tensor = clip_model.image_preprocess(image).unsqueeze(0).to("cuda")
            >>> features = clip_model.encode_image(image_tensor)
            >>> features.shape
            torch.Size([1, 512])
        """
        if isinstance(image, Image.Image):
            image = self.image_preprocess(image).unsqueeze(0).to(self.device)
        img_feats = self.model.encode_image(image).to(dtype)
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        return img_feats


class MobileCLIP(TextModel):
    """实现苹果 MobileCLIP 文本编码器，具备高效文本编码能力。

    使用苹果 MobileCLIP 模型实现 TextModel 接口，与标准 CLIP 相比计算开销更低，
    适用于需要高效推理的视觉-语言任务。

    Attributes:
        model (mobileclip.model.MobileCLIP): 已加载的 MobileCLIP 模型。
        tokenizer (callable): 文本分词函数。
        device (torch.device): 模型所在设备。
        config_size_map (dict): 大小标识符到模型配置名称的映射。

    Methods:
        tokenize: 将输入文本转换为 MobileCLIP token。
        encode_text: 将 token 编码为归一化特征向量。

    Examples:
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> text_encoder = MobileCLIP(size="s0", device=device)
        >>> tokens = text_encoder.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> features = text_encoder.encode_text(tokens)
    """

    config_size_map = {"s0": "s0", "s1": "s1", "s2": "s2", "b": "b", "blt": "b"}

    def __init__(self, size: str, device: torch.device) -> None:
        """初始化 MobileCLIP 文本编码器。

        使用苹果 MobileCLIP 模型实现 TextModel 接口，提供高效文本编码。

        Args:
            size (str): 模型大小标识符（如 's0'、's1'、's2'、'b'、'blt'）。
            device (torch.device): 加载模型的目标设备。
        """
        try:
            import mobileclip
        except ImportError:
            # 优先使用 Ultralytics fork，因为苹果官方 MobileCLIP 仓库的 torchvision 版本不兼容
            checks.check_requirements("git+https://github.com/ultralytics/mobileclip.git")
            import mobileclip

        super().__init__()
        config = self.config_size_map[size]
        file = f"mobileclip_{size}.pt"
        if not Path(file).is_file():
            from ultralytics import download

            download(f"https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/{file}")
        self.model = mobileclip.create_model_and_transforms(f"mobileclip_{config}", pretrained=file, device=device)[0]
        self.tokenizer = mobileclip.get_tokenizer(f"mobileclip_{config}")
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        """将输入文本转换为 MobileCLIP token。

        Args:
            texts (list[str]): 待分词的文本列表。

        Returns:
            (torch.Tensor): 分词后的 token 张量，形状为 (batch_size, sequence_length)。

        Examples:
            >>> model = MobileCLIP("s0", "cpu")
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
        """
        return self.tokenizer(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """将 token 编码为归一化特征向量。

        Args:
            texts (torch.Tensor): token 输入张量。
            dtype (torch.dtype, optional): 输出特征向量的数据类型。

        Returns:
            (torch.Tensor): 经 L2 归一化后的文本特征向量。

        Examples:
            >>> model = MobileCLIP("s0", device="cpu")
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = model.encode_text(tokens)
            >>> features.shape
            torch.Size([2, 512])  # 实际维度取决于模型大小
        """
        text_features = self.model.encode_text(texts).to(dtype)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features


class MobileCLIPTS(TextModel):
    """加载 TorchScript 追踪版本的 MobileCLIP。

    使用苹果 MobileCLIP 模型的 TorchScript 格式实现 TextModel 接口，
    通过优化后的推理性能提供高效文本编码能力。

    Attributes:
        encoder (torch.jit.ScriptModule): 已加载的 TorchScript MobileCLIP 文本编码器。
        tokenizer (callable): 文本分词函数。
        device (torch.device): 模型所在设备。

    Methods:
        tokenize: 将输入文本转换为 MobileCLIP token。
        encode_text: 将 token 编码为归一化特征向量。

    Examples:
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> text_encoder = MobileCLIPTS(device=device)
        >>> tokens = text_encoder.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> features = text_encoder.encode_text(tokens)
    """

    def __init__(self, device: torch.device, weight: str = "mobileclip_blt.ts"):
        """初始化 MobileCLIP TorchScript 文本编码器。

        使用苹果 MobileCLIP 模型的 TorchScript 格式实现 TextModel 接口，
        以优化的推理性能进行高效文本编码。

        Args:
            device (torch.device): 加载模型的目标设备。
            weight (str): TorchScript 模型权重文件路径。
        """
        super().__init__()
        from ultralytics.utils.downloads import attempt_download_asset

        self.encoder = torch.jit.load(attempt_download_asset(weight), map_location=device)
        self.tokenizer = clip.clip.tokenize
        self.device = device

    def tokenize(self, texts: list[str], truncate: bool = True) -> torch.Tensor:
        """将输入文本转换为 MobileCLIP token。

        Args:
            texts (list[str]): 待分词的文本列表。
            truncate (bool, optional): 是否截断超过上下文长度的文本，默认为 True，
                与 CLIP 行为一致，防止长标题导致运行时错误。

        Returns:
            (torch.Tensor): 分词后的 token 张量，形状为 (batch_size, sequence_length)。

        Examples:
            >>> model = MobileCLIPTS(device=torch.device("cpu"))
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> strict_tokens = model.tokenize(
            ...     ["a very long caption"], truncate=False
            ... )  # 若超过 77 个 token 则抛出 RuntimeError
        """
        return self.tokenizer(texts, truncate=truncate).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """将 token 编码为归一化特征向量。

        Args:
            texts (torch.Tensor): token 输入张量。
            dtype (torch.dtype, optional): 输出特征向量的数据类型。

        Returns:
            (torch.Tensor): 经 L2 归一化后的文本特征向量。

        Examples:
            >>> model = MobileCLIPTS(device="cpu")
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = model.encode_text(tokens)
            >>> features.shape
            torch.Size([2, 512])  # 实际维度取决于模型大小
        """
        # 注意：此处无需归一化，归一化已内嵌在 TorchScript 模型中
        return self.encoder(texts).to(dtype)


def build_text_model(variant: str, device: torch.device = None) -> TextModel:
    """根据指定的变体名称构建文本编码模型。

    Args:
        variant (str): 格式为 "base:size" 的模型变体（如 "clip:ViT-B/32" 或 "mobileclip:s0"）。
        device (torch.device, optional): 加载模型的目标设备。

    Returns:
        (TextModel): 实例化的文本编码模型。

    Examples:
        >>> model = build_text_model("clip:ViT-B/32", device=torch.device("cuda"))
        >>> model = build_text_model("mobileclip:s0", device=torch.device("cpu"))
    """
    base, size = variant.split(":")
    if base == "clip":
        return CLIP(size, device)
    elif base == "mobileclip":
        return MobileCLIPTS(device)
    elif base == "mobileclip2":
        return MobileCLIPTS(device, weight="mobileclip2_b.ts")
    else:
        raise ValueError(f"Unrecognized base model '{base}'. Supported models are 'clip', 'mobileclip', 'mobileclip2'.")
