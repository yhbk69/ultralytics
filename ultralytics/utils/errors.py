# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import emojis


class HUBModelError(Exception):
    """当无法从 Ultralytics HUB 找到或获取模型时抛出的异常。

    该自定义异常专门用于处理 Ultralytics YOLO 中模型获取相关的错误。
    错误消息会经过 emoji 处理，以提供更好的用户体验。

    属性:
        message (str): 异常抛出时显示的错误消息。

    方法:
        __init__: 使用自定义消息初始化 HUBModelError。

    示例:
        >>> try:
        ...     # 可能无法找到模型的代码
        ...     raise HUBModelError("Custom model not found message")
        ... except HUBModelError as e:
        ...     print(e)  # 显示带有 emoji 的错误消息
    """

    def __init__(self, message: str = "Model not found. Please check model URL and try again."):
        """初始化 HUBModelError 异常。

        当请求的模型未找到或无法从 Ultralytics HUB 获取时抛出此异常。
        消息会经过 emoji 处理，以提供更好的用户体验。

        参数:
            message (str, 可选): 异常抛出时显示的错误消息。
        """
        super().__init__(emojis(message))
