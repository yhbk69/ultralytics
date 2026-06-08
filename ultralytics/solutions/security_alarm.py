# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import colors


class SecurityAlarm(BaseSolution):
    """管理实时监控安全警报功能的类。

    此类扩展了 BaseSolution 类，提供监控帧中目标、在检测目标总数超过特定阈值时发送电子邮件通知，
    以及为可视化标注输出帧的功能。

    属性:
        email_sent (bool): 标记当前事件是否已发送电子邮件的标志。
        records (int): 触发警报的检测目标数量阈值。
        server (smtplib.SMTP): 用于发送电子邮件警报的 SMTP 服务器连接。
        to_email (str): 警报接收者的电子邮件地址。
        from_email (str): 警报发送者的电子邮件地址。

    方法:
        authenticate: 设置用于发送警报的电子邮件服务器身份验证。
        send_email: 发送包含详细信息和图像附件的电子邮件通知。
        process: 监控帧、处理检测结果，并在超过阈值时触发警报。

    示例:
        >>> security = SecurityAlarm()
        >>> security.authenticate("abc@gmail.com", "1111222233334444", "xyz@gmail.com")
        >>> frame = cv2.imread("frame.jpg")
        >>> results = security.process(frame)
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 SecurityAlarm 类，配置实时目标监控的参数。

        参数:
            **kwargs (Any): 传递给父类的额外关键字参数。
        """
        super().__init__(**kwargs)
        self.email_sent = False
        self.records = self.CFG["records"]
        self.server = None
        self.to_email = ""
        self.from_email = ""

    def authenticate(self, from_email: str, password: str, to_email: str) -> None:
        """验证电子邮件服务器以发送警报通知。

        此方法使用提供的凭据初始化与 SMTP 服务器的安全连接并登录。

        参数:
            from_email (str): 发送者的电子邮件地址。
            password (str): 发送者电子邮件账户的密码。
            to_email (str): 接收者的电子邮件地址。

        示例:
            >>> alarm = SecurityAlarm()
            >>> alarm.authenticate("sender@example.com", "password123", "recipient@example.com")
        """
        import smtplib

        self.server = smtplib.SMTP("smtp.gmail.com", 587)
        self.server.starttls()
        self.server.login(from_email, password)
        self.to_email = to_email
        self.from_email = from_email

    def send_email(self, im0, records: int = 5) -> None:
        """发送包含图像附件的电子邮件通知，指示检测到的目标数量。

        此方法对输入图像进行编码，撰写包含检测详细信息的电子邮件消息，并将其发送给指定的接收者。

        参数:
            im0 (np.ndarray): 要附加到电子邮件的输入图像或帧。
            records (int, optional): 要包含在电子邮件消息中的检测目标数量。

        示例:
            >>> alarm = SecurityAlarm()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> alarm.send_email(frame, records=10)
        """
        from email.mime.image import MIMEImage
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        import cv2

        img_bytes = cv2.imencode(".jpg", im0)[1].tobytes()  # 将图像编码为 JPEG

        # 创建电子邮件
        message = MIMEMultipart()
        message["From"] = self.from_email
        message["To"] = self.to_email
        message["Subject"] = "安全警报"

        # 添加文本消息正文
        message_body = f"Ultralytics 警报: 检测到 {records} 个目标。"
        message.attach(MIMEText(message_body))

        # 附加图像
        image_attachment = MIMEImage(img_bytes, name="ultralytics.jpg")
        message.attach(image_attachment)

        # 发送电子邮件
        try:
            self.server.send_message(message)
            LOGGER.info("电子邮件发送成功！")
        except Exception as e:
            LOGGER.error(f"发送电子邮件失败: {e}")

    def process(self, im0) -> SolutionResults:
        """监控帧、处理目标检测，并在满足阈值时触发警报。

        此方法处理输入帧、提取检测结果、用边界框标注帧，如果检测目标数量达到或超过指定阈值
        且尚未发送警报，则发送电子邮件通知。

        参数:
            im0 (np.ndarray): 待处理和标注的输入图像或帧。

        返回:
            (SolutionResults): 包含处理后图像 `plot_im`、'total_tracks'（跟踪目标总数）
                和 'email_sent'（是否触发了电子邮件警报）。

        示例:
            >>> alarm = SecurityAlarm()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> results = alarm.process(frame)
        """
        self.extract_tracks(im0)  # 提取轨迹
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # 初始化标注器

        # 遍历边界框和类别索引
        for box, cls in zip(self.boxes, self.clss):
            # 绘制边界框
            annotator.box_label(box, label=self.names[cls], color=colors(cls, True))

        total_det = len(self.clss)
        if total_det >= self.records and not self.email_sent and self.server:
            self.send_email(im0, total_det)
            self.email_sent = True

        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类函数显示输出

        # 返回 SolutionResults
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids), email_sent=self.email_sent)
