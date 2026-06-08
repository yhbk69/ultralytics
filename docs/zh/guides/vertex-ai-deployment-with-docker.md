---
comments: true
description: 了解如何使用 Docker 容器和 FastAPI 在 Google Cloud Vertex AI 上部署预训练的 YOLO26 模型，实现可扩展推理，并完全掌控预处理和后处理。
keywords: YOLO26, Vertex AI, Docker, FastAPI, 部署, 容器, GCP, Artifact Registry, Ultralytics, 云部署
---

# 使用 Ultralytics 在 Vertex AI 上部署预训练 YOLO 模型进行推理

本指南将演示如何将预训练的 YOLO26 模型与 Ultralytics 容器化，为其构建 FastAPI 推理服务器，并将模型及推理服务器部署到 Google Cloud Vertex AI 上。示例实现将聚焦于 YOLO26 的目标检测用例，但同样的原则也适用于 [其他 YOLO 模式](../modes/index.md)。

开始之前，你需要创建一个 Google Cloud Platform (GCP) 项目。作为新用户，你可以获得 $300 的 GCP 免费额度，这个额度足以测试一套运行中的部署方案，之后还可以扩展用于其他 YOLO26 用例，包括训练、批量推理和流式推理。

## 你将学到的内容

1. 使用 FastAPI 为 Ultralytics YOLO26 模型创建推理后端。
2. 创建一个 GCP Artifact Registry 仓库来存储 Docker 镜像。
3. 构建包含模型的 Docker 镜像并将其推送到 Artifact Registry。
4. 在 Vertex AI 中导入模型。
5. 创建 Vertex AI 端点并部署模型。

!!! tip "为什么要部署容器化模型？"

    - **通过 Ultralytics 完全掌控模型**：你可以使用自定义推理逻辑，完全控制预处理、后处理和响应格式。
    - **Vertex AI 处理其他一切**：自动扩展，同时灵活配置计算资源、内存和 GPU。
    - **原生 GCP 集成和安全性**：与 Cloud Storage、BigQuery、Cloud Functions、VPC 控制、IAM 策略和审计日志无缝集成。

## 前提条件

1. 在你的机器上安装 [Docker](https://docs.docker.com/engine/install/)。
2. 安装 [Google Cloud SDK](https://docs.cloud.google.com/sdk/docs/install-sdk) 并 [完成 gcloud CLI 身份验证](https://docs.cloud.google.com/docs/authentication/gcloud)。
3. 强烈建议先阅读 [Ultralytics Docker 快速入门指南](https://docs.ultralytics.com/guides/docker-quickstart)，因为在本指南中你需要扩展一个 Ultralytics 官方 Docker 镜像。

## 1. 使用 FastAPI 创建推理后端

首先，你需要创建一个 FastAPI 应用程序来处理 YOLO26 模型的推理请求。该应用程序将负责模型加载、图像预处理和推理（预测）逻辑。

### Vertex AI 合规基础

Vertex AI 要求你的容器实现两个特定端点：

1. **健康检查**端点（`/health`）：服务就绪时必须返回 HTTP 状态码 `200 OK`。
2. **预测**端点（`/predict`）：接收包含 **base64 编码**图像和可选参数的结构化预测请求。根据端点类型的不同，有相应的[载荷大小限制](https://docs.cloud.google.com/vertex-ai/docs/predictions/choose-endpoint-type)。

    `/predict` 端点的请求载荷应遵循以下 JSON 结构：

    ```json
    {
        "instances": [{ "image": "base64_encoded_image" }],
        "parameters": { "confidence": 0.5 }
    }
    ```

### 项目文件夹结构

我们的大部分构建将在 Docker 容器内进行，Ultralytics 也会加载预训练的 YOLO26 模型，因此本地文件夹结构可以保持简洁：

```txt
YOUR_PROJECT/
├── src/
│   ├── __init__.py
│   ├── app.py              # 核心 YOLO26 推理逻辑
│   └── main.py             # FastAPI 推理服务器
├── tests/
├── .env                    # 本地开发环境变量
├── Dockerfile              # 容器配置
├── LICENSE                 # AGPL-3.0 许可证
└── pyproject.toml          # Python 依赖和项目配置
```

!!! note "重要许可证说明"

    Ultralytics YOLO26 模型和框架基于 AGPL-3.0 许可，有重要的合规要求。请务必阅读 Ultralytics 文档中关于 [如何遵守许可条款](../help/contributing.md#how-to-comply-with-agpl-30) 的内容。

### 创建包含依赖的 pyproject.toml

为方便管理项目，创建一个包含以下依赖的 `pyproject.toml` 文件：

```toml
[project]
name = "YOUR_PROJECT_NAME"
version = "0.0.1"
description = "YOUR_PROJECT_DESCRIPTION"
requires-python = ">=3.10,<3.13"
dependencies = [
   "ultralytics>=8.3.0",
   "fastapi[all]>=0.89.1",
   "uvicorn[standard]>=0.20.0",
   "pillow>=9.0.0",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

- `uvicorn` 将用于运行 FastAPI 服务器。
- `pillow` 将用于图像处理，但不仅限于 PIL 图像——Ultralytics 支持 [多种其他格式](../modes/predict.md#inference-sources)。

### 使用 Ultralytics YOLO26 创建推理逻辑

现在项目结构和依赖已经就绪，你可以使用 Ultralytics Python API 实现核心的 YOLO26 推理逻辑。创建一个 `src/app.py` 文件来处理模型加载、图像处理和预测。

```python
# src/app.py

from ultralytics import YOLO

# 模型初始化和就绪状态
model_yolo = None
_model_ready = False


def _initialize_model():
    """初始化 YOLO 模型。"""
    global model_yolo, _model_ready

    try:
        # 使用 Ultralytics 基础镜像中的预训练 YOLO26n 模型
        model_yolo = YOLO("yolo26n.pt")
        _model_ready = True

    except Exception as e:
        print(f"初始化 YOLO 模型时出错: {e}")
        _model_ready = False
        model_yolo = None


# 模块导入时初始化模型
_initialize_model()


def is_model_ready() -> bool:
    """检查模型是否已准备好进行推理。"""
    return _model_ready and model_yolo is not None
```

这样在容器启动时会一次性加载模型，所有请求将共享该模型。如果你的模型需要处理较大的推理负载，建议在后续步骤向 Vertex AI 导入模型时选择内存更大的机器类型。

接下来，使用 `pillow` 创建两个用于输入和输出图像处理的工具函数。YOLO26 原生支持 PIL 图像。

```python
def get_image_from_bytes(binary_image: bytes) -> Image.Image:
    """将字节图像转换为 PIL RGB 格式。"""
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image
```

```python
def get_bytes_from_image(image: Image.Image) -> bytes:
    """将 PIL 图像转换为字节。"""
    return_image = io.BytesIO()
    image.save(return_image, format="JPEG", quality=85)
    return_image.seek(0)
    return return_image.getvalue()
```

最后，实现 `run_inference` 函数来处理目标检测。在本示例中，我们将从模型预测结果中提取边界框、类别名称和置信度分数。该函数将返回一个包含检测结果和原始数据的字典，用于进一步处理或标注。

```python
def run_inference(input_image: Image.Image, confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """使用 YOLO26n 模型对图像进行推理。"""
    global model_yolo

    # 检查模型是否就绪
    if not is_model_ready():
        print("模型尚未准备好进行推理")
        return {"detections": [], "results": None}

    try:
        # 进行预测并获取原始结果
        results = model_yolo.predict(
            imgsz=640, source=input_image, conf=confidence_threshold, save=False, augment=False, verbose=False
        )

        # 提取检测结果（边界框、类别名称和置信度）
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes.xyxy) > 0:
                boxes = result.boxes

                # 将张量转换为 numpy 以便处理
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)

                # 创建检测字典
                for i in range(len(xyxy)):
                    detection = {
                        "xmin": float(xyxy[i][0]),
                        "ymin": float(xyxy[i][1]),
                        "xmax": float(xyxy[i][2]),
                        "ymax": float(xyxy[i][3]),
                        "confidence": float(conf[i]),
                        "class": int(cls[i]),
                        "name": model_yolo.names.get(int(cls[i]), f"class_{int(cls[i])}"),
                    }
                    detections.append(detection)

        return {
            "detections": detections,
            "results": results,  # 保留原始结果用于标注
        }
    except Exception as e:
        # 如果出错，返回空结构
        print(f"YOLO 检测出错: {e}")
        return {"detections": [], "results": None}
```

可选地，你可以添加一个函数，使用 Ultralytics 内置的绘图方法对图像进行边界框和标签标注。如果你想在预测响应中返回标注后的图像，这将很有用。

```python
def get_annotated_image(results: list) -> Image.Image:
    """使用 Ultralytics 内置的 plot 方法获取标注图像。"""
    if not results or len(results) == 0:
        raise ValueError("没有可用于标注的结果")

    result = results[0]
    # 使用 Ultralytics 内置的 plot 方法，输出 PIL 格式
    return result.plot(pil=True)
```

### 使用 FastAPI 创建 HTTP 推理服务器

现在核心 YOLO26 推理逻辑已经就绪，你可以创建一个 FastAPI 应用程序来提供服务。这将包括 Vertex AI 所需的健康检查和预测端点。

首先，添加导入并配置 Vertex AI 的日志记录。由于 Vertex AI 将 stderr 视为错误输出，将日志输出到 stdout 更为合理。

```python
import sys

from loguru import logger

# 配置日志记录器
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")
```

为实现完整的 Vertex AI 合规性，在环境变量中定义所需端点并设置请求大小限制。对于生产部署，建议使用 [私有 Vertex AI 端点](https://docs.cloud.google.com/vertex-ai/docs/predictions/choose-endpoint-type)。这样你将获得更高的请求载荷限制（私有端点 10 MB，而公共端点仅为 1.5 MB），同时享有可靠的安全性和访问控制。

```python
# Vertex AI 环境变量
AIP_HTTP_PORT = int(os.getenv("AIP_HTTP_PORT", "8080"))
AIP_HEALTH_ROUTE = os.getenv("AIP_HEALTH_ROUTE", "/health")
AIP_PREDICT_ROUTE = os.getenv("AIP_PREDICT_ROUTE", "/predict")

# 请求大小限制（私有端点 10 MB，公共端点 1.5 MB）
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10 MB（字节）
```

添加两个 Pydantic 模型来验证请求和响应：

```python
# 请求/响应的 Pydantic 模型
class PredictionRequest(BaseModel):
    instances: list
    parameters: Optional[Dict[str, Any]] = None


class PredictionResponse(BaseModel):
    predictions: list
```

添加健康检查端点来验证模型就绪状态。**这对 Vertex AI 非常重要**，因为如果没有专门的健康检查，其编排器会 ping 随机套接字，无法判断模型是否已准备好进行推理。你的检查必须在成功时返回 `200 OK`，失败时返回 `503 Service Unavailable`：

```python
# 健康检查端点
@app.get(AIP_HEALTH_ROUTE, status_code=status.HTTP_200_OK)
def health_check():
    """Vertex AI 健康检查端点。"""
    if not is_model_ready():
        raise HTTPException(status_code=503, detail="模型未就绪")
    return {"status": "healthy"}
```

现在你已经具备了实现预测端点所需的一切。该端点将接收图像文件，运行推理并返回结果。请注意，图像必须使用 base64 编码，这会额外增加最多 33% 的载荷大小。

```python
@app.post(AIP_PREDICT_ROUTE, response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Vertex AI 预测端点。"""
    try:
        predictions = []

        for instance in request.instances:
            if isinstance(instance, dict):
                if "image" in instance:
                    image_data = base64.b64decode(instance["image"])
                    input_image = get_image_from_bytes(image_data)
                else:
                    raise HTTPException(status_code=400, detail="实例必须包含 'image' 字段")
            else:
                raise HTTPException(status_code=400, detail="无效的实例格式")

            # 提取 YOLO26 参数（如有提供）
            parameters = request.parameters or {}
            confidence_threshold = parameters.get("confidence", 0.5)
            return_annotated_image = parameters.get("return_annotated_image", False)

            # 使用 YOLO26n 模型运行推理
            result = run_inference(input_image, confidence_threshold=confidence_threshold)
            detections_list = result["detections"]

            # 为 Vertex AI 格式化预测结果
            detections = []
            for detection in detections_list:
                formatted_detection = {
                    "class": detection["name"],
                    "confidence": detection["confidence"],
                    "bbox": {
                        "xmin": detection["xmin"],
                        "ymin": detection["ymin"],
                        "xmax": detection["xmax"],
                        "ymax": detection["ymax"],
                    },
                }
                detections.append(formatted_detection)

            # 构建预测响应
            prediction = {"detections": detections, "detection_count": len(detections)}

            # 如果请求且存在检测结果，则添加标注图像
            if (
                return_annotated_image
                and result["results"]
                and result["results"][0].boxes is not None
                and len(result["results"][0].boxes) > 0
            ):
                import base64

                annotated_image = get_annotated_image(result["results"])
                img_bytes = get_bytes_from_image(annotated_image)
                prediction["annotated_image"] = base64.b64encode(img_bytes).decode("utf-8")

            predictions.append(prediction)

        logger.info(
            f"已处理 {len(request.instances)} 个实例，共发现 {sum(len(p['detections']) for p in predictions)} 个检测目标"
        )

        return PredictionResponse(predictions=predictions)

    except HTTPException:
        # 原样重新抛出 HTTPException（不要捕获并转换为 500）
        raise
    except Exception as e:
        logger.error(f"预测错误: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {e}")
```

最后，添加应用程序入口点来运行 FastAPI 服务器。

```python
if __name__ == "__main__":
    import uvicorn

    logger.info(f"服务器启动于端口 {AIP_HTTP_PORT}")
    logger.info(f"健康检查路由: {AIP_HEALTH_ROUTE}")
    logger.info(f"预测路由: {AIP_PREDICT_ROUTE}")
    uvicorn.run(app, host="0.0.0.0", port=AIP_HTTP_PORT)
```

现在你已经拥有一个完整的 FastAPI 应用程序，可以为 YOLO26 推理请求提供服务。你可以通过安装依赖并运行服务器在本地进行测试，例如使用 uv。

```bash
# 安装依赖
uv pip install -e .

# 直接运行 FastAPI 服务器
uv run src/main.py
```

要测试服务器，你可以使用 cURL 查询 `/health` 和 `/predict` 端点。将测试图像放入 `tests` 文件夹中，然后在终端中运行以下命令：

```bash
# 测试健康检查端点
curl http://localhost:8080/health

# 使用 base64 编码图像测试预测端点
curl -X POST -H "Content-Type: application/json" -d "{\"instances\": [{\"image\": \"$(base64 -i tests/test_image.jpg)\"}]}" http://localhost:8080/predict
```

你应该收到一个包含检测目标的 JSON 响应。首次请求时可能会稍有延迟，因为 Ultralytics 需要拉取和加载 YOLO26 模型。

## 2. 使用你的应用程序扩展 Ultralytics Docker 镜像

Ultralytics 提供了多个 Docker 镜像，可以用作应用程序镜像的基础镜像。Docker 会安装 Ultralytics 以及必要的 GPU 驱动程序。

要充分利用 Ultralytics YOLO 模型的能力，你应该选择针对 CUDA 优化的镜像进行 GPU 推理。不过，如果 CPU 推理足以满足你的任务需求，你也可以选择仅 CPU 镜像以节省计算资源：

- [Dockerfile](https://github.com/ultralytics/ultralytics/blob/main/docker/Dockerfile)：针对 CUDA 优化的 YOLO26 单/多 GPU 训练和推理镜像。
- [Dockerfile-cpu](https://github.com/ultralytics/ultralytics/blob/main/docker/Dockerfile-cpu)：仅 CPU 的 YOLO26 推理镜像。

### 为你的应用程序创建 Docker 镜像

在项目根目录创建一个 `Dockerfile`，内容如下：

```dockerfile
# 扩展 Ultralytics 官方 YOLO26 Docker 镜像
FROM ultralytics/ultralytics:latest

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 安装 FastAPI 和依赖
RUN uv pip install fastapi[all] uvicorn[standard] loguru

WORKDIR /app
COPY src/ ./src/
COPY pyproject.toml ./

# 安装应用程序包
RUN uv pip install -e .

RUN mkdir -p /app/logs
ENV PYTHONPATH=/app/src

# Vertex AI 端口
EXPOSE 8080

# 启动推理服务器
ENTRYPOINT ["python", "src/main.py"]
```

示例中使用了 Ultralytics 官方 Docker 镜像 `ultralytics:latest` 作为基础镜像。它已经包含了 YOLO26 模型和所有必要的依赖。服务器的入口点与本地测试 FastAPI 应用程序时使用的相同。

### 构建和测试 Docker 镜像

现在你可以使用以下命令构建 Docker 镜像：

```bash
docker build --platform linux/amd64 -t IMAGE_NAME:IMAGE_VERSION .
```

将 `IMAGE_NAME` 和 `IMAGE_VERSION` 替换为你期望的值，例如 `yolo26-fastapi:0.1`。请注意，如果你要部署到 Vertex AI，必须为 `linux/amd64` 架构构建镜像。如果你在 Apple Silicon Mac 或其他非 x86 架构上构建镜像，需要显式设置 `--platform` 参数。

镜像构建完成后，你可以在本地测试 Docker 镜像：

```bash
docker run --platform linux/amd64 -p 8080:8080 IMAGE_NAME:IMAGE_VERSION
```

你的 Docker 容器现在在端口 `8080` 上运行 FastAPI 服务器，随时可以接收推理请求。你可以使用与之前相同的 cURL 命令测试 `/health` 和 `/predict` 端点：

```bash
# 测试健康检查端点
curl http://localhost:8080/health

# 使用 base64 编码图像测试预测端点
curl -X POST -H "Content-Type: application/json" -d "{\"instances\": [{\"image\": \"$(base64 -i tests/test_image.jpg)\"}]}" http://localhost:8080/predict
```

## 3. 将 Docker 镜像上传到 GCP Artifact Registry

要在 Vertex AI 中导入容器化模型，你需要将 Docker 镜像上传到 Google Cloud Artifact Registry。如果你还没有 Artifact Registry 仓库，需要先创建一个。

### 在 Google Cloud Artifact Registry 中创建仓库

在 Google Cloud Console 中打开 [Artifact Registry 页面](https://console.cloud.google.com/artifacts)。如果你是首次使用 Artifact Registry，可能会提示你先启用 Artifact Registry API。

<p align="center">
  <img width="70%" src="https://github.com/lussebullar/temp-image-storage/releases/download/docs/create-artifact-registry-repo.png" alt="Google Cloud Artifact Registry 仓库创建">
</p>

1. 选择"创建仓库"（Create Repository）。
2. 输入仓库名称。选择所需区域，其他选项使用默认设置，除非你有特殊需求需要更改。

!!! note

    区域选择可能会影响机器的可用性以及非企业用户的某些计算限制。你可以在 Vertex AI 官方文档中找到更多信息：[Vertex AI 配额和限制](https://docs.cloud.google.com/vertex-ai/docs/quotas)

1. 仓库创建完成后，将你的 PROJECT_ID、Location（区域）和 Repository Name 保存到你的密钥保管库或 `.env` 文件中。稍后你需要使用它们来标记 Docker 镜像并将其推送到 Artifact Registry。

### 配置 Docker 对 Artifact Registry 的身份验证

将 Docker 客户端认证到你刚刚创建的 Artifact Registry 仓库。在终端中运行以下命令：

```sh
gcloud auth configure-docker YOUR_REGION-docker.pkg.dev
```

### 标记镜像并将其推送到 Artifact Registry

标记 Docker 镜像并将其推送到 Google Artifact Registry。

!!! note "为镜像使用唯一标签"

    建议每次更新镜像时使用唯一的标签。大多数 GCP 服务，包括 Vertex AI，都依赖镜像标签进行自动版本控制和扩展，因此使用语义化版本或基于日期的标签是一个好习惯。

使用 Artifact Registry 仓库 URL 标记你的镜像。将占位符替换为你之前保存的值。

```sh
docker tag IMAGE_NAME:IMAGE_VERSION YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPOSITORY_NAME/IMAGE_NAME:IMAGE_VERSION
```

将标记后的镜像推送到 Artifact Registry 仓库。

```sh
docker push YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPOSITORY_NAME/IMAGE_NAME:IMAGE_VERSION
```

等待推送过程完成。你应该可以在 Artifact Registry 仓库中看到你的镜像。

关于如何在 Artifact Registry 中操作镜像的更多具体说明，请参阅 Artifact Registry 文档：[推送和拉取镜像](https://docs.cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling)。

## 4. 在 Vertex AI 中导入模型

使用刚刚推送的 Docker 镜像，你现在可以在 Vertex AI 中导入模型。

1. 在 Google Cloud 导航菜单中，转到 Vertex AI > Model Registry。或者，在 Google Cloud Console 顶部的搜索栏中搜索 "Vertex AI"。
 <p align="center">
   <img width="80%" src="https://github.com/lussebullar/temp-image-storage/releases/download/docs/vertex-ai-import.png" alt="Vertex AI Model Registry 导入界面">
 </p>
1. 点击"导入"（Import）。
1. 选择"作为新模型导入"（Import as a new model）。
1. 选择区域。你可以选择与 Artifact Registry 仓库相同的区域，但具体选择应取决于你所在区域机器类型和配额的可用性。
1. 选择"导入现有模型容器"（Import an existing model container）。
 <p align="center">
   <img width="80%" src="https://github.com/lussebullar/temp-image-storage/releases/download/docs/import-model.png" alt="Vertex AI 导入模型对话框">
 </p>
1. 在"容器镜像"（Container image）字段中，浏览之前创建的 Artifact Registry 仓库，选择刚刚推送的镜像。
1. 向下滚动到"环境变量"（Environment variables）部分，输入你在 FastAPI 应用程序中定义的预测端点、健康检查端点和端口。
 <p align="center">
   <img width="60%" src="https://github.com/lussebullar/temp-image-storage/releases/download/docs/predict-health-port.png" alt="Vertex AI 环境变量配置">
 </p>
1. 点击"导入"（Import）。Vertex AI 将花费几分钟时间来注册模型并准备部署。导入完成后，你将收到电子邮件通知。

## 5. 创建 Vertex AI 端点并部署模型

!!! note "Vertex AI 中的端点与模型"

    在 Vertex AI 术语中，**端点（endpoints）**指的是**已部署**的模型，因为它们代表发送推理请求的 HTTP 端点，而**模型（models）**是存储在 Model Registry 中已训练好的机器学习工件。

要部署模型，你需要在 Vertex AI 中创建一个端点。

1.  在 Vertex AI 导航菜单中，转到"端点"（Endpoints）。选择导入模型时使用的区域。点击"创建"（Create）。
<p align="center">
  <img width="60%" src="https://github.com/lussebullar/temp-image-storage/releases/download/docs/endpoint-name.png" alt="Vertex AI 创建端点界面">
</p>
1.  输入端点名称。
1.  对于访问方式，Vertex AI 建议使用私有 Vertex AI 端点。除了安全优势外，选择私有端点还可以获得更高的载荷限制，但你需要配置 VPC 网络和防火墙规则以允许访问端点。请参阅 Vertex AI 文档中关于 [私有端点](https://docs.cloud.google.com/vertex-ai/docs/predictions/choose-endpoint-type) 的更多说明。
1.  点击"继续"（Continue）。
1.  在"模型设置"（Model settings）对话框中，选择之前导入的模型。现在你可以为模型配置机器类型、内存和 GPU 设置。如果预计推理负载较高，请确保分配足够的内存，避免 I/O 瓶颈，确保 YOLO26 的正常性能。
1.  在"加速器类型"（Accelerator type）中，选择用于推理的 GPU 类型。如果不确定选择哪个 GPU，可以从 NVIDIA T4 开始，它支持 CUDA。

    !!! note "区域和机器类型配额"

        请注意，某些区域的计算配额非常有限，因此你可能无法在所在区域选择某些机器类型或 GPU。如果这一点很重要，可以将部署区域更改为配额更大的区域。更多信息请参阅 Vertex AI 官方文档：[Vertex AI 配额和限制](https://docs.cloud.google.com/vertex-ai/docs/quotas)。

1.  选择机器类型后，可以点击"继续"（Continue）。此时，你可以选择在 Vertex AI 中启用模型监控——一项额外的服务，用于跟踪模型性能并提供行为洞察。此功能是可选的，会产生额外费用，请根据需求选择。点击"创建"（Create）。

Vertex AI 将花费几分钟时间（在某些区域最多 30 分钟）来部署模型。部署完成后，你将收到电子邮件通知。

## 6. 测试已部署的模型

部署完成后，Vertex AI 将为你提供一个示例 API 界面来测试模型。

要测试远程推理，你可以使用提供的 cURL 命令，或创建另一个 Python 客户端库向已部署的模型发送请求。请记住，在发送到 `/predict` 端点之前，需要将图像编码为 base64。

<p align="center">
  <img width="50%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/vertex-ai-endpoint-test-curl-yolo11.avif" alt="使用 cURL 测试 Vertex AI 端点">
</p>

!!! note "首次请求可能稍有延迟"

    与本地测试类似，首次请求时可能会稍有延迟，因为 Ultralytics 需要在运行中的容器内拉取和加载 YOLO26 模型。

你已经成功使用 Ultralytics 在 Google Cloud Vertex AI 上部署了预训练的 YOLO26 模型。

## 常见问题

### 我可以在不依赖 Docker 的情况下在 Vertex AI 上使用 Ultralytics YOLO 模型吗？

可以；但是，你首先需要将模型导出为 Vertex AI 兼容的格式，例如 TensorFlow、Scikit-learn 或 XGBoost。Google Cloud 提供了一份关于在 Vertex 上运行 `.pt` 模型的指南，其中全面概述了转换过程：[在 Vertex AI 上运行 PyTorch 模型](https://cloud.google.com/blog/topics/developers-practitioners/pytorch-google-cloud-how-deploy-pytorch-models-vertex-ai)。

请注意，最终设置将仅依赖 Vertex AI 标准服务层，不支持高级 Ultralytics 框架功能。由于 Vertex AI 完全支持容器化模型，并可根据你的部署配置自动扩展，因此你可以充分利用 Ultralytics YOLO 模型的全部能力，而无需将其转换为其他格式。

### 为什么 FastAPI 是服务 YOLO26 推理的好选择？

FastAPI 为推理工作负载提供了高吞吐量。异步支持允许在不阻塞主线程的情况下处理多个并发请求，这对计算机视觉模型的服务至关重要。

FastAPI 的自动请求/响应验证可减少生产推理服务中的运行时错误。这对于输入格式一致性很重要的目标检测 API 尤其有价值。

FastAPI 为推理管道增加的计算开销最小，为模型执行和图像处理任务留出了更多资源。

FastAPI 还支持 SSE（Server-Sent Events），这对于流式推理场景很有用。

### 为什么我需要多次选择区域？

这实际上是 Google Cloud Platform 的一项灵活性特性，你需要为使用的每个服务选择一个区域。对于在 Vertex AI 上部署容器化模型这一任务，最重要的区域选择是 Model Registry 的区域。它将决定模型部署可用的机器类型和配额。

此外，如果你要扩展部署方案并将预测数据或结果存储在 Cloud Storage 或 BigQuery 中，需要使用与 Model Registry 相同的区域，以最大程度减少延迟并确保数据访问的高吞吐量。
