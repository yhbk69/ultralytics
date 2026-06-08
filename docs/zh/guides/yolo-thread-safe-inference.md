---
comments: true
description: 了解如何在 Python 中确保 YOLO 模型推理的线程安全。通过最佳实践避免竞态条件，可靠地运行多线程任务。
keywords: YOLO 模型, 线程安全, Python 线程, 模型推理, 并发, 竞态条件, 多线程, 并行, Python GIL
---

# YOLO 模型线程安全推理

在多线程环境中运行 YOLO 模型需要特别注意以确保线程安全。Python 的 `threading` 模块允许你同时运行多个线程，但在多个线程中使用 YOLO 模型时，存在重要的安全问题需要关注。本页面将指导你创建线程安全的 YOLO 模型推理。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/jMbvN6uCIos"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何使用 Ultralytics YOLO 模型在 Python 中执行线程安全推理 | 多线程 🚀
</p>

## 理解 Python 线程

Python 线程是一种并行形式，允许你的程序同时运行多个操作。然而，Python 的全局解释器锁（GIL）意味着同一时间只有一个线程可以执行 Python 字节码。

<p align="center">
  <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/single-vs-multi-thread-examples.avif" alt="单线程 vs 多线程推理">
</p>

虽然这听起来像是一种限制，但线程仍然可以提供并发性，特别是对于 I/O 密集型操作，或者当使用会释放 GIL 的操作时（如 YOLO 底层 C 库执行的操作）。

## 共享模型实例的危险

在线程外部实例化 YOLO 模型并在多个线程之间共享该实例可能导致[竞态条件](https://www.ultralytics.com/glossary/algorithmic-bias)，即模型的内部状态因并发访问而被不一致地修改。当模型或其组件持有的状态并非设计为线程安全时，这一问题尤为严重。

### 非线程安全示例：单个模型实例

在 Python 中使用线程时，识别可能导致并发问题的模式非常重要。以下是你应该避免的做法：在多个线程之间共享单个 YOLO 模型实例。

```python
# 不安全：在多个线程之间共享单个模型实例
from threading import Thread

from ultralytics import YOLO

# 在线程外部实例化模型
shared_model = YOLO("yolo26n.pt")


def predict(image_path):
    """使用预加载的 YOLO 模型对图像进行目标预测，接收图像路径字符串作为参数。"""
    results = shared_model.predict(image_path)
    # 处理结果


# 启动共享同一模型实例的线程
Thread(target=predict, args=("image1.jpg",)).start()
Thread(target=predict, args=("image2.jpg",)).start()
```

在上面的示例中，`shared_model` 被多个线程使用，这可能导致不可预测的结果，因为 `predict` 可能被多个线程同时执行。

### 非线程安全示例：多个模型实例

同样，以下是一个使用多个 YOLO 模型实例的不安全模式：

```python
# 不安全：在多个线程之间共享多个模型实例仍可能导致问题
from threading import Thread

from ultralytics import YOLO

# 在线程外部实例化多个模型
shared_model_1 = YOLO("yolo26n_1.pt")
shared_model_2 = YOLO("yolo26n_2.pt")


def predict(model, image_path):
    """使用指定的 YOLO 模型对图像进行预测，返回结果。"""
    results = model.predict(image_path)
    # 处理结果


# 启动带有各自模型实例的线程
Thread(target=predict, args=(shared_model_1, "image1.jpg")).start()
Thread(target=predict, args=(shared_model_2, "image2.jpg")).start()
```

即使存在两个独立的模型实例，并发问题的风险仍然存在。如果 `YOLO` 的内部实现不是线程安全的，使用独立实例可能无法防止竞态条件，特别是当这些实例共享任何非线程局部的底层资源或状态时。

## 线程安全推理

要执行线程安全的推理，你应该在每个线程内部实例化一个独立的 YOLO 模型。这确保了每个线程都有自己隔离的模型实例，消除了竞态条件的风险。

### 线程安全示例

以下是如何在每个线程内部实例化 YOLO 模型以实现安全的并行推理：

```python
# 安全：在每个线程内部实例化单个模型
from threading import Thread

from ultralytics import YOLO


def thread_safe_predict(image_path):
    """以线程安全的方式使用新的 YOLO 模型实例对图像进行预测；接收图像路径作为输入。"""
    local_model = YOLO("yolo26n.pt")
    results = local_model.predict(image_path)
    # 处理结果


# 启动各自拥有独立模型实例的线程
Thread(target=thread_safe_predict, args=("image1.jpg",)).start()
Thread(target=thread_safe_predict, args=("image2.jpg",)).start()
```

在此示例中，每个线程创建自己的 `YOLO` 实例。这防止了任何线程干扰另一个线程的模型状态，从而确保每个线程安全地执行推理，而不会与其他线程产生意外的交互。

## 使用 ThreadingLocked 装饰器

Ultralytics 提供了一个 `ThreadingLocked` 装饰器，可用于确保函数的线程安全执行。该装饰器使用锁来确保同一时间只有一个线程可以执行被装饰的函数。

```python
from ultralytics import YOLO
from ultralytics.utils import ThreadingLocked

# 创建一个模型实例
model = YOLO("yolo26n.pt")


# 装饰 predict 方法使其线程安全
@ThreadingLocked()
def thread_safe_predict(image_path):
    """使用共享模型实例进行线程安全预测。"""
    results = model.predict(image_path)
    return results


# 现在你可以安全地从多个线程调用此函数
```

当你需要在多个线程之间共享一个模型实例但又希望确保同一时间只有一个线程可以访问它时，`ThreadingLocked` 装饰器特别有用。与为每个线程创建新的模型实例相比，这种方法可以节省内存，但可能会降低并发性，因为线程需要等待锁被释放。

## 结论

在 Python 的 `threading` 中使用 YOLO 模型时，始终在使用模型的线程内部实例化模型，以确保线程安全。这一实践避免了竞态条件，并确保你的推理任务可靠运行。

对于更高级的场景以及进一步优化多线程推理性能，可以考虑使用基于进程的并行方案，如 [multiprocessing](https://docs.python.org/3/library/multiprocessing.html)，或者利用带有专用工作进程的任务队列。

## 常见问题

### 如何在多线程 Python 环境中避免使用 YOLO 模型时的竞态条件？

要在多线程 Python 环境中防止 Ultralytics YOLO 模型的竞态条件，需要在每个线程内部实例化一个独立的 YOLO 模型。这确保每个线程拥有自己隔离的模型实例，避免模型状态的并发修改。

示例：

```python
from threading import Thread

from ultralytics import YOLO


def thread_safe_predict(image_path):
    """以线程安全的方式对图像进行预测。"""
    local_model = YOLO("yolo26n.pt")
    results = local_model.predict(image_path)
    # 处理结果


Thread(target=thread_safe_predict, args=("image1.jpg",)).start()
Thread(target=thread_safe_predict, args=("image2.jpg",)).start()
```

有关确保线程安全的更多信息，请访问 [YOLO 模型线程安全推理](#线程安全推理)。

### 在 Python 中运行多线程 YOLO 模型推理的最佳实践是什么？

要在 Python 中安全地运行多线程 YOLO 模型推理，请遵循以下最佳实践：

1. 在每个线程内部实例化 YOLO 模型，而非在多个线程之间共享单个模型实例。
2. 使用 Python 的 `multiprocessing` 模块进行并行处理，以避免与全局解释器锁（GIL）相关的问题。
3. 利用 YOLO 底层 C 库执行的操作来释放 GIL。
4. 当内存有限时，考虑对共享模型实例使用 `ThreadingLocked` 装饰器。

线程安全模型实例化示例：

```python
from threading import Thread

from ultralytics import YOLO


def thread_safe_predict(image_path):
    """使用新的 YOLO 模型实例以线程安全的方式运行推理。"""
    model = YOLO("yolo26n.pt")
    results = model.predict(image_path)
    # 处理结果


# 启动多个线程
Thread(target=thread_safe_predict, args=("image1.jpg",)).start()
Thread(target=thread_safe_predict, args=("image2.jpg",)).start()
```

更多背景信息，请参阅[线程安全推理](#线程安全推理)部分。

### 为什么每个线程应该拥有自己的 YOLO 模型实例？

每个线程应该拥有自己的 YOLO 模型实例以防止竞态条件。当单个模型实例在多个线程之间共享时，并发访问可能导致不可预测的行为和模型内部状态的修改。通过使用独立的实例，你可以确保线程隔离，使多线程任务可靠且安全。

详细指导请参阅[非线程安全示例：单个模型实例](#非线程安全示例单个模型实例)和[线程安全示例](#线程安全示例)部分。

### Python 的全局解释器锁（GIL）如何影响 YOLO 模型推理？

Python 的全局解释器锁（GIL）只允许同一时间有一个线程执行 Python 字节码，这可能限制 CPU 密集型多线程任务的性能。然而，对于 I/O 密集型操作或使用会释放 GIL 的库（如 YOLO 的底层 C 库）的进程，你仍然可以实现并发。为了获得更好的性能，可以考虑使用 Python 的 `multiprocessing` 模块实现基于进程的并行。

有关 Python 线程的更多信息，请参阅[理解 Python 线程](#理解-python-线程)部分。

### 对于 YOLO 模型推理，使用基于进程的并行是否比线程更安全？

是的，使用 Python 的 `multiprocessing` 模块进行 YOLO 模型推理的并行运行通常更安全且更高效。基于进程的并行创建独立的内存空间，避免了全局解释器锁（GIL）并降低了并发问题的风险。每个进程将独立运行，拥有自己的 YOLO 模型实例。

有关 YOLO 模型基于进程并行的更多详情，请参阅[线程安全推理](#线程安全推理)页面。
