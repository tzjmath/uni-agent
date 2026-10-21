# 最小例子：在 Docker 里跑通轨迹收集

这条路径和强化学习训练用的是同一条数据链路：模型在 Gateway 里生成 token，智能体在 Docker 容器里执行命令，最后把轨迹写到磁盘。跑完就可以用这些轨迹做 SFT。

官方的完整训练脚本按 64 张 GPU 来写。这里只跑 **1 道题、1 张 GPU**，用来确认整条链路是通的。

## 你会得到什么

日志目录里每个会话有三份文件：

- `task.log`：人能读的过程，包括在 Docker 里执行的命令和输出。
- `trajectory.json`：轮数、分数、token 长度。
- `trajectory.npz`：训练用的 token。`response_mask` 里 `1` 是模型自己说的话，`0` 是工具和 Docker 的输出。做 SFT 时只在 `1` 上计算损失。

容器本身不会被保存。任务结束后容器会被删掉。

## 怎么跑

在仓库根目录执行：

```bash
bash examples/mini_docker/run.sh
```

默认设置：

- 模型：`/home/tianzhengjie/models/Qwen2.5-Coder-7B-Instruct`
- 沙箱镜像：`uni-agent-python:3.12`（里面已经装了 tmux）
- Python：`/home/tianzhengjie/venvs/uni-agent/bin/python`
- 只使用第 0 张 GPU
- 日志：`~/data/uni_agent/logs/mini_docker`

这台机器的 CUDA 12.2 编译器编不过 Qwen3.5 需要的线性注意力内核，所以这条冒烟路径用 7B 模型。

题目很短：让智能体在容器里写下 `/workspace/result.txt`，内容必须是 `hello-from-docker`。写对了分数是 1，写错了是 0。无论分数多少，轨迹都会留下来。

## 跑完后看结果

```bash
find ~/data/uni_agent/logs/mini_docker -type f | sort
```

先打开 `task.log` 看命令，再用下面的代码看 token 有没有记全：

```python
import json
import numpy as np
from pathlib import Path

run = next(Path.home().joinpath("data/uni_agent/logs/mini_docker").rglob("trajectory.json")).parent
print(json.loads((run / "trajectory.json").read_text()))
data = np.load(run / "trajectory.npz")
mask = data["traj0_response_mask"]
print("model tokens", int(mask.sum()), "context tokens", int((mask == 0).sum()))
```
