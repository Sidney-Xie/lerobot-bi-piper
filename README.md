# LeRobot 双 Piper 实机部署

基于 Hugging Face LeRobot **v0.6.1**，保留原版库代码，增加双 Piper 驱动、只读推理测试、HDF5 轨迹回放、本地 RTC / gRPC 远程 / OpenPI WebSocket 闭环及运行日志。

- [中文版使用说明](docs/bi_piper/USAGE.zh-CN.md)：环境安装、硬件参数、模型准备、只读测试、回放和闭环命令。
- [代码来源与整理范围](docs/bi_piper/PACKAGING.zh-CN.md)：原版基线、纳入/排除清单及整理时修正。
- [GitHub 上传命令](docs/bi_piper/GITHUB.zh-CN.md)：发布现有本地仓库、Git LFS 与 submodule 克隆。
- [验证记录](docs/bi_piper/VALIDATION.zh-CN.md)：无硬件测试及验证边界。
- [LeRobot 原版 README](README.upstream.md) / [许可证](LICENSE)。

## 功能入口

| 用途 | 文件 |
| --- | --- |
| 双臂驱动 / 物理关节顺序 | `src/lerobot/robots/bi_piper/` |
| piper_x 检查点关节映射 | `src/lerobot/robots/bi_piperx/` |
| 只读推理，不发送动作 | `examples/bi_piper/infer_pi05_no_action.py` |
| 轨迹校验与回放 | `examples/bi_piper/replay_hdf5.py` |
| 闭环部署 | `examples/bi_piper/rollout_pi05_async_rtc.py` |
| 控制周期日志 | `src/lerobot/rollout/telemetry.py` |

## 安装与开始

```bash
uv sync --locked --python 3.12 \
  --extra dataset --extra piper --extra intelrealsense --extra pi --extra async

# 查看参数；实际运行前按中文说明准备检查点、轨迹、CAN 和相机。
uv run python examples/bi_piper/infer_pi05_no_action.py --help
uv run python examples/bi_piper/replay_hdf5.py --help
uv run python examples/bi_piper/rollout_pi05_async_rtc.py --help
```

建议依次进行配置预览、只读推理、轨迹校验/回放、闭环部署。回放与闭环均要求显式执行参数及准确的确认短语；闭环默认退出回位并保留力矩，具体行为见使用说明。

## OpenPI 与排除范围

`openpi/` 是指向 [官方 OpenPI](https://github.com/Physical-Intelligence/openpi) 的 Git submodule，固定提交 `215abfb217dbac7d5f1273282331b9b1866c0479`，初始未拉取源码。需要客户端时：

```bash
git submodule update --init openpi
uv pip install msgpack
uv pip install --no-deps -e openpi/packages/openpi-client
```

客户端使用 `uv run --no-sync` 运行，避免同步时移除手动安装的包。本仓库不包含 SVA、OpenPI 本地修改、ACT/Q 实验服务、训练检查点、轨迹或运行数据。官方 OpenPI 不自带原工作区新增的 Piper 配置；WebSocket 后端需要已匹配配置和权重的服务端。可直接使用本仓库 LeRobot RTC/gRPC 路径部署兼容检查点。

LeRobot 原版的 ACT 策略及原有 OpenPI 对齐测试仍随基线保留。上游版本、逐文件来源与哈希见 [来源清单](docs/bi_piper/SOURCE_MANIFEST.json)。
