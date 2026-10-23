# bi_piper 中文使用说明

本仓库以 LeRobot v0.6.1 为基础，增加双 Piper 驱动、只读推理、HDF5 轨迹回放以及本地/远程闭环部署。所有命令均在仓库根目录执行，OpenPI 服务端命令除外。

## 1. 目录与部署方式

| 路径 | 用途 |
| --- | --- |
| `src/lerobot/robots/bi_piper/` | 物理关节顺序的双臂驱动、CAN、限位及 HDF5 回放 |
| `src/lerobot/robots/bi_piperx/` | 在驱动边界交换左右臂各自的 j4/j5，适配 piper_x 策略 |
| `examples/bi_piper/infer_pi05_no_action.py` | 只读反馈、图像与动作预测日志 |
| `examples/bi_piper/replay_hdf5.py` | HDF5 检查与显式授权后的回放 |
| `examples/bi_piper/rollout_pi05_async_rtc.py` | 本地 RTC、gRPC 远程推理、OpenPI WebSocket 闭环启动器 |
| `src/lerobot/rollout/telemetry.py` | 控制周期 JSONL 日志 |
| `src/lerobot/async_inference/` | LeRobot gRPC 推理服务、策略常驻和客户端会话管理 |
| `openpi/` | 官方 OpenPI 的 Git submodule，初始状态未拉取 |

只读脚本固定使用 `bi_piperx`；闭环启动器默认也使用 `bi_piperx`。HDF5 回放固定使用物理顺序的 `bi_piper`。只有模型确实采用物理关节顺序时，才给闭环脚本添加 `--robot-type bi_piper`。选错类型会把 j4/j5 发送到错误关节，不能只根据机械臂外观选择。

## 2. 安装

环境要求：Python 3.12 或 3.13、uv、Linux SocketCAN；真机图像采集使用三路 Intel RealSense。本地 Pi0.5/RTC 还需要适合模型的 GPU 和 PyTorch 环境。

```bash
# 克隆后进入仓库；YOUR_ACCOUNT 替换为实际 GitHub 用户或组织。
git clone https://github.com/YOUR_ACCOUNT/lerobot-bi-piper.git
cd lerobot-bi-piper

# 仅使用 LeRobot 本地/远程后端时，可以暂不初始化 OpenPI。
uv sync --locked --python 3.12 \
  --extra dataset --extra piper --extra intelrealsense --extra pi --extra async
```

`piper` extra 包含 `piper-sdk`、Python CAN 依赖和 `h5py`。保留了 LeRobot 原有包名 `lerobot` 与命令行入口，不要在同一 Python 环境中混装其他版本的 LeRobot。

需要 OpenPI WebSocket 后端时，先完成上面的同步，再安装轻量客户端：

```bash
git submodule update --init openpi
uv pip install msgpack
uv pip install --no-deps -e openpi/packages/openpi-client
uv run --no-sync python -c "import websockets, msgpack, openpi_client; print('OpenPI 客户端依赖正常')"
```

OpenPI 客户端的依赖声明要求 NumPy < 2，而本仓库的 LeRobot 使用 NumPy 2，因此这里采用 `--no-deps`，并手动安装 `msgpack`。客户端仅用于序列化、图像处理和 WebSocket 通信；JAX 模型在单独的服务端环境运行。再次执行 `uv sync` 后，应重新执行这两条 `uv pip install`。本文的 OpenPI 调用使用 `uv run --no-sync`，以保留手动安装的客户端。

## 3. 准备模型、轨迹与硬件参数

仓库不含训练权重、数据集、HDF5 轨迹及运行日志。自行准备与双臂和相机一致的 Pi0.5 检查点，例如放在：

```text
outputs/pi05_robotwin_piper_x_sft_step40000/
├── config.json
├── model.safetensors
├── policy_preprocessor.json
├── policy_preprocessor_step_3_normalizer_processor.safetensors
├── policy_postprocessor.json
└── policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

启动器会检查以上文件是否存在，核对 `pi05` 类型、14 维状态/动作、三路图像、归一化及相对动作处理配置。即使使用远程后端，当前启动器也要求本地检查点目录完整；远程模式不会把本地权重加载进机器人进程。这里展示的目录名只是约定，不是随仓库提供的模型或可直接下载的模型地址。

三路图像特征名为 `cam_high`、`cam_left_wrist`、`cam_right_wrist`，默认 RGB 640×480、30 FPS；它们必须与检查点配置匹配。先列出相机序列号：

```bash
uv run lerobot-find-cameras realsense
```

以下变量请替换为本机实际值，后续命令可直接复用：

```bash
CHECKPOINT="$PWD/outputs/pi05_robotwin_piper_x_sft_step40000"
TRAJECTORY="$PWD/outputs/stack_blocks_two_replay/episode_0000000.hdf5"
HEAD_SERIAL="替换为头部相机序列号"
LEFT_SERIAL="替换为左腕相机序列号"
RIGHT_SERIAL="替换为右腕相机序列号"
TASK="Stack blocks"
```

双臂必须使用不同的 SocketCAN 接口，脚本默认 `can_left` / `can_right`、1 Mbit/s。网卡重命名、驱动及操作权限由部署机器配置；配置完成后检查：

```bash
ip -details -statistics link show dev can_left
ip -details -statistics link show dev can_right
```

脚本会在连接前检查 CAN 状态为 `ERROR-ACTIVE`、速率为 1 Mbit/s，并采样错误计数。接口名不同时传入 `--left-can` 与 `--right-can`。`--left-installation-pos`、`--right-installation-pos` 默认都为 1，应与实际安装方向对应。

## 4. 只读推理测试

先预览配置。`--dry-run` 需要完整的检查点，但不会加载模型、打开 CAN 或相机：

```bash
uv run python examples/bi_piper/infer_pi05_no_action.py \
  --checkpoint "$CHECKPOINT" \
  --head-camera "$HEAD_SERIAL" \
  --left-wrist-camera "$LEFT_SERIAL" \
  --right-wrist-camera "$RIGHT_SERIAL" \
  --task "$TASK" --dry-run
```

随后执行本地只读推理：

```bash
uv run python examples/bi_piper/infer_pi05_no_action.py \
  --checkpoint "$CHECKPOINT" \
  --head-camera "$HEAD_SERIAL" \
  --left-wrist-camera "$LEFT_SERIAL" \
  --right-wrist-camera "$RIGHT_SERIAL" \
  --task "$TASK" --duration 60 --device cuda
```

只读模式只读取关节反馈和图像，不设置运动模式、不使能力矩、不调用 `send_action()`，断开时也不改变现有力矩状态。它不会阻止其他进程或控制器已在执行的动作。

默认日志写入 `outputs/bi_piper_inference_logs/predictions_*.jsonl`，记录 14 维实测状态、图像摘要、完整预测动作块及推理耗时；不保存原始图像。可用 `--log-file` 指定路径，用 `--exposure` / `--gain` 调整三路曝光/增益，或用 `--head-exposure`、`--left-wrist-gain` 等覆盖单路参数。

已启动 OpenPI 服务时，复用上述参数并切换后端：

```bash
uv run --no-sync python examples/bi_piper/infer_pi05_no_action.py \
  --backend openpi --openpi-server-address 127.0.0.1:8000 \
  --checkpoint "$CHECKPOINT" \
  --head-camera "$HEAD_SERIAL" \
  --left-wrist-camera "$LEFT_SERIAL" \
  --right-wrist-camera "$RIGHT_SERIAL" \
  --task "$TASK" --duration 60
```

## 5. HDF5 轨迹回放

先只检查文件，默认不会连接机器人：

```bash
uv run python examples/bi_piper/replay_hdf5.py \
  --trajectory "$TRAJECTORY" --control-hz 25
```

`--control-hz` 的代码默认值是 **15 Hz**；若轨迹采集频率为 25 Hz，请如示例显式指定 25。更改该参数会改变回放速度。

推荐 HDF5 结构为 `joint_action/vector`，形状 `[帧数, 14]`，按左臂 j1…j6、左夹爪、右臂 j1…j6、右夹爪排列。关节单位是弧度；夹爪默认 0=闭合、1=张开。向量数据集必须在根属性或该数据集属性中包含：

```text
vector_order = "left_j1..j6,left_gripper,right_j1..j6,right_gripper"
joint_action_units = "radians"
gripper_encoding = "normalized_0_closed_1_open"
```

后两个属性未提供时使用上述默认值。也支持四个分离数据集 `joint_action/left_arm`、`left_gripper`、`right_arm`、`right_gripper`，以及用 `gripper_encoding="meters"` 标记的米制夹爪数据。加载时检查维度、有限值、物理范围和元数据一致性。

确认双臂接近选定首帧、工作空间已清空且急停可用后，才执行：

```bash
uv run python examples/bi_piper/replay_hdf5.py \
  --trajectory "$TRAJECTORY" --control-hz 25 \
  --execute --confirm "EXECUTE BI_PIPER HDF5 REPLAY"
```

支持 `--start-frame` / `--end-frame` 选择区间，结束索引不包含在回放内。默认相邻关节步长上限为 0.15 rad，夹爪为 0.10；同样的阈值用于检查启动实测姿态与首帧的距离。先检查文件，再连接、使能并检查姿态；姿态检查失败时不发送轨迹帧，但连接过程可能已经使能力矩。

回放采用绝对目标并关闭反馈相对限幅，避免机械臂滞后时改变录制轨迹。默认退出后保留力矩支撑，包括错误退出；应先人工支撑机械臂，再按现场流程恢复/去使能。`--disable-on-disconnect` 是显式选项，只适合已确保失去力矩也安全的安装。

## 6. 闭环部署

三个后端共用状态采集、动作发送、关节限制、退出回位和 JSONL 日志。默认控制频率 25 Hz、运行 10 秒；`--duration 0` 表示持续运行。默认 SDK 速度比例为 25，控制周期的关节变化上限为 1.0 rad、归一化夹爪变化上限为 0.50；它们是软件参数，并不构成碰撞保护，首次部署应按实际场景审查并调低。

### 6.1 本地 Pi0.5 + RTC

```bash
uv run python examples/bi_piper/rollout_pi05_async_rtc.py \
  --backend rtc --robot-type bi_piperx \
  --checkpoint "$CHECKPOINT" \
  --head-camera "$HEAD_SERIAL" \
  --left-wrist-camera "$LEFT_SERIAL" \
  --right-wrist-camera "$RIGHT_SERIAL" \
  --task "$TASK" --duration 10 --dry-run
```

预览会打印实际 `lerobot-rollout` 命令。默认 RTC `execution_horizon=25`、`queue_threshold=45`，检查点的 `chunk_size` 必须大于 45；可通过 `--execution-horizon` 和 `--queue-threshold` 调整，要求 `1 ≤ execution_horizon ≤ queue_threshold < chunk_size`。

确认配置与现场条件后，使用同一组参数，把 `--dry-run` 替换为：

```bash
--execute --confirm-motion "I UNDERSTAND THIS WILL MOVE BOTH PIPER ARMS"
```

只有两个执行参数同时给出且确认短语精确匹配时，启动器才进入真机路径。`--dry-run` 与 `--execute` 不能同时使用。RTC 在本机后台线程加载模型，`--device` 默认为 `cuda`。

### 6.2 LeRobot gRPC 远程推理

在 GPU 服务端安装本仓库对应版本及 `dataset`、`pi`、`async` extras，然后运行：

```bash
uv run python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 --port=8080 --fps=25 \
  --inference_latency=0.04 --obs_queue_timeout=2 \
  --policy_type=pi05 \
  --policy_path=/srv/models/pi05_robotwin_piper_x_sft_step40000 \
  --policy_device=cuda
```

服务端启动时预加载模型；匹配的客户端会话复用常驻策略。机器人端预览：

```bash
uv run python examples/bi_piper/rollout_pi05_async_rtc.py \
  --backend remote --robot-type bi_piperx \
  --checkpoint "$CHECKPOINT" \
  --server-address POLICY_HOST:8080 \
  --server-policy-path /srv/models/pi05_robotwin_piper_x_sft_step40000 \
  --server-policy-device cuda --actions-per-chunk 50 \
  --head-camera "$HEAD_SERIAL" \
  --left-wrist-camera "$LEFT_SERIAL" \
  --right-wrist-camera "$RIGHT_SERIAL" \
  --task "$TASK" --duration 10 --dry-run
```

`POLICY_HOST` 替换为服务器地址；`--server-policy-path` 是服务端可见路径，与机器人端 `--checkpoint` 可以不同，但模型和配置必须一致。实际运行仍须将 `--dry-run` 替换为上述两项执行参数。机器人进程使用 CPU，不在本地加载策略权重；远程后端不叠加 RTC 前缀引导。

默认连接/模型准备/RPC/首动作超时分别为 10/600/30/120 秒，可用 `--connect-timeout-s`、`--setup-timeout-s`、`--rpc-timeout-s`、`--first-action-timeout-s` 调整。客户端使用会话租约；每台机器人使用独立服务实例。传输涉及 pickle，按现有协议应只在受控可信网络使用，不直接暴露到公网。

### 6.3 OpenPI WebSocket 远程推理

OpenPI 是固定到 `215abfb217dbac7d5f1273282331b9b1866c0479` 的官方 submodule。原工作区对 OpenPI 的未提交修改、RobotWin/Piper 专用训练配置、SVA/ACT/Q 服务均未迁入。

服务端在独立 OpenPI 环境中按其 README 安装依赖，然后运行以下模板；必须把配置名和检查点路径替换为**已在该服务端注册并与 Piper 模型匹配**的值：

```bash
cd openpi
uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=YOUR_REGISTERED_PIPER_CONFIG \
  --policy.dir=/srv/models/YOUR_OPENPI_CHECKPOINT \
  --port=8000
```

官方固定版本不自带本工作区新增的 `pi05_robotwin_piper_x` 配置，因此这个服务端命令是接入模板，不能直接拿示例配置名启动 Piper。需要现成可运行的路径时，优先使用 6.1/6.2 的 LeRobot 检查点后端；若已有匹配的 OpenPI 服务，则直接使用客户端：

```bash
# 回到本仓库根目录，并完成第 2 节的客户端安装。
uv run --no-sync python examples/bi_piper/rollout_pi05_async_rtc.py \
  --backend openpi --robot-type bi_piperx \
  --checkpoint "$CHECKPOINT" \
  --openpi-server-address POLICY_HOST:8000 \
  --head-camera "$HEAD_SERIAL" \
  --left-wrist-camera "$LEFT_SERIAL" \
  --right-wrist-camera "$RIGHT_SERIAL" \
  --task "$TASK" --duration 10 --dry-run
```

实际运行同样替换 `--dry-run` 为两项执行参数。标准请求包含三路 `images`（CHW uint8、默认缩放补边至 224×224）、14 维 `state` 和 `prompt`；服务端应返回有限值 `actions[T,14]`，关节顺序/单位和绝对目标语义必须与所选驱动一致。默认最多接收 50 步动作块，队列低于 `--openpi-open-loop-horizon`（默认 1）时请求下一块；默认请求超时 30 秒。机器人端不加载 JAX，不发送 SVA 专用图像字段。

## 7. 日志与退出

闭环每周期日志位于 `outputs/bi_piper_inference_logs/closed_loop_*.jsonl`，或通过 `--log-file` 指定。日志在后台线程写入，包含状态、策略动作、处理后动作、驱动返回的发送动作、发送结果、控制周期、队列大小、动作块 ID 与推理延迟。暂时没有动作的周期也会记录，`action` 为 `null`；推理未完成时其结束时间和延迟为 `null`。

```bash
jq -r '[.step,.control_dt,.action_available,.chunk_id,.queue_size,.inference_latency_ms,.send_success] | @tsv' \
  outputs/bi_piper_inference_logs/closed_loop_*.jsonl
```

闭环默认尝试返回启动时测得的姿态，断开 CAN 后保留力矩。可显式用 `--no-return-to-initial-position` 关闭回位；`--disable-on-disconnect` 控制断开去使能。回位、停止和断开仍受硬件故障影响，应由现场操作员监督。

## 8. 不连接硬件的回归检查

```bash
uv sync --locked --extra dataset --extra piper --extra pi --extra async --extra test
uv run python -m pytest \
  tests/robots/test_bi_piper.py \
  tests/robots/test_bi_piper_gripper_effort.py \
  tests/robots/test_bi_piper_hdf5_replay.py \
  tests/robots/test_bi_piper_rollout_launcher.py \
  tests/test_remote_inference.py \
  tests/test_rollout_telemetry.py \
  tests/test_rollout.py \
  tests/processor/test_pipeline.py -q
```

若要额外验证 OpenPI 观测格式，按第 2 节重新安装可选客户端后执行 `uv run --no-sync python -m pytest tests/test_openpi_observation.py -q`。这些测试使用模拟 CAN/机器人/网络对象，不执行真机回放或闭环。

上游其他测试按 [LeRobot 原版说明](../../README.upstream.md) 配置；整理时的实际验证范围见 [验证记录](VALIDATION.zh-CN.md)。
