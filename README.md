# bi_piper 中文使用说明

本仓库以 LeRobot v0.6.1 为基础，增加双 Piper 驱动、只读推理、HDF5 轨迹回放、本地/远程闭环部署以及 HDF5 数据采集。所有命令均在仓库根目录执行，OpenPI 服务端命令除外。

## 1. 目录与部署方式

| 路径 | 用途 |
| --- | --- |
| `src/lerobot/robots/bi_piper/` | 物理关节顺序的双臂驱动、CAN、限位及 HDF5 回放 |
| `examples/bi_piper/infer_pi05_no_action.py` | 只读反馈、图像与动作预测日志 |
| `examples/bi_piper/replay_hdf5.py` | HDF5 检查与显式授权后的回放 |
| `examples/bi_piper/rollout_pi05_async_rtc.py` | 本地 RTC、gRPC 远程推理、OpenPI WebSocket 闭环启动器 |
| `src/lerobot/rollout/telemetry.py` | 控制周期 JSONL 日志 |
| `src/lerobot/async_inference/` | LeRobot gRPC 推理服务、策略常驻和客户端会话管理 |
| `openpi/` | 官方 OpenPI 的 Git submodule，初始状态未拉取 |

只读推理、闭环和 HDF5 回放统一使用 `bi_piper`。状态和动作均使用物理关节顺序，模型必须与此关节顺序、单位及夹爪定义一致。

## 2. 安装

环境要求：Python 3.12 或 3.13、uv、Linux SocketCAN；真机图像采集使用三路 Intel RealSense。本地 Pi0.5/RTC 还需要适合模型的 GPU 和 PyTorch 环境。

```bash
# 克隆后进入仓库；YOUR_ACCOUNT 替换为实际 GitHub 用户或组织。
git clone https://github.com/Sidney-Xie/lerobot-bi-piper.git
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
  --backend rtc --robot-type bi_piper \
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
  --backend remote --robot-type bi_piper \
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

OpenPI 是固定到 `215abfb217dbac7d5f1273282331b9b1866c0479` 的官方 submodule。

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
  --backend openpi --robot-type bi_piper \
  --checkpoint "$CHECKPOINT" \
  --openpi-server-address POLICY_HOST:8000 \
  --head-camera "$HEAD_SERIAL" \
  --left-wrist-camera "$LEFT_SERIAL" \
  --right-wrist-camera "$RIGHT_SERIAL" \
  --task "$TASK" --duration 10 --dry-run
```

实际运行同样替换 `--dry-run` 为两项执行参数。标准请求包含三路 `images`（CHW uint8、默认缩放补边至 224×224）、14 维 `state` 和 `prompt`；服务端应返回有限值 `actions[T,14]`，关节顺序/单位和绝对目标语义必须与所选驱动一致。默认最多接收 50 步动作块，队列低于 `--openpi-open-loop-horizon`（默认 1）时请求下一块；默认请求超时 30 秒。

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

上游其他测试按 [LeRobot 原版说明](README.upstream.md) 配置。

## 9. HDF5 数据采集

本模块参照 `piper_data_collection` 的双遍 replay-collect 流程实现，提供两种采集方式：

1. 两阶段回放：第一遍只执行轨迹，回初态后人工复位场景；第二遍执行同一轨迹，同时采集从臂实测状态、末端位姿和相机图像。
2. 主从遥操作：左右各一对主从臂，分别共用 `can_left`、`can_right`；由 Piper 控制器直接按 1:1 目标跟随，程序被动记录主臂指令、从臂反馈和图像。

两种方式使用同一个 HDF5 writer，均采用双臂物理关节顺序。

### 9.1. 安装与公共参数

在本仓库根目录执行：

```bash
uv sync --locked --extra piper --extra intelrealsense
uv run lerobot-collect-bi-piper --help
uv run lerobot-find-cameras realsense
```

如果环境尚未重新生成命令行入口，所有命令都可以将 `uv run lerobot-collect-bi-piper` 替换为 `uv run python -m lerobot.scripts.lerobot_collect_bi_piper`。也提供 `examples/bi_piper/collect_replay.py` 和 `collect_teleop.py` 两个便捷入口。

真实采集默认需要三路 RealSense RGB-D，640×480、30 FPS。替换下列相机序列号：

```bash
HEAD_SERIAL="替换为头部相机序列号"
LEFT_SERIAL="替换为左腕相机序列号"
RIGHT_SERIAL="替换为右腕相机序列号"
```

`--camera-stream-mode rgb` 只启用 RGB，不生成深度数据集。`--no-cameras` 是显式的仅状态采集模式，不能同时传相机序列号。默认 `--gripper-max-m 0.1` 适用于 100 mm 夹爪；70 mm 夹爪应在两种采集方式中都设置 `--gripper-max-m 0.07`。

两路 SocketCAN 必须以 1 Mbit/s 启动且控制器状态为 `ERROR-ACTIVE`：

```bash
ip -details -statistics link show dev can_left
ip -details -statistics link show dev can_right
```

采集启动器检查接口类型、速率和控制器状态，不代替现场的供电、终端电阻、安装固定和急停检查。下面的真实回放命令会使能和移动双臂，应由现场操作员监督。

### 9.2. 方式一：两阶段 replay-collect

#### 流程

每一条源 episode 按以下顺序执行：

1. 预检整批源文件、两遍动作计划及所有输出路径。
2. 双臂同步插值到第一遍首帧，按第一遍速度回放，**不采集样本**。
3. 双臂和夹爪同步插值回设定初态，读取反馈确认到位。
4. 等待 Enter，供操作员复位物体、恢复场景。
5. 双臂同步插值到第二遍首帧，回放并在每个源帧对应的控制命令处采一帧真实反馈与图像。
6. 到轨迹末端自动停采；双臂和夹爪回初态、确认到位后，原子保存 HDF5。
7. 等待 Enter，再进入下一条或结束批次。

默认源频率回落值为 25 Hz，两遍速度比例均为 0.5，控制命令频率约 30 Hz。第二遍有多少个源帧，就保存多少条样本；不根据另一个独立录制 FPS 重采样。减速后目标采样频率为 `source_hz × second_speed_scale`，相机 FPS 必须不低于该值。

#### 先预览，不连接设备

```bash
uv run lerobot-collect-bi-piper replay \
  /path/to/source_episodes \
  --output-dir outputs/bi_piper_collected/replay \
  --episode-start 0 --episode-end 9 \
  --dry-run
```

也可以逐个传入源 HDF5：

```bash
uv run python examples/bi_piper/collect_replay.py \
  /path/to/episode0.hdf5 /path/to/episode1.hdf5 \
  --output-dir outputs/bi_piper_collected/replay --dry-run
```

不加执行选项时也是预览模式，不打开 CAN、相机或模型，不写数据。预览不要求提供相机序列号。

#### 正式采集

```bash
uv run lerobot-collect-bi-piper replay \
  /path/to/source_episodes \
  --output-dir outputs/bi_piper_collected/replay \
  --episode-start 0 --episode-end 9 \
  --left-can can_left --right-can can_right \
  --head-camera "$HEAD_SERIAL" \
  --left-wrist-camera "$LEFT_SERIAL" \
  --right-wrist-camera "$RIGHT_SERIAL" \
  --source-hz 25 --control-hz 30 \
  --first-speed-scale 0.5 --second-speed-scale 0.5 \
  --task "Stack blocks" \
  --execute --confirm "EXECUTE BI_PIPER REPLAY COLLECT"
```

回放使用 PC 控制从臂。开始前应让主臂停止发送示教控制帧，或将主臂从对应共享总线断开，避免主臂广播与 PC 回放指令争用；不要同时运行遥操作和回放进程。

默认初态（每侧关节 rad、夹爪 m）为：

```text
[0, 0.05, -0.05, 0, 0, 0, gripper_max_m - 0.0005] × 2
```

可用 `--initial-state` 后接 14 个数字覆盖。初态和源目标均检查关节限位与夹爪范围。移动到首帧、回初态使用同步插值，默认关节速度不超过 `--move-joint-velocity 0.5` rad/s、夹爪速度不超过 `--move-gripper-velocity 0.05` m/s，随后等待反馈收敛；默认关节容差 0.08 rad、夹爪容差 0.005 m、到位等待超时 5 秒。

轨迹回放本身保持统一时间比例，不根据机械臂响应再次拉长时间。计划中相邻命令超过 `--max-joint-step-rad 0.15` 或 `--max-gripper-step-m 0.01` 时，在连接硬件前拒绝执行；可降低速度比例、提高控制频率或修正源轨迹。`--first-pass-no-grasp` 让第一遍夹爪保持全开，第二遍仍执行原始抓取动作。

#### 源文件和批量输出

支持 HDF5 的三种源布局：

| 源布局 | 字段 |
| --- | --- |
| 标准双臂向量 | `joint_action/vector`，形状 `[N,14]` |
| 分离字段 | `joint_action/left_arm`、`left_gripper`、`right_arm`、`right_gripper` |
| 原生 RoboTwin | `action/left_arm_joint_states`、`left_ee_joint_states`、`right_arm_joint_states`、`right_ee_joint_states` |

顺序固定为左六关节、左夹爪、右六关节、右夹爪。关节必须为弧度；夹爪默认是 0=闭合、1=张开的归一化值，若根属性 `gripper_encoding="meters"` 则按米读取。缺失 `vector_order` 时按 RoboTwin 规范顺序处理并记录该假设；显式声明的错误顺序会拒绝。NaN、无限值和超限目标均拒绝，不静默裁剪。暂不读取 PKL 轨迹。

源频率依次取 `--source-hz` 显式覆盖、根属性 `fps`、根属性 `control_hz`、`additional_info/frequency`、默认 25。目录中的 `.hdf5` / `.h5` 按 episode 数字排序；`--episode-start` / `--episode-end` 为包含首尾的编号范围。默认沿用源文件名，`--output-start-index 100` 可统一编号为 `episode_000100.hdf5` 起；已有文件或输出重名会在整批预检时拒绝，不自动覆盖。

#### 离线验证完整写盘链路

```bash
uv run lerobot-collect-bi-piper replay /path/to/episode0.hdf5 \
  --output-dir outputs/bi_piper_offline_check --offline
```

离线文件用源目标作为合成状态，明确标记 `sim=True`、`backend="offline"`、`action_source="synthetic_source_target"`，没有真实相机或末端反馈；用于检查格式和帧映射，不能混作真实示教数据。

### 9.3. 方式二：主从共享 CAN 遥操作采集

#### 接线与角色

| CAN 接口 | 共用该总线的设备 | 数据方向 |
| --- | --- | --- |
| `can_left` | 左主臂 + 左从臂 + PC CAN 适配器 | 主臂发目标，从臂跟随并发反馈，PC 监听 |
| `can_right` | 右主臂 + 右从臂 + PC CAN 适配器 | 同上 |

每一路只能有一台发送控制帧的主臂和一台发送反馈帧的从臂；左右两路保持独立。需提前按 Piper 官方流程配置输入/主臂与运动输出/从臂，使用标准、无偏移的 CAN ID。参考 SDK 角色配置分别为主臂 `MasterSlaveConfig(0xFA, 0, 0, 0)`、从臂 `MasterSlaveConfig(0xFC, 0, 0, 0)`；**配置角色时应物理隔离单个臂，再按对应角色分别设置，不能在已并联的共享总线上广播角色切换命令**。

采集程序不自动切换角色、使能力矩或设置 ID 偏移。固件跟随和主臂可示教状态应先由官方流程准备好。当前 SDK 标准解码器不支持偏移后的控制/反馈 ID；`dh_is_offset` 是几何设置，不能用来代替 CAN ID 配置。

“完全复刻”在这里指原厂主从协议的 **1:1 目标跟随**：不做关节换序、缩放、镜像、动作平滑或软件二次转发。从臂实际响应仍会存在机械和控制延迟，因此文件同时保留主臂目标与从臂实测值，不把二者伪装成完全相同的数据。

#### 预览配置

```bash
uv run lerobot-collect-bi-piper teleop \
  --output-dir outputs/bi_piper_collected/teleop \
  --left-can can_left --right-can can_right \
  --fps 25 --duration 60 --num-episodes 10 --dry-run
```

#### 正式采集

```bash
uv run lerobot-collect-bi-piper teleop \
  --output-dir outputs/bi_piper_collected/teleop \
  --left-can can_left --right-can can_right \
  --head-camera "$HEAD_SERIAL" \
  --left-wrist-camera "$LEFT_SERIAL" \
  --right-wrist-camera "$RIGHT_SERIAL" \
  --fps 25 --duration 60 --num-episodes 10 --start-index 0 \
  --task "Stack blocks" \
  --execute --confirm "RECORD BI_PIPER SHARED CAN TELEOP"
```

程序先等待左右总线同时出现新鲜的主臂控制帧和从臂反馈。每条 episode 按 Enter 开始，默认录制 60 秒；采样条数为 `ceil(duration × fps)`，结束后原子保存并等待下一条。`--start-index` 可用于继续编号。主臂关节目标从 `GetArmJointCtrl().joint_ctrl` 读取，夹爪目标从 `GetArmGripperCtrl().gripper_ctrl` 读取；从臂读取反馈接口，每侧只创建一个 SDK client，使用 `ConnectPort(piper_init=False)`。

进程只监听，不调用 `JointCtrl`、`GripperCtrl`、`EnableArm`、`MasterSlaveConfig` 或急停指令。等待 Enter、停止采集、Ctrl+C 或退出进程都**不会停止硬件主从跟随**；暂停动作需由现场操作员控制主臂，紧急情况使用物理急停。遥操作采集也不自动回零或去使能。

### 9.4. HDF5 数据格式

内部采集统一使用关节 rad、夹爪 m。写盘时，`joint_action` 与 `action` 中的夹爪除以 `gripper_max_m`，转为 0=闭合、1=张开的归一化值。夹爪按真实编码器开度记录，即使夹持力矩较大也不会改写为 0。

```text
episode_000000.hdf5
├── joint_action/
│   ├── vector                 [N,14]  从臂实测状态（夹爪归一化）
│   ├── left_arm, right_arm     [N,6]
│   └── left_gripper, right_gripper [N]
├── action                     [N,14]  第二遍已发送目标／主臂广播目标
├── observations/qpos          [N,14]  joint_action/vector 的硬链接
├── endpose/
│   ├── left_endpose, right_endpose [N,7] XYZ(m) + quaternion(wxyz)
│   ├── left_available, right_available [N]
│   └── left_gripper, right_gripper [N]
├── observation/{head_camera,left_camera,right_camera}/
│   ├── rgb                    [N]     每帧 JPEG 字节串
│   ├── depth                  [N,H,W] 可选，float64 毫米，已对齐 RGB
│   ├── intrinsic_cv           [N,3,3]
│   ├── extrinsic_cv           [N,3,4] 未标定填 NaN
│   └── cam2world_gl            [N,4,4] 未标定填 NaN
├── pointcloud                 [N,0]   未重建点云
├── recording_metadata/
│   ├── timestamp_ns           [N]     主机单调时钟，不是 Unix 时间
│   ├── source_frame_index     [N]     回放为 0..N-1；遥操作为 -1
│   ├── action_available       [N]
│   ├── camera_capture_timestamp_ns/{camera}
│   ├── camera_device_timestamp_ms/{camera}
│   └── camera_frame_number/{camera}
└── camera_calibration/{camera}/       内参、深度比例和深度→RGB外参
```

与参考仓库保持一致，`joint_action/vector` 写的是**真实从臂反馈**；独立的 `action` 保存专家目标。回放采样先发送对应控制目标，再读取反馈和图像，因此两者不必相等。文件根属性记录 `workflow`、`backend`、`sim`、`task`、`action_source`、`vector_order`、`gripper_encoding` 等语义。

回放文件的 `fps` 保留源频率，`capture_target_fps` 保存减速后的目标采样频率，`capture_fps` 根据实际时间戳计算。遥操作文件 `fps` 为目标采样频率。相机设备时钟和主机时钟是不同时间域，不能直接相减。SDK 关节反馈时间戳是多个 CAN 帧的聚合时间戳；左右臂和三台相机也没有硬件同步，保存各自时间信息用于后续对齐。

每次采样要求每台相机提供比上次更新的帧，RGB 和对齐深度来自同一 frameset；深度用设备实际 `depth_scale_m` 转换到毫米，不假定固定 1 mm 比例。RGB-only 模式不创建 `depth`。未标定世界外参保留 NaN 并标记不可用，不生成虚假的外参或点云。

数值和深度增量写入临时 HDF5，仅缓存压缩 JPEG；全部帧校验及回位成功后再发布最终文件。任何采集异常、相机超时或 Ctrl+C 都会丢弃**当前未完成**的 episode，已保存文件不受影响。回放异常会尝试急停并保留力矩支撑，不在故障后盲目回位；共享 CAN 模式异常只停止记录，不发送控制帧。默认 `--max-lateness-s 0.5`，超出控制/采样调度容忍时间时中止，而不静默跳过帧。

### 9.5. 不连接硬件的测试

```bash
uv sync --locked --extra piper --extra test
uv run python -m pytest \
  tests/robots/test_bi_piper_hdf5_recording.py \
  tests/robots/test_bi_piper_replay_collection.py \
  tests/robots/test_bi_piper_shared_can.py \
  tests/robots/test_bi_piper_collection.py \
  tests/robots/test_bi_piper_collection_cameras.py \
  tests/robots/test_bi_piper_collection_cli.py -q
```

这些测试使用模拟 SDK、相机和时钟，验证数据格式、源帧对齐、主从无写操作、状态机、异常清理及覆盖保护。真机采集仍需在安装完成后逐项验证。
