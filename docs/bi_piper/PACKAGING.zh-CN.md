# 代码整理说明

## 来源

- LeRobot 官方仓库：<https://github.com/huggingface/lerobot>。
- 原版基线：`v0.6.1`，提交 `7e241bd630a3719a56157a497ce5d08f244784f1`。
- OpenPI 官方仓库：<https://github.com/Physical-Intelligence/openpi>。
- OpenPI submodule 固定提交：`215abfb217dbac7d5f1273282331b9b1866c0479`。
- 整理日期：2026-10-23。

整理时原仓库处于 detached HEAD，双臂功能尚未提交。因此来源判定同时使用了 `git log`、`git diff HEAD` 和未跟踪文件清单，不能仅根据提交日志复制。

新仓库采用全新历史：先导入与 LeRobot 原版 Git tree 完全一致的快照，并标记 `lerobot-v0.6.1-base`；再提交筛选后的双臂改动；最后提交中文文档和来源清单。不迁移上游的完整历史。`LICENSE`、源文件版权声明及上游说明均保留；原版 README 另存为根目录 `README.upstream.md`。

生成的提交使用本仓库局部身份 `BiPiper repository export <bi-piper-export@localhost>`，用于区分自动整理提交。后续开发可设置自己的 `git config --local user.name` 和 `user.email`。

## 纳入范围

| 类别 | 纳入内容 |
| --- | --- |
| LeRobot 原版 | 完整基线源码、文档、示例、测试、构建配置及原版 Git LFS 测试资源 |
| 双臂驱动 | `bi_piper`、`bi_piperx`、只读模式、关节/夹爪转换、CAN 故障处理与限位 |
| 回放 | HDF5 解析、单位/形状/限位校验、首帧检查、绝对目标回放 |
| 推理与部署 | 只读推理、RTC 启动器、gRPC 引擎、标准 OpenPI WebSocket 客户端适配 |
| LeRobot 集成 | 机器人工厂和 CLI 注册、处理器修正、rollout 生命周期、策略常驻与会话租约 |
| 诊断 | 控制周期 JSONL、推理时间、队列与动作块信息，以及相关无硬件测试 |
| 依赖 | `piper` extra、对应锁文件、原工作区新增的 WebSocket 依赖 |

共筛选原工作区 27 个已修改文件和 22 个未跟踪源文件。逐文件来源与 SHA-256 见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。该清单同时记录整理后的哈希，便于区分直接复制的文件与在新仓库修正的文件。

## 排除范围

- `SVA/` 全目录。
- 原 `openpi/` 工作树、虚拟环境、第三方目录和所有本地未提交修改；主仓库只记录 `.gitmodules` 与 `160000` 类型的 gitlink。
- OpenPI 本地新增的 SVA/ACT/Q 推理服务、候选动作采样、性能实验及 RobotWin/Piper 专用训练配置。
- `examples/bi_piper/experiment.py`、`experiment.md`、`tests/robots/test_bi_piper_experiment.py`、`close_loop_experiment.sh`。
- 写死设备序列号、任务与执行授权的 `close_loop_execute.sh`。通用启动器已覆盖其核心用途。
- 训练权重、HDF5 轨迹、数据集、`outputs/`、虚拟环境、运行日志、缓存及本地预览图。

原版 LeRobot 中已有的 ACT 策略和 OpenPI 对齐测试属于基线源码，按“保留原版仓库代码”保留。保留的 OpenPI 机器人侧接口只负责标准协议通信，不包含 OpenPI 模型实现或 SVA 调用链。

## 整理时的修正

1. 删除 OpenPI 适配器中根据 `act.q_model` 元数据额外构造 `value_images` 的 SVA/Q 专用分支，并移除对应测试。
2. 将闭环启动器默认驱动调整为 `bi_piperx`，与默认 piper_x 检查点约定、只读脚本及现有配置测试一致；仍支持显式选择 `bi_piper`。
3. 保持原工作区 RTC 默认值 `execution_horizon=25`、`queue_threshold=45`，更新仍断言 15/30 的旧测试。
4. 保持回放代码默认 15 Hz，修正原先声称默认 25 Hz 的模块描述。中文命令在需要 25 Hz 时显式传参。
5. 修正相关导入顺序及移除未使用导入；OpenPI 观测测试在可选客户端未安装时跳过。

这些修改只发生在新仓库；原工作区的源码、模型、轨迹和 OpenPI 未提交改动均保持原样。

## Git LFS 测试资源

原版包含 50 个 LFS 测试资源路径，主要为小型图像、视频、测试张量及 RealSense 测试录制。新仓库保留原指针，并准备了本地 LFS 对象；它们与未打包的训练检查点不同。

首次发布时需要一起上传这些 LFS 对象，具体命令见 [GitHub 上传说明](GITHUB.zh-CN.md)。新仓库的 `git lfs fsck` 可验证这些资源的完整性。

## 复核命令

```bash
# 第一个提交对应原版快照，后续差异即双臂集成和整理文档。
git log --oneline --reverse
git diff --stat lerobot-v0.6.1-base..HEAD

# 应仅看到 openpi 的 160000 gitlink，不应列出 openpi/ 下的源码。
git ls-files --stage openpi
git submodule status

# 原版 LFS 资源完整性及工作区状态。
git lfs fsck
git status --short
```
