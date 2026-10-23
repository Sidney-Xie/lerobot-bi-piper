# 验证记录

验证日期：2026-10-23。

## 已完成

- 相关无硬件回归测试：**272 passed**，耗时约 14 秒。
- 检查范围：双臂驱动、夹爪力反馈、HDF5 回放、启动器参数/执行门槛、远程推理与服务端会话、JSONL 日志、OpenPI 标准观测格式、rollout 和 processor pipeline。
- Ruff 对所有筛选的 Python 文件及相关模块检查通过；格式检查通过。最后两处格式化前后 Python AST 一致。
- `uv lock --check --offline` 通过，现有 `pyproject.toml` 与锁文件一致。
- 9 条命令行冒烟检查通过：三个脚本的 `--help`、本地/OpenPI 只读预览、RTC/gRPC/OpenPI 闭环预览，以及一条合成 HDF5 的只校验调用。所有调用都没有执行标志。
- 原版导入提交的 Git tree 与 LeRobot `7e241bd630a3719a56157a497ce5d08f244784f1` 完全一致。
- 50 个原版 Git LFS 测试资源路径的内容哈希与指针一致，`git lfs fsck` 通过。
- OpenPI 索引项为 `160000` gitlink，URL 指向官方仓库，固定提交与原工作区 OpenPI 的已提交基线一致；未纳入其本地修改。
- 来源清单中的原工作区文件哈希与整理前一致。

## 验证环境

使用本机已有的 Python 3.12 环境，PyTorch 2.10.0、pytest 8.4.2，CPU 测试模式；`PYTHONPATH` 明确指向新仓库，避免导入原工作区的 LeRobot。

HDF5 测试使用临时目录中的 h5py 3.16.0；OpenPI 观测测试使用从固定官方提交导出的轻量 `openpi-client`，没有使用原工作区修改过的 OpenPI 服务端。缓存设置到可写临时目录。复现测试命令见 [使用说明第 8 节](USAGE.zh-CN.md#8-不连接硬件的回归检查)。

## 验证边界

整理过程中没有连接 CAN、相机或模型服务器，没有使能机械臂、执行轨迹或运行闭环，也没有推送 GitHub。测试结果验证软件接口与模拟场景，不替代安装完成后的真机验证。

由于执行环境网络受限，未从网络重新安装锁文件的全部依赖，也未在线拉取 OpenPI submodule；这不影响本地 submodule gitlink 的建立。未运行 LeRobot 全量测试、GPU 推理、模型精度对齐或实际网络服务联调。官方 OpenPI 版本没有本工作区的自定义 Piper 配置，这部分需由部署方准备。
