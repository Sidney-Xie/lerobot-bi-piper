# bi_piper 双臂部署示例

数据采集见 [两阶段回放与共享 CAN 主从遥操作采集](../../README.md#9-hdf5-数据采集)。

完整中文版说明见 [双臂使用说明](../../README.md)。

建议按以下顺序进行：

1. `infer_pi05_no_action.py --dry-run` 检查检查点和相机参数。
2. `infer_pi05_no_action.py` 读取反馈和图像，记录动作预测，不发送动作。
3. `replay_hdf5.py --trajectory PATH` 校验轨迹，再按说明添加执行确认参数。
4. `rollout_pi05_async_rtc.py --backend rtc|remote|openpi --dry-run` 预览闭环命令，再按说明执行。

每个脚本支持 `--help`。只读推理、闭环和 HDF5 回放统一使用 `bi_piper`，关节状态与动作均采用物理关节顺序，不交换 j4/j5。回放默认 **15 Hz**，闭环默认 **25 Hz**；按数据采集频率显式设置参数。

OpenPI 以官方 submodule 提供，主仓库只保留标准 WebSocket 接入，不包含 SVA、ACT/Q 实验启动器或服务端定制配置。模型权重、轨迹与设备序列号需自行准备。
