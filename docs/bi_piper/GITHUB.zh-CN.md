# 将新仓库上传 GitHub

本地新仓库已经初始化为 `main` 并提交，不需要重新 `git init` 或再次全量 `git add`。上传由你执行，尚未创建或推送任何 GitHub 仓库。

## 方式一：使用 GitHub CLI

先安装 Git、Git LFS 和 GitHub CLI，确认当前目录是整理后的仓库：

```bash
cd /home/xiexinyi/lerobot_robot_bi_piper/dist/lerobot-bi-piper
git status --short
git log --oneline -3
git lfs fsck

# 将下面一行改为你的 GitHub 用户或组织/仓库名。
GITHUB_REPO="YOUR_ACCOUNT/lerobot-bi-piper"

gh auth login
gh auth setup-git
gh repo create "$GITHUB_REPO" --private --source=. --remote=origin

# 先上传 LeRobot 原版测试资源，再上传 main 与基线标签。
git lfs push --all origin
git push -u origin main
git push origin lerobot-v0.6.1-base
```

如需公开仓库，将 `--private` 改为 `--public`。`gh repo create` 要求目标仓库尚不存在；若已创建，请使用方式二添加 remote。

## 方式二：网页创建仓库后上传

在 GitHub 创建一个空仓库，不要自动添加 README、LICENSE 或 `.gitignore`。然后运行：

```bash
cd /home/xiexinyi/lerobot_robot_bi_piper/dist/lerobot-bi-piper
GITHUB_REPO="YOUR_ACCOUNT/lerobot-bi-piper"

git remote add origin "https://github.com/${GITHUB_REPO}.git"
git lfs push --all origin
git push -u origin main
git push origin lerobot-v0.6.1-base
```

也可以使用 SSH remote：`git@github.com:YOUR_ACCOUNT/lerobot-bi-piper.git`。如已配置 `origin`，先用 `git remote -v` 确认指向自己的新仓库；需要更换时使用 `git remote set-url origin URL`，不要推送到 LeRobot 上游仓库。

## OpenPI submodule 与后续克隆

`.gitmodules` 已指向官方公开地址并固定到官方提交，因此只推送主仓库即可；不需要上传原工作区的 OpenPI 代码，也不要在新仓库重复执行 `git submodule add`。

其他机器获取仓库及 OpenPI 的方式：

```bash
git clone --recurse-submodules https://github.com/YOUR_ACCOUNT/lerobot-bi-piper.git
cd lerobot-bi-piper
git lfs install --local
git lfs pull
```

或者先普通克隆，在需要 WebSocket 后端时执行 `git submodule update --init openpi`。只使用 RTC/gRPC 后端不需要拉取 OpenPI。

训练检查点和 HDF5 文件需要单独分发；它们没有纳入 Git，本地部署时按 [中文使用说明](USAGE.zh-CN.md) 放置。
