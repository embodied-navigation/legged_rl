# AGENTS.md

本文件为 Codex 及其他编程智能体提供仓库级协作约定。规则适用于整个仓库；如果子目录中存在更具体的 `AGENTS.md`，则以距离目标文件最近的规则为准。所有开发、提交、评审和合并操作还必须遵守 `CONTRIBUTING.md`。

## 仓库定位

本仓库是 Unitree 足式机器人强化学习项目的集成工作区，主要通过 Git 子模块管理两个上游 fork：

- `modules/unitree_rl_lab`：基于 Isaac Lab 的训练、推理与部署代码。
- `modules/unitree_rl_gym`：基于 Isaac Gym 的训练、推理与部署代码。

根仓库只维护集成层内容，例如子模块版本、`.gitmodules`、文档和工作区级工具。功能代码应在对应子模块中修改并提交，再由根仓库更新子模块引用。

## 开始工作前

1. 阅读根目录 `README.md` 以及目标子模块自己的 README 和开发文档。
2. 阅读根目录 `CONTRIBUTING.md`，确认分支、提交和 PR 要求。
3. 使用 `git status --short` 检查根仓库状态；修改子模块时，也要在子模块内单独检查状态。
4. 不覆盖或清理用户已有的未提交修改。
5. 如果子模块尚未初始化，运行：

   ```bash
   git submodule update --init --recursive
   ```

## 修改约定

- 遵循 `main`、`develop`、`feature/*`、`release/*` 和 `hotfix/*` 分支规范；不得绕过 PR 直接修改受保护分支。
- 保持改动聚焦，不顺带重构无关代码。
- 遵循目标子模块现有的代码风格、目录结构和命名方式。
- 不直接复制两个子模块间的大段代码；确需共享时，先说明设计取舍。
- 修改子模块后，分别报告子模块内的改动和根仓库中的 gitlink 变化。
- 不提交训练日志、模型权重、数据集、缓存、构建产物或本机环境配置，除非任务明确要求。
- 文档默认使用中文；命令、路径、代码标识符和上游专有名词保持原样。

## 验证

验证范围应与改动风险相匹配，并优先使用目标子模块已有的工具。

### unitree_rl_lab

在 `modules/unitree_rl_lab` 中：

```bash
./unitree_rl_lab.sh -l
pre-commit run --all-files
```

Isaac Lab、Isaac Sim、GPU 或机器人资源不可用时，只运行当前环境可执行的静态检查，并明确说明未运行哪些验证及原因。

### unitree_rl_gym

在 `modules/unitree_rl_gym` 中，根据改动选择最小可行验证：

```bash
python legged_gym/scripts/train.py --help
python legged_gym/scripts/play.py --help
```

训练、仿真和部署通常依赖 Isaac Gym、MuJoCo、GPU 或机器人硬件；不要为了验证普通代码改动而启动长时间训练。

### 根仓库

修改文档或子模块配置后至少运行：

```bash
git diff --check
git submodule status
```

## 安全边界

- 未经明确要求，不运行真实机器人部署、网络控制或 Sim2Real 命令。
- 不执行可能移动机器人、写入硬件、占用机器人网络接口或影响急停状态的操作。
- 不下载大型数据集、模型或仿真资源，除非完成任务确有必要。
- 不修改 Git 远端、不推送分支、不创建发布版本，除非用户明确要求。

## 交付说明

完成任务时简要说明：

- 修改了哪些文件或子模块；
- 运行了哪些检查及结果；
- 哪些验证因缺少仿真器、GPU 或硬件而未运行；
- 是否还存在需要用户处理的子模块提交或根仓库 gitlink 更新。
