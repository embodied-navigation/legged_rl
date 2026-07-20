# legged_rl

用于足式机器人强化学习、训练与部署的集成工作区。

## 子仓库

- `modules/unitree_rl_lab`：[unitreerobotics/unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) 的 fork，基于 Isaac Lab。
- `modules/unitree_rl_gym`：[unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) 的 fork，基于 Isaac Gym。

## 克隆仓库

克隆仓库及其所有子仓库：

```bash
git clone --recurse-submodules git@github.com:embodied-navigation/legged_rl.git
```

如果已经克隆了主仓库，可通过以下命令初始化并更新子仓库：

```bash
git submodule update --init --recursive
```

## 参与开发

开发、分支、提交、代码评审、合并和版本发布规则请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)，版本变化记录见 [CHANGELOG.md](CHANGELOG.md)。Codex 等编程智能体还应遵守 [AGENTS.md](AGENTS.md)。
