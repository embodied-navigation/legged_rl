# legged_rl

用于足式机器人强化学习、训练与部署的集成工作区。

## 子仓库

- `modules/unitree_rl_lab`：[unitreerobotics/unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) 的 fork，基于 Isaac Lab。
- `modules/unitree_rl_gym`：[unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) 的 fork，基于 Isaac Gym。
- `modules/unitree_mujoco`：[unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) 的 fork，用于 SDK2/DDS Sim2Sim 验证。

## 克隆仓库

克隆仓库及其所有子仓库：

```bash
git clone --recurse-submodules git@github.com:embodied-navigation/legged_rl.git
```

如果已经克隆了主仓库，可通过以下命令初始化并更新子仓库：

```bash
git submodule update --init --recursive
```

## H2 Sim2Sim

H2 的 MuJoCo 3.3.6、C++ controller、策略导出和固定指令验收流程见：

- [设计文档](docs/superpowers/specs/2026-07-21-h2-sim2sim-design.md)
- [实施计划与验证记录](docs/superpowers/plans/2026-07-21-h2-sim2sim.md)

Sim2Sim 默认使用 DDS domain 1 和 loopback 接口，不连接真实机器人网络。checkpoint、ONNX、运行日志和 `artifacts/` 不提交仓库。

## 参与开发

开发、分支、提交、代码评审、合并和版本发布规则请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)，版本变化记录见 [CHANGELOG.md](CHANGELOG.md)。Codex 等编程智能体还应遵守 [AGENTS.md](AGENTS.md)。
