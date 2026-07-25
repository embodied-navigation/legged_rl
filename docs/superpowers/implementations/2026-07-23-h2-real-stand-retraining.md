# H2 蹲姿参数重训练实施记录

## 状态

- 训练状态：已完成5000 iterations。
- 日志目录：`modules/unitree_rl_lab/logs/rsl_rl/h2_velocity/2026-07-25_11-29-25/`。
- 固定指令集验收：已执行，基本验收未通过。
- 模型、TensorBoard 日志和评估 JSON 仅保存在训练服务器，不提交仓库。
- 本地 `unitree_rl_lab` 基线提交：`6695516 feat: tune H2 crouched locomotion training`。

本轮最初按实机对称站姿设计，实际调参后采用训练效果较好的 `1.05 m` 对称蹲姿。当前实现和修订后的计划以实际训练配置为准。

## 实际训练配置

### 资产、初始状态与执行器

- URDF：`assets/urdf/h2_description/H2_stand.urdf`。
- 根链接初始高度：`1.05 m`。
- 默认关节位置：
  - `.*_hip_pitch_joint = -0.15`
  - `.*_knee_joint = 0.30`
  - `.*_ankle_pitch_joint = -0.20`
- 动作维度：15。
- Hip/Knee：`Kp=300`、`Kd=5`。
- Ankle Roll/Pitch：`Kp=60`、`Kd=3`。
- Waist：`Kp=300`、`Kd=5`。

### 环境和奖励调参

- `push_robot = None`。
- 静止环境比例：`0.2`。
- 初始命令：
  - `lin_vel_x=(0.2, 0.3)`
  - `lin_vel_y=(0.0, 0.0)`
  - `ang_vel_z=(-0.5, 0.5)`
- curriculum 命令上限：
  - `lin_vel_x=(-0.3, 0.5)`
  - `lin_vel_y=(-0.3, 0.3)`
  - `ang_vel_z=(-0.5, 0.5)`
- policy 和 critic 使用五帧 observation history，不使用 `gait_phase` observation。
- 关键奖励参数：
  - `track_lin_vel_xy=1.3`
  - `base_angular_velocity=-0.05`
  - `joint_acc=-2.5e-8`
  - `action_rate=-0.002`
  - `dof_pos_limits=-2.0`
  - base height target `0.97`
- base height termination threshold：`0.5`。

### PPO 和正式训练

- Runner 默认 `max_iterations=30000`，正式训练通过命令行限制为5000轮。
- `num_steps_per_env=24`。
- `save_interval=100`。
- experiment：`h2_velocity`。
- 当前 H2 runner 继承 Base PPO 配置，不单独覆盖 `policy`、`algorithm` 或 `clip_actions`。
- 本轮使用调整后的参数重新训练，不恢复旧 checkpoint。

## 已完成检查

在本地执行：

```bash
cd modules/unitree_rl_lab
python3 -m pytest -q test/test_h2_locomotion_static.py
python3 -m compileall -q \
  source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py \
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py \
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/velocity_env_cfg.py \
  test/test_h2_locomotion_static.py
git diff --check
```

结果：

```text
21 passed
compileall PASS
unitree_rl_lab git diff --check PASS
root git diff --check PASS
```

本地未安装 Isaac Lab，因此没有执行 Isaac Lab 加载、仿真或 checkpoint 推理。

仓库规定的任务注册检查也受本地环境限制：

```text
./unitree_rl_lab.sh -l
[Error] No conda environment activated.

python3 scripts/list_envs.py
ModuleNotFoundError: No module named 'gymnasium'
```

任务注册、Isaac Lab 加载和 checkpoint 推理必须在101训练服务器的
`legged_rl_unitree_rl_lab` 环境中完成。

## 训练记录

101训练服务器记录：

- 根仓库 SHA：`2eaaaf9ba9c93ae466115d1f01983fe42ff4f5b2`。
- `unitree_rl_lab` SHA：`4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`。
- 最终 checkpoint：`model_4999.pt`，大小 `7,869,941 bytes`。
- seed：`42`。
- 训练时间：`2026-07-25 11:29:56` 至 `13:49:52`，约 `2 h 19 min 56 s`。
- 最终 mean reward：`10.7365`。
- 最终 mean episode length：`1000.0`。
- `Episode_Termination/time_out=0.998658`。
- `Episode_Termination/base_height=0.001342`。
- `Metrics/base_velocity/error_vel_xy=0.196283`。
- `Metrics/base_velocity/error_vel_yaw=0.467955`。
- 最终 `value_function loss=0.005816`，`mean_noise_std=0.431364`，没有在最终轮出现 NaN、Inf 或非法 action std。

训练时服务器仓库不是干净提交状态。日志保存了 `git/unitree_rl_lab.diff` 和参数快照，复现时必须同时保留这些文件，不能只引用上述 SHA。服务器工作区还包含与本轮训练无关的 ONNX Runtime 二进制变化，不应提交。

训练产物位于：

```text
/home/gaojie/workspace/legged_rl/modules/unitree_rl_lab/logs/rsl_rl/h2_velocity/2026-07-25_11-29-25/
```

## 固定指令集验收

已使用最终 checkpoint 执行：

```bash
python scripts/rsl_rl/evaluate_h2.py --headless \
  --task Unitree-H2-Velocity \
  --checkpoint logs/rsl_rl/h2_velocity/2026-07-25_11-29-25/model_4999.pt \
  --num_envs 256 \
  --duration 20 \
  --seed 42 \
  --output /tmp/h2_2026-07-25_11-29-25_eval.json
```

聚合结果：

| 指标 | 结果 | 标准 | 判定 |
|---|---:|---:|---|
| 20秒 episode 存活率 | `97.61%` | `>= 90%` | 通过 |
| XY 线速度 RMSE | `0.1801 m/s` | `<= 0.20 m/s` | 通过 |
| Yaw 角速度 RMSE | `0.1552 rad/s` | `<= 0.25 rad/s` | 通过 |
| 非期望接触 episode 比例 | `100%` | `<= 5%` | 失败 |
| Soft-limit invalid rate | `4.5637%` | `<= 0.1%` | 失败 |

共检查 `2,013,795` 个样本，`invalid_count=91,903`。基本验收和优化目标均未通过。

分场景结果：

| 场景 | 存活率 | 线速度 RMSE | Yaw RMSE | Invalid rate |
|---|---:|---:|---:|---:|
| stand | `100%` | `0.0117` | `0.0187` | `0.4855%` |
| forward | `100%` | `0.2530` | `0.0943` | `0.5438%` |
| backward | `100%` | `0.2018` | `0.0151` | `0.4977%` |
| left | `100%` | `0.2004` | `0.0324` | `0.4563%` |
| right | `100%` | `0.2002` | `0.0182` | `0.5199%` |
| yaw_left | `100%` | `0.0213` | `0.2692` | `31.0270%` |
| yaw_right | `100%` | `0.0140` | `0.2939` | `0.5047%` |
| mixed | `80.86%` | `0.2635` | `0.2004` | `2.1524%` |

主要诊断：

- 所有场景都报告 `torso_link`、左右 `shoulder_roll_link` 非期望接触，需要先确认 fixed 上半身的自碰撞/接触过滤和评估统计语义。
- `yaw_left` 出现 `78,180` 次左膝 soft-limit violation，是 aggregate invalid rate 的主要来源，并表现出明显左右转向不对称。
- 其余多数场景持续出现左右 ankle roll soft-limit violation。
- `mixed` 场景存活率只有 `80.86%`，并伴随膝、踝越限以及 pelvis/knee 非期望接触。
- 不应通过放宽 URDF 硬限位消除失败；下一轮应优先检查接触过滤、左膝动作/关节映射、默认姿态对 yaw 的偏置以及相应奖励约束。
