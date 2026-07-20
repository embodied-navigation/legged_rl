# H2 强化学习步态训练设计

## 1. 目标

在 `modules/unitree_rl_lab` 中新增 H2 平地全向速度跟踪任务 `Unitree-H2-Velocity`，使用 RSL-RL PPO 训练 H2 双腿和腰部共 15 个可动关节。第一阶段只覆盖 Isaac Lab 训练、推理和固定指令集性能评估，不包含 MuJoCo Sim2Sim、Sim2Real 或真实机器人控制。

训练基线使用 RTX 4090、4096 个并行环境、每环境每次采样 24 步和最多 5000 次迭代。

## 2. 方案选择

采用现有 `Unitree-H1-Velocity` 任务作为模板，新增独立 H2 任务，不重构 H1/G1 的公共配置。这样可以复用已经验证的 Manager-Based RL、观测、奖励和 RSL-RL 训练结构，同时把 H2 的资产、动作、执行器和奖励差异限制在独立目录中。

未采用以下方案：

- 不先抽取通用 humanoid 基类，避免扩大已有任务的回归范围；
- 不从零构建最小任务，避免遗漏现有速度跟踪任务的关键约定；
- 不在子模块中复制 H2 URDF 或 mesh，避免重复维护大型资产。

## 3. 资产依赖

H2 任务直接使用根仓库中的：

- `assets/urdf/h2_description/H2_simple.urdf`
- `assets/urdf/h2_description/meshes/*.stl`

`unitree_rl_lab` 从自身文件位置向父目录查找 `assets/urdf/h2_description/H2_simple.urdf`，不使用本机绝对路径。路径不存在时立即报错，并提示 H2 任务必须在完整 `legged_rl` 工作区中运行。

该决定意味着：H1、G1 和 Go2 任务仍可独立使用子模块，H2 任务不能在单独克隆的 `unitree_rl_lab` 中运行。

## 4. 代码结构

子模块新增：

- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/__init__.py`：注册 `Unitree-H2-Velocity`；
- `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/velocity_env_cfg.py`：H2 场景、动作、观测、命令、奖励、事件和终止配置；
- H2 专用 RSL-RL runner 配置：固定实验名和 5000 次迭代；
- 固定指令集评估入口：加载 checkpoint 并输出验收指标。

子模块修改：

- `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`：增加 `UNITREE_H2_CFG` 和工作区资产路径解析；
- 必要的包导入文件：确保 Gym 自动注册 H2 任务。

功能代码和配置先提交到 `unitree_rl_lab` 子模块，再由根仓库更新 gitlink。根仓库不直接承载训练功能代码。

## 5. 机器人与动作配置

`UNITREE_H2_CFG` 使用 `UnitreeUrdfFileCfg` 导入 `H2_simple.urdf`，保留全部 15 个可动关节：

- 左腿 6 个；
- 右腿 6 个；
- `waist_yaw_joint`、`waist_roll_joint`、`waist_pitch_joint`。

动作采用关节位置目标，并显式声明顺序，不依赖 URDF 解析顺序：

```text
left_hip_pitch_joint
left_hip_roll_joint
left_hip_yaw_joint
left_knee_joint
left_ankle_roll_joint
left_ankle_pitch_joint
right_hip_pitch_joint
right_hip_roll_joint
right_hip_yaw_joint
right_knee_joint
right_ankle_roll_joint
right_ankle_pitch_joint
waist_yaw_joint
waist_roll_joint
waist_pitch_joint
```

双腿 action scale 为 `0.25`，腰部 action scale 为 `0.10`。初始姿态采用轻微屈膝站立，腰部保持零位；base 高度根据 sole 最低面确定，避免初始悬空或穿地。

执行器第一版使用分组 Ideal PD 初值：hip/knee 刚度较高，ankle 使用中等刚度，waist 使用适中刚度和较小动作幅度。effort/velocity limit 不得超过 URDF 数值，armature 使用保守小值。所有临时 PD 参数必须在代码注释中标明需要根据 H2 电机和控制器规格复核。

## 6. 环境与命令

环境使用 Isaac Lab `ManagerBasedRLEnv`：

- 纯平面地形；
- 4096 个并行环境；
- `sim.dt=0.005`；
- decimation 为 4；
- episode 长度为 20 秒；
- 不启用 terrain curriculum 或高度扫描器；
- 保留摩擦随机化、轻量 base mass 随机化和周期推扰；
- 仅使用速度命令 curriculum，从小指令逐步扩展到完整范围。

完整命令范围：

- `lin_vel_x`: `[-0.3, 1.0] m/s`；
- `lin_vel_y`: `[-0.3, 0.3] m/s`；
- `ang_vel_z`: `[-0.5, 0.5] rad/s`；
- 保留少量静止指令环境，用于学习稳定站立。

## 7. 观测、奖励与终止

策略观测沿用 H1 的结构：

- base angular velocity；
- projected gravity；
- velocity command；
- relative joint position；
- relative joint velocity；
- last action；
- gait phase。

critic 额外使用 base linear velocity 和 joint effort 等 privileged observations。训练时启用与 H1 同类的观测噪声，评估时关闭噪声。

奖励包含：

- XY 线速度与 yaw 角速度跟踪；
- 存活奖励；
- base 竖直速度、roll/pitch 角速度和平姿态惩罚；
- joint acceleration、action rate 和关节限位惩罚；
- 双脚交替步态、滑移、离地高度和接触力；
- 腰部偏离默认姿态惩罚；
- 非足部接触惩罚。

删除 H1 的手臂关节偏离奖励，因为 H2 手臂已在 URDF 中固定。pelvis 或 torso 触地时终止 episode；超时属于正常终止。

足部接触、步态和离地高度项必须显式使用 H2 的 `left_ankle_pitch_link` 与 `right_ankle_pitch_link`，不得依赖会匹配其他 ankle link 的宽泛表达式。

## 8. PPO 配置

H2 使用独立 runner 配置：

- `num_steps_per_env=24`；
- `max_iterations=5000`；
- `save_interval=100`；
- `experiment_name="h2_velocity"`；
- 基准 seed 为 `42`；
- actor hidden dims 为 `[512, 256, 128]`；
- critic hidden dims 为 `[512, 256, 128]`；
- 其余 PPO 参数继承现有 `BasePPORunnerCfg`。

第一版只要求单 seed 达标。多 seed 均值和方差属于后续稳定性验证，不阻塞本设计的首次验收。

## 9. 验证流程

### 9.1 配置验证

- `./unitree_rl_lab.sh -l` 列出 `Unitree-H2-Velocity`；
- 单环境能够创建并重置；
- 导入后可动关节集合与动作维度均严格为 15；
- 关节顺序与本设计一致；
- URDF、mesh、link 或 joint 不匹配时立即失败。

### 9.2 短训验证

使用少量环境运行 10～20 次迭代，确认：

- 无 NaN；
- 无 action/observation 维度错误；
- 无 CUDA、PhysX 或仿真崩溃；
- 无异常初始接触或明显关节越限；
- checkpoint 能被 `play.py` 加载。

### 9.3 正式训练

在 RTX 4090 上使用 4096 个环境、24 个采样步和 5000 次迭代完成 seed 42 基准训练。训练日志必须记录最终 checkpoint、墙钟时间和关键 reward 曲线。

## 10. 固定指令集评估

评估入口加载最终 checkpoint，关闭 observation noise、push 和 domain randomization，每类指令运行 20 秒：

- 静止；
- 前进 `0.5 m/s`；
- 后退 `-0.2 m/s`；
- 横移 `+0.2 m/s`；
- 横移 `-0.2 m/s`；
- 偏航 `+0.3 rad/s`；
- 偏航 `-0.3 rad/s`；
- 固定 seed 的随机混合指令。

输出以下聚合指标：

- episode 存活率；
- XY 线速度 RMSE；
- yaw 角速度 RMSE；
- 非足部接触 episode 比例；
- NaN、关节越限和仿真异常计数。

性能验收标准：

- 20 秒 episode 存活率不低于 90%；
- XY 线速度 RMSE 不高于 `0.20 m/s`；
- yaw 角速度 RMSE 不高于 `0.25 rad/s`；
- 非足部接触 episode 比例不高于 5%；
- 静止指令下无持续漂移或明显原地踏步；
- 训练和评估无 NaN、崩溃或异常关节越限。

评估脚本在资产路径缺失、checkpoint 与动作维度不匹配或指标计算不完整时返回非零状态，不输出误导性的部分通过结果。

## 11. 非目标

本设计不包含：

- 粗糙地形、台阶或 terrain curriculum；
- H2 动作模仿；
- MuJoCo Sim2Sim；
- 策略部署和 Sim2Real；
- 真实机器人通信、控制或安全验证；
- 多随机种子统计；
- 已有 H1/G1/Go2 配置重构。

## 12. 本地开发与服务器验证

开发笔记本未安装 Isaac Lab，因此本地只执行：

- 代码编辑；
- Python 语法和不依赖 Isaac Lab 的静态检查；
- URDF/XML 与资源路径检查；
- Git diff、提交和子模块状态检查。

本地静态检查不能替代 Isaac Lab 运行验证，也不能作为 H2 任务可训练或可推理的完成证据。

Isaac Lab 验证统一在 RTX 4090 服务器上进行：

```text
host: 172.16.1.101
workspace: /home/gaojie/workspace/legged_rl
unitree_rl_lab: /home/gaojie/workspace/legged_rl/modules/unitree_rl_lab
conda environment: env_isaaclab
```

在服务器终端中进入环境：

```bash
cd /home/gaojie/workspace/legged_rl/modules/unitree_rl_lab
conda activate env_isaaclab
```

服务器负责执行：

- `./unitree_rl_lab.sh -l` 任务注册检查；
- 单环境创建和重置；
- 10～20 次迭代短训；
- 4096 环境、5000 次迭代正式训练；
- checkpoint 推理；
- 固定指令集性能评估。

开发期通过 VS Code SFTP 同步到服务器：

```text
remotePath: /home/gaojie/workspace/legged_rl
uploadOnSave: true
```

SFTP 只作为文件传输通道，Git 仍是唯一版本基准。每次服务器验证前必须确认目标文件已上传；正式训练前要求本地代码已经提交，并记录被测 Git commit。训练日志和 checkpoint 保留在服务器，不提交到 Git。

`.vscode/` 不纳入版本控制，避免提交公司内网地址、用户名和服务器路径。如果 SSH/SFTP 或服务器环境不可用，Isaac Lab 验证应明确标记为阻塞，不得用本地静态检查替代。
