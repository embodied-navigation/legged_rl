# H2 Sim2Sim 设计

**日期：** 2026-07-21
**状态：** 已确认
**目标：** 使用 `unitree_mujoco` 在笔记本的 MuJoCo 3.3.6 环境中，验证 `unitree_rl_lab` 训练出的 H2 强化学习速度步态策略，并建立可复用于后续 Sim2Real 的 C++/SDK2/DDS 部署链路。

## 1. 背景与范围

当前 H2 速度步态策略在 RTX 4090 服务器的 Isaac Lab 环境中完成训练，首版 Sim2Sim 使用以下正式 checkpoint：

```text
logs/rsl_rl/h2_velocity/2026-07-21_16-54-04/model_4999.pt
```

笔记本没有安装 Isaac Lab，但已经安装 MuJoCo：

```text
~/.mujoco/mujoco-3.3.6
```

本设计选择 `unitree_mujoco` 的 C++ simulator 和 Unitree SDK2/DDS 低层控制链路，而不是 Python 直接控制 MuJoCo。这样不仅验证策略在不同物理引擎中的表现，还验证观测构造、关节映射、PD 控制、控制周期和部署通信接口，为后续 Sim2Real 保留同一套 H2 controller。

首版包含人工 GUI 验证和固定指令回归，但不涉及真实机器人控制。

## 2. 总体架构

```text
RTX 4090 训练服务器
  model_4999.pt
       │ Isaac Lab 导出
       ▼
  policy.onnx + deploy.yaml
       │ SFTP / rsync，不进入 Git
       ▼
笔记本 unitree_rl_lab/deploy/robots/h2
  H2 C++ Controller
  - ONNX Runtime 推理
  - Passive / FixedStand / Policy 状态机
  - 观测构造与动作映射
  - 固定指令与手柄输入
  - 指标记录
       │ Unitree SDK2 DDS
       │ domain_id=1, interface=lo
       ▼
modules/unitree_mujoco
  - H2 MJCF 与 scene
  - H2 motor index / actuator 映射
  - LowState 发布
  - LowCmd 接收
  - MuJoCo 3.3.6 仿真与 GUI
```

仓库职责如下：

- 根仓库负责 `modules/unitree_mujoco` 子模块引用、设计文档、实施计划和工作区级说明。
- `embodied-navigation/unitree_mujoco` fork 固定官方已经提供的 H2 MJCF、场景及 SDK2 simulator bridge 版本；只有验证发现缺陷时才增加必要修复。
- `modules/unitree_rl_lab` 负责可复用于 Sim2Real 的 H2 ONNX controller。
- checkpoint、ONNX、导出参数、运行日志和指标产物均保存在 Git 忽略目录中。

`unitree_mujoco` 以 Unitree SDK2 的 `LowState`/`LowCmd` 为低层接口，符合本设计的部署目标。经实施前核对，官方 [`unitree_mujoco`](https://github.com/unitreerobotics/unitree_mujoco) 主分支已经包含 `unitree_robots/h2/h2_mujoco.xml`、`scene.xml`、31 个 actuator/sensor 和基于 `unitree_hg` 消息的 bridge。团队 fork 直接固定并验证该上游实现，不再从 [`unitree_model`](https://github.com/unitreerobotics/unitree_model) 重复复制模型；只需核对模型来源、版本和许可证。

## 3. 模型与关节职责

MuJoCo 使用完整 H2 模型参与动力学计算，不把机器人精简为 15 关节模型：

- 策略控制双腿 12 个关节和腰部 3 个关节。
- 其他可驱动关节由 controller 使用 PD 保持官方模型的默认姿态。
- controller 必须填充完整 SDK2 `LowCmd`，不得遗漏非策略关节。
- 需要建立并验证 `Isaac Lab joint → MJCF joint → actuator → SDK2 motor index` 显式映射。

每个关节需要核对名称、顺序、旋转方向、零位、角度单位、软硬限位、gear、力矩限制和默认姿态。单关节低幅测试通过前，不允许运行整机策略。

## 4. 策略输入输出契约

策略输入严格复现训练时的 56 维 policy observation：

| Observation | 维度 | 缩放 |
|---|---:|---:|
| `base_ang_vel` | 3 | `0.2` |
| `projected_gravity` | 3 | `1.0` |
| `velocity_commands` | 3 | `1.0` |
| `joint_pos_rel` | 15 | `1.0` |
| `joint_vel_rel` | 15 | `0.05` |
| `last_action` | 15 | `1.0` |
| `gait_phase` | 2 | `1.0` |
| 合计 | 56 | — |

策略输出为 15 维，顺序固定为：

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

训练控制周期为：

```text
sim.dt 0.005s × decimation 4 = policy step 0.02s = 50Hz
```

每个策略动作保持至下一个 20ms 周期。动作转换使用训练导出的默认关节偏置和 scale：

- hip、knee、ankle：`0.25`
- waist：`0.10`

gait phase 使用与训练一致的 `0.6s` 周期，并在进入 `Policy` 时从零开始。ONNX 导出应包含训练时的 observation normalizer，controller 不重复归一化。

## 5. 导出与模型交付

策略在安装 Isaac Lab 的 4090 服务器上导出：

```text
model_4999.pt
  → policy.onnx
  → deploy.yaml
```

导出结果通过 SFTP 或 rsync 同步到笔记本：

```text
modules/unitree_rl_lab/deploy/robots/h2/config/policy/velocity/
  exported/policy.onnx
  params/deploy.yaml
```

该目录中的模型和生成参数不提交 Git。原始 checkpoint 也不复制进仓库。C++ controller 使用 ONNX Runtime 加载策略，笔记本无需安装 Isaac Lab。

controller 启动时必须 fail-fast 校验：

- ONNX 输入为 56 维、输出为 15 维。
- `deploy.yaml` 中的动作和观测顺序符合 H2 编译期映射。
- 15 个训练关节均能唯一映射到 MJCF、actuator 和 SDK2 motor index。
- 控制周期、offset、scale、PD 和 limit 参数完整有效。
- DDS 状态、IMU 和 observation 不含 NaN 或无穷值。

任何映射缺失、状态超时或非法数值都应阻止进入 `Policy`。

## 6. 状态机与安全边界

H2 controller 使用三个明确状态：

```text
Passive
   │ 手动触发，状态与通信检查通过
   ▼
FixedStand
   │ 3 秒平滑插值到默认站姿
   │ 姿态稳定并由用户确认
   ▼
Policy
   │ ONNX 50Hz 推理
   │
   ├─ 通信超时 / NaN / 严重越界 / 跌倒 ─→ Passive
   └─ 用户停止 ─────────────────────────→ Passive
```

- `Passive` 不运行策略，使用安全阻尼或 simulator 允许的最低安全输出，配合虚拟挂带放置 H2。自动回归中，simulator 检测到 controller 首次发送非零 stiffness 后延迟 4 秒释放挂带并显式清零外力；人工按键 9 仍可用于 GUI 验证。
- `FixedStand` 从当前关节位置以 3 秒插值进入训练默认站姿，不允许目标角度阶跃。
- 进入 `Policy` 时清零 `last_action`、gait phase 和指令滤波状态。
- `Policy` 的前 1 秒将策略动作权重从 0 平滑增加到 1。
- 非策略关节在状态切换时保持连续的 PD 目标。
- DDS 超时、IMU 非法、观测或动作出现 NaN、严重关节越界、torso 高度过低或姿态超限时，立即停止策略并进入安全状态。

Sim2Sim 默认固定使用：

```text
domain_id=1
interface=lo
```

不得沿用现有 G1 controller 中默认 DDS domain 0 的行为。改变 domain 或网络接口必须显式传参，避免误连真实机器人。未经单独授权，本任务不运行任何真实机器人或 Sim2Real 命令。

## 7. 指令输入

controller 支持两种互斥输入源：

1. `unitree_mujoco` 发布的 `wireless_controller` 手柄输入，用于 GUI 交互。
2. YAML 固定速度序列，用于可重复的自动验收。

两种输入都经过相同的限幅和斜坡处理：

```text
lin_vel_x: [-0.3, 1.0] m/s
lin_vel_y: [-0.3, 0.3] m/s
ang_vel_z: [-0.5, 0.5] rad/s
```

启动参数必须明确选择输入源，禁止两套输入同时覆盖速度命令。

## 8. 最小文件职责

根仓库新增：

```text
.gitmodules
modules/unitree_mujoco/
docs/superpowers/specs/2026-07-21-h2-sim2sim-design.md
docs/superpowers/plans/2026-07-21-h2-sim2sim.md
```

`unitree_mujoco` fork 复用并固定上游已有的 H2 simulator 资源：

```text
unitree_robots/h2/h2_mujoco.xml
unitree_robots/h2/scene.xml
simulate/config.yaml                 # 支持 robot: h2
simulate SDK2 bridge                 # 31 actuator 时使用 unitree_hg bridge
```

不额外复制 H2 URDF，也不重复导入官方已经包含的 H2 MJCF。只有映射、消息类型或仿真行为验证发现问题时，才在 fork 中提交最小修复，并保留官方来源和许可证说明。

`unitree_rl_lab` 沿用已有机器人部署结构：

```text
deploy/robots/h2/CMakeLists.txt
deploy/robots/h2/main.cpp
deploy/robots/h2/include/Types.h
deploy/robots/h2/src/State_RLBase.cpp
deploy/robots/h2/config/config.yaml
```

共享 FSM、ONNX runner、Isaac Lab observation/action 实现继续复用 `deploy/include`。指标记录优先集成到共享环境或 H2 controller，不为 H2 重写一套部署框架。

## 9. 验证方案

### 9.1 静态映射验证

- 验证 MJCF joint、actuator 和 SDK2 motor index 一一对应。
- 验证名称、方向、零位、限位、gear 和 torque limit。
- 验证 ONNX 56 维输入和 15 维输出。
- 验证 `deploy.yaml` 的控制周期、scale、offset 和 PD 参数。
- 使用单关节小幅命令确认正负方向。

### 9.2 MuJoCo 基础验证

- MuJoCo 3.3.6 能加载 H2 scene。
- DDS 状态频率稳定，无 NaN、时间倒退或持续丢失。
- 虚拟挂带可以升降和释放机器人。
- `FixedStand` 连续保持 20 秒，无跌倒、异常接触或关节硬限位越界。

### 9.3 策略烟雾验证

- 零速度进入 `Policy` 并持续 10 秒。
- 然后以 `0.2m/s` 前进 10 秒。
- 不出现 NaN、动作维度错误、通信超时、严重限位越界或立即跌倒。
- 失败时保留最近一段 observation、action、joint state 和 IMU 数据。

### 9.4 固定指令回归

| 场景 | 指令 | 持续时间 |
|---|---|---:|
| 静止 | `(0.0, 0.0, 0.0)` | 10s |
| 前进 | `lin_vel_x=+0.3m/s` | 10s |
| 后退 | `lin_vel_x=-0.2m/s` | 10s |
| 左侧移 | `lin_vel_y=+0.15m/s` | 10s |
| 右侧移 | `lin_vel_y=-0.15m/s` | 10s |
| 左转 | `ang_vel_z=+0.3rad/s` | 10s |
| 右转 | `ang_vel_z=-0.3rad/s` | 10s |
| 停止恢复 | `(0.0, 0.0, 0.0)` | 10s |

首版硬性通过条件：

- 完成完整序列，不崩溃且不出现 NaN。
- DDS 和 control loop 无持续超时。
- 无非足部严重接触。
- 无关节越过硬限位。
- 完成测试后机器人仍保持站立。

首版记录但暂不设置硬阈值：

- 各场景存活率。
- 线速度和 yaw RMSE。
- soft-limit 越界次数及关节分布。
- torso 高度以及 roll/pitch 峰值。
- 关节峰值速度、力矩和 action rate。
- controller 实际周期、DDS 延迟和丢包。
- 足部滑移和非预期接触。

接触分类和接触力由 MuJoCo 进程直接记录；controller 记录策略指令、状态、动作、速度跟踪和关节限位。两个进程使用同一个运行目录，避免从 DDS 状态推断 simulator 内部接触。

运行产物写入 Git 忽略目录：

```text
artifacts/h2_sim2sim/<timestamp>/
  metrics.json
  timeseries.csv
  run.log
```

首轮结果作为 MuJoCo 基线。获得真实数据后，再为速度 RMSE、soft-limit rate 和姿态稳定性设置合理阈值。

## 10. 失败诊断顺序

出现“策略加载成功但机器人不动或倒下”时，按以下顺序定位：

```text
关节映射与方向
→ 初始姿态与默认 offset
→ 观测坐标系和 IMU 四元数
→ 控制周期与 gait phase
→ PD、torque 和 actuator 差异
→ 接触、摩擦和模型惯量
→ 最后才考虑重新训练策略
```

该顺序用于避免把部署实现错误误判为强化学习策略问题。

## 11. 实施与提交顺序

1. 创建 `embodied-navigation/unitree_mujoco` fork，并以子模块加入根仓库。
2. 固定并验证上游已有的 H2 MJCF、scene、31 actuator/sensor 和 `unitree_hg` SDK2 bridge；只在测试失败时提交最小修复。
3. 在 4090 服务器导出当前正式策略及部署参数。
4. 在 `unitree_rl_lab` 中实现 H2 controller、状态机适配和指标记录。
5. 在笔记本依次完成静态、FixedStand、烟雾和固定指令验证。
6. 分别提交 `unitree_mujoco` 和 `unitree_rl_lab` 子模块 PR。
7. 子模块提交可获取并完成评审后，根仓库提交 gitlink、文档和集成 PR。
