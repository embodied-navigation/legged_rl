# H2 实机站姿参数重训练设计

## 1. 目标

基于 H2 真实机器人在运动模式静止状态下采集的关节位置、PD 增益和根链接高度，更新现有 `Unitree-H2-Velocity` 训练基线，并从头训练新的 15-DoF 平地速度策略。

本轮只改变机器人模型、默认站姿、初始高度和执行器 PD 参数。动作缩放、奖励、速度命令、环境随机化和 PPO 参数保持不变，以便与原训练结果有效对比。

## 2. 方案选择

继续使用现有任务和实验名称：

```text
Task: Unitree-H2-Velocity
Experiment: h2_velocity
```

不新增任务注册或实验名称。RSL-RL 使用新的时间戳日志目录隔离本轮训练，旧 checkpoint 和日志保留。

实机站立角度写入 Isaac Lab 的 `init_state.joint_pos`，不写入 URDF 的 `joint origin`。策略动作继续使用 `use_default_offset=True`，因此策略输出表示相对默认站姿的关节位置偏移。

未采用把站姿角度烘焙进关节 origin 并平移关节限位的方案。该方案会改变关节坐标系和零位定义，复合髋关节尤其容易产生错误。

## 3. H2_stand URDF

`assets/urdf/h2_description/H2_stand.urdf` 从 `H2_simple.urdf` 重新构建：

- 腿部 12 个关节和腰部 3 个关节保持可动，共 15 DoF；
- 15 个可动关节的 origin、axis 和 limit 与 `H2_simple.urdf` 保持一致；
- 不把腿部或腰部默认站姿写入 joint origin；
- 上半身全部固定，不进入动作空间；
- 上半身使用左右对称的实机静止近似姿态。

固定上半身姿态如下：

| 关节 | 左侧 rad | 右侧 rad |
|---|---:|---:|
| Shoulder Pitch | `0.0` | `0.0` |
| Shoulder Roll | `0.4` | `-0.4` |
| Shoulder Yaw | `-0.4` | `0.4` |
| Elbow | `1.1` | `1.1` |
| Wrist Roll | `0.0` | `0.0` |
| Wrist Pitch | `0.0` | `0.0` |
| Wrist Yaw | `0.0` | `0.0` |
| Head Pitch | `0.0` | — |
| Head Yaw | `0.0` | — |

固定姿态应通过正确的关节变换构建，不改变下肢和腰部的运动学定义。

## 4. 资产路径

H2 资产路径由原来的 `H2_simple.urdf` 改为 `H2_stand.urdf`。按照本轮代码可追溯性要求，旧参数注释保留，新参数写在其下方：

```python
# H2_URDF_RELATIVE_PATH = Path("assets/urdf/h2_description/H2_simple.urdf")
H2_URDF_RELATIVE_PATH = Path("assets/urdf/h2_description/H2_stand.urdf")
```

路径必须相对完整的 `legged_rl` 工作区解析，不使用本机或服务器绝对路径。

## 5. 默认站姿和初始高度

实机单次采样存在左右偏差。训练默认姿态使用采样值的对称化近似，避免把静态偏差引入期望步态：

| 关节 | 左侧 rad | 右侧 rad |
|---|---:|---:|
| Hip Pitch | `0.065` | `0.065` |
| Hip Roll | `0.09` | `-0.09` |
| Hip Yaw | `0.30` | `-0.30` |
| Knee | `0.09` | `0.09` |
| Ankle Roll | `-0.02` | `0.02` |
| Ankle Pitch | `0.04` | `0.04` |

腰部默认位置为：

```text
waist_yaw_joint:   0.0
waist_roll_joint:  0.0
waist_pitch_joint: 0.08
```

根链接初始高度由 `1.05 m` 改为实测值 `0.97 m`。服务器加载验证时必须检查双脚是否自然接触地面。若出现悬空或穿透，只校正 `init_state.pos.z`，不通过修改默认关节角度掩盖高度问题。

旧的默认高度和关节位置配置保留为注释，新配置紧随其后。

## 6. 执行器参数

训练使用实机运动模式下采集的 PD 增益：

| 执行器组 | Kp | Kd |
|---|---:|---:|
| Hip | `300` | `5` |
| Knee | `300` | `5` |
| Ankle Roll | `60` | `3` |
| Ankle Pitch | `60` | `3` |
| Waist Yaw | `300` | `5` |
| Waist Roll/Pitch | `300` | `5` |

以下参数保持当前配置不变：

- effort limit；
- velocity limit；
- armature；
- 腿部 action scale `0.25`；
- 腰部 action scale `0.10`。

被替换的 stiffness 和 damping 值按要求注释保留，新值写在其下方。本轮不引入 PD 增益随机化；若后续需要提高 Sim2Real 鲁棒性，再单独设计小范围参数随机化实验。

## 7. 保持不变的训练配置

为隔离实机参数带来的影响，第一轮不修改：

- 观测项、观测噪声和 privileged observations；
- 奖励项与权重；
- 终止条件；
- 摩擦、质量和推扰随机化；
- 速度命令范围及 curriculum；
- 仿真步长、decimation 和 episode 长度；
- PPO 网络、优化器和每环境采样步数。

本轮不因训练结果修改 URDF 硬限位，也不通过同步调整奖励掩盖模型或控制参数问题。

## 8. 从头训练与日志隔离

本轮策略从头训练，不加载任何旧模型：

- 不传入 `--resume`；
- 不传入 `--checkpoint`；
- 使用 `seed=42`；
- 由 RSL-RL 创建新的时间戳日志目录；
- 不删除或覆盖旧训练模型。

虽然任务和 experiment 名称不变，但新旧训练可通过时间戳目录、Git commit 和训练记录区分。正式训练前必须记录被测根仓库和 `unitree_rl_lab` 子模块提交。

## 9. 验证流程

### 9.1 本地静态验证

开发笔记本未安装 Isaac Lab，本地只执行：

- `H2_stand.urdf` XML 解析；
- 非 fixed 关节数量严格为 15；
- 可动关节名称和顺序与动作配置一致；
- 上半身关节全部 fixed；
- 腿部和腰部 origin、axis、limit 与 `H2_simple.urdf` 一致；
- H2 资源路径和静态测试；
- `git diff --check` 和子模块状态检查。

本地静态检查不能替代 Isaac Lab 加载或训练验证。

### 9.2 服务器冒烟训练

在 RTX 4090 服务器运行：

```text
num_envs=64
max_iterations=300
seed=42
```

冒烟阶段检查：

- URDF 和 `Unitree-H2-Velocity` 正常加载；
- 动作维度严格为 15；
- 初始姿态对称，双脚无明显悬空或穿透；
- 无初始自碰撞导致的弹飞；
- 无 NaN、CUDA、PhysX 或仿真崩溃；
- episode length 呈合理趋势；
- 生成的 checkpoint 能够重新加载。

300 轮冒烟模型不要求满足正式速度跟踪指标。

### 9.3 服务器正式训练

冒烟验证通过后运行：

```text
num_envs=4096
max_iterations=5000
seed=42
```

每环境采样步数保持现有默认值。训练必须完整保存最终 checkpoint，并记录日志目录、Git commit、墙钟时间和关键奖励曲线。

## 10. 固定指令集评估

正式训练完成后使用现有 H2 固定指令集评估入口，不新增另一套指标计算方式。

基本验收标准：

| 指标 | 标准 |
|---|---:|
| 20 秒 episode 存活率 | `>= 90%` |
| XY 线速度 RMSE | `<= 0.20 m/s` |
| Yaw 角速度 RMSE | `<= 0.25 rad/s` |
| 非期望接触 episode 比例 | `<= 5%` |
| Soft-limit invalid rate | `<= 0.1%` |

本轮进一步优化目标：

| 指标 | 目标 |
|---|---:|
| 20 秒 episode 存活率 | `100%` |
| Soft-limit invalid rate | `0%` |

基本验收用于判断模型是否可用；优化目标不作为实现收尾的硬门槛。若基本验收通过但优化目标未达到，应保留模型并记录具体差距。若基本验收失败，应先按具体失败指标分析模型、控制或奖励问题，不直接放宽 URDF 硬限位。

## 11. 非目标

本设计不包含：

- 修改任务名称或注册新 Task；
- 修改动作维度或增加上半身控制；
- 修改奖励、速度命令或 PPO 参数；
- PD 增益随机化；
- 粗糙地形和台阶训练；
- MuJoCo Sim2Sim 验收；
- Sim2Real 或真实机器人控制；
- 多随机种子统计；
- 删除、覆盖或继续训练旧 checkpoint。

## 12. 实施边界

功能代码修改发生在 `modules/unitree_rl_lab`，URDF 和设计文档修改发生在根仓库。子模块功能提交完成后，根仓库需要单独更新 gitlink。

训练日志、checkpoint、Isaac Sim 缓存和本机环境配置不提交 Git。真实机器人采集值仅作为离线参数输入，本设计不授权任何真实机器人控制或网络部署操作。
