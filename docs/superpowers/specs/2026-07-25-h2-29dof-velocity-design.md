# H2 29-DoF 全身速度任务设计

## 1. 背景与目标

当前 `Unitree-H2-Velocity` 使用 `H2_stand.urdf`：机器人只有双腿、腰部共 15 个活动关节，上半身关节固定。新增任务需要使用完整 H2 动力学模型，让双臂参与行走平衡，但不让策略控制两个头部关节。

第一版目标是：

- 保留现有 15-DoF 平地速度任务及其兼容入口；
- 新增使用完整 `H2.urdf` 的 H2 平地速度任务；
- 完整模型保留 31 个物理活动关节，策略控制其中 29 个；
- 从头训练，不迁移现有 15-DoF checkpoint；
- 保持现有地形、速度命令、步态目标和 5000 次训练框架，便于与 15-DoF 基线对比；
- 完成 Isaac Lab 训练和评估，不在本阶段承诺 29-DoF Sim2Sim 或 Sim2Real。

## 2. 方案选择

采用独立维护两个任务的方案。15-DoF 和 29-DoF 分别拥有独立环境配置，不在本次改动中抽取共享基类。

该方案会产生少量地形、命令和腿部奖励配置重复，但能让新任务的动作、观测和奖励调整不影响已经验证的 15-DoF 任务。等 29-DoF 训练稳定后，再依据实际重复程度决定是否抽取公共配置。

不采用以下方案：

- 共享配置基类：需要同时重构现有任务，Isaac Lab 嵌套 `configclass` 的复制与覆盖会增加回归风险；
- 单环境动态切换：不同动作维度被运行时条件隐藏，训练、导出和部署更容易选错构型。

## 3. 任务与目录结构

H2 任务目录调整为：

```text
tasks/locomotion/robots/h2/
├── __init__.py
├── evaluation.py
├── h2_15dof/
│   ├── __init__.py
│   └── velocity_env_cfg.py
└── h2_29dof/
    ├── __init__.py
    └── velocity_env_cfg.py
```

注册以下任务：

| Task ID | 资产 | 物理活动关节 | 动作维度 | 用途 |
| --- | --- | ---: | ---: | --- |
| `Unitree-H2-15dof-Velocity` | `H2_stand.urdf` | 15 | 15 | 显式的原任务名称 |
| `Unitree-H2-29dof-Velocity` | `H2.urdf` | 31 | 29 | 新的全身速度任务 |
| `Unitree-H2-Velocity` | `H2_stand.urdf` | 15 | 15 | 15-DoF 兼容别名 |

兼容任务和显式 15-DoF 任务必须指向同一环境配置类，不能形成第三套配置。

PPO runner 拆分为 `H2_15DOFPPORunnerCfg` 和 `H2_29DOFPPORunnerCfg`，并使用不同的 `experiment_name`。原 `H2PPORunnerCfg` 暂时作为 15-DoF runner 的兼容别名，防止已有命令、日志恢复和工具立即失效。

## 4. 机器人资产与 articulation

机器人配置拆分为：

```python
UNITREE_H2_15DOF_CFG
UNITREE_H2_29DOF_CFG
```

原 `UNITREE_H2_CFG` 暂时作为 `UNITREE_H2_15DOF_CFG` 的兼容别名。

29-DoF 任务使用完整 `H2.urdf`。`H2.urdf` 和 `H2_dae.urdf` 都有 31 个活动关节，主要区别是视觉网格格式；第一版沿用以 STL 为主的 `H2.urdf`，不把 DAE 格式视为不同的动力学模型。

完整模型的关节职责为：

- 双腿 12 个、腰部 3 个、双臂 14 个关节由策略控制，共 29 维动作；
- `head_pitch_joint` 和 `head_yaw_joint` 保留在 articulation 中；
- 两个头部关节由独立 PD actuator 保持默认零位，不进入策略动作空间；
- 29 个动作关节使用显式有序列表和 `preserve_order=True`，该顺序同时构成训练、导出和未来部署接口契约。

执行器按物理能力拆分：

- 腿部和腰部沿用当前已验证分组；
- shoulder pitch 单独分组；
- shoulder roll/yaw、elbow 和 wrist roll 按 URDF effort/velocity limit 配置；
- wrist pitch/yaw 使用其较低的 effort limit 和更小动作尺度；
- head 使用独立的高阻尼保持控制。

所有 effort 和 velocity limit 不得超过 URDF。新增 PD 参数是训练初值，必须通过关节跟踪误差、振荡和动作饱和率验证，不能直接作为实机参数。

## 5. 默认姿态与动作尺度

腿部和腰部沿用当前微蹲站姿：

```text
双侧 hip_pitch       -0.15
双侧 knee             0.30
双侧 ankle_pitch     -0.20
腰部三个关节          0
```

完整 URDF 的上半身默认姿态根据 `H2_stand.urdf` 相对 `H2.urdf` 修改过的 fixed-joint `origin rpy` 还原：

```text
左 shoulder_roll     +0.40
右 shoulder_roll     -0.40
左 shoulder_yaw      -0.40
右 shoulder_yaw      +0.40
双侧 elbow           +1.10

shoulder_pitch        0
wrist_roll/pitch/yaw  0
head_pitch/yaw        0
```

这些值是完整模型的关节位置目标，不是直接复制 fixed joint 的原始 `origin rpy`。

第一版动作尺度为：

```text
腿部             沿用 15-DoF 配置
腰部             0.10
shoulder pitch   0.15
shoulder roll    0.12
shoulder yaw     0.12
elbow            0.15
wrist roll       0.08
wrist pitch/yaw  0.04
```

动作尺度允许肩和肘辅助平衡，同时限制腕部高速或大幅摆动。

## 6. 观测契约

Actor 只观察 29 个受控关节。关节位置和速度必须通过与动作顺序一致的显式列表选择，不能使用匹配全部关节的表达式。

每帧 policy observation 为：

| 观测项 | 维度 |
| --- | ---: |
| `base_ang_vel` | 3 |
| `projected_gravity` | 3 |
| `velocity_commands` | 3 |
| 受控关节相对位置 | 29 |
| 受控关节相对速度 | 29 |
| `last_action` | 29 |
| 合计 | 96 |

`history_length=5` 时 actor 输入为 480 维。训练、play 和导出测试必须断言 480 维 observation 与 29 维 action 契约。

Critic 可以使用全部 31 个关节的位置、速度和力矩，包括两个头部关节，并继续使用 base linear velocity 等 privileged observations。Critic 的关节选择也应显式表达，避免资产变化时静默改变网络输入。

## 7. 奖励、终止与接触

29-DoF 环境沿用 15-DoF 基准的核心奖励：

- XY 线速度与 yaw 角速度跟踪；
- 存活、base 高度与平面姿态；
- 步态、足部离地高度、足部滑动与接触力；
- 腿部关节限制、关节加速度和动作变化；
- 非足部接触惩罚。

新增上半身约束：

- shoulder 相对默认姿态的中等偏离惩罚；
- elbow 相对默认姿态的中等偏离惩罚；
- wrist 相对默认姿态的较强偏离惩罚；
- head 相对零位的保持惩罚；
- 上半身关节速度和加速度惩罚；
- 上半身 action-rate 惩罚；
- 关节力矩或机械功率惩罚。

第一版不提供预定义周期摆臂轨迹。肩和肘可根据速度跟踪和平衡需要自然运动，但默认姿态、动作幅度和平滑项限制无意义摆动，腕部约束最强。

手、腕、肘、头部、pelvis 和 torso 的地面接触均属于 undesired contact。pelvis 或 torso 低于安全高度时终止 episode。完整 URDF 导入后必须检查自碰撞设置，至少避免手臂穿入躯干；第一版不增加手部任务、动作模仿、搬运或复杂地形。

新增奖励的初始权重以仓库现有 G1 29-DoF 配置和 H2 15-DoF 权重为基线，在实施计划中明确到每一项。权重调整只能发生在分段训练阶段，并记录修改前后的 seed、迭代数和诊断指标；不得在一次实验中同时改变速度命令、核心腿部奖励和上肢约束。

## 8. 训练与评估

29-DoF 策略从头训练。由于 actor 输入和输出维度都发生变化，本阶段不实现 15-DoF checkpoint 的部分权重迁移。

训练保持现有平地任务的速度命令、4096 个正式并行环境、seed 42 和最多 5000 次迭代。执行顺序为：

1. 32 个环境、2 次迭代的导入和 shape smoke test；
2. 约 100–300 次迭代，检查存活、NaN/Inf、头部保持和上肢振荡；
3. 约 1000 次迭代，检查速度跟踪趋势、非足部接触和动作饱和率；
4. 前述指标合理后，运行 4096 环境、5000 次正式训练。

固定命令评估沿用现有基线：

```text
survival_rate        ≥ 0.95
linear_velocity_rmse ≤ 0.25
yaw_velocity_rmse    ≤ 0.30
invalid_count        = 0
```

另外记录以下 29-DoF 诊断指标：

- undesired-contact episode rate；
- shoulder、elbow、wrist RMS velocity；
- 各关节动作饱和率；
- 头部角度 RMS；
- 平均关节力矩或机械功率；
- 与 15-DoF 在同命令、同 seed 下的速度跟踪和存活率对比。

第一版只记录上肢指标，不预设未经数据验证的通过阈值。第一轮训练形成分布后再制定阈值。

## 9. 验证

纯静态测试必须覆盖：

- `H2_stand.urdf` 恰好有 15 个活动关节；
- `H2.urdf` 恰好有 31 个活动关节；
- 29 维动作列表不包含 head joints，没有重复，并且全部存在于 `H2.urdf`；
- 初始角度位于各自 joint limits 内；
- actuator effort/velocity limits 不超过 URDF；
- policy 每帧观测为 96 维，历史观测为 480 维；
- 三个 Gym task ID 正确注册；
- 兼容任务与显式 15-DoF 任务使用同一配置；
- 两个 runner 的实验目录互不冲突；
- 原有 15-DoF 静态测试继续通过。

Isaac Lab 环境中至少运行：

```bash
./unitree_rl_lab.sh -l

python scripts/rsl_rl/train.py \
  --headless \
  --task Unitree-H2-29dof-Velocity \
  --num_envs 32 \
  --max_iterations 2
```

短运行必须确认 articulation 为 31 个活动关节、action dimension 为 29、observation shape 符合契约、没有 NaN/Inf、头部维持零位，并检查 reset 后手臂没有穿入身体或地面。15-DoF 显式任务和兼容任务也需要完成最小 smoke test。

本地缺少 Isaac Lab、GPU 或仿真资源时，只运行静态检查，并明确记录未运行的验证及原因。

## 10. 错误处理

- 15-DoF 和 29-DoF 环境分别校验自己的 URDF 路径，缺失时错误信息必须包含 task ID、解析后的绝对路径和完整工作区要求；
- 初始化时校验物理活动关节数、动作关节集合、动作顺序和观测维度，任何不一致都应立即失败，不能依赖训练若干步后暴露 shape 错误；
- reset 或 step 出现 NaN/Inf 时计入 `invalid_count` 并使评估失败；
- 29-DoF checkpoint 进入现有 15-DoF 导出或部署入口时，必须因 observation/action shape 不匹配而明确拒绝，不能截断或填充张量继续运行；
- 三个 task ID 的日志必须打印最终解析出的任务、资产和动作维度，降低兼容别名造成的误用风险。

## 11. 部署边界

现有 H2 部署和 Sim2Sim 代码固定为 56 维 observation、15 维 action 和 15 项 joint map，只支持当前 15-DoF 策略。

29-DoF checkpoint 必须禁止通过现有部署入口误加载。29-DoF 的 observation/action 编码、SDK motor ID 映射、安全限制、ONNX shape、MuJoCo 验证和实机状态机需要后续独立设计，不属于本次训练任务范围。

## 12. 文档与兼容性

更新 H2 训练文档，明确：

- `Unitree-H2-Velocity` 是 15-DoF 兼容别名；
- 新实验应显式选择 `Unitree-H2-15dof-Velocity` 或 `Unitree-H2-29dof-Velocity`；
- 名称中的 29-DoF 表示策略动作维度，物理模型仍有 31 个活动关节；
- 两个任务依赖根工作区 `assets/urdf/h2_description`，不能从独立子模块克隆直接运行；
- 29-DoF 第一版仅承诺 Isaac Lab 训练与评估，不承诺部署能力。
