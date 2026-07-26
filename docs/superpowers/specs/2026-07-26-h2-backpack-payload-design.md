# H2 5.5 kg Backpack Payload Design

## Goal

为现有 H2 15-DoF 与 29-DoF 速度任务增加固定在躯干后的 5.5 kg
背包负载，同时保持任务 ID、动作/观测契约、奖励函数和 runner
experiment name 不变。已有无背包策略作为迁移训练起点，不从头训练。

## Decisions

- 保留现有任务 ID：
  - `Unitree-H2-Velocity`
  - `Unitree-H2-15dof-Velocity`
  - `Unitree-H2-29dof-Velocity`
- 不增加 Backpack 专用任务 ID。
- 不增加 `H2_USE_BACKPACK` 等运行时开关。
- 不增加 `--run_name`。
- 任务默认加载 5.5 kg 背包 URDF。
- 无背包 URDF 继续保留，`unitree.py` 中以邻近注释记录其可选路径。
- 奖励函数、目标高度、速度跟踪权重、课程和终止条件全部保持不变。
- 15-DoF 与 29-DoF 分别从已有 25,000 轮 checkpoint 恢复训练。

## Asset layout

消除原文件名中 DoF 契约不明确的问题：

```text
H2_stand.urdf  -> H2_15dof.urdf
H2.urdf        -> H2_29dof.urdf
```

在无背包资产基础上新增：

```text
H2_15dof_backpack.urdf
H2_29dof_backpack.urdf
```

最终四份资产的含义固定如下：

| Asset | Movable joints | Payload |
|---|---:|---:|
| `H2_15dof.urdf` | 15 | none |
| `H2_29dof.urdf` | 31 physical / 29 policy-controlled | none |
| `H2_15dof_backpack.urdf` | 15 | 5.5 kg |
| `H2_29dof_backpack.urdf` | 31 physical / 29 policy-controlled | 5.5 kg |

重命名必须使用 Git move，并同步静态测试、文档和所有硬编码引用。
四份 URDF 继续共享原有相对 mesh 目录，不移动机器人本体 mesh。

## Backpack body

参考 `/home/csp/H2_with_backpack.urdf` 中的质量、惯量和安装位姿。参考
文件所引用的 `meshes/beibao.STL` 不存在于当前可用资产中，因此第一版使用
简化 box visual。

```xml
<link name="backpack_link">
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="5.5"/>
    <inertia
      ixx="0.024298"
      ixy="0"
      ixz="-0.000132"
      iyy="0.013846"
      iyz="0"
      izz="0.014324"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.16 0.30 0.36"/>
    </geometry>
  </visual>
</link>

<joint name="backpack_fixed_joint" type="fixed">
  <origin xyz="-0.1 0 0.1" rpy="0 0 3.141592653589793"/>
  <parent link="torso_link"/>
  <child link="backpack_link"/>
</joint>
```

第一版不为背包增加 collision。质量、质心和惯量仍参与动力学，但不会引入
背包与手臂或躯干的自碰撞。后续若需要训练手臂避障，应作为独立设计增加
简化 collision 并重新验证碰撞过滤和初始化。

fixed joint 不得改变 movable joint 数量、顺序、执行器覆盖或策略动作维度。

## Default asset selection

`unitree.py` 直接选择背包资产，不增加配置开关：

```python
# H2_15DOF_URDF_RELATIVE_PATH = Path(
#     "assets/urdf/h2_description/H2_15dof.urdf"
# )
H2_15DOF_URDF_RELATIVE_PATH = Path(
    "assets/urdf/h2_description/H2_15dof_backpack.urdf"
)

# H2_29DOF_URDF_RELATIVE_PATH = Path(
#     "assets/urdf/h2_description/H2_29dof.urdf"
# )
H2_29DOF_URDF_RELATIVE_PATH = Path(
    "assets/urdf/h2_description/H2_29dof_backpack.urdf"
)
```

因此原任务 ID 和旧 checkpoint 的 shape 契约保持兼容，但任务默认动力学由
无负载变为携带 5.5 kg 背包。若人工切回无背包路径，必须作为明确的代码
变更提交，不能依赖未记录的本地环境变量。

## Checkpoint migration

所有 smoke、诊断和长训练均从已有策略恢复，不从头训练。

15-DoF 基线：

```text
logs/rsl_rl/h2_15dof_velocity/2026-07-25_20-51-29/model_25000.pt
```

29-DoF 基线：

```text
logs/rsl_rl/h2_29dof_velocity/2026-07-26_01-58-44/model_25000.pt
```

实施前必须在 101 服务器确认实际文件名。若实际为 `25000.pt`，命令使用真实
文件名，不创建别名或猜测。

32-env smoke 只验证资产、shape 和 checkpoint 可加载性，不作为正式训练链
起点。正式迁移训练应重新从原始 25,000 轮 checkpoint 启动：

```text
无背包 model_25000
├── 32-env resume smoke（验证分支）
└── 4096-env backpack migration
    └── subsequent long training
```

恢复 checkpoint 后，actor、critic、optimizer 和迭代编号继续沿用；仿真
环境改为背包 URDF。

## Validation

### Static

- 四份 URDF 均通过 XML 解析。
- 无背包 15/29-DoF 资产分别保持 15/31 个 movable joints。
- 对应背包资产的 movable joint 名称和顺序与无背包版本完全一致。
- 背包资产各增加且只增加一个 fixed joint。
- 两份背包资产的质量、惯量、box 尺寸和固定变换完全一致。
- 无背包资产不包含 `backpack_link`。
- 活动代码和当前用户文档中不再引用 `H2_stand.urdf` 或 `H2.urdf`；历史
  specs/plans 保留当时的文件名以维持审计记录。
- 15/29-DoF action 与 observation 静态契约继续通过。

### 101 server smoke

在已验证的 `legged_rl_unitree_rl_lab` 环境中执行：

- 15-DoF：从指定 25,000 checkpoint 恢复，32 env，2–5 iterations。
- 29-DoF：从指定 25,000 checkpoint 恢复，32 env，2–5 iterations。

两者必须无 NaN/Inf、checkpoint shape mismatch、joint-regex mismatch、
missing actuator 或初始化错误。

### Migration training

15-DoF 与 29-DoF 的 resume smoke 通过后，各自的正式训练链重新从原始
无背包 25,000 checkpoint 恢复，使用 4096 env 训练到累计约 50,000
iterations。因为 RSL-RL 的 `--max_iterations` 表示恢复后的额外迭代数，
从 `model_25000.pt` 启动时应设置约 25,000，而不是 50,000。

正式长训练先观察前 300 个迁移 iterations，作为在线门控窗口。重点检查：

- survival 与 base-height termination；
- base pitch/roll；
- linear/yaw velocity RMSE；
- ankle joint-limit diagnostics；
- action saturation；
- undesired contacts；
- 背包引起的持续后倾、膝伸直或踝关节补偿。

前 300 个迁移 iterations 通过后，不重启、不更换基线，继续同一背包训练链
直到累计约 50,000 iterations。15-DoF 与 29-DoF 必须使用各自独立的
checkpoint 和日志目录，严禁交叉加载模型。

最终 checkpoint 的实际文件名由 RSL-RL 的零基编号和保存时机决定，可能是
`model_49999.pt`、`model_49998.pt` 或相邻编号。验收以训练日志显示的累计
迭代数和实际生成的最终文件为准，不为得到字面上的 `50000.pt` 而重命名
checkpoint。

## Compatibility and scope

- Gym task ID、runner experiment name、action/observation shape 不变。
- 无背包 checkpoint 可加载，但运行环境的质量分布已改变。
- 不修改奖励、课程、网络结构、history 长度或部署协议。
- 不执行真实机器人或 Sim2Real。
- 不提交 checkpoint、TensorBoard 日志、评估 JSON、缓存或转换产物。
- 当前 15-DoF Sim2Sim 部署若共享该 URDF 路径，需要在实施中显式检查；
  本设计不授权把背包策略部署到真实机器人。
