# H2 Real-Stand Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用 H2 实机运动模式的对称化站姿、`0.97 m` 初始高度和实测 PD 参数，重建 `H2_stand.urdf`，更新现有 `Unitree-H2-Velocity` 配置，并在 RTX 4090 服务器从头完成 300 轮冒烟训练、5000 轮正式训练和固定指令集验收。

**Architecture:** 根仓库维护 `H2_stand.urdf`、设计与计划文档；`modules/unitree_rl_lab` 维护资产路径、默认状态、执行器、H2 runner 稳定性保护和静态契约测试。任务名、experiment、15维动作、奖励、命令、随机化、PPO 网络和优化器不变。实机站姿写入 `init_state.joint_pos`，不改下肢与腰部 joint origin、axis 或 limit；上半身以对称姿态折叠为 fixed joint。

**Tech Stack:** URDF/XML、Python 3、pytest、Isaac Lab、RSL-RL PPO、RTX 4090

**Design:** `docs/superpowers/specs/2026-07-23-h2-real-stand-retraining-design.md`

---

## 执行边界

- 本地笔记本未安装 Isaac Lab，只执行 XML、Python AST、pytest 静态契约和 Git 检查。
- Isaac Lab 加载、300 轮冒烟、5000 轮正式训练和评估只在 RTX 4090 服务器执行。
- 不传 `--resume` 或 `--checkpoint` 启动训练。
- 不删除或覆盖旧日志与 checkpoint。
- 不提交训练日志、模型、缓存、SFTP 配置或本机环境文件。
- 不修改 `Unitree-H2-Velocity` 的动作缩放、奖励、命令、随机化、终止条件、PPO 网络或优化器参数。
- 不修改 URDF 硬限位来提高验收结果。
- 修改子模块前单独确认其状态；不得覆盖用户已有修改。

### Task 1: 建立新机器人参数的失败契约测试

**Files:**

- Modify: `modules/unitree_rl_lab/test/test_h2_locomotion_static.py`
- Reference: `assets/urdf/h2_description/H2.urdf`
- Reference: `assets/urdf/h2_description/H2_simple.urdf`
- Reference: `docs/superpowers/specs/2026-07-23-h2-real-stand-retraining-design.md`

- [ ] **Step 1: 将测试资产目标改为 H2_stand**

把当前 `H2_URDF` 改为：

```python
H2_URDF = WORKSPACE_ROOT / "assets/urdf/h2_description/H2_stand.urdf"
H2_SIMPLE_URDF = WORKSPACE_ROOT / "assets/urdf/h2_description/H2_simple.urdf"
H2_FULL_URDF = WORKSPACE_ROOT / "assets/urdf/h2_description/H2.urdf"
```

保留 `EXPECTED_JOINTS` 的 15 个关节及原顺序。

- [ ] **Step 2: 增加可动关节运动学不变测试**

解析 `H2_simple.urdf` 和 `H2_stand.urdf`，对 `EXPECTED_JOINTS` 逐一断言：

- joint type 相同；
- parent 与 child 相同；
- origin 的 xyz/rpy 相同；
- axis 相同；
- limit 的 lower、upper、effort、velocity 相同。

该测试必须能捕获旧版 `H2_stand.urdf` 中把腿部站姿写入 origin 并平移 limit 的问题。

- [ ] **Step 3: 增加上半身 fixed 姿态测试**

定义期望角度：

```python
EXPECTED_FIXED_POSE = {
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.4,
    "left_shoulder_yaw_joint": -0.4,
    "left_elbow_joint": 1.1,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": -0.4,
    "right_shoulder_yaw_joint": 0.4,
    "right_elbow_joint": 1.1,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,
    "head_yaw_joint": 0.0,
}
```

测试不能只比较 rpy 文本。为每个关节：

1. 从完整 `H2.urdf` 读取原始 origin 和 axis；
2. 计算 `T_expected = T_origin @ R_axis(q)`；
3. 从 `H2_stand.urdf` 读取 fixed origin 得到 `T_actual`；
4. 使用数值容差比较两个齐次变换；
5. 断言 stand joint 为 fixed，且不含 axis 和 limit。

这样可验证 fixed joint 真正表示原 revolute joint 在目标 `q` 下的姿态。

- [ ] **Step 4: 更新默认状态契约**

使用 AST 解析 `UNITREE_H2_CFG`，断言：

```python
pos == (0.0, 0.0, 0.97)
joint_pos == {
    "left_hip_pitch_joint": 0.065,
    "left_hip_roll_joint": 0.09,
    "left_hip_yaw_joint": 0.30,
    "left_knee_joint": 0.09,
    "left_ankle_roll_joint": -0.02,
    "left_ankle_pitch_joint": 0.04,
    "right_hip_pitch_joint": 0.065,
    "right_hip_roll_joint": -0.09,
    "right_hip_yaw_joint": -0.30,
    "right_knee_joint": 0.09,
    "right_ankle_roll_joint": 0.02,
    "right_ankle_pitch_joint": 0.04,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.08,
}
```

继续禁止 `".*"` catch-all，避免具体站姿被覆盖。

- [ ] **Step 5: 更新资产路径和执行器契约**

断言有效资产路径为 `H2_stand.urdf`，并断言执行器：

```text
H2_HIP:             stiffness=300.0, damping=5.0
H2_KNEE:            stiffness=300.0, damping=5.0
H2_ANKLE_ROLL:      stiffness=60.0,  damping=3.0
H2_ANKLE_PITCH:     stiffness=60.0,  damping=3.0
H2_WAIST_YAW:       stiffness=300.0, damping=5.0
H2_WAIST_ROLL_PITCH stiffness=300.0, damping=5.0
```

effort limit、velocity limit、armature 和 joint selector 必须与当前配置完全相同。

- [ ] **Step 6: 运行测试并确认按预期失败**

Run:

```bash
cd modules/unitree_rl_lab
pytest -q test/test_h2_locomotion_static.py \
  -k "urdf or articulation or initial_joint"
```

Expected: FAIL，失败原因只应是旧 `H2_stand.urdf` 的 joint origin/limit、旧资产路径、旧默认姿态、高度或 PD 参数不满足新契约。

- [ ] **Step 7: 提交子模块测试**

```bash
git add test/test_h2_locomotion_static.py
git commit -m "test: define H2 real-stand training contract"
```

### Task 2: 重建 H2_stand URDF

**Files:**

- Modify: `assets/urdf/h2_description/H2_stand.urdf`
- Reference: `assets/urdf/h2_description/H2_simple.urdf`
- Reference: `assets/urdf/h2_description/H2.urdf`

- [ ] **Step 1: 以 H2_simple 为结构基线**

让 `H2_stand.urdf` 的 link、inertial、visual、collision、传感器 frame 和15个可动关节以 `H2_simple.urdf` 为基线，robot 名称为：

```xml
<robot name="H2_stand">
```

删除旧文件中“把实测角度写入 joint origin、同步平移 limit”的说明与实现。

- [ ] **Step 2: 恢复15个可动关节的原始定义**

以下关节的 type、origin、parent、child、axis 和 limit 必须逐字或数值等价于 `H2_simple.urdf`：

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

- [ ] **Step 3: 将左臂固定到对称姿态**

根据完整 `H2.urdf` 的 joint origin 和 axis，按 URDF 变换顺序把以下角度折叠进 fixed joint origin：

```text
left_shoulder_pitch_joint:  0.0
left_shoulder_roll_joint:   0.4
left_shoulder_yaw_joint:   -0.4
left_elbow_joint:           1.1
left_wrist_roll_joint:      0.0
left_wrist_pitch_joint:     0.0
left_wrist_yaw_joint:       0.0
```

joint type 保持 fixed，并删除 axis 与 limit。不要把数值直接按名称塞入任意 rpy 分量；必须使用 `T_origin @ R_axis(q)` 得到等价 fixed origin。

- [ ] **Step 4: 将右臂和头部固定到对称姿态**

以同样方式应用：

```text
right_shoulder_pitch_joint:  0.0
right_shoulder_roll_joint:  -0.4
right_shoulder_yaw_joint:    0.4
right_elbow_joint:           1.1
right_wrist_roll_joint:      0.0
right_wrist_pitch_joint:     0.0
right_wrist_yaw_joint:       0.0
head_pitch_joint:            0.0
head_yaw_joint:              0.0
```

- [ ] **Step 5: 运行 URDF 契约测试**

Run:

```bash
cd modules/unitree_rl_lab
pytest -q test/test_h2_locomotion_static.py \
  -k "urdf"
```

Expected: URDF 可动关节、运动学不变和上半身 fixed 姿态测试 PASS；资产路径与机器人配置测试仍可能失败。

- [ ] **Step 6: 检查根仓库 URDF 差异**

Run:

```bash
cd ../..
git diff --check
git diff -- assets/urdf/h2_description/H2_stand.urdf
```

Expected: 只有 robot 名称、旧错误站姿归零和上半身 fixed 姿态变换属于预期差异；collision 简化结果不应意外改变。

- [ ] **Step 7: 暂存根仓库 URDF**

```bash
git add assets/urdf/h2_description/H2_stand.urdf
```

根仓库提交延后到子模块配置提交和 gitlink 更新完成后统一处理。

### Task 3: 更新 H2 资产、默认状态和 PD 参数

**Files:**

- Modify: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`
- Modify: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`
- Modify: `modules/unitree_rl_lab/README.md`
- Test: `modules/unitree_rl_lab/test/test_h2_locomotion_static.py`

- [ ] **Step 1: 切换 H2 URDF 路径**

按已确认格式保留旧行注释并新增有效行：

```python
# H2_URDF_RELATIVE_PATH = Path("assets/urdf/h2_description/H2_simple.urdf")
H2_URDF_RELATIVE_PATH = Path("assets/urdf/h2_description/H2_stand.urdf")
```

不得写成重复目录，也不得使用绝对路径。

- [ ] **Step 2: 更新初始高度和15关节默认位置**

保留旧高度和旧 joint_pos 行为注释，在其下方使用显式关节名写入设计中的完整对称站姿：

```python
# pos=(0.0, 0.0, 1.05),
pos=(0.0, 0.0, 0.97),
```

不得用左右正负不同的关节共享一个 regex 值，不得增加 `".*"` catch-all。

- [ ] **Step 3: 更新 Hip 和 Knee PD**

保留旧 stiffness/damping 为相邻注释，新值为：

```text
H2_HIP:  stiffness=300.0, damping=5.0
H2_KNEE: stiffness=300.0, damping=5.0
```

effort limit、velocity limit 和 armature 不变。

- [ ] **Step 4: 更新 Ankle PD**

新值为：

```text
H2_ANKLE_ROLL:  stiffness=60.0, damping=3.0
H2_ANKLE_PITCH: stiffness=60.0, damping=3.0
```

Roll 和 Pitch 继续分组，以保留当前不同的 effort limit。

- [ ] **Step 5: 更新 Waist PD**

新值为：

```text
H2_WAIST_YAW:        stiffness=300.0, damping=5.0
H2_WAIST_ROLL_PITCH: stiffness=300.0, damping=5.0
```

两个组继续保留各自当前 effort limit。

- [ ] **Step 6: 更新 README 资产说明**

把 H2 任务依赖的资产从 `H2_simple.urdf` 更新为 `H2_stand.urdf`。训练命令、任务名、4096环境和5000轮保持不变。

- [ ] **Step 7: 增加 H2 runner 数值稳定性保护**

在 `H2PPORunnerCfg` 中设置：

```python
clip_actions = 1.0
policy = RslRlPpoActorCriticCfg(
    init_noise_std=1.0,
    noise_std_type="log",
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation="elu",
)
```

测试必须断言这两个稳定性参数，防止恢复为无界动作和可变成负数的 scalar std。

- [ ] **Step 8: 运行完整 H2 静态测试**

Run:

```bash
cd modules/unitree_rl_lab
pytest -q test/test_h2_locomotion_static.py
```

Expected: PASS。

- [ ] **Step 9: 运行子模块静态检查**

Run:

```bash
python -m compileall -q \
  source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py \
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py \
  test/test_h2_locomotion_static.py
git diff --check
git status --short
```

Expected: compileall 和 diff check PASS；只有本任务文件以及 Task 1 测试提交后的预期差异。

- [ ] **Step 10: 提交子模块实现**

```bash
git add \
  source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py \
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py \
  README.md
git commit -m "feat: use H2 real-stand training parameters"
```

### Task 4: 完成本地集成检查

**Files:**

- Test: `assets/urdf/h2_description/H2_stand.urdf`
- Test: `modules/unitree_rl_lab/test/test_h2_locomotion_static.py`
- Test: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`

- [ ] **Step 1: 确认子模块干净且提交正确**

Run:

```bash
git -C modules/unitree_rl_lab status --short
git -C modules/unitree_rl_lab log -2 --oneline
```

Expected: 子模块工作区干净，最近提交分别为新契约测试和实机站姿参数实现。

- [ ] **Step 2: 运行根仓库检查**

Run:

```bash
git diff --check
git submodule status
git status --short
```

Expected: 根仓库仅包含 `H2_stand.urdf`、设计/计划文档和 `unitree_rl_lab` gitlink 的预期变化。

- [ ] **Step 3: 提交根仓库集成变更**

```bash
git add \
  assets/urdf/h2_description/H2_stand.urdf \
  docs/superpowers/specs/2026-07-23-h2-real-stand-retraining-design.md \
  docs/superpowers/plans/2026-07-23-h2-real-stand-retraining.md \
  modules/unitree_rl_lab
git commit -m "feat: integrate H2 real-stand retraining baseline"
```

- [ ] **Step 4: 记录服务器被测版本**

Run:

```bash
git rev-parse HEAD
git -C modules/unitree_rl_lab rev-parse HEAD
```

Expected: 记录根仓库和子模块 commit SHA，后续训练结果必须关联这两个 SHA。

### Task 5: 在 RTX 4090 服务器执行 Isaac Lab 加载检查

**Files:**

- Validate: `/home/gaojie/workspace/legged_rl`
- Validate: `/home/gaojie/workspace/legged_rl/modules/unitree_rl_lab`

- [ ] **Step 1: 同步并核对服务器代码**

使用已配置的 SFTP 或 Git 同步已提交代码。SFTP 同步后，在服务器核对关键文件内容和 commit；不得假定 `uploadOnSave` 已完整同步。

Run on server:

```bash
cd /home/gaojie/workspace/legged_rl
git status --short
git submodule status
```

Expected: 服务器代码与 Task 4 记录的版本一致；没有训练日志或本机配置混入待提交文件。

- [ ] **Step 2: 进入 Isaac Lab 环境**

Run:

```bash
cd /home/gaojie/workspace/legged_rl/modules/unitree_rl_lab
conda activate legged_rl_unitree_rl_lab
```

若服务器实际仍使用已确认可用的环境名，则先用 `conda env list` 核对；不得误用其他工程的 `env_isaaclab` 后继续记录结果。

- [ ] **Step 3: 检查任务注册**

Run:

```bash
./unitree_rl_lab.sh -l | grep Unitree-H2-Velocity
```

Expected: 输出包含 `Unitree-H2-Velocity`。

- [ ] **Step 4: 运行单环境单轮加载测试**

Run:

```bash
python scripts/rsl_rl/train.py --headless \
  --task Unitree-H2-Velocity \
  --num_envs 1 \
  --max_iterations 1 \
  --seed 42
```

Expected: URDF 转换成功，动作维度为 15，无 joint/link、初始状态、CUDA 或 PhysX 错误，并保存可加载 checkpoint。

- [ ] **Step 5: 可视化检查初始姿态**

Run:

```bash
python scripts/rsl_rl/play.py \
  --task Unitree-H2-Velocity \
  --num_envs 1 \
  --checkpoint /path/to/one-iteration/model.pt
```

Expected: 上半身为对称固定姿态，双脚自然接触地面，无明显悬空、穿透、初始自碰撞或生成后弹飞。若仅高度不合适，只调整 `init_state.pos.z`，并重新执行 Task 1～5 的相关检查。

### Task 6: 执行 300 轮冒烟训练

**Files:**

- Generate only: `modules/unitree_rl_lab/logs/rsl_rl/h2_velocity/<timestamp>/`

- [ ] **Step 1: 启动从头冒烟训练**

Run on server:

```bash
cd /home/gaojie/workspace/legged_rl/modules/unitree_rl_lab
conda activate legged_rl_unitree_rl_lab
python scripts/rsl_rl/train.py --headless \
  --task Unitree-H2-Velocity \
  --num_envs 64 \
  --max_iterations 300 \
  --seed 42
```

命令中不得出现 `--resume` 或 `--checkpoint`。

- [ ] **Step 2: 检查冒烟结果**

Expected:

- 完成 300 iterations；
- 无 NaN、CUDA、PhysX 或仿真崩溃；
- 无 action/observation 维度错误；
- mean episode length 呈合理趋势；
- 保存最终 checkpoint；
- 日志位于新的时间戳目录。

300轮模型不要求通过正式验收指标。

- [ ] **Step 3: 加载冒烟 checkpoint**

Run:

```bash
python scripts/rsl_rl/play.py --headless \
  --task Unitree-H2-Velocity \
  --num_envs 16 \
  --checkpoint logs/rsl_rl/h2_velocity/<smoke-timestamp>/model_299.pt
```

Expected: checkpoint 成功加载并完成推理，不出现维度或配置不兼容。

- [ ] **Step 4: 记录冒烟结果**

在执行记录中保存：

- 根仓库和子模块 SHA；
- 日志目录；
- 最终 checkpoint；
- 训练墙钟时间；
- 最终 mean reward、mean episode length；
- 是否存在 base height termination、关节越限或非期望接触异常。

不要把模型或日志提交 Git。

### Task 7: 执行 5000 轮正式训练

**Files:**

- Generate only: `modules/unitree_rl_lab/logs/rsl_rl/h2_velocity/<timestamp>/`

- [ ] **Step 1: 确认正式训练前置条件**

必须同时满足：

- Task 1～5 全部通过；
- 300轮冒烟完整结束；
- 冒烟无 NaN 或仿真崩溃；
- 初始姿态可视化合理；
- 服务器版本与记录 SHA 一致。

- [ ] **Step 2: 启动正式训练**

Run on server:

```bash
cd /home/gaojie/workspace/legged_rl/modules/unitree_rl_lab
conda activate legged_rl_unitree_rl_lab
python scripts/rsl_rl/train.py --headless \
  --task Unitree-H2-Velocity \
  --num_envs 4096 \
  --max_iterations 5000 \
  --seed 42
```

Expected: 从头创建新的 `h2_velocity/<timestamp>`，完成5000轮并保存最终 checkpoint。不得从旧模型恢复。

- [ ] **Step 3: 记录正式训练结果**

保存：

- 日志目录和最终 checkpoint 路径；
- Git SHA 和 seed；
- 墙钟时间；
- 最终 mean reward、mean episode length；
- `Episode_Termination/time_out` 和 `base_height`；
- 速度跟踪、关节限位、接触和步态奖励；
- 是否出现 NaN、崩溃或异常中断。

### Task 8: 执行固定指令集验收

**Files:**

- Generate only: server-local evaluation JSON
- Reference: `modules/unitree_rl_lab/scripts/rsl_rl/evaluate_h2.py`

- [ ] **Step 1: 运行正式评估**

Run on server:

```bash
cd /home/gaojie/workspace/legged_rl/modules/unitree_rl_lab
conda activate legged_rl_unitree_rl_lab
python scripts/rsl_rl/evaluate_h2.py --headless \
  --task Unitree-H2-Velocity \
  --checkpoint logs/rsl_rl/h2_velocity/<formal-timestamp>/model_4999.pt \
  --num_envs 256 \
  --duration 20 \
  --seed 42 \
  --output /tmp/h2_real_stand_eval.json
```

- [ ] **Step 2: 检查基本验收**

全部满足才通过基本验收：

```text
survival rate >= 0.90
linear velocity RMSE <= 0.20 m/s
yaw velocity RMSE <= 0.25 rad/s
undesired contact episode rate <= 0.05
invalid rate <= 0.001
```

评估脚本当前直接输出 `invalid_count`；执行记录必须同时记录评估总样本数和由此计算的 invalid rate，不能只报告异常计数。

- [ ] **Step 3: 检查本轮优化目标**

单独报告：

```text
20-second survival rate == 1.0
soft-limit invalid rate == 0.0
```

基本验收通过但优化目标未达到时，模型仍视为可用；记录失败场景和差距供下一轮优化，不修改硬限位或临时放宽基本标准。

- [ ] **Step 4: 分场景诊断**

至少报告：

- stand；
- forward/backward；
- left/right；
- yaw_left/yaw_right；
- mixed。

重点比较左右横移和左右转向是否存在明显不对称，并检查 ankle roll soft-limit、pelvis/torso 非期望接触和 base height termination。

### Task 9: 收尾记录与仓库检查

**Files:**

- Create: `docs/superpowers/implementations/2026-07-23-h2-real-stand-retraining.md`

- [ ] **Step 1: 编写简短实施记录**

记录：

- 实际修改文件和 commit；
- 最终站姿、高度和 PD 参数；
- 本地检查结果；
- 服务器冒烟日志与 checkpoint；
- 正式训练日志与 checkpoint；
- 聚合和分场景评估结果；
- 基本验收与优化目标是否分别通过；
- 未完成项或下一轮优化建议。

不得把完整训练日志或模型复制进仓库。

- [ ] **Step 2: 运行最终检查**

Run:

```bash
git diff --check
git submodule status
git status --short
git -C modules/unitree_rl_lab status --short
```

Expected: 实施记录是唯一新增未提交文件，子模块工作区干净，根仓库 gitlink 指向已提交的子模块实现。

- [ ] **Step 3: 提交实施记录**

```bash
git add docs/superpowers/implementations/2026-07-23-h2-real-stand-retraining.md
git commit -m "docs: record H2 real-stand retraining results"
```

- [ ] **Step 4: 最终交付说明**

报告：

- 根仓库和子模块提交；
- 修改文件；
- 本地与服务器验证；
- 冒烟、正式训练和评估结果；
- 未提交的训练产物位置；
- 是否仍需推送分支或创建 PR。
