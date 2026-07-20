# H2 Locomotion Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `unitree_rl_lab` 中新增使用根仓库 `H2_simple.urdf` 的 15-DoF 平地速度跟踪任务，并在 RTX 4090 服务器完成注册、短训、5000 次训练和固定指令集性能验收。

**Architecture:** 以 H1 velocity task 为模板新增独立 H2 task，保留现有 Manager-Based RL 和 RSL-RL 数据流。机器人配置在 H2 环境创建时校验根工作区资产，避免 H2 路径缺失影响已有任务；本地只做静态检查，Isaac Lab 运行检查全部在服务器完成。

**Tech Stack:** Python 3.10、Isaac Lab 2.3、Isaac Sim 5.1、RSL-RL 2.3、PyTorch、Gymnasium、URDF、pytest

---

## 文件结构

子模块创建或修改：

- Modify: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`
- Create: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/__init__.py`
- Create: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/velocity_env_cfg.py`
- Modify: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`
- Create: `modules/unitree_rl_lab/scripts/rsl_rl/evaluate_h2.py`
- Create: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/evaluation.py`
- Create: `modules/unitree_rl_lab/test/test_h2_locomotion_static.py`
- Modify: `modules/unitree_rl_lab/README.md`

根仓库修改：

- Modify: `docs/superpowers/specs/2026-07-20-h2-locomotion-training-design.md` only if implementation reveals a reviewed design correction
- Create: `docs/superpowers/plans/2026-07-20-h2-locomotion-training.md`
- Update: `modules/unitree_rl_lab` gitlink after the submodule commit is complete

不要提交 `.vscode/`、`logs/`、checkpoint、导出策略、缓存或服务器环境文件。

### Task 1: 建立本地静态契约测试

**Files:**

- Create: `modules/unitree_rl_lab/test/test_h2_locomotion_static.py`
- Reference: `assets/urdf/h2_description/H2_simple.urdf`

- [ ] **Step 1: 在子模块创建功能分支**

Run:

```bash
git -C modules/unitree_rl_lab switch -c feature/add-h2-locomotion
```

Expected: 子模块当前分支为 `feature/add-h2-locomotion`，工作区干净。

- [ ] **Step 2: 写入不依赖 Isaac Lab 的失败测试**

创建 `modules/unitree_rl_lab/test/test_h2_locomotion_static.py`，使用标准库 `ast`、`pathlib` 和 `xml.etree.ElementTree`。定义：

```python
from __future__ import annotations

import ast
from pathlib import Path
import xml.etree.ElementTree as ET


SUBMODULE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SUBMODULE_ROOT.parents[1]
PACKAGE_ROOT = SUBMODULE_ROOT / "source/unitree_rl_lab/unitree_rl_lab"
H2_URDF = WORKSPACE_ROOT / "assets/urdf/h2_description/H2_simple.urdf"
H2_ENV_CFG = PACKAGE_ROOT / "tasks/locomotion/robots/h2/velocity_env_cfg.py"
H2_REGISTRY = PACKAGE_ROOT / "tasks/locomotion/robots/h2/__init__.py"
UNITREE_CFG = PACKAGE_ROOT / "assets/robots/unitree.py"
PPO_CFG = PACKAGE_ROOT / "tasks/locomotion/agents/rsl_rl_ppo_cfg.py"

EXPECTED_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_roll_joint", "left_ankle_pitch_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_roll_joint", "right_ankle_pitch_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]


def parse_python(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_h2_urdf_contract():
    root = ET.parse(H2_URDF).getroot()
    movable = [
        joint.attrib["name"] for joint in root.findall("joint")
        if joint.attrib["type"] in {"revolute", "continuous", "prismatic"}
    ]
    assert movable == EXPECTED_JOINTS
    assert root.find("./link[@name='left_ankle_pitch_link']") is not None
    assert root.find("./link[@name='right_ankle_pitch_link']") is not None


def test_h2_python_files_exist_and_parse():
    for path in (H2_ENV_CFG, H2_REGISTRY, UNITREE_CFG, PPO_CFG):
        assert path.is_file(), path
        parse_python(path)


def test_h2_registration_and_runner_contract():
    registry = H2_REGISTRY.read_text(encoding="utf-8")
    runner = PPO_CFG.read_text(encoding="utf-8")
    assert 'id="Unitree-H2-Velocity"' in registry
    assert "H2PPORunnerCfg" in runner
    assert "max_iterations = 5000" in runner
    assert 'experiment_name = "h2_velocity"' in runner


def test_h2_environment_contract():
    source = H2_ENV_CFG.read_text(encoding="utf-8")
    for joint in EXPECTED_JOINTS:
        assert joint in source
    assert "left_ankle_pitch_link" in source
    assert "right_ankle_pitch_link" in source
    assert "MeshPlaneTerrainCfg" in source
    assert "num_envs=4096" in source
```

- [ ] **Step 3: 运行测试并确认因 H2 Python 文件不存在而失败**

Run:

```bash
python3 -m pytest modules/unitree_rl_lab/test/test_h2_locomotion_static.py -q
```

Expected: `test_h2_urdf_contract` PASS；其他测试 FAIL，错误指出 H2 task 或 runner 尚不存在。

- [ ] **Step 4: 提交静态契约测试**

```bash
git -C modules/unitree_rl_lab add test/test_h2_locomotion_static.py
git -C modules/unitree_rl_lab commit -m "test: define H2 locomotion contracts"
```

### Task 2: 新增 H2 articulation 配置

**Files:**

- Modify: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`
- Modify: `modules/unitree_rl_lab/test/test_h2_locomotion_static.py`

- [ ] **Step 1: 为工作区路径和 actuator 分组增加失败断言**

在静态测试增加：

```python
def test_h2_articulation_contract():
    source = UNITREE_CFG.read_text(encoding="utf-8")
    assert "def resolve_workspace_asset" in source
    assert "UNITREE_H2_CFG" in source
    assert "H2_simple.urdf" in source
    for group in ("H2_HIP", "H2_KNEE", "H2_ANKLE", "H2_WAIST"):
        assert f'"{group}"' in source
```

Run: `python3 -m pytest modules/unitree_rl_lab/test/test_h2_locomotion_static.py::test_h2_articulation_contract -q`

Expected: FAIL because `UNITREE_H2_CFG` does not exist.

- [ ] **Step 2: 实现非破坏性的工作区资产解析**

在 `unitree.py` 顶部增加 `from pathlib import Path`，并在目录常量后增加：

```python
H2_URDF_RELATIVE_PATH = Path("assets/urdf/h2_description/H2_simple.urdf")


def resolve_workspace_asset(relative_path: Path) -> str:
    """Resolve a root-workspace asset without breaking non-H2 imports."""
    source = Path(__file__).resolve()
    for parent in source.parents:
        candidate = parent / relative_path
        if candidate.is_file():
            return str(candidate)
    return str(source.parent / "_missing_legged_rl_workspace" / relative_path)


H2_URDF_PATH = resolve_workspace_asset(H2_URDF_RELATIVE_PATH)
```

该函数在路径不存在时返回预期路径但不在 import 阶段抛错；H2 环境 Task 3 将在实际创建 H2 配置时强校验，因此单独使用 H1/G1 不受影响。

- [ ] **Step 3: 添加 H2 机器人配置**

在 `UNITREE_H1_CFG` 后新增 `UNITREE_H2_CFG`。使用 `UnitreeUrdfFileCfg(asset_path=H2_URDF_PATH)`；初始 base position 为 `(0.0, 0.0, 1.05)`，joint position 为 hip pitch `-0.15`、knee `0.30`、ankle pitch `-0.15`、其余关节 `0.0`。

actuator 配置必须使用四个互斥组：

```python
actuators={
    "H2_HIP": IdealPDActuatorCfg(
        joint_names_expr=[".*_hip_.*"],
        effort_limit=360.0,
        velocity_limit=20.0,
        stiffness=150.0,
        damping=4.0,
        armature=0.01,
    ),
    "H2_KNEE": IdealPDActuatorCfg(
        joint_names_expr=[".*_knee_joint"],
        effort_limit=360.0,
        velocity_limit=20.0,
        stiffness=200.0,
        damping=5.0,
        armature=0.01,
    ),
    "H2_ANKLE": IdealPDActuatorCfg(
        joint_names_expr=[".*_ankle_.*"],
        effort_limit=19.0,
        velocity_limit=28.61,
        stiffness=40.0,
        damping=2.0,
        armature=0.01,
    ),
    "H2_WAIST": IdealPDActuatorCfg(
        joint_names_expr=["waist_.*_joint"],
        effort_limit=120.0,
        velocity_limit=28.375,
        stiffness=100.0,
        damping=4.0,
        armature=0.01,
    ),
},
```

`joint_sdk_names` 必须等于计划头部的 15 关节顺序。注释说明 PD 是训练初值，需按 H2 实机规格复核。

- [ ] **Step 4: 运行本地静态测试**

Run:

```bash
python3 -m pytest modules/unitree_rl_lab/test/test_h2_locomotion_static.py -q
python3 -m compileall -q modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py
```

Expected: articulation contract PASS；尚未实现的 task tests 仍 FAIL；compileall 退出 0。

- [ ] **Step 5: 提交 articulation 配置**

```bash
git -C modules/unitree_rl_lab add source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py test/test_h2_locomotion_static.py
git -C modules/unitree_rl_lab commit -m "feat: add H2 articulation configuration"
```

### Task 3: 新增 H2 平地速度跟踪环境与注册

**Files:**

- Create: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/__init__.py`
- Create: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/velocity_env_cfg.py`
- Modify: `modules/unitree_rl_lab/test/test_h2_locomotion_static.py`

- [ ] **Step 1: 从 H1 配置建立 H2 文件**

复制 H1 `velocity_env_cfg.py` 为 H2 文件，然后做以下确定性变更：

- import `Path`；
- import `UNITREE_H2_CFG as ROBOT_CFG` and `H2_URDF_PATH`；
- terrain generator 只包含 `MeshPlaneTerrainCfg(proportion=1.0)`；
- 删除 height scanner 定义、观测和 update period；
- `ActionsCfg` 使用单个 `JointPositionActionCfg`，显式 15 个 `joint_names`、`preserve_order=True`、`use_default_offset=True`；
- scale 使用以下互斥映射，确保每个关节只匹配一个 scale：

```python
scale={
    ".*_hip_.*": 0.25,
    ".*_knee_joint": 0.25,
    ".*_ankle_.*": 0.25,
    "waist_.*_joint": 0.10,
}
```
- foot body names 只使用 `left_ankle_pitch_link` 和 `right_ankle_pitch_link`；
- 删除 arms deviation reward；
- torso deviation reward 改为三个 `waist_.*_joint`；
- illegal contact termination body names 为 `pelvis` 和 `torso_link`；
- 删除 terrain curriculum，只保留 `lin_vel_cmd_levels`；
- `scene.num_envs=4096`；`dt=0.005`、decimation 4、episode 20 秒；
- 命令 limit ranges 为 x `(-0.3,1.0)`、y `(-0.3,0.3)`、yaw `(-0.5,0.5)`。

- [ ] **Step 2: 在环境创建时强校验 H2 资产**

在 `RobotEnvCfg.__post_init__()` 最前面增加：

```python
if not Path(H2_URDF_PATH).is_file():
    raise FileNotFoundError(
        "H2 asset not found at "
        f"{H2_URDF_PATH}. Run Unitree-H2-Velocity from the complete legged_rl workspace."
    )
```

- [ ] **Step 3: 注册任务**

创建 `h2/__init__.py`：

```python
import gymnasium as gym

gym.register(
    id="Unitree-H2-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:H2PPORunnerCfg"
        ),
    },
)
```

现有 `tasks/locomotion/robots/__init__.py` 为空且 `locomotion/__init__.py` 使用 `from .robots import *`。确认项目的 package importer 会递归导入新 package；服务器 `-l` 是最终证据，不增加无依据的显式 import。

- [ ] **Step 4: 加强静态环境断言**

静态测试必须断言：15 个关节按顺序出现；`preserve_order=True`；两只 ankle pitch link 精确出现；不存在 `height_scanner`、`terrain_levels` 或 arms deviation；完整命令范围存在。

- [ ] **Step 5: 运行静态测试与编译**

Run:

```bash
python3 -m pytest modules/unitree_rl_lab/test/test_h2_locomotion_static.py -q
python3 -m compileall -q modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2
```

Expected: environment and registration contracts PASS；PPO contract 仍 FAIL。

- [ ] **Step 6: 提交 H2 环境**

```bash
git -C modules/unitree_rl_lab add source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2 test/test_h2_locomotion_static.py
git -C modules/unitree_rl_lab commit -m "feat: add H2 velocity environment"
```

### Task 4: 新增 H2 PPO runner 配置

**Files:**

- Modify: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`
- Modify: `modules/unitree_rl_lab/test/test_h2_locomotion_static.py`

- [ ] **Step 1: 实现 H2 runner subclass**

在 `BasePPORunnerCfg` 后新增：

```python
@configclass
class H2PPORunnerCfg(BasePPORunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "h2_velocity"
    seed = 42
```

网络和 algorithm 保持继承，不复制父类配置。

- [ ] **Step 2: 运行全部本地静态测试**

Run:

```bash
python3 -m pytest modules/unitree_rl_lab/test/test_h2_locomotion_static.py -q
python3 -m compileall -q modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab
git -C modules/unitree_rl_lab diff --check
```

Expected: 全部 PASS，compileall 和 diff check 退出 0。

- [ ] **Step 3: 提交 PPO 配置**

```bash
git -C modules/unitree_rl_lab add source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py test/test_h2_locomotion_static.py
git -C modules/unitree_rl_lab commit -m "feat: add H2 PPO runner configuration"
```

### Task 5: 在 RTX 4090 服务器完成注册和短训门禁

**Files:**

- Verify only: `/home/gaojie/workspace/legged_rl/modules/unitree_rl_lab`
- Logs only: `modules/unitree_rl_lab/logs/rsl_rl/h2_velocity/` on server

- [ ] **Step 1: 确认 SFTP 同步的代码版本**

在本地记录子模块 HEAD：

```bash
git -C modules/unitree_rl_lab rev-parse HEAD
```

在服务器终端执行：

```bash
cd /home/gaojie/workspace/legged_rl/modules/unitree_rl_lab
git diff -- source/unitree_rl_lab/unitree_rl_lab test
```

Expected: 服务器工作树内容包含本地 H2 文件；若 SFTP 没有自动上传 Codex 修改，使用明确的 SFTP 上传或在 VS Code 保存目标文件后再检查。

- [ ] **Step 2: 激活环境并列出任务**

```bash
cd /home/gaojie/workspace/legged_rl/modules/unitree_rl_lab
conda activate env_isaaclab
./unitree_rl_lab.sh -l
```

Expected: 输出包含 `Unitree-H2-Velocity`。

- [ ] **Step 3: 单环境无策略启动**

Run:

```bash
python scripts/rsl_rl/train.py --headless --task Unitree-H2-Velocity \
  --num_envs 1 --max_iterations 1 --seed 42
```

Expected: URDF 成功导入；action dimension 为 15；运行 1 iteration 后退出；无 missing joint/link、NaN 或 PhysX error。

- [ ] **Step 4: 运行短训**

Run:

```bash
python scripts/rsl_rl/train.py --headless --task Unitree-H2-Velocity \
  --num_envs 256 --max_iterations 20 --seed 42
```

Expected: 20 iterations 完成；产生 checkpoint；无 NaN、CUDA/PhysX crash 或 action/observation dimension error。

- [ ] **Step 5: 使用 play 加载短训 checkpoint**

```bash
python scripts/rsl_rl/play.py --headless --task Unitree-H2-Velocity \
  --num_envs 16 --checkpoint <短训实际 checkpoint 路径>
```

Expected: checkpoint 加载成功并运行；手工终止后无加载或维度错误。checkpoint 路径必须替换为上一步日志中打印的实际路径，不写入仓库。

### Task 6: 新增固定指令集评估入口

**Files:**

- Create: `modules/unitree_rl_lab/scripts/rsl_rl/evaluate_h2.py`
- Create: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/evaluation.py`
- Modify: `modules/unitree_rl_lab/test/test_h2_locomotion_static.py`

- [ ] **Step 1: 为纯指标函数写本地失败测试**

在 `h2/evaluation.py` 中实现不依赖 Isaac Lab 的 `H2EvaluationAccumulator`：累计 active mask、XY velocity squared error、yaw squared error、undesired contact mask 和 invalid mask；`summary()` 返回 survival rate、两个 RMSE、undesired-contact episode rate 和 invalid count。评估脚本只消费该类，Isaac Lab import 不进入纯指标模块。

在静态测试通过 `importlib.util.spec_from_file_location` 直接导入 `evaluation.py`，用两步小 tensor fixture 验证公式，避免本地通过 package import 间接加载 Isaac Lab。

- [ ] **Step 2: 实现评估 CLI**

CLI 参数必须包含：`--task`（默认 `Unitree-H2-Velocity`）、`--checkpoint`、`--num_envs`（默认 256）、`--duration`（默认 20 秒）、`--seed`（默认 42）、`--output`（可选 JSON）。复用 `play.py` 的 AppLauncher、env cfg、runner 和 checkpoint 加载流程。

评估时必须关闭 observation corruption、push event、physics material/base mass randomization，设置固定 seed，并依次运行：stand、forward、backward、left、right、yaw_left、yaw_right、mixed。每个场景直接写入 command manager 的 `base_velocity` command tensor，不等待随机 command resampling。

- [ ] **Step 3: 定义指标与退出码**

输出 JSON 字段：

```json
{
  "checkpoint": "...",
  "seed": 42,
  "scenarios": {},
  "aggregate": {
    "survival_rate": 0.0,
    "linear_velocity_rmse": 0.0,
    "yaw_velocity_rmse": 0.0,
    "undesired_contact_episode_rate": 0.0,
    "invalid_count": 0
  },
  "passed": false
}
```

通过条件：survival ≥ 0.90、linear RMSE ≤ 0.20、yaw RMSE ≤ 0.25、undesired contact rate ≤ 0.05、invalid count = 0。未通过返回 1，配置/checkpoint/指标错误返回 2，通过返回 0。

- [ ] **Step 4: 本地运行纯指标测试和静态检查**

Run:

```bash
python3 -m pytest modules/unitree_rl_lab/test/test_h2_locomotion_static.py -q
python3 -m compileall -q modules/unitree_rl_lab/scripts/rsl_rl/evaluate_h2.py modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/evaluation.py
git -C modules/unitree_rl_lab diff --check
```

Expected: 全部 PASS。

- [ ] **Step 5: 服务器用短训 checkpoint 验证评估入口**

```bash
conda activate env_isaaclab
python scripts/rsl_rl/evaluate_h2.py --headless \
  --checkpoint <短训实际 checkpoint 路径> \
  --num_envs 32 --duration 2 --output /tmp/h2_short_eval.json
```

Expected: 脚本完整执行并生成 schema 正确的 JSON；短训策略允许指标不通过，但不得崩溃或缺字段。

- [ ] **Step 6: 提交评估入口**

```bash
git -C modules/unitree_rl_lab add scripts/rsl_rl/evaluate_h2.py source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/evaluation.py test/test_h2_locomotion_static.py
git -C modules/unitree_rl_lab commit -m "feat: add H2 locomotion evaluation"
```

### Task 7: 更新文档、运行正式训练并集成子模块

**Files:**

- Modify: `modules/unitree_rl_lab/README.md`
- Update: root `modules/unitree_rl_lab` gitlink

- [ ] **Step 1: 更新 README**

增加 H2 task 命令、完整工作区资产依赖和服务器训练示例：

```bash
conda activate env_isaaclab
python scripts/rsl_rl/train.py --headless --task Unitree-H2-Velocity \
  --num_envs 4096 --max_iterations 5000 --seed 42
```

明确 H2 任务不能从单独克隆的子模块运行，并注明 PD 参数需要按实机规格复核。

- [ ] **Step 2: 运行子模块最终静态检查**

```bash
python3 -m pytest modules/unitree_rl_lab/test/test_h2_locomotion_static.py -q
python3 -m compileall -q modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab modules/unitree_rl_lab/scripts/rsl_rl/evaluate_h2.py
git -C modules/unitree_rl_lab diff --check
```

Expected: 全部 PASS。

- [ ] **Step 3: 提交 README**

```bash
git -C modules/unitree_rl_lab add README.md
git -C modules/unitree_rl_lab commit -m "docs: document H2 locomotion training"
```

- [ ] **Step 4: 在服务器执行正式训练**

```bash
cd /home/gaojie/workspace/legged_rl/modules/unitree_rl_lab
conda activate env_isaaclab
python scripts/rsl_rl/train.py --headless --task Unitree-H2-Velocity \
  --num_envs 4096 --max_iterations 5000 --seed 42
```

Expected: 5000 iterations 完成，保存最终 checkpoint；记录墙钟时间和日志路径；无 NaN 或仿真崩溃。

- [ ] **Step 5: 运行正式性能评估**

```bash
python scripts/rsl_rl/evaluate_h2.py --headless \
  --checkpoint <正式训练最终 checkpoint 路径> \
  --num_envs 256 --duration 20 --seed 42 \
  --output /tmp/h2_final_eval.json
```

Expected: exit 0 且 aggregate 指标满足全部设计阈值。若 exit 1，保留日志并返回奖励/控制参数设计，不把实现标记为完成。

- [ ] **Step 6: 更新根仓库 gitlink**

确认子模块提交可由团队访问后，在根仓库执行：

```bash
git add modules/unitree_rl_lab docs/superpowers/plans/2026-07-20-h2-locomotion-training.md
git commit -m "feat(unitree_rl_lab): add H2 locomotion training"
```

- [ ] **Step 7: 根仓库最终检查**

```bash
git diff --check
git submodule status
git status --short
```

Expected: 无未提交 H2 改动；`.vscode/` 保持未跟踪且未提交；gitlink 指向已提交的 H2 子模块 commit。
