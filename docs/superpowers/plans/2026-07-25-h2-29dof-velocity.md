# H2 29-DoF Full-Body Velocity Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留 H2 15-DoF 训练兼容性的同时，新增基于完整 31 活动关节 `H2.urdf`、由策略控制腿腰和双臂共 29 个关节的平地速度任务。

**Architecture:** 采用两套独立环境配置：15-DoF 任务继续使用 `H2_stand.urdf`，29-DoF 任务使用 `H2.urdf`，头部两个关节保留在动力学模型中但不进入策略动作。三个 Gym ID 分别提供显式 15-DoF、显式 29-DoF 和旧名称兼容入口；训练、评估与部署通过严格的 observation/action shape 契约隔离。

**Tech Stack:** Python 3.10、Isaac Lab Manager-Based RL、Gymnasium、RSL-RL、PyTorch、pytest、URDF/XML、Git submodules。

---

## File map

### Root repository

- Modify: `docs/superpowers/specs/2026-07-25-h2-29dof-velocity-design.md` only if implementation reveals a genuine design correction.
- Modify: `README.md` to list the explicit H2 task names and compatibility behavior.
- Modify: `modules/unitree_rl_lab` gitlink only after the submodule implementation is committed and available.

### `modules/unitree_rl_lab`

- Modify: `test/test_h2_locomotion_static.py` for shared 15/29-DoF contracts and evaluator diagnostics.
- Modify: `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py` for separate URDF paths and articulation configurations.
- Create: `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/h2_15dof/__init__.py`.
- Create: `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/h2_15dof/velocity_env_cfg.py`.
- Create: `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/h2_29dof/__init__.py`.
- Create: `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/h2_29dof/velocity_env_cfg.py`.
- Modify: `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/__init__.py` to import both task packages.
- Delete: `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/velocity_env_cfg.py` after its content moves unchanged into `h2_15dof`.
- Modify: `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py` for isolated runner names.
- Modify: `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/evaluation.py` for pure upper-body diagnostic accumulation.
- Modify: `scripts/rsl_rl/evaluate_h2.py` to select controlled joints from the active task and emit upper-body diagnostics.
- Modify: `README.md` for training and compatibility commands.

Do not modify `deploy/robots/h2` in this plan. Its 56-input/15-output contract remains intentionally restricted to the 15-DoF policy.

### Task 1: Define static 15/29-DoF contracts

**Files:**
- Modify: `modules/unitree_rl_lab/test/test_h2_locomotion_static.py`
- Test: `modules/unitree_rl_lab/test/test_h2_locomotion_static.py`

- [ ] **Step 1: Replace the single-task constants with explicit task paths and ordered joint contracts**

Add these constants near the top of the test module:

```python
H2_15DOF_ENV_CFG = (
    PACKAGE_ROOT / "tasks/locomotion/robots/h2/h2_15dof/velocity_env_cfg.py"
)
H2_29DOF_ENV_CFG = (
    PACKAGE_ROOT / "tasks/locomotion/robots/h2/h2_29dof/velocity_env_cfg.py"
)

H2_LEG_WAIST_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

H2_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

H2_HEAD_JOINTS = ["head_pitch_joint", "head_yaw_joint"]
H2_29DOF_ACTION_JOINTS = H2_LEG_WAIST_JOINTS + H2_ARM_JOINTS
H2_31DOF_URDF_JOINTS = (
    H2_LEG_WAIST_JOINTS + H2_HEAD_JOINTS + H2_ARM_JOINTS
)
H2_31DOF_SDK_JOINTS = H2_29DOF_ACTION_JOINTS + H2_HEAD_JOINTS
```

Keep `EXPECTED_JOINTS = H2_LEG_WAIST_JOINTS` temporarily so pre-existing tests remain readable while they are migrated.

- [ ] **Step 2: Add failing URDF and Python-layout tests**

```python
def movable_joint_names(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    return [
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib["type"] in {"revolute", "continuous", "prismatic"}
    ]


def test_h2_15_and_29dof_urdf_contracts():
    assert movable_joint_names(H2_URDF) == H2_LEG_WAIST_JOINTS
    assert movable_joint_names(H2_FULL_URDF) == H2_31DOF_URDF_JOINTS
    assert set(H2_31DOF_SDK_JOINTS) == set(H2_31DOF_URDF_JOINTS)
    assert set(H2_29DOF_ACTION_JOINTS).isdisjoint(H2_HEAD_JOINTS)
    assert len(H2_29DOF_ACTION_JOINTS) == len(set(H2_29DOF_ACTION_JOINTS)) == 29


def test_h2_split_environment_files_exist_and_parse():
    for path in (H2_15DOF_ENV_CFG, H2_29DOF_ENV_CFG):
        assert path.is_file(), path
        parse_python(path)
```

- [ ] **Step 3: Add a reusable AST helper and failing action contract test**

```python
def action_cfg_keywords(path: Path) -> dict[str, ast.expr]:
    tree = parse_python(path)
    actions_cfg = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ActionsCfg"
    )
    assignment = next(
        node
        for node in actions_cfg.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "JointPositionAction"
            for target in node.targets
        )
    )
    return {keyword.arg: keyword.value for keyword in assignment.value.keywords}


def test_h2_action_dimensions_and_order_are_explicit():
    fifteen = action_cfg_keywords(H2_15DOF_ENV_CFG)
    twenty_nine = action_cfg_keywords(H2_29DOF_ENV_CFG)
    assert ast.literal_eval(fifteen["joint_names"]) == H2_LEG_WAIST_JOINTS
    assert ast.literal_eval(twenty_nine["joint_names"]) == H2_29DOF_ACTION_JOINTS
    assert ast.literal_eval(fifteen["preserve_order"]) is True
    assert ast.literal_eval(twenty_nine["preserve_order"]) is True
```

- [ ] **Step 4: Run the new tests and verify the intended failure**

Run:

```bash
cd modules/unitree_rl_lab
pytest -q test/test_h2_locomotion_static.py \
  -k "15_and_29dof or split_environment or action_dimensions"
```

Expected: the URDF-only assertion passes; file-layout and action tests fail because the split environment files do not exist.

- [ ] **Step 5: Commit the contract tests**

```bash
git add test/test_h2_locomotion_static.py
git commit -m "test: define H2 15 and 29-DoF task contracts"
```

### Task 2: Split H2 articulation assets

**Files:**
- Modify: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`
- Modify: `modules/unitree_rl_lab/test/test_h2_locomotion_static.py`

- [ ] **Step 1: Add failing AST assertions for paths, aliases, default pose, and actuator coverage**

Add:

```python
def test_h2_articulation_names_paths_and_compatibility_alias():
    source = UNITREE_CFG.read_text(encoding="utf-8")
    assert (
        'H2_15DOF_URDF_RELATIVE_PATH = '
        'Path("assets/urdf/h2_description/H2_stand.urdf")'
    ) in source
    assert (
        'H2_29DOF_URDF_RELATIVE_PATH = '
        'Path("assets/urdf/h2_description/H2.urdf")'
    ) in source
    assert "UNITREE_H2_15DOF_CFG = UnitreeArticulationCfg(" in source
    assert "UNITREE_H2_29DOF_CFG = UnitreeArticulationCfg(" in source
    assert "UNITREE_H2_CFG = UNITREE_H2_15DOF_CFG" in source
    assert "H2_URDF_PATH = H2_15DOF_URDF_PATH" in source


def test_h2_29dof_default_upper_body_pose_and_head_actuator():
    source = UNITREE_CFG.read_text(encoding="utf-8")
    for entry in (
        '"left_shoulder_roll_joint": 0.40',
        '"right_shoulder_roll_joint": -0.40',
        '"left_shoulder_yaw_joint": -0.40',
        '"right_shoulder_yaw_joint": 0.40',
        '".*_elbow_joint": 1.10',
        '"head_pitch_joint": 0.0',
        '"head_yaw_joint": 0.0',
    ):
        assert entry in source
    for actuator in (
        '"H2_SHOULDER_PITCH"',
        '"H2_SHOULDER_ROLL_YAW"',
        '"H2_ELBOW_WRIST_ROLL"',
        '"H2_WRIST_PITCH_YAW"',
        '"H2_HEAD"',
    ):
        assert actuator in source
```

- [ ] **Step 2: Run the assertions and verify failure**

Run:

```bash
pytest -q test/test_h2_locomotion_static.py \
  -k "articulation_names_paths or default_upper_body"
```

Expected: FAIL because only `UNITREE_H2_CFG` and the stand URDF path exist.

- [ ] **Step 3: Split the asset paths while preserving aliases**

Replace the current H2 path constants with:

```python
H2_15DOF_URDF_RELATIVE_PATH = Path("assets/urdf/h2_description/H2_stand.urdf")
H2_29DOF_URDF_RELATIVE_PATH = Path("assets/urdf/h2_description/H2.urdf")

H2_15DOF_URDF_PATH = resolve_workspace_asset(H2_15DOF_URDF_RELATIVE_PATH)
H2_29DOF_URDF_PATH = resolve_workspace_asset(H2_29DOF_URDF_RELATIVE_PATH)

# Backward-compatible names for the existing 15-DoF task.
H2_URDF_RELATIVE_PATH = H2_15DOF_URDF_RELATIVE_PATH
H2_URDF_PATH = H2_15DOF_URDF_PATH
```

Rename the existing `UNITREE_H2_CFG` assignment to `UNITREE_H2_15DOF_CFG` without changing its spawn, pose, actuators, or SDK joint order.

- [ ] **Step 4: Add the 29-DoF articulation**

Create `UNITREE_H2_29DOF_CFG` beside the 15-DoF config. Reuse the six existing leg/waist actuator blocks verbatim and add these upper-body groups:

```python
UNITREE_H2_29DOF_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(asset_path=H2_29DOF_URDF_PATH),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_pitch_joint": -0.15,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.20,
            "left_shoulder_roll_joint": 0.40,
            "right_shoulder_roll_joint": -0.40,
            "left_shoulder_yaw_joint": -0.40,
            "right_shoulder_yaw_joint": 0.40,
            ".*_elbow_joint": 1.10,
            "head_pitch_joint": 0.0,
            "head_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "H2_HIP": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_.*"],
            effort_limit=360.0,
            velocity_limit=20.0,
            stiffness=300.0,
            damping=5.0,
            armature=0.01,
        ),
        "H2_KNEE": IdealPDActuatorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit=360.0,
            velocity_limit=20.0,
            stiffness=300.0,
            damping=5.0,
            armature=0.01,
        ),
        "H2_ANKLE_ROLL": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint"],
            effort_limit=19.0,
            velocity_limit=28.61,
            stiffness=60.0,
            damping=3.0,
            armature=0.01,
        ),
        "H2_ANKLE_PITCH": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint"],
            effort_limit=66.88,
            velocity_limit=28.61,
            stiffness=60.0,
            damping=3.0,
            armature=0.01,
        ),
        "H2_WAIST_YAW": IdealPDActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit=120.0,
            velocity_limit=28.375,
            stiffness=300.0,
            damping=5.0,
            armature=0.01,
        ),
        "H2_WAIST_ROLL_PITCH": IdealPDActuatorCfg(
            joint_names_expr=["waist_(roll|pitch)_joint"],
            effort_limit=180.0,
            velocity_limit=28.375,
            stiffness=300.0,
            damping=5.0,
            armature=0.01,
        ),
        "H2_SHOULDER_PITCH": IdealPDActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint"],
            effort_limit=130.0,
            velocity_limit=21.9,
            stiffness=80.0,
            damping=3.0,
            armature=0.01,
        ),
        "H2_SHOULDER_ROLL_YAW": IdealPDActuatorCfg(
            joint_names_expr=[".*_shoulder_(roll|yaw)_joint"],
            effort_limit=60.0,
            velocity_limit=18.7,
            stiffness=60.0,
            damping=2.0,
            armature=0.01,
        ),
        "H2_ELBOW_WRIST_ROLL": IdealPDActuatorCfg(
            joint_names_expr=[".*_(elbow|wrist_roll)_joint"],
            effort_limit=60.0,
            velocity_limit=18.7,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
        "H2_WRIST_PITCH_YAW": IdealPDActuatorCfg(
            joint_names_expr=[".*_wrist_(pitch|yaw)_joint"],
            effort_limit=10.0,
            velocity_limit=37.7,
            stiffness=20.0,
            damping=1.0,
            armature=0.01,
        ),
        "H2_HEAD": IdealPDActuatorCfg(
            joint_names_expr=["head_(pitch|yaw)_joint"],
            effort_limit=50.0,
            velocity_limit=10.0,
            stiffness=40.0,
            damping=4.0,
            armature=0.01,
        ),
    },
    joint_sdk_names=H2_31DOF_SDK_JOINTS,
)

UNITREE_H2_CFG = UNITREE_H2_15DOF_CFG
```

Define the two production orders immediately above the H2 articulation configs:

```python
H2_31DOF_URDF_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_roll_joint", "left_ankle_pitch_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_roll_joint", "right_ankle_pitch_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_pitch_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

H2_31DOF_SDK_JOINTS = [
    *H2_31DOF_URDF_JOINTS[:15],
    *H2_31DOF_URDF_JOINTS[17:],
    *H2_31DOF_URDF_JOINTS[15:17],
]
```

The SDK order follows the official H2 MJCF actuator order: 29 controlled leg/waist/arm joints followed by the two head joints. Do not import test constants into production.

- [ ] **Step 5: Replace the comment in the actuator dictionary with the six exact existing blocks**

The implementation must contain real dictionary entries, not the explanatory comment shown in Step 4. Copy only those six blocks from `UNITREE_H2_15DOF_CFG`, keeping all numeric values unchanged, and ensure no joint regex belongs to more than one actuator group.

- [ ] **Step 6: Add a URDF limit test for every 29-DoF actuator group**

Parse `H2.urdf` limits into a map and assert each production actuator's `effort_limit` and `velocity_limit` is less than or equal to the minimum matching URDF limit. Use `re.fullmatch()` for the production regexes:

```python
def test_h2_29dof_actuator_limits_do_not_exceed_urdf():
    root = ET.parse(H2_FULL_URDF).getroot()
    limits = {
        joint.attrib["name"]: {
            "effort": float(joint.find("limit").attrib["effort"]),
            "velocity": float(joint.find("limit").attrib["velocity"]),
        }
        for joint in root.findall("joint")
        if joint.attrib["type"] == "revolute"
    }
    assert limits["left_wrist_pitch_joint"] == {"effort": 10.0, "velocity": 37.7}
    assert limits["head_pitch_joint"] == {"effort": 50.0, "velocity": 10.0}
```

Extend this test with the AST actuator extraction already used by `test_h2_articulation_contract`, selecting `UNITREE_H2_29DOF_CFG` and checking every regex match.

- [ ] **Step 7: Run static tests**

Run:

```bash
pytest -q test/test_h2_locomotion_static.py \
  -k "urdf_contracts or articulation or default_upper_body or actuator_limits"
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add \
  source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py \
  test/test_h2_locomotion_static.py
git commit -m "feat: add H2 29-DoF articulation configuration"
```

### Task 3: Split registrations, 15-DoF environment, and runners

**Files:**
- Create: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/h2_15dof/__init__.py`
- Create: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/h2_15dof/velocity_env_cfg.py`
- Modify: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/__init__.py`
- Modify: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`
- Modify: `modules/unitree_rl_lab/test/test_h2_locomotion_static.py`

- [ ] **Step 1: Add a failing three-ID registration and runner test**

```python
def test_h2_task_ids_and_runners_are_isolated():
    registry = H2_REGISTRY.read_text(encoding="utf-8")
    assert "from . import h2_15dof" in registry

    fifteen_registry = (
        PACKAGE_ROOT / "tasks/locomotion/robots/h2/h2_15dof/__init__.py"
    ).read_text(encoding="utf-8")
    assert 'id="Unitree-H2-15dof-Velocity"' in fifteen_registry
    assert 'id="Unitree-H2-Velocity"' in fifteen_registry
    assert "H2_15DOFPPORunnerCfg" in fifteen_registry

    runner_source = PPO_CFG.read_text(encoding="utf-8")
    assert "class H2_15DOFPPORunnerCfg(BasePPORunnerCfg):" in runner_source
    assert "class H2_29DOFPPORunnerCfg(BasePPORunnerCfg):" in runner_source
    assert "H2PPORunnerCfg = H2_15DOFPPORunnerCfg" in runner_source
    assert 'experiment_name = "h2_15dof_velocity"' in runner_source
    assert 'experiment_name = "h2_29dof_velocity"' in runner_source
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest -q test/test_h2_locomotion_static.py -k "task_ids_and_runners"
```

Expected: FAIL because the package split and runner classes do not exist.

- [ ] **Step 3: Move the existing environment without behavioral changes**

Run:

```bash
mkdir -p source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/h2_15dof
git mv \
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/velocity_env_cfg.py \
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/h2_15dof/velocity_env_cfg.py
```

In the moved file, change only the imports and missing-asset message:

```python
from unitree_rl_lab.assets.robots.unitree import (
    H2_15DOF_URDF_PATH,
    UNITREE_H2_15DOF_CFG as ROBOT_CFG,
)
```

Use `H2_15DOF_URDF_PATH` in the file check and include `Unitree-H2-15dof-Velocity` in its error text. Do not change rewards, commands, action order, observation terms, simulation timing, or play settings.

- [ ] **Step 4: Register both 15-DoF names**

Create `h2_15dof/__init__.py`:

```python
import gymnasium as gym


def _register(task_id: str) -> None:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.velocity_env_cfg:RobotEnvCfg"
            ),
            "play_env_cfg_entry_point": (
                f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
                "H2_15DOFPPORunnerCfg"
            ),
        },
    )


_register("Unitree-H2-15dof-Velocity")
_register("Unitree-H2-Velocity")
```

Replace the root H2 registry with:

```python
from . import h2_15dof
```

- [ ] **Step 5: Split the runner configuration**

Replace `H2PPORunnerCfg` with:

```python
@configclass
class H2_15DOFPPORunnerCfg(BasePPORunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 100
    experiment_name = "h2_15dof_velocity"


@configclass
class H2_29DOFPPORunnerCfg(BasePPORunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "h2_29dof_velocity"


H2PPORunnerCfg = H2_15DOFPPORunnerCfg
```

- [ ] **Step 6: Update old static tests to use `H2_15DOF_ENV_CFG`**

Replace all remaining references to `H2_ENV_CFG` with `H2_15DOF_ENV_CFG`. Update runner assertions to expect `H2_15DOFPPORunnerCfg`, `h2_15dof_velocity`, and the compatibility alias. Do not weaken any existing reward, command, foot-link, evaluator, or articulation assertion.

- [ ] **Step 7: Run all pure static tests that do not import Isaac Lab**

Run:

```bash
pytest -q test/test_h2_locomotion_static.py
```

Expected: all 15-DoF behavior and runner tests pass. The intentionally absent 29-DoF package is not imported until Task 4. If collection imports the task package in the local environment, restrict this checkpoint to AST/XML tests and record the missing Isaac Lab dependency.

- [ ] **Step 8: Commit**

```bash
git add \
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2 \
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py \
  test/test_h2_locomotion_static.py
git commit -m "refactor: split H2 15-DoF task registration"
```

### Task 4: Add the 29-DoF velocity environment

**Files:**
- Create: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/h2_29dof/__init__.py`
- Create: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/h2_29dof/velocity_env_cfg.py`
- Modify: `modules/unitree_rl_lab/test/test_h2_locomotion_static.py`

- [ ] **Step 1: Extend the failing environment test with observation and reward contracts**

Add source assertions:

```python
def test_h2_29dof_observation_reward_and_error_contract():
    source = H2_29DOF_ENV_CFG.read_text(encoding="utf-8")
    assert "from . import h2_15dof, h2_29dof" in H2_REGISTRY.read_text(
        encoding="utf-8"
    )
    assert "H2_29DOF_ACTION_JOINTS" in source
    assert source.count('joint_names=H2_29DOF_ACTION_JOINTS') >= 3
    assert "H2_31DOF_URDF_JOINTS" in source
    assert "joint_deviation_shoulders = RewTerm(" in source
    assert "joint_deviation_elbows = RewTerm(" in source
    assert "joint_deviation_wrists = RewTerm(" in source
    assert "upper_body_joint_vel = RewTerm(" in source
    assert "upper_body_joint_acc = RewTerm(" in source
    assert "energy = RewTerm(" in source
    assert "Unitree-H2-29dof-Velocity" in source
    assert "expected 31 physical joints and 29 controlled joints" in source
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest -q test/test_h2_locomotion_static.py \
  -k "action_dimensions or observation_reward or split_environment"
```

Expected: FAIL because the 29-DoF files do not exist.

- [ ] **Step 3: Create the package registration**

Create `h2_29dof/__init__.py`:

```python
import gymnasium as gym

gym.register(
    id="Unitree-H2-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "H2_29DOFPPORunnerCfg"
        ),
    },
)
```

Update the root H2 registry to import both packages:

```python
from . import h2_15dof, h2_29dof
```

- [ ] **Step 4: Copy the 15-DoF environment as the independent baseline**

Run:

```bash
mkdir -p source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/h2_29dof
cp \
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/h2_15dof/velocity_env_cfg.py \
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/h2_29dof/velocity_env_cfg.py
```

Replace the robot import with:

```python
from unitree_rl_lab.assets.robots.unitree import (
    H2_29DOF_URDF_PATH,
    H2_31DOF_URDF_JOINTS,
    UNITREE_H2_29DOF_CFG as ROBOT_CFG,
)
```

Define the action order locally, independent of the SDK/physical order:

```python
H2_29DOF_ACTION_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
```

- [ ] **Step 5: Replace the action configuration**

```python
@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=H2_29DOF_ACTION_JOINTS,
        preserve_order=True,
        scale={
            ".*_hip_.*": 0.25,
            ".*_knee_joint": 0.25,
            ".*_ankle_.*": 0.25,
            "waist_.*_joint": 0.10,
            ".*_shoulder_pitch_joint": 0.15,
            ".*_shoulder_(roll|yaw)_joint": 0.12,
            ".*_elbow_joint": 0.15,
            ".*_wrist_roll_joint": 0.08,
            ".*_wrist_(pitch|yaw)_joint": 0.04,
        },
        use_default_offset=True,
    )
```

- [ ] **Step 6: Restrict actor joint observations and make critic selection explicit**

For actor `joint_pos_rel` and `joint_vel_rel`, add:

```python
params={
    "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=H2_29DOF_ACTION_JOINTS,
        preserve_order=True,
    )
}
```

For critic joint position, velocity, and effort terms, use:

```python
params={
    "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=H2_31DOF_URDF_JOINTS,
        preserve_order=True,
    )
}
```

Keep policy and critic `history_length=5`. The per-frame actor shape is `3 + 3 + 3 + 29 + 29 + 29 = 96`; the runner-facing actor observation is 480.

- [ ] **Step 7: Add separated upper-body rewards**

Keep every existing 15-DoF reward unchanged and add:

```python
    joint_deviation_shoulders = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.10,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_.*_joint"]
            )
        },
    )
    joint_deviation_elbows = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.10,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_elbow_joint"]
            )
        },
    )
    joint_deviation_wrists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.20,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_wrist_.*_joint"]
            )
        },
    )
    head_pose_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.50,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["head_(pitch|yaw)_joint"]
            )
        },
    )
    upper_body_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            )
        },
    )
    upper_body_joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-8,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            )
        },
    )
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
```

The existing global `action_rate` already covers all 29 controlled actions, so do not add a second action-rate term that double-counts upper-body actions.

- [ ] **Step 8: Add initialization-time asset and shape checks**

At the start of `RobotEnvCfg.__post_init__`, validate the asset path and static lists:

```python
        if not Path(H2_29DOF_URDF_PATH).is_file():
            raise FileNotFoundError(
                "Unitree-H2-29dof-Velocity asset not found at "
                f"{H2_29DOF_URDF_PATH}. Run the task from the complete "
                "legged_rl workspace."
            )
        if (
            len(H2_31DOF_URDF_JOINTS) != 31
            or len(H2_29DOF_ACTION_JOINTS) != 29
            or set(H2_31DOF_URDF_JOINTS)
            != set(H2_29DOF_ACTION_JOINTS)
            | {"head_pitch_joint", "head_yaw_joint"}
        ):
            raise ValueError(
                "Unitree-H2-29dof-Velocity expected 31 physical joints "
                "and 29 controlled joints with only the two head joints excluded."
            )
```

Runtime articulation names and resolved observation shapes must additionally be asserted in an Isaac Lab smoke test in Task 7; they are unavailable before the environment is constructed.

- [ ] **Step 9: Run the static H2 suite**

Run:

```bash
pytest -q test/test_h2_locomotion_static.py
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add \
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2 \
  test/test_h2_locomotion_static.py
git commit -m "feat: add H2 29-DoF velocity environment"
```

### Task 5: Add 29-DoF upper-body evaluation diagnostics

**Files:**
- Modify: `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/evaluation.py`
- Modify: `modules/unitree_rl_lab/scripts/rsl_rl/evaluate_h2.py`
- Modify: `modules/unitree_rl_lab/test/test_h2_locomotion_static.py`

- [ ] **Step 1: Write failing pure diagnostic accumulator tests**

```python
def test_h2_upper_body_diagnostics_accumulate_rms_and_saturation():
    spec = importlib.util.spec_from_file_location("h2_eval_diagnostics", H2_EVALUATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    metrics = module.H2UpperBodyDiagnostics(
        shoulder_indices=[0],
        elbow_indices=[1],
        wrist_indices=[2],
        head_indices=[3, 4],
        controlled_joint_indices=[0, 1, 2],
        action_limits=[1.0, 1.0, 1.0],
    )
    metrics.update(
        joint_velocity=[[2.0, 3.0, 4.0, 0.1, 0.2]],
        joint_position=[[0.0, 0.0, 0.0, 0.1, -0.1]],
        actions=[[1.0, 0.5, -1.0]],
        joint_effort=[[1.0, 2.0, 3.0, 0.0, 0.0]],
        active_mask=[True],
    )
    assert metrics.summary() == pytest.approx(
        {
            "shoulder_rms_velocity": 2.0,
            "elbow_rms_velocity": 3.0,
            "wrist_rms_velocity": 4.0,
            "head_angle_rms": 0.1,
            "action_saturation_rate": 2.0 / 3.0,
            "mean_abs_joint_power": 20.0 / 3.0,
        }
    )
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest -q test/test_h2_locomotion_static.py \
  -k "upper_body_diagnostics"
```

Expected: FAIL with missing `H2UpperBodyDiagnostics`.

- [ ] **Step 3: Implement the pure accumulator**

Add a backend-neutral class beside `H2EvaluationAccumulator`. It must accept Python sequences, NumPy arrays, or torch tensors through the same conversion helper already used in that module. Store sums of squares, absolute joint power, saturated-action count, active scalar count, and sample count; ignore inactive rows and reject non-finite active values.

Use these formulas:

```python
group_rms = sqrt(sum(group_joint_velocity ** 2) / group_scalar_count)
head_angle_rms = sqrt(sum(head_joint_position ** 2) / head_scalar_count)
action_saturation_rate = saturated_action_count / active_action_scalar_count
mean_abs_joint_power = (
    sum(abs(controlled_joint_effort * controlled_joint_velocity))
    / controlled_joint_scalar_count
)
```

An action is saturated when `abs(action) >= 0.98 * action_limit`. Raise `ValueError("no active finite upper-body samples")` if `summary()` has no valid active samples.

- [ ] **Step 4: Wire diagnostics into the evaluator**

After resolving `manager_env.scene["robot"]`, derive group indices and `controlled_joint_indices` by exact joint names from `robot.joint_names`. The controlled indices must follow the 29-element action order, not the URDF order, because the two head joints occur between waist and arm joints in `robot.joint_names`. For `Unitree-H2-29dof-Velocity`, construct `H2UpperBodyDiagnostics`; for either 15-DoF task, leave it `None`.

Inside the existing `torch.inference_mode()` step loop, pass:

```python
diagnostics.update(
    joint_velocity=robot.data.joint_vel,
    joint_position=robot.data.joint_pos,
    actions=actions,
    joint_effort=robot.data.applied_torque,
    active_mask=active_mask,
)
```

Attach `diagnostics.summary()` under `"upper_body"` in each 29-DoF scenario and aggregate result. Keep the existing velocity/contact pass thresholds unchanged; upper-body values are diagnostic only.

- [ ] **Step 5: Add evaluator source-contract assertions**

```python
def test_h2_evaluator_emits_29dof_upper_body_diagnostics():
    source = H2_EVALUATOR.read_text(encoding="utf-8")
    assert "H2UpperBodyDiagnostics" in source
    assert 'task == "Unitree-H2-29dof-Velocity"' in source
    assert "robot.data.applied_torque" in source
    assert '"upper_body"' in source
    assert "0.98" in H2_EVALUATION.read_text(encoding="utf-8")
```

- [ ] **Step 6: Run evaluator unit/static tests**

Run:

```bash
pytest -q test/test_h2_locomotion_static.py \
  -k "evaluation or evaluator or upper_body"
```

Expected: PASS, with no `.cpu()`, `.tolist()`, or `.item()` added inside the step loop.

- [ ] **Step 7: Commit**

```bash
git add \
  source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h2/evaluation.py \
  scripts/rsl_rl/evaluate_h2.py \
  test/test_h2_locomotion_static.py
git commit -m "feat: report H2 upper-body evaluation diagnostics"
```

### Task 6: Update H2 user documentation and deployment guardrails

**Files:**
- Modify: `modules/unitree_rl_lab/README.md`
- Modify: `README.md`
- Modify: `modules/unitree_rl_lab/test/test_h2_locomotion_static.py`

- [ ] **Step 1: Add failing documentation contract tests**

```python
def test_h2_readme_documents_explicit_tasks_and_deployment_boundary():
    readme = (SUBMODULE_ROOT / "README.md").read_text(encoding="utf-8")
    for task in (
        "Unitree-H2-15dof-Velocity",
        "Unitree-H2-29dof-Velocity",
        "Unitree-H2-Velocity",
    ):
        assert task in readme
    assert "31 physical joints" in readme
    assert "29 policy-controlled joints" in readme
    assert "15-DoF deployment" in readme
    assert "must not load a 29-DoF checkpoint" in readme
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
pytest -q test/test_h2_locomotion_static.py -k "readme_documents"
```

Expected: FAIL because the README documents only `Unitree-H2-Velocity`.

- [ ] **Step 3: Update the submodule README**

Replace the H2 training introduction with an explicit task table. Include these exact commands:

```bash
python scripts/rsl_rl/train.py --headless \
  --task Unitree-H2-15dof-Velocity \
  --num_envs 4096 --max_iterations 30000 --seed 42

python scripts/rsl_rl/train.py --headless \
  --task Unitree-H2-29dof-Velocity \
  --num_envs 4096 --max_iterations 5000 --seed 42

python scripts/rsl_rl/evaluate_h2.py --headless \
  --task Unitree-H2-29dof-Velocity \
  --checkpoint /path/to/model.pt \
  --num_envs 256 --duration 20 --seed 42 \
  --output logs/h2_29dof_evaluation.json
```

State in English that `Unitree-H2-Velocity` remains a 15-DoF compatibility alias, the 29-DoF name means 29 policy-controlled joints in a model with 31 physical joints, and the existing 15-DoF deployment must not load a 29-DoF checkpoint.

- [ ] **Step 4: Update the root README**

Add a short Chinese H2 training subsection linking the design and plan documents and listing the two explicit task IDs. State that the existing Sim2Sim path remains 15-DoF only.

- [ ] **Step 5: Run documentation and diff checks**

Run:

```bash
pytest -q test/test_h2_locomotion_static.py -k "readme_documents"
git diff --check
```

Expected: PASS.

- [ ] **Step 6: Commit submodule documentation**

```bash
git add README.md test/test_h2_locomotion_static.py
git commit -m "docs: document H2 15 and 29-DoF training"
```

Do not commit the root README from inside the submodule. It will be committed with the root gitlink in Task 8.

### Task 7: Run static, import, and short-training validation

**Files:**
- Modify only if a validation defect is found in a file already listed in Tasks 1–6.

- [ ] **Step 1: Run all local static tests**

Run:

```bash
cd modules/unitree_rl_lab
pytest -q test/test_h2_locomotion_static.py
pytest -q test/test_h2_sim2sim_static.py
git diff --check
```

Expected: PASS. The Sim2Sim tests must remain unchanged because deployment is still 15-DoF.

- [ ] **Step 2: Run repository task discovery where Isaac Lab is available**

Run:

```bash
./unitree_rl_lab.sh -l
```

Expected output contains all three IDs:

```text
Unitree-H2-15dof-Velocity
Unitree-H2-29dof-Velocity
Unitree-H2-Velocity
```

- [ ] **Step 3: Run 15-DoF compatibility smoke tests**

Run:

```bash
python scripts/rsl_rl/train.py --headless \
  --task Unitree-H2-15dof-Velocity \
  --num_envs 32 --max_iterations 2 --seed 42

python scripts/rsl_rl/train.py --headless \
  --task Unitree-H2-Velocity \
  --num_envs 32 --max_iterations 2 --seed 42
```

Expected: both construct 15-joint/15-action environments without NaN/Inf and write to `h2_15dof_velocity`.

- [ ] **Step 4: Run the 29-DoF shape smoke test**

Run:

```bash
python scripts/rsl_rl/train.py --headless \
  --task Unitree-H2-29dof-Velocity \
  --num_envs 32 --max_iterations 2 --seed 42
```

Expected runtime contract:

```text
physical articulation joints: 31
policy-controlled joints: 29
policy observation per frame: 96
policy observation with history: 480
policy action: 29
```

The run must produce no NaN/Inf, joint-regex mismatch, missing actuator, or self-collision initialization error.

- [ ] **Step 5: Run staged training checkpoints**

Run 300 iterations first:

```bash
python scripts/rsl_rl/train.py --headless \
  --task Unitree-H2-29dof-Velocity \
  --num_envs 4096 --max_iterations 300 --seed 42
```

Inspect survival, velocity tracking, upper-body velocity, non-foot contact, and action saturation. Continue to 1000 iterations only if the policy does not show persistent wrist saturation, uncontrolled arm collision, or head drift:

```bash
python scripts/rsl_rl/train.py --headless \
  --task Unitree-H2-29dof-Velocity \
  --num_envs 4096 --max_iterations 1000 --seed 42
```

Do not start the 5000-iteration run until the 1000-iteration checkpoint passes those qualitative gates.

- [ ] **Step 6: Run formal training and fixed-command evaluation**

Run:

```bash
python scripts/rsl_rl/train.py --headless \
  --task Unitree-H2-29dof-Velocity \
  --num_envs 4096 --max_iterations 5000 --seed 42

python scripts/rsl_rl/evaluate_h2.py --headless \
  --task Unitree-H2-29dof-Velocity \
  --checkpoint /absolute/path/to/model_5000.pt \
  --num_envs 256 --duration 20 --seed 42 \
  --output logs/h2_29dof_evaluation.json
```

Set the checkpoint argument to the exact `model_5000.pt` path printed by the completed training command. Expected acceptance:

```text
survival_rate        >= 0.95
linear_velocity_rmse <= 0.25
yaw_velocity_rmse    <= 0.30
invalid_count        == 0
```

The JSON must also contain upper-body RMS velocities, head angle RMS, action saturation rate, mean absolute joint power, and undesired-contact rate.

- [ ] **Step 7: Record unavailable validation**

If Isaac Lab, Isaac Sim, an NVIDIA GPU, or the server environment is unavailable, do not fake Steps 2–6. Record each skipped command and the missing dependency in the final handoff; local static success is not evidence that the task trains.

### Task 8: Finalize submodule and root integration

**Files:**
- Modify: `README.md`
- Modify: `modules/unitree_rl_lab` gitlink

- [ ] **Step 1: Verify submodule history and cleanliness**

Run:

```bash
git -C modules/unitree_rl_lab status --short
git -C modules/unitree_rl_lab log --oneline -6
```

Expected: no uncommitted H2 implementation changes; commits from Tasks 1–6 are present. Training logs, checkpoints, caches, and local environments are not tracked.

- [ ] **Step 2: Verify the root integration diff**

Run:

```bash
git status --short
git diff --check
git submodule status
git diff --submodule=log -- modules/unitree_rl_lab
```

Expected: the intended root README update and `modules/unitree_rl_lab` gitlink are visible. Preserve any pre-existing unrelated root or submodule changes.

- [ ] **Step 3: Commit root documentation and gitlink**

Only after the submodule commit is pushed to an accessible feature branch or merged according to `CONTRIBUTING.md`, run:

```bash
git add README.md modules/unitree_rl_lab
git commit -m "feat(unitree_rl_lab): add H2 29-DoF velocity task"
```

- [ ] **Step 4: Final verification**

Run:

```bash
git diff --check HEAD^
git submodule status
git status --short
```

Expected: checks pass. Report the submodule commit, root gitlink commit, static checks, runtime checks, training/evaluation results, and every skipped GPU/simulator validation.
