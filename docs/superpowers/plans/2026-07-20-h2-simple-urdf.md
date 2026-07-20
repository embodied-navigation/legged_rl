# H2 Simple URDF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于完整 `H2.urdf` 人工创建 `H2_simple.urdf`，保留双腿和腰部共 15 个可动关节，固定头部与双臂，并用 box 替换所有启用的 mesh collision。

**Architecture:** 直接复制并人工编辑现有 URDF，不增加生成脚本、配置文件、测试代码或验证报告。保留全部 link、visual、inertial、camera 和 IMU frame，只修改目标 joint 的类型以及启用的 collision geometry。

**Tech Stack:** URDF/XML、文本编辑器、现有 URDF 查看器

---

## 范围

**唯一实现交付物：** `assets/urdf/h2_description/H2_simple.urdf`

不得修改：

- `assets/urdf/h2_description/H2.urdf`
- `assets/urdf/h2_description/H2_dae.urdf`
- `assets/urdf/h2_description/meshes/`

不创建生成器、YAML/JSON 配置、自动化测试或验证报告；不执行关节扫描、落体、PD 站立、Isaac/MuJoCo 交叉验证或真实机器人验证。

### Task 1: 从原始 URDF 创建简化资产

**Files:**

- Create: `assets/urdf/h2_description/H2_simple.urdf`
- Reference: `assets/urdf/h2_description/H2.urdf`

- [ ] **Step 1: 复制原始 URDF**

Run:

```bash
cp assets/urdf/h2_description/H2.urdf \
  assets/urdf/h2_description/H2_simple.urdf
```

Expected: `H2_simple.urdf` 存在，且复制后与 `H2.urdf` 内容一致。

- [ ] **Step 2: 修改 robot 名称**

将根元素：

```xml
<robot name="H2">
```

改为：

```xml
<robot name="H2_simple">
```

- [ ] **Step 3: 确认保留 15 个可动关节**

以下关节保持 `type="revolute"`，并保持原有 `origin`、`parent`、`child`、`axis` 和 `limit` 不变：

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

- [ ] **Step 4: 将头部和左臂关节固定在零位**

把以下 9 个 joint 的 `type="revolute"` 改为 `type="fixed"`：

```text
head_pitch_joint
head_yaw_joint
left_shoulder_pitch_joint
left_shoulder_roll_joint
left_shoulder_yaw_joint
left_elbow_joint
left_wrist_roll_joint
left_wrist_pitch_joint
left_wrist_yaw_joint
```

每个 joint 保留原有 `origin`、`parent` 和 `child`，删除整个 `<axis .../>` 与 `<limit .../>` 元素。保留原 joint origin 即表示固定在 `q=0` 姿态。

- [ ] **Step 5: 将右臂关节固定在零位**

对以下 7 个 joint 做同样处理：

```text
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
right_wrist_roll_joint
right_wrist_pitch_joint
right_wrist_yaw_joint
```

Expected: 共 16 个原 revolute joint 转为 fixed；15 个腿部和腰部 joint 仍为 revolute。原有 camera、IMU 和 hand fixed joint 不变。

### Task 2: 人工简化启用的 mesh collision

**Files:**

- Modify: `assets/urdf/h2_description/H2_simple.urdf`
- Reference: `assets/urdf/h2_description/meshes/*.stl`

- [ ] **Step 1: 保留现有 primitive collision**

以下碰撞体保持不变：

- `left_hip_roll_link`、`right_hip_roll_link` 的 sphere；
- `left_knee_link`、`right_knee_link` 的 cylinder。

不要启用原 URDF 中被 XML 注释包围的 collision。

- [ ] **Step 2: 替换骨盆、腿部和躯干的启用 mesh collision**

人工查看 visual mesh，并将以下 link 中启用的 mesh collision 替换为一个或少量 box：

```text
pelvis
left_hip_yaw_link
left_ankle_pitch_link
right_hip_yaw_link
right_ankle_pitch_link
torso_link
```

box 写法：

```xml
<collision>
  <origin xyz="X Y Z" rpy="R P Y"/>
  <geometry>
    <box size="L W H"/>
  </geometry>
</collision>
```

`X Y Z`、`R P Y` 和 `L W H` 必须根据 visual mesh 在对应 link 局部坐标系中的实际外形人工填写。允许一个 link 使用多个 `<collision>`，避免用过大的单个 box 覆盖明显空白区域。

- [ ] **Step 3: 单独调整双脚碰撞体**

每个 `ankle_pitch_link` 至少使用两个 box：

- `sole`：薄 box，描述实际有效支撑面；
- `foot_body`：较小 box，描述脚背和脚主体。

人工确认左右 sole 的长度、宽度、厚度、最低面高度和镜像关系。不得为了提高训练稳定性而明显放大支撑面。

- [ ] **Step 4: 替换头部和双臂的启用 mesh collision**

将以下 link 中启用的 mesh collision 替换为 box：

```text
head_yaw_link
left_shoulder_pitch_link
left_shoulder_roll_link
left_shoulder_yaw_link
left_elbow_link
left_wrist_yaw_link
left_hand_link
right_shoulder_pitch_link
right_shoulder_roll_link
right_shoulder_yaw_link
right_elbow_link
right_wrist_yaw_link
right_hand_link
```

上臂和前臂使用细长 box，手部使用小 box。默认零位姿态下，box 不得明显侵入 torso、大腿或另一侧手臂。

- [ ] **Step 5: 确认没有启用的 mesh collision**

Run:

```bash
python - <<'PY'
from pathlib import Path
import xml.etree.ElementTree as ET

path = Path("assets/urdf/h2_description/H2_simple.urdf")
root = ET.parse(path).getroot()
meshes = root.findall("./link/collision/geometry/mesh")
assert not meshes, f"enabled mesh collisions remain: {len(meshes)}"
print("PASS: no enabled mesh collision")
PY
```

Expected: 输出 `PASS: no enabled mesh collision`。

### Task 3: 执行最小验证

**Files:**

- Verify: `assets/urdf/h2_description/H2_simple.urdf`

- [ ] **Step 1: 检查 XML 和可动关节集合**

Run:

```bash
python - <<'PY'
from pathlib import Path
import xml.etree.ElementTree as ET

path = Path("assets/urdf/h2_description/H2_simple.urdf")
root = ET.parse(path).getroot()
movable_types = {"revolute", "continuous", "prismatic"}
actual = [
    joint.attrib["name"]
    for joint in root.findall("joint")
    if joint.attrib["type"] in movable_types
]
expected = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_roll_joint", "left_ankle_pitch_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_roll_joint", "right_ankle_pitch_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]
assert actual == expected, f"unexpected movable joints: {actual}"
print("PASS: XML valid and movable joint set is 15-DoF")
PY
```

Expected: 输出 `PASS: XML valid and movable joint set is 15-DoF`。

- [ ] **Step 2: 在现有查看器中加载资产**

使用项目当前可用的 URDF 查看器加载：

```text
assets/urdf/h2_description/H2_simple.urdf
```

只检查：

- URDF 能正常加载；
- visual 与 collision 大致对齐；
- 头部和双臂处于原模型零位；
- 默认姿态没有明显自穿透；
- 双脚 sole 位于相同地面高度；
- 不存在明显的非足部初始地面接触。

- [ ] **Step 3: 执行仓库级文档与状态检查**

Run:

```bash
git diff --check
git submodule status
```

Expected: `git diff --check` 无输出；两个子模块未发生修改或 gitlink 变化。

## 完成标准

- [ ] 只新增 `assets/urdf/h2_description/H2_simple.urdf` 这一项实现资产。
- [ ] 原始 URDF、mesh 和 inertial 数据未修改。
- [ ] 双腿和腰部共 15 个关节保持可动。
- [ ] 头部和双臂共 16 个关节固定在零位。
- [ ] 所有启用的 mesh collision 已人工替换为 box。
- [ ] 原有 hip sphere 和 knee cylinder 保持不变。
- [ ] XML、关节集合和最小可视化检查通过。
