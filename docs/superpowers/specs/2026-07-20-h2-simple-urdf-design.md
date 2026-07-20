# H2 Simple URDF 设计说明

## 目标

基于完整的 `assets/urdf/h2_description/H2.urdf` 创建简化训练资产 `H2_simple.urdf`，降低上半身动作复杂度和 mesh collision 的接触计算开销，同时保留腰部对全身姿态调节的能力。

## 自由度设计

模型保留 15 个可动关节：

- 左腿 6 个关节；
- 右腿 6 个关节；
- `waist_yaw_joint`、`waist_roll_joint`、`waist_pitch_joint`。

头部和双臂共 16 个关节改为 `fixed`，固定在原 URDF 的零位姿态。转换时保留 joint 的 `origin`、`parent` 和 `child`，删除不再适用于 fixed joint 的 `axis` 和 `limit`。

所有 link、visual、inertial、camera 和 IMU frame 均保持不变。

## 碰撞模型

所有启用的 mesh collision 均人工替换为 box：

- pelvis、torso 和大腿等较大部位使用一个或少量 box；
- 每只脚使用 `sole` 和 `foot_body` 两个 box；
- 头部、手臂、手腕和手部使用与局部外形相符的 box；
- 保留左右 hip 原有的 sphere；
- 保留左右 knee 原有的 cylinder；
- 不启用原 URDF 中被注释的 collision。

初次检查发现 `pelvis_lower_box` 与左右 `hip_roll_link` sphere 发生干涉。最终将该 box 的 Y 尺寸由 `0.18 m` 收窄为 `0.13 m`，零位下左右间隙分别约为 `9 mm` 和 `12 mm`。

## 文件范围

实现资产：

- `assets/urdf/h2_description/H2_simple.urdf`

实施计划：

- `docs/superpowers/plans/2026-07-20-h2-simple-urdf.md`

原始 `H2.urdf`、`H2_dae.urdf` 和 `meshes/` 不作修改。

## 验收结果

- XML 可正常解析，robot name 为 `H2_simple`；
- 可动关节集合严格为双腿和腰部共 15 个关节；
- 头部和双臂共 16 个关节固定在原零位；
- link 的 visual 和 inertial 与原模型一致；
- collision 包含 25 个 box、2 个 sphere 和 2 个 cylinder，不含启用的 mesh；
- 左右 sole 的尺寸与位置满足镜像关系；
- 零位下 pelvis/torso collision 与 hip collision 无干涉；
- 正视、侧视和俯视叠加检查未发现明显碰撞体错位。

当前环境未安装 MuJoCo，因此未使用 MuJoCo viewer 加载；未进行关节扫描、落体、PD 站立、Isaac 交叉验证或真实机器人验证。
