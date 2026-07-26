# URDF 模型

本目录用于存放机器人 URDF 描述文件及其配套网格资源。提交新模型时，应将 URDF 和模型引用的 `meshes` 目录放在同一个机器人描述目录中，避免使用本机绝对路径。

## H2 机器人

H2 模型位于 `h2_description/`：

- `H2_29dof.urdf`：使用 STL 网格的 29-DoF 策略/31 活动关节机器人描述。
- `H2_29dof_backpack.urdf`：默认训练资产，在 29-DoF 模型上增加固定的
  5.5 kg 背包负载。
- `H2_15dof.urdf`：上半身固定、腿腰 15 个活动关节的无背包资产。
- `H2_15dof_backpack.urdf`：默认 15-DoF 训练资产，增加固定的 5.5 kg
  背包负载。
- `H2_dae.urdf`：可视化模型使用 DAE 网格，碰撞模型仍使用 STL。
- `H2_simple.urdf`：保留双腿和腰部 15 个可动关节的训练模型，上半身关节固定，启用的碰撞体均为 primitive。
- `meshes/`：URDF 引用的 DAE 和 STL 网格文件。

## 检查 URDF

提交或修改 URDF 后，应至少检查 XML 语法、网格路径、连杆与关节层级、关节轴和限位、惯性参数以及碰撞几何。可以使用以下任一方式查看模型。

### 使用 MuJoCo

安装 MuJoCo Python 包后，可以从仓库根目录打开 URDF：

```bash
python -m mujoco.viewer --mjcf assets/urdf/h2_description/H2_29dof_backpack.urdf
```

如果需要检查 DAE 可视化网格，可将文件替换为：

```bash
python -m mujoco.viewer --mjcf assets/urdf/h2_description/H2_dae.urdf
```

加载时应关注终端中的解析警告和错误，并在查看器中检查模型姿态、关节运动、碰撞体和质量分布。MuJoCo 能成功加载模型并不代表 URDF 的动力学参数一定正确，仍需结合机器人规格进行核对。

### 使用在线 Robot Viewer

打开 [Robot Viewer](https://viewer.robotsfan.com/)，导入 URDF 及其引用的 `meshes` 资源，即可在浏览器中查看机器人结构。推荐检查：

- visual 与 collision 几何是否对齐；
- 连杆坐标系、关节轴和旋转方向是否正确；
- 关节限位是否符合机器人规格；
- 网格是否完整，是否存在路径或材质加载错误；
- 不同姿态下是否出现异常穿模。

在线工具适合快速检查模型结构和外观；涉及动力学、接触和控制效果时，应继续在本地 MuJoCo 或目标仿真环境中验证。
