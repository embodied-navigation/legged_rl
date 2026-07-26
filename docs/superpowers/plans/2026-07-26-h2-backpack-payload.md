# H2 5.5 kg Backpack Payload Implementation Plan

**Goal:** 默认让现有 H2 15/29-DoF 任务加载 5.5 kg 背包资产，并从已有
25,000 轮无背包 checkpoint 迁移训练，同时保持任务、奖励和策略 shape 不变。

**Design:** `docs/superpowers/specs/2026-07-26-h2-backpack-payload-design.md`

## Task 1: Establish asset contract tests

**Files**

- Modify: `modules/unitree_rl_lab/test/test_h2_locomotion_static.py`

- [ ] 定义四份目标 URDF 路径和共享 backpack 常量。
- [ ] 增加失败测试，要求旧文件名不存在、新文件名存在。
- [ ] 断言 15/29-DoF 无背包与背包资产的 movable joint 名称及顺序相同。
- [ ] 断言背包只增加 `backpack_fixed_joint`，且 parent/child、位姿正确。
- [ ] 断言质量为 5.5 kg、惯量、box 尺寸正确且没有 collision。
- [ ] 断言无背包资产不存在 `backpack_link`。
- [ ] 运行目标测试并确认因资产尚未迁移而按预期失败。
- [ ] 在 `unitree_rl_lab` 子模块提交测试。

## Task 2: Rename base assets and add backpack variants

**Files**

- Rename: `assets/urdf/h2_description/H2_stand.urdf`
  → `assets/urdf/h2_description/H2_15dof.urdf`
- Rename: `assets/urdf/h2_description/H2.urdf`
  → `assets/urdf/h2_description/H2_29dof.urdf`
- Create: `assets/urdf/h2_description/H2_15dof_backpack.urdf`
- Create: `assets/urdf/h2_description/H2_29dof_backpack.urdf`

- [ ] 使用 `git mv` 重命名两份无背包资产。
- [ ] 从对应无背包版本复制两份 backpack URDF。
- [ ] 在两个 backpack URDF 末尾增加相同的 `backpack_link` 与 fixed joint。
- [ ] 使用 5.5 kg、批准的惯量、`0.16 0.30 0.36` box visual。
- [ ] 不添加 backpack collision。
- [ ] 用 XML 解析器检查四份资产。
- [ ] 运行 Task 1 的资产契约测试并确认通过。
- [ ] 在根仓库提交资产变更。

## Task 3: Switch articulation defaults and update references

**Files**

- Modify:
  `modules/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`
- Modify: `modules/unitree_rl_lab/test/test_h2_locomotion_static.py`
- Modify: relevant README/docs containing active old asset names

- [ ] 先将子模块工作分支建立在 `origin/main` 的已合并提交上。
- [ ] 把活动路径切换到 `H2_15dof_backpack.urdf` 和
  `H2_29dof_backpack.urdf`。
- [ ] 在每个活动路径上方保留对应无背包路径的邻近注释。
- [ ] 不增加 task ID、配置开关、runner name 或命令行参数。
- [ ] 更新测试常量和文档中的有效旧名称引用。
- [ ] 使用 `rg` 确认活动代码和当前用户文档中没有
  `H2_stand.urdf`/`H2.urdf` 引用；历史 specs/plans 保留审计记录。
- [ ] 运行 H2 locomotion 与 Sim2Sim 静态测试。
- [ ] 在子模块提交并推送功能分支。

## Task 4: Local integration verification

- [ ] `python3 -m pytest -q test/test_h2_locomotion_static.py`
- [ ] `python3 -m pytest -q test/test_h2_sim2sim_static.py`
- [ ] `python3 -m compileall -q source/unitree_rl_lab scripts/rsl_rl`
- [ ] 在根仓库运行 `git diff --check`。
- [ ] 检查根仓库、子模块状态和 gitlink，保留无关用户修改。
- [ ] 记录静态测试中因本地缺少 Isaac/PyTorch 而跳过的项目。

## Task 5: Synchronize exact commits to server 101

- [ ] 推送根仓库与子模块功能分支，或用可审计 bundle 同步。
- [ ] 在独立服务器工作区检出被测根提交和子模块提交。
- [ ] 记录 root SHA、submodule SHA、工作树状态、conda 环境、GPU 和驱动。
- [ ] 确认两个基线 checkpoint 实际存在：

```text
logs/rsl_rl/h2_15dof_velocity/2026-07-25_20-51-29/model_25000.pt
logs/rsl_rl/h2_29dof_velocity/2026-07-26_01-58-44/model_25000.pt
```

- [ ] 若实际文件名不同，记录并使用真实文件名。

## Task 6: Resume-based smoke validation

- [ ] 从 15-DoF 25,000 checkpoint 恢复，32 env，2–5 iterations。
- [ ] 从 29-DoF 25,000 checkpoint 恢复，32 env，2–5 iterations。
- [ ] 确认恢复迭代编号延续，而不是初始化新策略。
- [ ] 确认 15/29 actions、actor/critic observation shape 不变。
- [ ] 确认无 NaN/Inf、shape mismatch、joint-regex mismatch、missing actuator
  或初始化错误。
- [ ] smoke 产生的 checkpoint 只作为验证产物，不用于正式迁移训练。

示例命令中的 `--max_iterations` 是额外迭代数：

```bash
python scripts/rsl_rl/train.py --headless \
  --task Unitree-H2-29dof-Velocity \
  --num_envs 32 --max_iterations 2 --seed 42 \
  --resume \
  --load_run 2026-07-26_01-58-44 \
  --checkpoint model_25000.pt
```

## Task 7: Run backpack migration to 50,000 iterations

- [ ] 15-DoF 重新从其原始无背包 25,000 checkpoint 恢复。
- [ ] 29-DoF 重新从其原始无背包 25,000 checkpoint 恢复。
- [ ] 两者均使用 4096 env，设置约 25,000 个额外 iterations，使累计训练
  达到约 50,000。
- [ ] 将每个正式训练链的前 300 个迁移 iterations 作为在线门控窗口。
- [ ] 检查 survival、base-height termination、base pitch/roll。
- [ ] 检查 linear/yaw tracking、踝关节限位和 action saturation。
- [ ] 检查 undesired contact，以及持续后倾、膝伸直或踝补偿。
- [ ] 若性能持续退化或出现数值/碰撞问题，停止对应任务的长训练。
- [ ] 若门控通过，保持同一进程/训练链继续到累计约 50,000，不从头开始。
- [ ] 记录实际最终 checkpoint 文件名；不假设或重命名为字面上的
  `50000.pt`。

## Task 8: Record evidence and integrate

- [ ] 在设计或计划文档末尾记录服务器 SHAs、命令、日志目录、checkpoint、
  seed、num_envs、耗时和指标。
- [ ] 不提交 checkpoint、日志、评估 JSON 或缓存。
- [ ] 先创建并合并 `unitree_rl_lab` 子模块 PR。
- [ ] 将根 gitlink 更新到子模块可从默认分支获取的合并提交。
- [ ] 提交根资产、文档和 gitlink。
- [ ] 创建根仓库到 `develop` 的 PR，并链接子模块 PR。
- [ ] 合并前运行 `git diff --check`、`git submodule status` 和状态检查。

## Out of scope

- 修改奖励、目标高度或速度跟踪权重。
- 新增 Backpack task ID、`H2_USE_BACKPACK` 或 `--run_name`。
- 背包 collision 或真实 STL mesh。
- 质量随机化。
- 真实机器人、Sim2Real 或 29-DoF 部署。
