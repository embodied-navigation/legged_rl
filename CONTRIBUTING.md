# 开发与贡献指南

本文定义 `legged_rl` 及其子仓库的开发、提交、评审和合并流程。参与者提交代码前应阅读本指南；自动化智能体还应遵守根目录 `AGENTS.md`。

## 1. 分支模型

- `main`：生产和发布分支，只包含已发布或可立即发布的稳定版本。每个正式版本必须在此分支创建标签。
- `develop`：日常集成分支，包含下一版本已经评审通过的改动。功能分支通常从这里创建并合回这里。
- `feature/<简短描述>`：功能、重构、测试和普通文档开发分支，从 `develop` 创建，合并回 `develop`。
- `release/v<版本号>`：发布准备分支，从 `develop` 创建，只接受版本号、文档、发布说明和阻塞性缺陷修复，完成后合并到 `main`，再同步回 `develop`。
- `hotfix/<简短描述>`：线上紧急修复分支，从 `main` 创建，完成后合并到 `main`，再同步回 `develop`。

分支名使用小写英文和连字符，例如：

- `feature/add-go2-task`
- `feature/update-installation`
- `release/v1.2.0`
- `hotfix/g1-joint-limit`

一个分支只解决一个主题。功能开发、无关重构和依赖升级应拆分处理。禁止直接向 `main` 和 `develop` 推送代码。

## 2. 开发流程

普通开发从最新 `develop` 创建功能分支：

```bash
git switch develop
git pull --ff-only
git submodule update --init --recursive
git switch -c feature/<简短描述>
```

开发时遵循以下原则：

- 优先做小而完整的改动，避免无关格式化或重构扩大 diff。
- 遵循目标子仓库已有的格式化、静态检查和测试配置。
- 新增行为应补充相应测试；无法自动测试时，在 PR 中给出可重复的手工验证步骤。
- 公共接口、配置格式、训练参数或默认行为发生变化时，同步更新文档。
- 不提交密钥、机器路径、数据集、模型权重、训练日志、缓存和构建产物。
- 涉及真实机器人或 Sim2Real 的验证必须明确环境、安全措施和操作者授权。

## 3. 子模块开发流程

功能代码应提交到对应 fork，而不是只在主仓库中留下未提交的子模块修改。

1. 在目标子模块中基于其主分支创建分支并完成修改。
2. 在子模块仓库提交、推送并创建 PR。
3. 子模块 PR 通过检查和评审后先合并。
4. 在本仓库分支中将子模块更新到已合并的提交。
5. 提交根仓库的 gitlink 更新，并在 PR 中链接子模块 PR。

检查主仓库和子模块状态：

```bash
git status --short
git submodule status
git -C modules/unitree_rl_lab status --short
git -C modules/unitree_rl_gym status --short
git -C modules/unitree_mujoco status --short
git -C modules/deploy status --short
```

根仓库不得引用仅存在于个人本地、临时分支或无法被团队获取的子模块提交。

## 4. 代码与文档规范

### Python

- 以目标子仓库现有配置为准，不引入第二套冲突的格式化规则。
- 新代码使用清晰的类型、命名和小函数；注释解释原因，不重复代码表面含义。
- 不使用宽泛异常捕获掩盖错误；错误信息应包含定位问题所需的上下文。
- 不无依据地改变随机种子、奖励、观测、动作空间或训练默认值。

### 配置与实验

- 配置变更应说明兼容性、预期影响和复现实验所需参数。
- 性能或训练效果声明应附任务名、随机种子、硬件、迭代次数和关键指标。
- 大型生成物保存在仓库外部，并在文档中记录获取方式和校验信息。

### 文档

- 本集成仓库文档默认使用中文；代码标识符、命令和专有名词保持原样。
- 命令示例应可复制，路径应与当前仓库结构一致。
- 修改行为时同步修改相关 README、配置说明或部署文档。

## 5. 提交规范

提交信息采用 Conventional Commits 格式：

```text
<类型>(<可选范围>): <简短说明>
```

示例：

```text
feat(unitree_rl_gym): add Go2 terrain curriculum
fix(unitree_rl_lab): correct G1 joint limit mapping
docs: add submodule development workflow
chore: bump unitree_rl_gym submodule
```

要求：

- 标题使用祈使语气，清楚说明改动，不以句号结尾。
- 每个提交应是可评审、可回退的逻辑单元。
- 破坏性变更使用 `!`，并在正文中增加 `BREAKING CHANGE:` 说明迁移方法。
- 不提交调试代码、无意义的 `WIP` 提交或混合多个无关主题的提交。
- 合并前可整理提交历史，但不要改写其他人已基于其开发的共享历史。

## 6. Pull Request 规范

PR 应保持聚焦，并包含：

- 问题背景和变更目标；
- 主要实现和重要设计取舍；
- 已运行的验证命令及结果；
- 未运行的验证、原因和潜在风险；
- 配置、兼容性、训练效果或部署安全影响；
- 关联 issue、子模块 PR，以及必要的日志、指标或截图。

创建 PR 前：

```bash
git diff --check
git submodule status
```

同时运行目标子仓库适用的格式化、静态检查和测试。不要为了通过检查而降低测试强度、屏蔽错误或删除断言。

## 7. 评审规范

评审按以下优先级进行：

1. 正确性、机器人安全和数据安全；
2. 行为兼容性、训练可复现性和子模块提交可获取性；
3. 测试覆盖、错误处理和可维护性；
4. 性能、代码风格和文档完整性。

作者应回应所有阻塞意见。重要修改完成后重新请求评审；评审者不应批准自己无法理解或无法确认风险边界的改动。

## 8. 合并规范

- PR 必须通过必需检查、解决冲突并获得至少一名非作者评审者批准。
- `feature/*` 合并到 `develop` 默认使用 **Squash and merge**，PR 标题应符合提交规范。
- `release/*` 合并到 `main` 使用 **Create a merge commit**，保留清晰的版本发布边界；发布完成后将 `main` 同步回 `develop`。
- `hotfix/*` 合并到 `main` 使用 **Create a merge commit**；修复发布后将 `main` 同步回 `develop`。
- 不将未经发布准备的 `develop` 直接合并到 `main`。
- 作者不得在存在未解决阻塞意见、失败检查或未知子模块提交时合并。
- 合并子模块更新前，确认对应子仓库提交已进入其受保护主分支。
- 合并后删除已完成的 `feature/*`、`release/*` 或 `hotfix/*` 远端分支。

紧急修复也必须通过 PR。确需缩短评审流程时，应记录原因、风险、回滚方案，并在合并后补齐测试和复盘。

## 9. 版本发布规范

版本号遵循语义化版本 `MAJOR.MINOR.PATCH`：

- `MAJOR`：存在不兼容的接口、配置、模型或部署行为变更；
- `MINOR`：向后兼容的新功能；
- `PATCH`：向后兼容的缺陷修复；
- 预发布版本使用 `-rc.N`，例如 `v1.2.0-rc.1`。

本集成仓库以 Git 标签和 GitHub Release 作为版本发布记录。正式标签格式为 `vX.Y.Z`，必须是指向 `main` 的 annotated tag。`CHANGELOG.md` 中的 `Unreleased` 内容应在发布时归入对应版本，并记录发布日期。

### 正式发布流程

1. 从最新 `develop` 创建 `release/vX.Y.Z`。
2. 冻结新功能，只处理发布阻塞问题。
3. 确认两个子模块均指向可获取、已评审且准备发布的提交。
4. 更新 `CHANGELOG.md`、版本引用和发布文档。
5. 完成静态检查、测试以及适用的训练、仿真或部署验证。
6. 创建 `release/vX.Y.Z` 到 `main` 的 PR，并记录验证证据和回滚方案。
7. 合并后在 `main` 的发布提交上创建并推送 annotated tag：

   ```bash
   git switch main
   git pull --ff-only
   git tag -a vX.Y.Z -m "release: vX.Y.Z"
   git push origin vX.Y.Z
   ```

8. 根据 `CHANGELOG.md` 创建 GitHub Release；发布产物应附来源、兼容性和校验信息。
9. 将 `main` 合并回 `develop`，解决版本文件和发布说明差异。
10. 验证使用该标签执行 `git submodule update --init --recursive` 能完整获取工作区。

标签一经公开不得移动或覆盖。发布错误应通过新补丁版本修复；不得删除并重建同名正式版本。

### 紧急修复发布

从最新 `main` 创建 `hotfix/<简短描述>`，完成评审和验证后合并到 `main`，发布新的 `PATCH` 版本，再将 `main` 同步回 `develop`。不得只修复 `main` 而遗漏 `develop`。

## 10. 建议的仓库保护设置

在 GitHub 中为 `main` 和 `develop` 启用：

- 禁止 force push 和删除分支；
- 必须通过 Pull Request 合并；
- 至少一名审批者；
- 新提交后撤销旧审批；
- 必需状态检查通过后才能合并；
- 必须解决全部 review conversation；
- 自动删除已合并分支。

另外限制 `main` 只接受 `release/*` 和 `hotfix/*` 的 PR，并限制正式版本标签的创建、更新和删除权限。

仓库保护属于 GitHub 设置，本文不会自动启用这些选项，需要仓库管理员配置。
