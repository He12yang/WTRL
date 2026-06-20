# offline_pitch_rl 项目 Git 使用手册

> **适用对象**：Git 新手 | **语言**：中文 | **更新日期**：2026-06-20  
> **仓库路径**：`c:/Users/He/Desktop/强化学习新工作/ChatGPT/offline_pitch_rl`

---

## 1. 速查卡 — 6 个命令搞定日常

| 场景 | 命令 | 说明 |
|------|------|------|
| 查看状态 | `git status` | 改了哪些文件？哪些已暂存？ |
| 查看改动 | `git diff` | 具体改了什么内容？ |
| 暂存文件 | `git add config.py` | 把文件加入"购物车" |
| 提交 | `git commit -m "消息"` | 永久保存到仓库 |
| 查看历史 | `git log --oneline` | 一行一条提交记录 |
| 撤销修改 | `git checkout -- config.py` | 放弃对某文件的修改 |

> **核心心法**：`git status` 是你最好的朋友 — 任何操作前后都可以跑它，它会告诉你当前状态和下一步该做什么。

---

## 2. 本项目 Git 状态说明

### 2.1 当前状态

```
当前分支：  master
首次提交：  a6a1e45 — "初始提交：CQL离线强化学习风力发电机桨距角控制项目"
追踪文件：  22 个源文件
远程仓库：  尚未配置
```

### 2.2 `.gitignore` 排除规则

以下文件和目录**不会被纳入 Git 管理**：

| 排除项 | 原因 |
|--------|------|
| `output/*.pth`, `output/*.onnx` | 模型权重文件过大（几十MB），不适合 Git |
| `output/*.json`, `output/*.png` | 训练自动生成的输出文件 |
| `*.csv`（如 `PI_OfflineData.csv`） | 原始数据文件，不应频繁提交 |
| `__pycache__/` | Python 字节码缓存，自动生成 |
| `.vscode/`, `.idea/` | IDE 个人配置 |
| `Thumbs.db`, `.DS_Store` | 操作系统自动生成 |

### 2.3 哪些文件在 Git 管理下？

```
根目录：    config.py, train.py, export_onnx.py, .gitignore
data/：     dataset.py, normalizer.py
models/：   actor.py, cql.py
env/：      reward.py
utils/：    save_scaler.py, load_scaler.py
visualization/： evaluate.py, plot_training.py, *.png（评估图表）
doc/：      technical_report.md, technical_report.tex, technical_report.pdf
```

---

## 3. 日常开发流程（三步走）

### 典型场景：修改了 `config.py` 中的超参数

```bash
# 第一步：查看状态 — 确认改了什么
git status

# 第二步：查看具体改动 — 逐行确认
git diff config.py

# 第三步：暂存并提交
git add config.py
git commit -m "调整CQL_ALPHA从5.0降至1.0，降低保守性惩罚"
```

### 典型场景：同时修改了多个文件

```bash
# 逐个添加（推荐，精确控制）
git add models/actor.py models/cql.py train.py

# 或一次性添加所有修改（谨慎，确保 .gitignore 正确）
git add .

git commit -m "优化Actor网络结构：第三层256→128"
```

### 典型场景：新增了一个文件

```bash
# 新建了 data/augmentation.py
git add data/augmentation.py
git commit -m "添加数据增强模块 augmentation.py"
```

> **提交粒度建议**：每完成一个独立功能点就提交一次，不要攒很多天的工作量。小步快跑，出问题容易定位。

---

## 4. 提交信息规范

### 推荐格式

```
<类型>: <简短描述>

<详细说明（可选）>
```

### 本项目常用类型

| 类型 | 适用场景 | 示例 |
|------|----------|------|
| `feat` | 新功能 | `feat: 添加TD3+BC自适应行为克隆正则化` |
| `fix` | 修bug | `fix: 修复CQL logsumexp维度计算错误` |
| `tune` | 调超参数 | `tune: CQL_ALPHA 5.0→1.0，REWARD_SCALE 0.1→0.01` |
| `refactor` | 重构代码 | `refactor: 将build_reward提取到env/reward.py` |
| `docs` | 文档更新 | `docs: 添加Git使用说明文档` |
| `clean` | 清理代码 | `clean: 删除未使用的utils占位文件` |

### 实践示例

```bash
# 好的提交信息
git commit -m "tune: 降低Critic学习率至1e-4，与Actor保持一致以提升训练平衡"

# 不好的提交信息（太笼统）
git commit -m "修改了一些东西"
```

---

## 5. 实用场景速查

### 5.1 撤销未暂存的修改

```bash
# 改坏了 config.py，想恢复到上次提交的状态
git checkout -- config.py

# 恢复所有修改（危险操作！）
git checkout -- .
```

### 5.2 从暂存区移除文件（但不删除文件）

```bash
# 不小心 git add 了不该提交的文件
git reset HEAD PI_OfflineData.csv
```

### 5.3 修改上一次提交（追加文件或改消息）

```bash
# 忘记添加某个文件
git add 遗漏的文件.py
git commit --amend -m "修正后的提交信息"

# 仅修改提交信息
git commit --amend -m "新的提交信息"
```

### 5.4 查看历史

```bash
git log --oneline                    # 简洁版：一行一条
git log --oneline --graph --all      # 图形化版：显示分支拓扑
git log -p                           # 详细版：显示每次提交的具体改动
git log --since="2026-06-15"        # 查看某日期之后的提交
```

### 5.5 查看某次提交的详情

```bash
git show a6a1e45          # 查看该提交的完整内容
git show a6a1e45 --stat   # 只看该提交改了哪些文件
```

### 5.6 比较版本差异

```bash
git diff HEAD~1           # 与上一次提交比较
git diff a6a1e45 HEAD     # 与指定提交比较
git diff config.py        # 只看某个文件的差异
```

### 5.7 回退到历史版本

```bash
# 临时回退（工作区回到历史版本，但保留修改）
git checkout a6a1e45

# 回到最新版本
git checkout master
```

### 5.8 创建分支（做实验性改动）

```bash
# 创建一个新分支用于实验
git branch experiment

# 切换到新分支
git checkout experiment

# 或者一步到位：创建并切换
git checkout -b experiment

# 在实验分支上修改、提交...
# 如果实验成功，合并回 master
git checkout master
git merge experiment

# 如果实验失败，删掉分支
git branch -d experiment
```

---

## 6. 远程仓库配置（推送到 GitHub）

### 6.1 前置条件

- 已注册 GitHub 账号（本项目用户：`He12yang`）
- 在 GitHub 网页上创建了一个空仓库（**不要**勾选 README、.gitignore、LICENSE）

### 6.2 关联并推送

```bash
# 第一步：关联远程仓库
git remote add origin https://github.com/He12yang/offline_pitch_rl.git

# 第二步：将 master 分支改名为 main（GitHub 默认分支名）
git branch -M main

# 第三步：推送代码
git push -u origin main
```

### 6.3 后续日常推送

```bash
# 第一次配置后，之后只需要：
git push

# 从远程拉取更新（如果有多人协作或你在多台电脑上工作）
git pull
```

### 6.4 查看远程仓库信息

```bash
git remote -v     # 查看远程仓库地址
```

---

## 7. 本项目目录 Git 管理边界

```
offline_pitch_rl/
├── .git/                    ← [Git内部] 仓库数据库，不要手动修改
├── .gitignore               ← [已追踪] Git忽略规则
│
├── config.py                ← [已追踪] 全局配置
├── train.py                 ← [已追踪] 训练主脚本
├── export_onnx.py           ← [已追踪] ONNX导出
│
├── data/
│   ├── dataset.py           ← [已追踪] 数据加载
│   └── normalizer.py        ← [已追踪] 归一化器
│
├── models/
│   ├── actor.py             ← [已追踪] Actor网络
│   └── cql.py               ← [已追踪] Critic网络
│
├── env/
│   └── reward.py            ← [已追踪] 奖励函数
│
├── utils/
│   ├── save_scaler.py       ← [已追踪] 保存工具
│   └── load_scaler.py       ← [已追踪] 加载工具
│
├── visualization/
│   ├── evaluate.py          ← [已追踪] 评估脚本
│   ├── plot_training.py     ← [已追踪] 训练曲线
│   └── *.png                ← [已追踪] 评估生成的图表
│
├── doc/
│   ├── technical_report.md  ← [已追踪] 技术报告
│   ├── technical_report.tex ← [已追踪] LaTeX源文件
│   ├── technical_report.pdf ← [已追踪] PDF报告
│   └── git_guide.md         ← [已追踪] 本Git手册
│
├── output/                  ← [被忽略] 训练输出（模型/日志/图片）
│   ├── actor_best.pth
│   ├── actor.onnx
│   ├── normalization.json
│   └── training_history.json
│
├── PI_OfflineData.csv       ← [被忽略] 原始数据
└── __pycache__/             ← [被忽略] Python缓存
```

---

## 8. 常见问题排查

### Q1: `git status` 显示乱码中文文件名？

```bash
git config --global core.quotepath false
```

### Q2: 提交后发现忘记排除某个文件？

```bash
# 1. 编辑 .gitignore，添加规则
echo "secret.key" >> .gitignore

# 2. 从 Git 追踪中移除（但保留本地文件）
git rm --cached secret.key

# 3. 提交
git add .gitignore
git commit -m "将secret.key加入.gitignore"
```

### Q3: Windows 换行符警告（LF will be replaced by CRLF）？

这是正常现象，不影响使用。Git 在 Windows 上会自动转换换行符。如果想消除警告：

```bash
git config --global core.autocrlf true
```

### Q4: `git commit` 后想反悔？

```bash
# 撤销提交但保留修改（推荐）
git reset --soft HEAD~1      # 文件回到暂存区

# 撤销提交且清空暂存区
git reset HEAD~1             # 文件回到工作区（未暂存）

# 彻底删除提交和修改（危险！）
git reset --hard HEAD~1
```

### Q5: 合并分支时发生冲突怎么办？

```bash
# 1. 查看哪些文件冲突
git status

# 2. 打开冲突文件，手动编辑 — 找到 <<<<<<< ======= >>>>>>> 标记
# 3. 删除标记，保留你想要的代码
# 4. 标记为已解决
git add 冲突文件.py

# 5. 完成合并
git commit -m "解决合并冲突"
```

### Q6: 不小心 `git add .` 暂存了不该提交的文件？

```bash
# 从暂存区移除所有文件
git reset HEAD

# 然后逐个添加需要的文件
git add train.py config.py
```

### Q7: 如何查看 `.gitignore` 是否正确生效？

```bash
# 查看被忽略的文件列表
git status --ignored

# 测试某个文件是否会被忽略
git check-ignore -v output/actor_best.pth
```

---

## 9. 附录：完整工作流程图

```
                    ┌──────────────────────────┐
                    │     修改代码              │
                    │  (编辑 .py 文件)           │
                    └──────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │   git status             │
                    │   查看改了哪些文件         │
                    └──────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │   git diff               │
                    │   逐行确认改动内容         │
                    └──────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │   git add <file>         │
                    │   暂存需要提交的文件       │
                    └──────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │   git commit -m "..."     │
                    │   永久保存到本地仓库       │
                    └──────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │   git push  (可选)        │
                    │   推送到 GitHub 远程仓库   │
                    └──────────────────────────┘
```

---

> **提示**：本文档本身就是用 Git 管理的 — 修改本文档后记得 `git add doc/git_guide.md && git commit -m "docs: 更新Git使用手册"`。
