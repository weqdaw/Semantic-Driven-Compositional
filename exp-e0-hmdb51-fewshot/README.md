# E0：HMDB51 少样本冻结视觉基线

**当前状态：已实现并运行真实 CLIP 开发基线。最终测试仍封存。**

公开项目总览见[根README](../README.md)，最新E0/E1开发结果见[结果综述](docs/RESULTS_OVERVIEW.md)。本页下方的第一轮原始报告链接属于本地生成产物，不随代码发布。

**第二轮已于2026-11-17完成60组开发运行。** 用户明确委托Codex AI做保守来源审核：350段训练视频归为119个来源阻断组（无人工签字或confirmed来源认证），支持池/开发来源ID互斥。V0/V1的1/3/5-shot Macro-F1均值分别为33.62/42.28/46.97%与33.69/42.41/46.75%，本轮未见稳定增益。见[第二轮结果](reports/e0-round2-report-20261117T081008202832Z-19e962f0/round2-results.md)、[AI来源审核说明](data/source_groups/ai-review-20261117/AI_SOURCE_AUDIT.md)和当前产物指针`reports/round2_latest.json`。来源审核、划分和V0/V1配对使用独立入口`scripts/07_round2.py`，复现步骤见[第二轮V1操作说明](docs/ROUND2_V1_GUIDE.md)。下文原有指标和01—06命令属于第一轮。

目标：在固定的 HMDB51 五类子集上，以冻结 CLIP 图像特征建立 1／3／5-shot 分类基线，为后续语义、时间和关系实验提供可复用对照。

2026-11-18已完成第二轮补充消融及E1概念开发实验。原序V1未见稳定优势；E1中C1优于C0但尚未体现相对视觉或随机对照的整体语义增量，见[结果综述](docs/RESULTS_OVERVIEW.md)。独立入口`scripts/09_e1_concepts.py`提供16个固定姿态概念、C0/C1及5种匹配随机投影R0/R1；复现按[E1操作指南](docs/E1_CONCEPT_GUIDE.md)执行，新增360组、引用60组。原01—08行为和历史结果保留，最终测试仍封存。

## 从这里开始

**下一轮 E1.1 已增加独立实现，尚未执行正式实验。** 脚本10提供共同正交几何对照G1、固定等权融合VC1/VG1，以及180行概念画面审核材料导出。新增330组并复用V1/C1共60组，操作顺序与解释边界见[E1.1指南](docs/E11_GUIDE.md)。

官方 Split 1训练视频每类70段，均通过解码资格检查；每类固定10段开发、60段支持池。每个K使用种子0–9的10个不同支持集，评价同一组50段开发视频。

| K | Accuracy | Macro-F1 |
| --- | --- | --- |
| 1 | 38.60% ± 5.58% | 35.14% ± 7.09% |
| 3 | 47.40% ± 5.08% | 45.51% ± 4.98% |
| 5 | 50.80% ± 5.09% | 49.30% ± 5.64% |

±为跨支持集的样本标准差（ddof=1），不是置信区间。30组运行完整重复一次，全部逐视频得分和指标一致；新进程重新编码一段真实视频，帧/视频特征逐位一致，骨干参数哈希不变。37项自动测试通过，已完成独立代码审查。

查看[完整报告与图表](reports/e0-results.md)、[本次运行记录](docs/E0_RUN_RECORD.json)、[验收清单](docs/ACCEPTANCE_CHECKLIST.md)。本机各产物具体路径在`reports/local_run.json`。

本地6个clap文件缺少官方前导`#`；已与官方原始视频逐一比较内容哈希，使用[显式只读别名](docs/provenance/verified_path_aliases.json)，未重命名外部数据。五类1,165个视频全部对应官方列表，未排除视频，未发现字节级重复。不同文件的原始来源片段分组尚未独立核实；本结果不宣称来源互斥。

## 复现

[环境说明](environment/README.md)记录安装和官方输入来源。本机已建好`.venv`和`configs/paths.local.yaml`。以下命令在仓库根目录的PowerShell执行；每步打印下一步需要的真实路径，用它替换示例中的“实际运行ID”：

```powershell
$E0Python = '.\.venv\Scripts\python.exe'
& $E0Python -m pytest -q
& $E0Python scripts/01_audit_dataset.py --aliases docs/provenance/verified_path_aliases.json
$E0Audit = 'data/manifests/audit-实际运行ID/audit.json'
& $E0Python scripts/02_build_manifests.py --audit $E0Audit
$E0Manifest = 'data/manifests/manifest-实际运行ID/bundle.json'
& $E0Python scripts/04_sample_support_sets.py --manifest $E0Manifest
$E0Supports = 'data/support_sets/supports-实际运行ID/support_sets.json'
& $E0Python scripts/03_extract_features.py --manifest $E0Manifest --stage smoke --device cuda
$E0SmokeIndex = 'artifacts/features/实际缓存身份/index-smoke.json'
& $E0Python scripts/05_run_v0.py --manifest $E0Manifest --support-sets $E0Supports --feature-index $E0SmokeIndex --stage smoke
& $E0Python scripts/03_extract_features.py --manifest $E0Manifest --stage development --device cuda
$E0Index = 'artifacts/features/实际缓存身份/index-development.json'
& $E0Python scripts/05_run_v0.py --manifest $E0Manifest --support-sets $E0Supports --feature-index $E0Index
$E0Batch = 'runs/e0-v0-development-实际运行ID'
& $E0Python scripts/06_summarize_results.py --batch $E0Batch
```

复用本机已冻结输入直接重跑30组（生成新目录）：

```powershell
$E0LocalRun = Get-Content -Raw -Encoding UTF8 reports/local_run.json | ConvertFrom-Json
& ./.venv/Scripts/python.exe scripts/05_run_v0.py --manifest $E0LocalRun.manifest --support-sets $E0LocalRun.supports --feature-index $E0LocalRun.feature_index
```

失败返回非零退出码并保留错误记录；不能跳过缺失输入继续正式评价。缓存绑定视频、检查点、预处理、解码、代码和协议身份，损坏缓存会报错。CPU/GPU属于不同特征条件，需重新提取。

## 已确定的 E0 协议

| 项目 | 决定 |
| --- | --- |
| 数据位置 | F:/Semantic-Driven-dataset/hmdb51，只读引用 |
| 类别顺序 | clap、jump、run、walk、wave；ID 固定为 0–4 |
| 官方划分 | Split 1；标签1为训练候选，2为最终测试，0不用于本折 |
| 开发集 | 从官方训练中每类留10段，固定种子20261116 |
| 支持集 | 剩余训练池每类抽K段，K=1/3/5，种子0–9，嵌套抽样 |
| 视频表示 | 全视频16段分层中点采帧，冻结CLIP ViT-B/32；帧特征均值＋总体标准差 |
| 分类器 | 多输出L2岭分类器，固定λ=1，含不惩罚截距，取最高类别分数 |
| 首轮评价 | 开发集；最终测试在完整比较方案冻结后才开启 |

本地数据文件数、目录存在性均不能证明官方版本完整或视频可解码。官方列表导入后必须审计缺失、重复和标签冲突。

## 目录职责

- `configs/`：固定协议与本机路径配置。
- `src/hmdb51_e0/`：审计、抽样、编码、缓存、岭分类、指标与报告模块。
- `scripts/`：六个阶段入口，均支持`--help`。
- `data/`：划分标注、本地清单、支持集记录；不存放原始视频副本。
- `artifacts/`：权重和特征缓存，默认不进入版本控制。
- `runs/`：批次快照、环境和每次运行产物。
- `reports/`：已验证的可读报告及图表。
- `templates/`：仅含表头的文件格式样例，不是实际实验产物。
- `tests/`：数值、身份、错误处理、视频解码与完整流程测试。
- `environment/`：环境准备与版本冻结要求。
- `references/`：数据与方法来源。

V1可复用16×512的逐帧Z；V0为1024维视频向量。后续语义方法须复用开发和支持ID；不能挑种子、类别或开发视频。CLIP已有图文预训练，后续比较检验显式概念的额外收益。当前未初始化Git，使用源码快照哈希记录代码身份。大型权重、数据、缓存及运行输出均忽略版本控制。
