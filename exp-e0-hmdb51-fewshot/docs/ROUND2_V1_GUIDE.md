# E0 第二轮操作说明：来源分组与 V0 / V1 配对基线

2026-11-18补充：重复均值与帧序打乱使用独立脚本08和配置，不修改本页原60组协议或缓存。见[第二轮补充消融说明](ROUND2_ABLATION_GUIDE.md)。

状态：**2026-11-17已完成AI辅助保守来源审核及60组真实开发运行。** 当前产物指针为reports/round2_latest.json，结果见reports/e0-round2-report-20261117T081008202832Z-19e962f0/round2-results.md。350段训练被划为119个审核来源阻断组，支持池95组/300段、开发24组/50段，来源ID无交集；没有人工签字或完整来源独立性认证。以下命令用于复现，旧审核材料和第一轮结果保留。

## 1. 实验目标和固定条件

先建立可追溯的来源映射，在官方 Split 1 的 eligible 训练视频内部建立来源组互斥的支持池/开发集，再比较普通分段时间表示的增量。原始五类当前有350段 eligible 官方训练视频，具体数量以父清单为准；最终测试150段继续封存。tag=0视频不使用。

这仍是固定五类、开发阶段、视频分类实验，不是动作定位、跨类FSAR或溺水部署评估。

| 项目 | 第二轮约定 |
| --- | --- |
| 类别顺序 | clap / jump / run / walk / wave，ID 0—4 |
| 来源组 | 同一原始影片或上传视频；跨动作类别统一编号 |
| 划分 | 完整来源组互斥；目标每类10段开发，允许实际数量不同 |
| 最低数量 | 每类开发≥1、支持池≥5；每类至少存在可分到两侧的来源 |
| 划分确定 | PCG64，种子20261117；512次仅依据类别计数的局部搜索，不看特征或模型得分 |
| shot | 每类K段完整视频，K=1/3/5，不保证每段来自不同来源 |
| 支持种子 | 0—9；相同种子内1/3-shot是5-shot前缀 |
| 特征 | 原E0冻结CLIP ViT-B/32的同一16帧Z；不重新编码 |
| 分类器 | 原有均值样本损失的多输出岭分类，λ=1，截距不惩罚 |
| 正式运行数 | V0和V1各30组，共60组 |
| 最终测试 | 无开启入口；不解码、不导入特征、不拟合 |

V0为全局均值和总体标准差（2d）；V1为相同两块，加上早/中/晚三段均值（5d）。16帧按1—6、7—11、12—16分段。各块分别除以`L2范数+1e-6`，拼接后再除以整体L2范数。统计量用float64累积、最终float32保存。官方d=512，对应1024/2560维。V0直接调用原有汇总函数，保持算术实现一致。

主比较为新分组协议内V1−V0；旧E0的38.60%/47.40%/50.80%只作历史参考，不能把新V1与旧V0之差全部归因于时间。

## 2. 启动前确认输入

使用现有`.venv`，无需安装新依赖。当前真实模型环境为Python 3.11.9，沿用已有环境；不要把系统Python 3.10或Vault工具使用的Python环境混入实验。

新增入口为`scripts/07_round2.py`，保留原01—06脚本。所有子命令都可以单独查看`--help`，只打印帮助不会启动实验。`--config`和`--paths`放在子命令后。

在PowerShell先执行以下初始化。它从第一轮`reports/summary.json`定位其真实批次，使用批次内冻结快照，避免选择较早的`local_run.json`记录。第二轮不会覆盖这个汇总入口。

```powershell
$E0Root = 'D:\desktop\Semantic-Driven-Compositional\exp-e0-hmdb51-fewshot'
$E0Python = Join-Path $E0Root '.venv\Scripts\python.exe'
$E0Round2 = Join-Path $E0Root 'scripts\07_round2.py'
$E0Summary = Get-Content -LiteralPath (Join-Path $E0Root 'reports\summary.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$E0ParentBatch = $E0Summary.batch
$E0ParentManifest = Join-Path $E0ParentBatch 'manifest_snapshot.json'
$E0ParentFeatureIndex = Join-Path $E0ParentBatch 'feature_index_snapshot.json'

& $E0Python $E0Round2 --help
```

本机路径仍从`configs/paths.local.yaml`读取。原始视频目录、旧缓存目录及权重路径都是输入。不要把任何输出配置到原始数据、旧缓存、权重或标注目录内。

## 3. 第一步：生成待审核来源表

```powershell
$E0Review = & $E0Python $E0Round2 audit-sources --parent-manifest $E0ParentManifest
if ($LASTEXITCODE -ne 0) { throw '来源候选生成失败，停止后续步骤。' }
$E0Review = "$E0Review".Trim()
Write-Output $E0Review
```

输出位于`data/source_groups/source-audit-时间戳-ID/`：

- `source_groups.csv`：全部eligible官方训练视频的候选来源，初始状态全部`unresolved`。
- `candidate_audit.json`：候选来源分布、第一轮支持池/开发候选重合情况及证据限制。

命令仅读取父清单，不打开视频、不训练模型。候选提取规则是文件名中动作/属性后缀之前的原片名，去前导`#`并统一大小写；跨类别使用同一编号。无法解析的条目保持空候选来源，等待人工确认。

**这个步骤产生的是待审核表，不是已完成的来源审计。** 同前缀不自动证明重复，不同前缀也不自动证明独立。

## 4. 来源证据审核：这一环节不能跳过

默认由人工核查。2026-11-17用户明确委托AI完成审核，因此本次允许使用有逐组证据的AI辅助保守审核；reviewer必须明确注明AI，不能登记为人工签字或把关键帧检查写成完整原片溯源。本次350条记录均为conservative，未声称真实来源独立性得到完整认证。具体审计见data/source_groups/ai-review-20261117/。

使用原始来源资料、联系图或必要的视频检查核对来源。代码不替代人工判断，也不为提高分数筛选视频。建议先按`candidate_source_id`排序；同一候选来源的多条记录需一起审核，注意跨类别来源和不同名称的同源视频。

以下字段是父清单身份，不要修改：

`video_id`、`class_name`、`video_sha256`、`original_role`、`candidate_source_id`。

填写以下字段：

| 字段 | 填写要求 |
| --- | --- |
| `source_group_id` | 小写字母/数字/下划线/连字符，首字符为字母或数字，最长96字符。程序已有的`src_...`仅为建议编号；确认不同候选同源时，应合并为同一ID。`unknown/unresolved/tbd/todo/none/null/na/n_a/pending/unassigned`为保留占位符，不可使用 |
| `verification_status` | `confirmed`表示已获得可靠来源证据；`conservative`表示已人工核验后采用保守合组；未解决保留`unresolved` |
| `evidence` | 真实证据描述或可回查引用；保守分组写明依据及不确定性，不能仅复制“已确认” |
| `reviewer` | 实际审核者；AI辅助审核必须明确写明AI及用户委托，不填作人工审核人 |
| `reviewed_at` | 真实ISO审核时间且包含时区，例如`2026-11-16T23:00:00+08:00` |

注意：

1. 不要为了通过程序批量把`unresolved`改成`confirmed`。审核状态描述的是你实际掌握的证据。
2. 任何未解决条目都会阻塞冻结。没有候选前缀的条目，必须补充可靠人工来源证据并标为`confirmed`，不能把它自动当成独立单例。
3. 本版采用保守策略：同一候选前缀不可拆到不同来源ID。若有证据必须拆分，需要先修订分组协议，而非绕过校验。
4. 来源ID跨类别使用：同一影片的run/walk不能按动作类别分开。
5. `conservative`可以进入明确标注的探索性分组实验，但不能据此宣称真实来源、人物或摄像头完全独立。
6. 保留所有行，以UTF-8或带BOM的UTF-8 CSV保存；程序同时接受两者。

来源提取规则v2补充官方训练列表中实际存在的`_l_`后缀，因此likebeckam与Crash的两条记录可以保留原文件名并归入相应候选前缀；不需要将它们误标为已确认原始URL。旧候选表保留，新审核表独立生成，变更在审计中登记。

## 5. 第二步：冻结分组清单和支持集合

完成真实审核并保存`$E0Review`指向的CSV之后执行：

```powershell
$E0Manifest2 = & $E0Python $E0Round2 freeze-split --parent-manifest $E0ParentManifest --source-review $E0Review
if ($LASTEXITCODE -ne 0) { throw '来源审核或分组冻结失败，停止。' }
$E0Manifest2 = "$E0Manifest2".Trim()

$E0Supports2 = & $E0Python $E0Round2 sample-supports --manifest $E0Manifest2
if ($LASTEXITCODE -ne 0) { throw '支持集合冻结失败，停止。' }
$E0Supports2 = "$E0Supports2".Trim()
```

清单输出在`data/manifests/round2/round2-grouped-manifest-.../`：`bundle.json`、`video_manifest.csv`、`source_review_snapshot.csv`及`source_group_audit.json`。清单内保存父E0快照、审核表、配置和身份哈希。

支持集合输出在`data/support_sets/round2/round2-supports-.../`：`support_sets.json/csv`及`source_coverage.csv`。同一K/种子的名单由V0/V1共享，视频shot数量与独立来源数量分别记录。

**在拟合前查看`source_group_audit.json`**：各类开发/支持视频数、来源组数，以及两侧来源ID交集应为空。程序允许开发数不是10，但不允许拆组凑数。若固定搜索未找到可行方案，会明确报错；这不等于数学上证明不可行。应检查来源映射、组大小与约束，在看模型结果前登记协议变更。不要尝试多个划分后挑模型分数最高者。

## 6. 第三步：核验并复用帧特征

```powershell
$E0FeatureIndex2 = & $E0Python $E0Round2 prepare-features --manifest $E0Manifest2 --parent-feature-index $E0ParentFeatureIndex
if ($LASTEXITCODE -ne 0) { throw '特征身份核验或导入失败，停止。' }
$E0FeatureIndex2 = "$E0FeatureIndex2".Trim()
```

该阶段需要读取350段训练视频的字节来重新核验大小和SHA，但不解码、不初始化CLIP、不使用GPU、不拟合分类器。处理内容包括：

- 父协议、原视频SHA、16帧索引、解码/采样/汇总条件、原特征实现哈希；
- 父缓存NPZ哈希、元数据、维度、有限值、归一化、V0可重算一致性；
- 父索引完整覆盖eligible官方训练、无错误，且冻结标记为布尔True；官方编码器前后参数哈希须与编码器身份中的初始哈希相同；
- 正式CLI要求官方OpenAI CLIP ViT-B/32检查点身份；
- 在独立目录保存同一Z及其V0/V1，保留父特征索引快照和派生身份。

新缓存位于`artifacts/round2_features/<派生身份>/`；每视频NPZ包含`Z`、`V0`、`V1`。旧的`artifacts/features/`不改动。完整成功的派生缓存可以核验后复用；同身份导入中断时可重试，程序保留`previous-attempt-...json`并核验已完成条目后继续。损坏或身份不匹配的条目不会静默替换；修复输入后也可指定新的独立`--output`目录。不要删除原始数据或篡改缓存元数据。

若使用自定义`--output`，后续运行需要把同一目录作为`--feature-root`传入。示例采用默认路径，无须另设参数。

## 7. 第四步：正式运行第二轮实验

**下面这条`run`命令会拟合真实分类器。2026-11-17已完成本机首轮V0/V1配对运行；再次执行会生成新的运行目录。**

```powershell
$E0Batch2 = & $E0Python $E0Round2 run --manifest $E0Manifest2 --support-sets $E0Supports2 --feature-index $E0FeatureIndex2
if ($LASTEXITCODE -ne 0) { throw '第二轮实验失败，保留失败目录并检查日志。' }
$E0Batch2 = "$E0Batch2".Trim()
```

该命令默认运行V0和V1各30组，不单独挑选种子。它只拟合轻量岭分类器，骨干不更新。结果保存到新的`runs/e0-round2-development-.../`，不会覆盖第一轮批次。

批次保存配置、环境、代码身份、分组清单、特征索引及支持集合快照。每个子运行保存`support_set.csv`、`predictions.csv`、`metrics.json`、`ridge.npz`、`provenance.json`和日志；逐视频预测增加表示名称与来源ID。失败留下非成功状态，后续汇总会拒绝缺项或失败批次。

建议把本次PowerShell变量中的路径保存到自己的实验笔记，跨会话恢复时重新赋值，不从文件名猜测缓存身份。

## 8. 第五步：生成配对报告

```powershell
$E0Report2 = & $E0Python $E0Round2 summarize --batch $E0Batch2
if ($LASTEXITCODE -ne 0) { throw '配对结果核验/汇总失败。' }
Write-Output $E0Report2
```

新报告在`reports/e0-round2-report-.../`：

- `round2-results.md`、`summary.json/csv`；
- `paired_deltas.csv`：每个K、每个种子的V1−V0；
- `paired_summary.csv`：差值均值、样本标准差、中位数及正/平/负种子数；
- `support_source_coverage.csv`；
- V0/V1各自的学习曲线与混淆矩阵。

汇总会验证全部60组、共享清单/特征/支持身份、预测哈希，并从逐视频得分重新计算指标。原来的`reports/summary.json`和第一轮图表保持不变。

判断时优先看同K配对Macro-F1差值、类别召回及种子方向。种子评价相同开发视频，标准差不是泛化置信区间。V1增加了维度及时间位置统计，提升不能单独证明顺序推理；若需机制诊断，可另立帧序打乱对照，本版没有把该可选对照混入主实验。

## 9. 修改范围与兼容性

新增模块：`source_groups.py`负责候选与人工审核；`round2_protocol.py`负责配置、父身份、分组清单和支持抽样；`temporal_features.py`负责V1；`round2_features.py`负责核验导入及独立缓存；`round2_runner.py`和`round2_reporting.py`负责配对运行/复算；`round2_cli.py`和`round2_io.py`负责入口及输出边界。

新增配置`configs/e0.round2.example.yaml`及入口`scripts/07_round2.py`。原V0的`clip_features.py`、`feature_cache.py`、`pipeline.py`、配置校验及01—06脚本均保留原实现，避免旧特征代码身份失效。

README、主实验规范和数据契约增加第二轮入口；新增自动测试使用临时合成记录、特征及小视频，没有真实实验成绩。修改前备份在本机`.round2-backup/`，不进入Git。

仅验证程序的命令：

```powershell
& $E0Python -m pytest "$E0Root\tests" -q
& $E0Python $E0Round2 --help
& $E0Python $E0Round2 run --help
```

这些检查通过表示程序行为受到测试覆盖，不代表第二轮实际准确率、来源审核完成或最终测试通过。
