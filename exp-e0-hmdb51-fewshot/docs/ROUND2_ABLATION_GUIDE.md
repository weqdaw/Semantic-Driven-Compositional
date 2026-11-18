# E0第二轮补充实验：重复均值与帧序打乱

日期：2026-11-18。真实补充实验已完成，结论及后续阶段见[E0收束记录](E0_CLOSEOUT.md)。原第二轮60组结果继续保留，入口仍为`reports/round2_latest.json`。本页保留复现步骤，使用`scripts/08_round2_ablations.py`；原01—07入口行为不变。

## 1. 研究问题与条件

第二轮V1增加三块分段均值，同时改变维度和标准差的相对权重。本轮用两个控制区分这些因素；不是为保证某条件获得高分而调参。

| 条件 | 输入块 | 512维帧特征下的维度 | 运行方式 |
| --- | --- | ---: | --- |
| V0 | μ、σ | 1024 | 引用并复核原第二轮30组 |
| V1 | μ、σ、早段均值、中段均值、晚段均值 | 2560 | 引用并复核原第二轮30组 |
| V1_repeat | μ、σ、μ、μ、μ | 2560 | 新增30组 |
| V1_shuffle | 打乱16帧后按V1汇总 | 2560 | 5扰动种子×30＝150组 |

μ为全局均值、σ为总体标准差。统计用float64累积，每块除以`L2范数+1e-6`，拼接后整体L2归一化，保存float32。V1仍按6／5／5分段。重复块不增加独立信息，但改变内积及各原始统计块的相对权重；同2560维也不能理解为具有相同有效秩。

新增拟合共180组；总报告引用60组＋新增180组。每个运行只用5K段支持视频，评价相同50段开发视频。采用原分组protocol、来源审核、300段支持候选、K=1/3/5、支持种子0—9、嵌套支持ID和λ=1。150段最终测试保持封存。源视频、来源表、划分及原特征不修改。

## 2. 打乱规则与复现

扰动种子默认0—4，独立于支持集合种子0—9。每段视频的随机种子由以下对象经项目`digest`函数产生SHA-256后，将十六进制摘要转为整数获得：

```json
{"algorithm":"sha256_video_seed_pcg64_permutation_v1","video_id":"class/original_name.avi","shuffle_seed":0}
```

`digest`对JSON键排序、无额外空格、UTF-8编码。使用显式`numpy.random.Generator(numpy.random.PCG64(seed)).permutation(16)`。算法及NumPy版本进入特征身份；不使用Python进程随机化hash，也不读取任务真值、预测分数或支持角色。

完整video_id用于稳定身份，包含类目录，但算法不解析类名，类别文本不提供特征信号。相同视频／扰动种子在所有K和支持集合中保持同一排列；支持、开发均独立按同一规则生成。先对双方打乱后分别拟合和评价，不使用原序训练后只打乱开发输入。

缓存保存int16、零基排列；`permutations.csv`同时保存排列、原始采样帧索引及打乱后的索引。偶然保留某些相邻关系或分段位置是随机排列的可能结果，不按效果重抽。此操作保留帧集合和总体统计，仅扰动位置／连续分段结构，不能称为抹去所有运动信息。

## 3. 环境与只读核验

在本仓库根目录的PowerShell执行，使用既有实验`.venv`，无需下载权重或安装依赖。脚本从自身位置解析仓库路径。

```powershell
Set-Location -LiteralPath 'D:\desktop\Semantic-Driven-Compositional\exp-e0-hmdb51-fewshot'
$AblationPython = '.\.venv\Scripts\python.exe'
$AblationReference = (Get-Content -LiteralPath '.\reports\round2_latest.json' -Raw -Encoding UTF8 | ConvertFrom-Json).batch
& $AblationPython scripts/08_round2_ablations.py check --reference-batch $AblationReference
if ($LASTEXITCODE -ne 0) { throw '第二轮输入核验失败，停止。' }
```

`check`逐预测复核原60组，验证350段缓存、冻结状态、来源／支持身份及原缓存代码条件；输出`fits_executed: 0`和`planned_new_runs: 180`。它不会生成控制特征、拟合模型、解码视频或访问最终测试特征。

如原第二轮特征根不在默认`artifacts/round2_features`，在`check`和`prepare`显式传相同`--feature-root`。`--paths`可指定已有本机路径配置；不要将包含本机路径的文件公开分发。

## 4. 准备特征：不拟合

```powershell
$AblationIndexOutput = & $AblationPython scripts/08_round2_ablations.py prepare --reference-batch $AblationReference
if ($LASTEXITCODE -ne 0) { throw '补充特征准备失败，停止。' }
$AblationIndex = ($AblationIndexOutput | Select-Object -Last 1).Trim()
Write-Output $AblationIndex
```

默认在`artifacts/round2_ablations/e0-ablation-features-实际ID/`生成：

- `index-development.json`：成功／失败状态、父批次证据摘要、原协议、编码器、完整条件配置、代码哈希和NumPy版本。
- `features.npz`：V1_repeat为350×2560，V1_shuffle为5×350×2560；排列为5×350×16；不复制Z。
- `permutations.csv`：350×5＝1750行实际排列与对应帧索引。

350段均来自已合格的官方训练池。无需重新编码CLIP或读取原始视频画面。每次准备创建新目录，不覆盖成功或失败目录；失败记录保留，修正输入后重新执行。修改代码或扰动种子后必须生成新缓存，不篡改旧身份。

## 5. 显式启动拟合

**本节会执行真实180组补充开发拟合；仅准备代码时不执行。**

```powershell
$AblationBatchOutput = & $AblationPython scripts/08_round2_ablations.py run --reference-batch $AblationReference --feature-index $AblationIndex
if ($LASTEXITCODE -ne 0) { throw '补充实验失败，停止，不汇总失败批次。' }
$AblationBatch = ($AblationBatchOutput | Select-Object -Last 1).Trim()
Write-Output $AblationBatch
```

新目录为`runs/e0-round2-ablations-实际ID/`。批次保存原60组所需的小型证据副本于`reference/`，保存本轮代码快照、环境、条件索引和排列；子运行保存支持集合、逐视频预测、复算指标、岭分类器、provenance和日志。原批次目录与指针均不改动。

`run`只读取准备好的控制向量并拟合岭分类器，编码器不更新。生产`check/prepare/run`拒绝合成或未核验骨干；pytest仅在临时目录使用明确标记的合成输入验证流程。

## 6. 汇总与结果阅读

```powershell
& $AblationPython scripts/08_round2_ablations.py summarize --batch $AblationBatch
if ($LASTEXITCODE -ne 0) { throw '完整性或逐预测复算失败。' }
```

打印新报告路径`reports/e0-round2-ablation-report-实际ID/ablation-results.md`。汇总器拒绝缺项、重复运行、不同支持名单、混合协议、错误真值／argmax、保存指标不符、损坏文件及未成功批次。报告可以从批次快照复算，无需重新读取原视频或原特征NPZ。

| 文件 | 用途 |
| --- | --- |
| summary.csv / summary.json | 四表示×三个K的主汇总；打乱先在支持集合内平均 |
| per_run_metrics.csv | 全240组指标，区分引用／新增、支持种子／扰动种子 |
| per_support_metrics.csv | 120行主聚合单位：四表示×三K×十支持集合 |
| paired_deltas.csv / paired_summary.csv | 四项主要配对及跨支持集合统计 |
| summary_by_shuffle_seed.csv | 5扰动×3K逐扰动汇总，不挑最佳 |
| paired_by_shuffle_seed.csv | V1−shuffle的150项原始扰动配对 |
| permutation_predictions.csv | 全240组逐视频预测，共12000行重复评价，不是12000段视频 |
| confusion_counts.json | 各运行原始混淆计数，不用聚合混淆矩阵代替平均Macro-F1 |
| permutations.csv | 全部实际排列和原始／打乱后采样帧索引 |
| learning_curves.png / paired_differences.png | 主学习曲线和配对增量，标明真实／程序验证身份 |

打乱主指标计算顺序为：对每个K／支持种子，先分别计算5次扰动的分类指标，再平均这5个指标；最后对10个支持集合计算均值和ddof=1样本SD。不是将50次当成50个独立支持集合，也不是平均5套分类得分再argmax的预测集成。各次扰动的原始预测和结果始终保留。

## 7. 事前比较与结论边界

- **V1−V1_repeat**：观察连续早／中／晚统计是否优于重复内容及权重重分配；相同维度不能完全匹配有效容量。
- **V1−V1_shuffle**：观察原序位置／连续分段结构是否带来增量；不能单独归因于有向时间推理。
- **V1_repeat−V0**：评估重复均值及相对标准差权重变化是否改变表现。
- **V1−V0**：重现已有开发配对，作为历史一致性对照。

以上均报告Accuracy、Macro-F1及每类召回，列配对均值、中位数和正／平／负支持集合数。结果接近只说明当前条件未见稳定增量，不证明时间无用或完全等效。原开发集已参与研究诊断，所有结果属探索性开发；不能无限追分后仍称独立验证。

来源审核继续限定为用户委托的AI辅助保守分组，没有人工签字或完整来源URL恢复；来源ID互斥不自动等于人物／摄像头／原素材完全独立。本轮不验证显式概念、组合零样本或溺水检测效果。

## 8. 实现验收

```powershell
& $AblationPython -m pytest -q
& $AblationPython scripts/08_round2_ablations.py --help
```

合成测试覆盖数值、排列回放、完整180组运行和两层种子聚合、文件及语义身份篡改、最终测试阻断。真实只读`check`单独记录，不能将合成测试报告写成真实实验结果。实现验证记录见[ROUND2_ABLATION_VERIFICATION.md](ROUND2_ABLATION_VERIFICATION.md)。
