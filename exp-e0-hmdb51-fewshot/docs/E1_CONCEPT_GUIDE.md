# E1实验说明：显式概念基线与匹配随机投影

日期：2026-11-18。正式E1开发实验已完成，公开结果及边界见[结果综述](RESULTS_OVERVIEW.md)。本页保留复现步骤，[E1实现验收](E1_VERIFICATION.md)记录的是实现阶段事实。[E0收束记录](E0_CLOSEOUT.md)保留前序背景。脚本09独立于01—08，不改写旧Python模块或缓存身份。

## 1. 本轮回答的问题

E1检验：在相同冻结CLIP、视频、来源划分及少量支持标签下，显式可读概念是否提供超出普通视觉统计和同维随机瓶颈的价值；概念分段时间统计是否比概念无序汇总更有用。

这是固定五类的少样本分类开发实验，不是组合零样本评测。当前不实现C2关系模块，不接溺水录像、检测框或跟踪器。

## 2. 概念库与资源边界

首版配置为[16个姿态概念](../configs/e1.concepts.v1.yaml)，包括双手接近／分开、手臂抬起／侧伸、双腿分开／接近、脚部接触地面状态、躯干姿态等。每条含稳定ID、中文名、单条英文描述和可见性边界。

概念由助手基于已知类名与一般姿态知识拟定，E0汇总结果已作为开发背景；未按E1成绩筛选提示，未使用最终测试，不冒充人工概念真值。默认描述不直接使用clap/jump/run/walk/wave作为类别提示。不能只给五个类别各写一句同义描述就声称完成概念组合。

原画面多人、遮挡及中心裁剪会影响概念可见性；尤其脚部概念不能在脚不可见时强判不存在。单帧姿态也不是完整行为的充分条件。保持首版提示不变完成整套对照；修改文本须登记版本、原因与人工成本，并重新生成独立缓存，不按最终测试或单个支持种子挑最佳文本。

通用预训练及概念编制成本单独披露。训练标签仍是当次5K个视频标签；开发标签、来源审核及候选池建库标注不计作免费的零标注资源。

## 3. 表示与实验矩阵

设已归一化帧向量Z为16×512，文本向量E为16×512，每条文本向量独立L2单位化。概念序列`C=Z E^T`，为16×16的余弦相似度。

**不对每帧概念向量再做L2归一化，不做softmax、概率化、阈值化或全数据集标准化。** 均值与总体标准差用float64统计，每个汇总块除以`L2范数+1e-6`，拼接后整体L2，最后保存float32。

| 条件 | 汇总 | 默认维度 | 新拟合数 |
| --- | --- | ---: | ---: |
| V0 | 原视觉均值／标准差 | 1024 | 0，引用30组 |
| V1 | 原视觉均值／标准差＋6/5/5分段均值 | 2560 | 0，引用30组 |
| C0 | 概念均值／总体标准差 | 32 | 30 |
| C1 | C0＋概念早／中／晚均值 | 80 | 30 |
| R0 | 随机方向相似度，采用C0相同汇总 | 32 | 5×30＝150 |
| R1 | 随机方向相似度，采用C1相同汇总 | 80 | 5×30＝150 |

新增共360组，报告共引用／新增420组，对同一50段开发视频产生21000条重复评价。所有组共享原300段支持候选、K=1/3/5、支持种子0—9、嵌套名单及原λ=1岭分类器。150段最终测试和665段unused不进入本轮编码／拟合／评价。

随机方向矩阵按`PCG64(seed).normal(size=(16,512))`生成，每行L2单位化，种子固定为0—4。R与C都用单位方向的余弦、相同汇总和分类器。同一矩阵跨所有视频、K和支持集合复用，不按视频重抽，不读取标签或分数。协议1.0拒绝减少、增加或替换投影种子；这些种子独立于支持种子。

## 4. 文本编码与身份

文本分支使用既有本机官方OpenAI CLIP ViT-B/32检查点，不重新编码图像或下载新模型。检查文本／图像检查点、维度、CLIP源码与完整参数哈希一致；文本分支保持eval、requires_grad=False及无梯度推理，编码前后参数不变。记录tokenizer源码、BPE资源、77 token上下文、拒绝截断等条件。

`check`只核验既有缓存、检查点文件及本机包可用性，不加载模型、不编码文本。`prepare`才加载文本分支、编码16条描述并派生特征。默认cuda；可显式指定cpu，但仍须满足参数与联合空间核验，文本设备及版本进入缓存身份，不能混合不同条件。

## 5. 开始前只读核验

以下命令在PowerShell中逐步执行。无需更改既有本机路径配置。

```powershell
Set-Location -LiteralPath 'D:\desktop\Semantic-Driven-Compositional\exp-e0-hmdb51-fewshot'
$E1Python = '.\.venv\Scripts\python.exe'
$E1Reference = (Get-Content -LiteralPath '.\reports\round2_latest.json' -Raw -Encoding UTF8 | ConvertFrom-Json).batch
& $E1Python scripts/09_e1_concepts.py check --reference-batch $E1Reference
if ($LASTEXITCODE -ne 0) { throw 'E1输入核验失败，停止。' }
```

成功输出350段训练范围缓存、50段开发、16概念、32／80维、360组计划新增运行、60组引用，`text_encoding_executed=false`及`fits_executed=0`。不重新冻结数据或抽支持集。自定义原缓存根时，`check/prepare`均传相同`--feature-root`。

## 6. 准备概念缓存：无分类器拟合

```powershell
$E1IndexOutput = & $E1Python scripts/09_e1_concepts.py prepare --reference-batch $E1Reference --device cuda
if ($LASTEXITCODE -ne 0) { throw 'E1文本或缓存准备失败，停止。' }
$E1Index = ($E1IndexOutput | Select-Object -Last 1).Trim()
Write-Output $E1Index
```

在`artifacts/e1_concepts/e1-concept-features-实际ID/`生成新目录：

- `index-development.json`：阶段／错误、完整条件配置、概念内容及源文件哈希、父批次证据摘要、图像／文本身份、双侧冻结记录、逐视频Z数组哈希、代码和NumPy版本。
- `features.npz`：Z(350×16×512)、E(16×512)、P(5×16×512)、C(350×16×16)，以及C0/C1/R0/R1。Z从已验证旧缓存复制，用于后续独立复核，不重读视频。
- `concept_catalog.json/yaml`：实际编码提示的完整快照。
- `concept_diagnostics.csv`：原始余弦的范围／均值／标准差，无类别筛选、阈值或人工概念正确率。
- `concept_audit_template.csv`：预设种子选30段开发视频×16概念＝480条空白可见性记录，不包含模型分数或任务真值字段。

加载时核对Z哈希与单位范数，从Z和E重算C、C0/C1，并从Z和每个P重算R0/R1；仅修正NPZ哈希不能让错配向量通过。每次准备创建新目录，失败状态保留，不覆盖旧缓存。修改代码、概念或设备后重新准备。

## 7. 概念画面审核

复制`concept_audit_template.csv`到单独的审核文件填写，不覆盖冻结模板。`judgment`建议使用`present/absent/uncertain`，并填写证据、审核者和带时区时间。查看实际视频及16帧输入，保留“不确定”，不要把高余弦值自动填写为存在。

表中video_id的目录及文件名可能透露类别，所以这是可见性记录模板，**不能称严格盲审**。需要严格盲审时应另行提供去标识呈现材料。本轮没有自动制作或宣称完成人工审核；报告默认明确标记`human_concept_validation=not_performed`。

审核可用于说明概念质量；若依据这些开发视频改提示，记录额外开发与人工成本，并将后续结果继续限定为探索性开发。审核文件不会自动进入模型拟合，也不会被当成新的概念监督。

## 8. 显式启动360组拟合

**本节将执行真实E1实验；只修改代码时不执行。**

```powershell
$E1BatchOutput = & $E1Python scripts/09_e1_concepts.py run --reference-batch $E1Reference --feature-index $E1Index
if ($LASTEXITCODE -ne 0) { throw 'E1拟合失败，停止，不汇总失败批次。' }
$E1Batch = ($E1BatchOutput | Select-Object -Last 1).Trim()
Write-Output $E1Batch
```

新批次位于`runs/e1-concept-development-实际ID/`。保存360组运行、原60组复算证据于`reference/`、概念库／审核模板／诊断、完整小型特征库副本、当前代码和环境。每组保存支持ID、来源组、逐视频得分、岭分类器、指标、配置、日志及文件哈希。不更新原`round2_latest.json`。

## 9. 汇总与主比较

```powershell
& $E1Python scripts/09_e1_concepts.py summarize --batch $E1Batch
if ($LASTEXITCODE -ne 0) { throw 'E1完整性或逐预测复算失败。' }
```

报告写入`reports/e1-concept-report-实际ID/e1-results.md`，summary.json记录实际批次路径。报告器检查完整运行网格、支持／来源／开发身份、概念快照、环境、代码、特征及模型文件，并用保存特征和分类器重新前向计算所有新增分数，再逐预测复算指标。独立汇总不重新拟合，也不需要外部视频或原Z文件。

| 文件 | 作用 |
| --- | --- |
| summary.csv/json | 六表示×三K的主汇总 |
| per_run_metrics.csv | 420组，保留每个投影／支持种子及是否引用 |
| per_support_metrics.csv | 180个主聚合单位；R在同支持集合内先平均5次指标 |
| paired_deltas.csv / paired_summary.csv | 五项比较，共150项逐支持配对与15项汇总 |
| summary_by_projection_seed.csv | R0/R1×5投影×3K＝30项原始投影汇总，不挑最佳 |
| paired_by_projection_seed.csv | C0−R0与C1−R1的300项逐投影配对 |
| time_increment_gaps.csv / time_increment_summary.csv | (C1−C0)−(R1−R0)，30项逐支持差值与3项汇总 |
| all_predictions.csv / confusion_counts.json | 21000条重复评价得分及原始混淆计数 |
| concept_catalog.json / concept_class_means.csv | 实际描述及80个开发类别—概念均值；不是概念真值 |
| learning_curves.png / paired_differences.png / concept_class_means.png | 分数、配对和概念关联图；标明真实或合成验证身份 |

关键比较：C0−V0（整体无序瓶颈）、C1−V1（整体时间概念）、C1−C0（概念时间增量）、C0−R0与C1−R1（匹配非语义对照）。视觉／概念维度不同，因此只看前两项无法把提升归因于语义。C1相对C0同时改变分段内容、维度和块权重，不能仅凭提升证明顺序机制；时间增量差提供进一步诊断，不等于严格因果识别。

随机控制先对每个支持集合的5次分类指标取平均，再跨10个支持集合计算均值及ddof=1样本SD；不是将50次当成50个独立数据重复，也不是平均预测分数后分类的集成。

## 10. 结果解释与下一阶段

- 概念组与随机组相近：不能据此声称显式语义有额外收益，保留一般降维／容量解释。
- 概念组稳定优于匹配随机组，且画面审核支持概念适用：形成继续研究的探索性信号，仍需独立评价。
- C1优于C0：还需查看随机时间增量；不直接声称关系推理或组合泛化。
- 分类好但概念审核差：不能据权重或热图称解释可信。
- 当前所有方法偏低：优先复查输入可见性、采样及概念适用，不反复更换开发集或追调提示。

全部比较冻结后再另行登记最终测试；本轮入口不开放最终测试。C2需在上述结果基础上另立关系与等容量控制，当前不提前叠加复杂模块。

E1结果之后的诊断实现见[E1.1指南](E11_GUIDE.md)：新增共同正交几何对照与固定融合，并导出实际裁剪画面的人工审核材料。该入口不改变本指南的E1定义或历史结果。

## 11. 验证

```powershell
& $E1Python -m pytest -q
& $E1Python scripts/09_e1_concepts.py --help
```

合成流程测试只写pytest临时目录，不作为真实E1成绩。真实验证包含只读check与原生文本分支冒烟：16条描述编码、联合空间参数核对、同进程重复一致、超长描述拒绝截断，没有任务分类器拟合。详细记录见[E1实现验收](E1_VERIFICATION.md)。
