# E1.2：概念可见性与支持集尺度匹配

本机E1.2 development已完成，本文保留复现步骤。后续类别语义参照见[E1.3指南](E13_GUIDE.md)。主分析固定3-shot，不扩展到1/5-shot，不改变原16个概念、CLIP、来源分组或支持／开发ID。最终测试封存。实现阶段验收记录见[E12_VERIFICATION.md](E12_VERIFICATION.md)。

## 研究问题

E1.1中C1优于G1，但VC1没有稳定超过V1；C1的视频表示高度趋同，而G1具有更大的样本间变化。本轮检验：控制支持集内特征尺度及正则化作用后，显式概念是否改善视觉分类？原180行画面审核另行检验概念可见性，不能由分类分数替代。

## 固定变量与数值规则

每组使用原15段支持视频（五类×3-shot）与原50段development，support_seed=0..9。G1沿用E1.1共同正交方向，R1沿用同一E1祖先的独立随机方向，各5种子0..4。控制类型与control_seed共同确定方向，不把G1/R1的同号种子当作相同向量。

对每个支持集合、每个块X分别计算：

`mean_X = mean(X_support, axis=0)`

`scale_X = sqrt(mean(sum((X_support - mean_X)^2, axis=1)))`

`SX = (X - mean_X) / scale_X`

使用一个块级标量，不做逐维标准化／白化。只用当前15段支持，不用其余候选视频、开发特征、开发标签或人工审核标签估计。尺度<=1e-6时记录失败并终止批次，不补零、不漏掉条件。

融合为`concat(SV, SB)/sqrt(2)`；不再做逐视频L2，因此融合后支持样本平均中心化平方范数为1，两块各贡献0.5。开发样本应用支持参数，其范数和平均变化量不强行匹配。数值计算及保存的均值／尺度为float64。

原岭分类器已经内部中心化；单纯减均值不是新收益来源。本轮改变的是块级缩放与融合几何。固定lambda=1，唯一另列的原V1_lambda2对照使用lambda=2，不进行搜索。半幅视觉＋零辅助块的lambda1预测，应与原视觉lambda2逐得分一致；自动测试验证该等价性。

## 条件与数量

| 名称 | 表示 | lambda | 控制种子 | 新拟合次数 |
|---|---|---:|---|---:|
| V1_lambda2 | 原V1，不缩放 | 2 | 无 | 10 |
| SV1 | 支持尺度匹配V1 | 1 | 无 | 10 |
| SC1 | 支持尺度匹配C1 | 1 | 无 | 10 |
| SG1 | 支持尺度匹配G1 | 1 | 0..4 | 50 |
| SR1 | 支持尺度匹配R1 | 1 | 0..4 | 50 |
| SVC1 | SV1＋SC1 | 1 | 无 | 10 |
| SVG1 | SV1＋SG1 | 1 | 0..4 | 50 |
| SVR1 | SV1＋SR1 | 1 | 0..4 | 50 |

合计240次新拟合。另引用3-shot原始V1/C1/G1/R1/VC1/VG1共180次，报告420次、21,000条重复开发预测（真实开发视频仍只有50段）。不重拟合原条件。

主比较：SVC1−SV1、SVC1−SVG1、SVC1−SVR1。次要诊断：SC1−SG1、SC1−SR1、SV1−V1、SC1−C1、SG1−G1、SR1−R1、SVC1−VC1、VC1−V1_lambda2、V1_lambda2−V1。

探索性继续门槛预先固定：SVC1−SV1的Macro-F1均值>=2个百分点、至少8/10支持集合为正，且SVC1均值优于SVG1和SVR1。它不是显著性检验，也不自动确认语义机制。尺度匹配不保证完整协方差或有效维度匹配。

## 输入与产物边界

输入必须是完整成功的E1.1运行批次，包含原E1祖先、G1特征、原模型/预测/代码快照。单独e11-results.md不足以运行。脚本11的check会验证完整链、方向、保存模型前向与指标；正式CLI拒绝合成输入。无需重新编码图像／文本或加载检查点权重。

prepare只复制已核验输入，不拟合尺度、不拟合分类器。它不生成一个全局归一化特征库，因为不同支持集合的均值与尺度不同。缓存与批次各保存完整E1.1证据快照，会占用额外磁盘空间且包含较多小文件，默认被Git忽略。

run才为各组支持集合计算参数并拟合。每组保存resolved_config.json、support_set.csv、scalers.npz、scaling.json、ridge.npz、predictions.csv、metrics.json、provenance.json与日志。scaling.json记录精确支持ID、块维数、变换前后中心化变化量；V1_lambda2不应用归一化，其scalers.npz为空。

summarize从父原始特征和每组支持ID重算尺度，逐数组核对保存参数，并复核保存岭模型的前向得分及指标。更改参数或预测后仅更新文件SHA不能通过验证。每个随机条件先在支持集合内平均5次指标，再跨10组支持集合统计样本SD，另保留逐种子结果。

## 本机启动步骤

以下在实验根目录PowerShell执行。示例报告是本机已有产物；新机器需要换成自己的真实E1.1批次。各命令失败时停止后续步骤，不使用残缺输出。

```powershell
Set-Location 'D:\desktop\Semantic-Driven-Compositional\exp-e0-hmdb51-fewshot'
$E12Python = '.\.venv\Scripts\python.exe'
$E11Summary = Get-Content -Raw -Encoding UTF8 'reports/e11-diagnostic-report-20261118T132055143489Z-b7975d0d/summary.json' | ConvertFrom-Json
$E11Batch = $E11Summary.batch
& $E12Python scripts/11_e12_support_scale.py check --e11-batch $E11Batch
if ($LASTEXITCODE -ne 0) { throw 'E1.2 preflight failed' }
```

### 人工审核与独立审核汇总

继续编辑原`data/e11_audits/e11-frame-audit-20261118T131913153411Z-85f541f2/judgments.csv`，勿修改空白template、图片、原描述和model_scores.after_review.csv。若还未导出材料，按[E1.1指南](E11_GUIDE.md)使用脚本10的export-audit命令。

judgment填present/absent/uncertain；填写reviewer、带时区ISO reviewed_at和evidence。以实际CLIP裁剪图判断，原图作上下文。不可判定不是absent。建议先完成判断，再查看模型分数。原始文件名可能暴露类别，不宣称严格盲审。

```powershell
$E12AuditDirectory = 'data/e11_audits/e11-frame-audit-20261118T131913153411Z-85f541f2'
& $E12Python scripts/11_e12_support_scale.py audit --e11-batch $E11Batch --audit-directory $E12AuditDirectory
if ($LASTEXITCODE -ne 0) { throw 'Audit verification failed' }
```

该命令不填写判断，不覆盖审核材料。空表可生成明确标记“未完成”的进度报告；只有180行判断与审核信息全部有效时才生成分数对照。原始导出清单human_review_completed=false保持原样，完成状态由当前实际行复核，不由手改清单决定。

按六概念报告present/absent/uncertain/blank数、可判定率、不同视频数、分数中位数与描述性AUROC。只有同时存在present/absent才计算AUROC，uncertain排除且单列；180行存在同视频相关性，不当作180独立样本。它不训练分类器、不筛选概念、不改提示、不自动调阈值。

### 准备输入快照

```powershell
$E12Index = & $E12Python scripts/11_e12_support_scale.py prepare --e11-batch $E11Batch
if ($LASTEXITCODE -ne 0) { throw 'E1.2 preparation failed' }
$E12Index
```

保存到artifacts/e12_support_scale下的新目录。准备后改变任何源代码会阻止使用旧快照运行，需要显式重新prepare。

### 显式拟合240次

```powershell
$E12Batch = & $E12Python scripts/11_e12_support_scale.py run --e11-batch $E11Batch --feature-index $E12Index
if ($LASTEXITCODE -ne 0) { throw 'E1.2 fitting failed; inspect failure.log' }
$E12Batch
```

审核表在软件上不阻塞拟合，两者独立；未完成审核时不得声称概念可见性已经验证。失败批次保留，不静默续跑／跳过；修复后创建新批次。

### 验证与报告

```powershell
$E12Report = & $E12Python scripts/11_e12_support_scale.py summarize --batch $E12Batch --audit-directory $E12AuditDirectory
if ($LASTEXITCODE -ne 0) { throw 'E1.2 report validation failed' }
$E12Report
```

审核材料尚未准备好时可以省略--audit-directory，报告会明确显示未核验。若传入损坏材料则报错，不静默忽略。正式报告在reports/e12-support-scale-report新目录；不会覆盖E1.1。另保存原审核表和模板快照与摘要，便于追踪当时的人工证据。

## 研究解释

报告含14行条件汇总、420行单次指标、140行支持集合指标、120行配对差值、12行配对汇总、35行逐控制种子汇总和350行逐块尺度诊断。原始混淆矩阵保存在各run的metrics.json。

若概念与随机条件同时改善，优先考虑共同尺度因素；若SVC1同时优于SV1和两种随机融合，则形成继续研究信号；概念独立分类改善但融合无益，需要继续检查互补性；人工审核差时另行登记概念/输入修订，不在本轮追调。

当前development已多次用于诊断；后续提升仍属开发证据。扩展K=1/5需另行登记并完整运行条件，不只挑获胜条件。方法冻结后再规划独立评价。本轮不证明未见类别zero-shot、未见组合或真实溺水事件检测。
