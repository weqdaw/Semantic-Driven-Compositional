# E2.0 概念可见性与视觉落地诊断

状态：实现及验证完成；正式E2.0审核包尚未准备，人工判断尚未执行。E0—E1.5代码、配置、数据和结果保留，最终测试继续封存。验证范围与测试记录见[E20_VERIFICATION](E20_VERIFICATION.md)。

## 1. 方向与研究问题

E1阶段发现类别文本T0强于当前支持校正，E1.5实例保留也未稳定优于压缩均值。E2首轮不继续搜索融合系数，而验证中间概念是否可信：

1. 原图是否有足够证据判断姿态？
2. CLIP实际输入的中心裁剪是否移除了手、脚或地面关系？
3. 原概念分数能否区分人工present和absent，且优于固定错配通道？

这是一轮诊断，不以动作分类Macro-F1提升为验收目标。它落实2026-11-24知识库中的E2.0设计；启动前须固定协议，不根据审核结果调词、调阈值或挑帧。

## 2. 固定资源与对照

输入为完整成功E1.1批次和原审核包。沿用10视频、3个原缓存帧位置[2,7,13]、6概念[pose01,pose02,pose03,pose10,pose11,pose12]，共30帧／180帧—概念对、60张已有原图与CLIP裁剪图。当前样本来自9个AI保守来源组；不是10个已完全溯源的独立视频来源。

概念依次为双手胸前靠近、双手身前分开、一手高于肩部、双脚离地、一脚着地另一脚抬起、双脚着地。它们是可见姿态，不等同于拍手、挥手、跳跃或跑走的完整动作真值。主体或部位关系无法唯一判断时记uncertain，不拼接不同人的身体部位。

| 条件 | 内容 | 用途 |
|---|---|---|
| H-crop | 人工审核原CLIP实际输入图 | 概念分数的对应标签 |
| H-original | 人工审核同帧原图 | 可见性、裁剪证据保留诊断 |
| S-correct | 原帧原概念余弦 | 主要分数对齐条件 |
| S-wrong | 固定循环错配[02,03,10,11,12,01] | 有限的标签对应对照，不搜索最差映射 |

**模型分数来自crop，只与H-crop配对。没有新的原图模型编码，不能把H-original当成原图模型性能。** 不重新采样、解码、编码或训练；六概念全量保留，不把审核标签并入3-shot训练。

## 3. 流程、匿名材料和人工字段

固定顺序：

```text
prepare -> crop180 -> freeze crop -> original180 -> freeze original
        -> review-crop36 -> freeze review-crop
        -> review-original36 -> freeze review-original
        -> adjudication分歧行 -> freeze adjudication -> summarize
```

第一人360判断，第二人72判断，裁决只增加实际分歧行。用PCG64种子20261124从匿名排序的10视频抽6视频，再各抽1帧；身份与分数都不参与抽样。每个阶段只给审核者当前文件夹；不得将`private/`、已经开放的其他视图、模型分数或报告提前交给审核者。

匿名图像文件名不包含类别或电影名，像素字节与原图完全相同。`private/`保存原映射、模型分数、原审核包和代码／环境快照。此分目录约定是工作流遮蔽，不是操作系统访问权限，也不证明完全双盲。

只编辑当前文件夹的`judgments.csv`中下列六列，保留全部身份／描述／路径列及行顺序：

| 字段 | 要求 |
|---|---|
| judgment | present / absent / uncertain |
| reason_code | present/absent填clear；uncertain选crop_missing、occlusion、blur、ambiguous_person、ground_unknown或other |
| reviewer | 稳定的实际审核者标识；第二人与第一人不同，比较忽略大小写 |
| reviewed_at | 带时区ISO，如2026-11-24T15:00:00+08:00 |
| evidence | 当前视图的可见依据；other必须具体说明 |
| elapsed_seconds | 有限、非负秒数，记录真实审核成本 |

不存在／看不清分开：absent要求相关身体部位及关系可判读；裁剪缺失、遮挡、地面不可见不能填absent。不要从动作类名反推姿态。空白行全部人工字段保持空白，不能只预填审核者后把它当已审核。

第二人须独立填写，名称区分仅是记录约束，无法替代真实独立审核。AI辅助标签不能冒充第二位人工；如使用辅助工具，需在evidence中清楚标记来源并在研究报告另行说明，不能声称满足独立人工验证。程序不自动识别或认证实际审核者身份。

freeze前可以修改当前判断；freeze后原表和私有冻结副本均绑定SHA，后续编辑会报错。发现冻结判断有误时，保留原包并另建新版本／记录修订，不手改manifest或SHA。空白、非法日期、身份修改、重复／缺失行、同名复核者均被拒绝。只有全部当前行完成才能开放下一视图；没有分歧也须显式冻结空裁决表。

## 4. 本机启动和填写步骤

以下供用户准备正式执行时使用；本次修改不自动创建真实审核包。继续使用现有`.venv`与`configs/paths.local.yaml`，无新依赖。当前`.venv`实际Python3.11.9，满足项目pyproject的>=3.11要求；不把文档中的建议版本当成本机实际版本。

### 4.1 只读核验旧材料

```powershell
Set-Location 'D:\desktop\Semantic-Driven-Compositional\exp-e0-hmdb51-fewshot'
$E20Python = '.\.venv\Scripts\python.exe'
$E20E11Summary = Get-Content -Raw -Encoding UTF8 'reports/e11-diagnostic-report-20261118T132055143489Z-b7975d0d/summary.json' | ConvertFrom-Json
$E20E11Batch = $E20E11Summary.batch
$E20OldAudit = 'data/e11_audits/e11-frame-audit-20261118T131913153411Z-85f541f2'
& $E20Python scripts/15_e20_concept_grounding.py check --e11-batch $E20E11Batch --audit-directory $E20OldAudit
if ($LASTEXITCODE -ne 0) { throw 'E2.0 read-only preflight failed' }
```

原E1.1父链、原图／裁剪图、帧索引、原概念分数均核验，旧人工判断不会自动导入E2。该步骤不解码视频、加载模型或创建正式审核包。

### 4.2 准备独立匿名包

```powershell
$E20Packet = & $E20Python scripts/15_e20_concept_grounding.py prepare --e11-batch $E20E11Batch --audit-directory $E20OldAudit
if ($LASTEXITCODE -ne 0) { throw 'E2.0 packet preparation failed' }
$E20Packet
Invoke-Item (Join-Path $E20Packet 'crop/index.html')
```

默认新建`data/e20_audits/e20-audit-*`，配置为`configs/e20.concept-grounding.yaml`。先把`crop/`单独交给第一人，填写其CSV的180行。其他文件和图像不要移动／修改，CSV使用UTF-8无BOM保存，保留表头、列名及行序。每个判断都要填写全部人工字段。当前包依赖原E1.1与原审核目录继续可访问且身份一致；不是离线独立分发的完整父批次。

### 4.3 冻结crop，开放original

仅在第一人完成crop后执行：

```powershell
& $E20Python scripts/15_e20_concept_grounding.py freeze --packet $E20Packet --view crop
if ($LASTEXITCODE -ne 0) { throw 'Complete valid crop judgments before opening originals' }
Invoke-Item (Join-Path $E20Packet 'original/index.html')
```

第一人再填写`original/judgments.csv`180行，不回改crop。原图只能帮助第二阶段判断，不能用于重写第一阶段当时看不到的证据。

### 4.4 冻结original，开放第二人的crop复核

```powershell
& $E20Python scripts/15_e20_concept_grounding.py freeze --packet $E20Packet --view original
if ($LASTEXITCODE -ne 0) { throw 'Complete valid original judgments first' }
Invoke-Item (Join-Path $E20Packet 'review-crop/index.html')
```

将`review-crop/`单独交给第二人，其36行不要参考第一人标签、原图或分数。

### 4.5 冻结第二人的crop，开放其original复核

```powershell
& $E20Python scripts/15_e20_concept_grounding.py freeze --packet $E20Packet --view review-crop
if ($LASTEXITCODE -ne 0) { throw 'Complete independent crop review first' }
Invoke-Item (Join-Path $E20Packet 'review-original/index.html')
```

第二人填写`review-original/judgments.csv`36行，保留同一审核者标识。

### 4.6 冻结第二人的original，处理分歧

```powershell
& $E20Python scripts/15_e20_concept_grounding.py freeze --packet $E20Packet --view review-original
if ($LASTEXITCODE -ne 0) { throw 'Complete independent original review first' }
Invoke-Item (Join-Path $E20Packet 'adjudication/index.html')
```

`adjudication/judgments.csv`只列判断标签不同的记录，保留双方原标签和reason。讨论后填写最后六个人工字段；可由参与者共同裁决，仍保留原独立记录。裁决仍只依据该行的view，不用原图或模型分数推断裁剪中不可见的姿态。冻结的裁决在分歧行替换主审核标签用于最终描述性指标，原始主审核／复核快照保留，可追踪标签来源；未冻结裁决不参与统计。相同标签的reason差异不自动新增裁决行，但可在报告中讨论。没有分歧时CSV只有表头，不需造一条记录。

### 4.7 冻结裁决，生成报告

```powershell
& $E20Python scripts/15_e20_concept_grounding.py freeze --packet $E20Packet --view adjudication
if ($LASTEXITCODE -ne 0) { throw 'Resolve all disagreement rows first' }
$E20Report = & $E20Python scripts/15_e20_concept_grounding.py summarize --packet $E20Packet
if ($LASTEXITCODE -ne 0) { throw 'E2.0 report verification failed' }
$E20Report
```

查看进度：

```powershell
& $E20Python scripts/15_e20_concept_grounding.py status --packet $E20Packet
if ($LASTEXITCODE -ne 0) { throw 'E2.0 packet validation failed' }
```

summarize可在任意有效阶段显式调用。主审核两视图未冻结时，只生成进度，没有分数/AUC输出。两视图已冻结但复核未完成时，只能得到单人描述性诊断，继续门槛保持false；报告目录不要交给仍需盲审的复核者。输出始终新建`reports/e20-grounding-report-*`，不覆盖原审核包。

## 5. 指标与推进规则

每概念分别报告：两视图present/absent/uncertain/blank、可判定率及uncertain原因，原图present→裁剪present保留率及全部三值转换，crop已知标签上的正确／错配AUROC和非插值AP、正例比例、分数中位数、跨视频覆盖、来源聚类区间。

不确定行排除二分类排名但必须保留在覆盖分母。只有单一标签的AUC/AP记NA；AUROC同分计0.5，AP按唯一分数阈值聚合，不任意打破同分。不同概念原分数不混池计算一个AUC。

bootstrap使用来源组、种子20261124、1000次；正确／错配共享重采样。少于5来源或有效重采样不足800时不输出区间，报告有效次数；最多9来源的区间不当显著性或独立确认。

项目预设的单概念可用条件：crop可判定率>=70%，至少5present/5absent且各覆盖至少3视频，正确AUROC>=.70，正确−错配AUROC>=.05。差值的阈值比较仅容许1e-12绝对浮点误差，避免0.70−0.65被机器误判为低于0.05；保存的指标不据此舍入。至少3/6概念满足，且独立复核／裁决均完成，才达到探索性继续门槛。阈值是本项目资源分配约定，不是文献公认标准。

根据结果进入E2.1：原图也不可判读则修订概念适用性；裁剪缺失突出则单独研究保留全身输入；crop可见但分数差则研究概念探测／描述对齐；可靠概念具备覆盖后再检验时间汇总及动作分类互补。审核产生的选择是开发信息，后续须用隔离评价验证，不将它自动推广为组合推理或溺水检测能力。

## 6. 产物与身份限制

审核包：`input_manifest.json`、固定`preregistration.md`、`private/`源审核快照／匿名映射／原分数／概念定义／代码环境／冻结副本，分阶段审核目录和CSV/HTML。所有图像与原包像素字节相同。原输入路径、源树SHA、代码与固定配置均绑定；准备后改源码或原证据会拒绝继续，需要明确新版本而不是改哈希。

结果目录：`e20-results.md`、`summary.json`、`progress.csv`、判断快照、输入manifest快照与来源记录。完成两视图后另有`concept_coverage.csv`、`crop_retention.csv`、`concept_alignment.csv`、`matched_scores.csv`、`agreement_rows.csv`（有复核时）和`error_cases.md`。匹配表含私有原视频／来源身份，只提供给分析者。

正式CLI所有入口拒绝合成材料和最终测试。自动测试使用明确合成身份，程序报告标“仅程序验证”。SHA及审核者名称只能检查保存内容的一致性，不能证明实际审核行为或抵御重写全部外部证据的攻击。验收记录见[E20_VERIFICATION](E20_VERIFICATION.md)。
