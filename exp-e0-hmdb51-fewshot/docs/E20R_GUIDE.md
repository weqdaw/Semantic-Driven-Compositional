# E2.0-R：概念适用性与样本覆盖修复

本轮代码用于回答：换成画面中更适用的概念并扩大来源覆盖后，概念能否被判断、CLIP分数是否对应概念、保留完整视野的输入能否改善可见性。它不训练动作分类器，不以分类准确率为验收目标。旧E2.0的AI探索结果保留，新包不导入旧判断。

2026-11-24已按用户授权完成主队列的真实解码、GPU编码及1080项AI预审核，三个主视图已冻结。原生／加边／原图可判定率为23.33%／23.89%／25.83%，尚未达到推进门槛。独立人工复核与裁决未执行，完整空白人工主包另行保留；可选定向补样未执行。结果见[AI探索汇总](../reports/e20r-coverage-report-20261124T131037646147Z-75255f3d/ai-exploratory-results.md)，精确计划、人工包、AI包和报告路径见[data/e20r_plans/execution-current.json](../data/e20r_plans/execution-current.json)。下文命令用于从头复现，继续现有队列应使用指针中的原路径。

## 固定设计

| 项目 | 约定 |
|---|---|
| 数据范围 | 原E1.1父链中的support_pool；每类4视频，共20视频 |
| 来源 | 确定性PCG64种子20261124，按类轮询并优先较少使用的来源，每来源最多2视频，至少10来源；不按模型分数选择 |
| 隔离 | 排除原审核视频和已审核的开发来源组。既有来源分组是AI保守阻断组，不是人工确认的源视频认证；final_test没有来源审核，继续封存，不声称已验证与其来源独立 |
| 帧 | 每视频原16帧缓存的零基位置2、7、13，共60帧；不是重新随机选帧，不增加shot数 |
| 概念 | pose01胸前双手靠近、pose02身前双手分开、pose03一手高于肩、pose04一臂侧展、pose06双臂低于肩、pose14躯干前倾 |
| 文本 | 复用原16概念库中的精确英文提示和E向量，不重新编码文本。新操作规则只明确可见性判据，不能解释为提示优化实验 |
| 视图 | native：原生CLIP中心裁剪224；letterbox：保留宽高比、双三次缩放至长边224、RGB(123,117,104)居中加边；original：原图，只作人工参照 |
| 模型 | 同一冻结OpenAI CLIP ViT-B/32及原生归一化；两个224输入分别编码，单位图像向量与原单位E向量余弦。无梯度、无文本重编码、无阈值搜索 |
| 主审核 | 60帧×6概念×3视图=1080项 |
| 复核 | 固定6个不同视频，每视频1帧，6×6×3=108项；再审核实际分歧 |

本轮换了样本和部分概念，不能将新旧AUROC变化单独归因于概念修订。native与letterbox是本轮内部同帧对照；固定错配概念为[02,03,04,06,14,01]，仅是有限的对应关系控制。

`representative`是预定、非定向补样主队列的程序名称；按类和来源平衡抽样不等于总体概率代表性。覆盖比例描述本队列，不外推为HMDB51、其他摄像头或真实溺水场景中的概念出现率。

## 环境与入口

复用本机`.venv`；当前实测为Python 3.11.9，与项目`requires-python>=3.11`相容，无新增依赖。使用现有PyTorch、OpenAI CLIP、Pillow、OpenCV、NumPy、PyYAML。正式编码需要`configs/paths.local.yaml`中的官方权重文件；使用现有GPU环境，CPU可通过`--device cpu`指定。编码器核对原检查点、参数、实现、预处理和库版本；不匹配时停止，不自动接受新的特征条件。

以下在PowerShell逐段执行。只读检查不会创建正式产物；`prepare`会读取视频生成图片，`encode`会加载模型。编码可以在审核前完成，但分数保存在private，不应交给审核者。

```powershell
Set-Location 'D:\desktop\Semantic-Driven-Compositional\exp-e0-hmdb51-fewshot'
$E20RPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$E20RScript = 'scripts/16_e20r_coverage_repair.py'
$E20RPaths = 'configs/paths.local.yaml'
$E20RE11 = (Get-Content -Raw -Encoding UTF8 'reports/e11-diagnostic-report-20261118T132055143489Z-b7975d0d/summary.json' | ConvertFrom-Json).batch
$E20ROldAudit = 'data/e11_audits/e11-frame-audit-20261118T131913153411Z-85f541f2'

& $E20RPython $E20RScript --help
& $E20RPython $E20RScript check --paths $E20RPaths --e11-batch $E20RE11 --audit-directory $E20ROldAudit
if ($LASTEXITCODE -ne 0) { throw 'E2.0-R preflight failed' }
```

只读预检已在本机通过：20视频、20保守来源组、60帧、1080主判断、108复核判断。此处的计数是确定性选择检查，不是新实验结果，也不证明每个概念都有足够正负例。原视频SHA和解码帧数在prepare时逐个验证。

## 创建主队列及编码

```powershell
$E20RPlan = & $E20RPython $E20RScript plan --paths $E20RPaths --e11-batch $E20RE11 --audit-directory $E20ROldAudit
if ($LASTEXITCODE -ne 0) { throw 'E2.0-R plan failed' }
$E20RPlan = "$E20RPlan".Trim()

$E20RPacket = & $E20RPython $E20RScript prepare --paths $E20RPaths --plan $E20RPlan
if ($LASTEXITCODE -ne 0) { throw 'E2.0-R prepare failed' }
$E20RPacket = "$E20RPacket".Trim()
$E20RPlan
$E20RPacket

& $E20RPython $E20RScript encode --paths $E20RPaths --packet $E20RPacket --device cuda
if ($LASTEXITCODE -ne 0) { throw 'E2.0-R encoding failed' }
& $E20RPython $E20RScript status --paths $E20RPaths --packet $E20RPacket
```

plan保存于`data/e20r_plans/e20r-plan-*/plan.json`，绑定原输入树、文本向量、源码、选择种子及所选视频；prepare创建`data/e20r_audits/e20r-audit-*`，保存原图与两个输入像素、匿名映射、文本向量、源码和环境。encode增加两个输入的特征、逐帧概念余弦、模型身份及冻结前后参数哈希。

每次plan/prepare都创建新目录；编码和冻结拒绝覆盖。失败保留错误记录并返回2，不能将失败目录继续作为成功材料。不要在同一包上并发执行编码、冻结或编辑。输入、代码改变后应建新版本；不要改旧哈希绕过检查。重新打开终端时将已打印的plan和packet绝对路径赋给变量即可，不要自动选“最新目录”覆盖既有队列。

## 审核、冻结与复核

依次只开放当前视图：

```text
native360 → letterbox360 → original360
→ review-native36 → review-letterbox36 → review-original36 → adjudication实际分歧
```

打开当前文件夹的`index.html`看图，编辑同目录的`judgments.csv`。程序不会自动填写判断，也不会将补样目标当成真实标签。公开文件夹不包含类别、视频名称、来源或分数；private是分析者材料。当前视图开放顺序减少信息泄漏，但同人先后看图仍可能有记忆效应，不等于完全盲法。

仅修改以下字段，保留全部行、行序、身份列与UTF-8 CSV格式：

| 字段 | 内容 |
|---|---|
| judgment | present / absent / uncertain；看不清、身体部位缺失、主体不明确使用uncertain |
| reason_code | 可判定时clear；不确定时crop_missing / occlusion / blur / ambiguous_person / ground_unknown / other |
| reviewer | 审核者标识，复核者与所有主审核者标识不同 |
| reviewer_origin | human或ai，明确记录来源，同一标识不能跨记录改变来源 |
| reviewed_at | 带时区ISO日期，如2026-11-24T15:00:00+08:00 |
| evidence | 当前图像的可见证据或不可判定原因，不能仅抄动作类别 |
| elapsed_seconds | 有限非负秒数；AI不能冒填人类耗时 |

每一阶段完成后单独执行对应命令；下一阶段目录只有冻结前一阶段后才生成，**不要一次执行全部冻结命令**：

```powershell
# 完成native/judgments.csv后执行；其余阶段按顺序替换--view
& $E20RPython $E20RScript freeze --paths $E20RPaths --packet $E20RPacket --view native
if ($LASTEXITCODE -ne 0) { throw 'Review freeze failed' }
```

后续`--view`依次为`letterbox`、`original`、`review-native`、`review-letterbox`、`review-original`、`adjudication`。第二人独立填写复核判断。最后生成的裁决表只含双方标签不同的行，保留双方标签、理由及来源；无分歧也需要显式冻结空裁决表。可使用不同的裁决者，但只能如实填写来源。不同AI标识不能满足人工复核链。

## 汇总与判据

```powershell
& $E20RPython $E20RScript summarize --paths $E20RPaths --packet $E20RPacket
if ($LASTEXITCODE -ne 0) { throw 'E2.0-R summarize failed' }
```

报告在`reports/e20r-coverage-report-*/e20r-results.md`；任何阶段都可汇总，但主三视图未冻结或尚未编码时只有进度。已冻结裁决替换对应分析标签，双方原记录始终保留；未冻结裁决不进入指标。

输出包含：

- `concept_coverage.csv`：18组概念×视图覆盖及不确定原因。
- `concept_alignment.csv`：12组概念×模型输入的AUROC/AP、固定错配对照、正负视频覆盖与来源聚类区间。
- `paired_inputs.csv`：6个概念的共同可判定且标签一致子集比较、输入间矛盾数及新增可判定数。两个视图分别计算的总体AUROC使用的样本可能不同，不能直接当成纯输入收益。
- `input_retention.csv`：原图与两个输入的三值标签转换与正例保留。
- `matched_scores.csv`、`agreement_rows.csv`：逐帧分数与复核一致性。
- `coverage_gaps.csv`：距各5正5负、各3视频的缺口；`supplement-selection.template.csv`是空表头，不是已选择数据。
- `summary.json`、判断快照、输入清单快照及`provenance.json`。

单类AUROC/AP为NA，不确定项不伪装成负例。bootstrap沿用来源组重采样1000次，至少5来源、800次有效才给区间；描述性区间不代替独立泛化或显著性结论。

同一输入至少3概念同时满足：可判定率≥.70、≥5正且≥5负并各来自≥3视频、AUROC≥.70、相对固定错配≥.05，且主队列完成人工主审核、独立人工复核和裁决，才达到探索性继续门槛。AI标签仍可生成探索性统计，但门槛始终为false；第二人名称不同只是记录约束，不能证明真实独立身份。

## 可选：覆盖不足时定向补样

先查看主队列`coverage_gaps.csv`，再依据可见图像证据从其余support_pool候选视频填写选择CSV。字段固定为`video_id,target_concept_id,target_label,selection_reason,selector,selection_origin`。target_label为present/absent，selection_origin为human/ai；注明选择依据、不要依据模型高低分挑选。代码核验身份与来源隔离，不能自动证明选择者未查看过模型分数。

每个补样队列1—12个唯一视频，每来源最多2视频，排除开发来源、原审核视频和主队列全部来源。一个视频可列多个目标，但不允许同一视频/概念/目标的重复记录。目标只是选择元数据，所有新审核仍为空白、按三视图完整审核。

```powershell
$E20RSelection = 'data/e20r_plans/supplement-selection.csv' # 先复制空模板并手工填写
$E20RSuppPlan = & $E20RPython $E20RScript plan --paths $E20RPaths --e11-batch $E20RE11 --audit-directory $E20ROldAudit --base-plan $E20RPlan --supplement-selection $E20RSelection
if ($LASTEXITCODE -ne 0) { throw 'Supplement plan failed' }
$E20RSuppPlan = "$E20RSuppPlan".Trim()
$E20RSuppPacket = & $E20RPython $E20RScript prepare --paths $E20RPaths --plan $E20RSuppPlan
if ($LASTEXITCODE -ne 0) { throw 'Supplement prepare failed' }
$E20RSuppPacket = "$E20RSuppPacket".Trim()
```

后续对`$E20RSuppPacket`分别执行encode、审核、freeze、summarize。补样队列标为`coverage_supplement`并单独汇总，不与主队列合并估计自然出现率，也不能满足主队列推进门槛。补样视频不自动进入分类训练。当前版本每份补样计划相对其指定主队列隔离；多份补样计划之间可能重复，需选择者检查并保持各份独立报告。

## 历史E2.0兼容性

历史包绑定准备时的全部源码SHA。新增E20R模块会改变当前全局源码身份，因此根目录脚本15可能拒绝历史包；不要修改旧清单或放宽验证。需要重看旧包时，使用其保存的入口并显式指定当前路径配置与新的报告输出：

```powershell
$E20HistoricalPacket = 'data/e20_audits/e20-audit-20261124T093751917704Z-d6dd6905'
$E20HistoricalScript = Join-Path $E20HistoricalPacket 'private/code_snapshot/scripts/15_e20_concept_grounding.py'
& $E20RPython -B $E20HistoricalScript status --paths (Resolve-Path $E20RPaths).Path --packet (Resolve-Path $E20HistoricalPacket).Path
# 如需旧版汇总，使用summarize并显式添加--output (Join-Path (Get-Location) 'reports')；生成新目录。
```

必须保留`-B`，禁止在不可变代码快照内部生成`__pycache__`；历史summarize等其他命令也相同。E20R独立读取已核验的E1.1与原审核证据，不调用旧E2.0包的当前源码校验，不覆盖其图片、标签或AI探索报告。新增代码验证见[E20R_VERIFICATION](E20R_VERIFICATION.md)。
