# E2.0-R2 有效人体帧审核与目标人物ROI对照

本轮固定E20R的20视频、60帧、六概念与原文本向量。先审核目标及人体证据，再比较原生中心裁剪、全图加边、目标人物ROI加边。目标ID绑定到具体帧，跨三输入保持同一人，不声称跨帧身份追踪。所有帧保留在分母；没有可用ROI时使用空分数、NaN特征和显式掩码，不用虚假的零向量填补。

已于2026-11-24完成AI探索执行：54个可用ROI、174张编码、1080项AI主审核；完整空白人工目标包另存，独立人工审核未完成。结果见[实验汇总](../reports/e20r2-person-roi-report-20261124T152738486400Z-7d59282f/ai-exploratory-results.md)，实际包与报告路径见[执行指针](../data/e20r2_runs/execution-current.json)。下文命令从头新建队列，继续已有队列应使用指针中的路径。

本机现有.venv为Python3.11.9，项目支持>=3.11；复用现有PyTorch/CLIP/Pillow/NumPy环境，无新增依赖。原视频只读，final_test封存。脚本17只提供development实验，不训练分类器，不重新编码文本。

## 固定实验条件

|条件|输入|
|---|---|
|native|原CLIP中心裁剪224|
|letterbox|全原图保留宽高比，加边224，RGB(123,117,104)|
|roi|原图中目标可见人体框四周各扩10%框宽高，裁到原图边界后同样加边224|

ROI需要新增定位监督，同时改变人物尺度和背景；本轮只能验证人物聚焦的整体作用，不能单独证明去背景或语义推理增强。绿色目标框仅供审核参考，模型编码不含框线。

每帧原图预先指定可见人体面积最大者，并列取最左；不能根据动作类别、概念真假或模型分数挑人。多个人物仍可选择明确目标；不能明确指定时标ambiguous。框只包含可见人体，不补画外部分。不同帧可因电影切镜出现不同人物，target_id用“视频匿名ID-帧ID”，不将不同人物误称同一轨迹。

有效人体条件事前固定为：目标可确定、双肩/双手/腰部均明确可见、原图目标框高至少80像素。80像素是本轮工程约定，不是通用标准。全60帧继续报告主指标，有效人体子集只作条件分析；不能只保留好帧声称覆盖改善。视频内有效帧重采样为独立后续实验，不在本轮更换帧。

## 准备

```powershell
Set-Location 'D:\desktop\Semantic-Driven-Compositional\exp-e0-hmdb51-fewshot'
$R2Python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$R2Script = 'scripts/17_e20r2_person_roi.py'
$R2Parent = (Get-Content -Raw -Encoding UTF8 'data/e20r_plans/execution-current.json' | ConvertFrom-Json).human_packet
& $R2Python -B $R2Script check --parent $R2Parent
if ($LASTEXITCODE -ne 0) { throw 'R2 preflight failed' }
$R2Packet = & $R2Python -B $R2Script prepare --parent $R2Parent
if ($LASTEXITCODE -ne 0) { throw 'R2 prepare failed' }
$R2Packet = "$R2Packet".Trim()
$R2Packet
```

推荐从E20R空白人工主包继承同样的像素与来源，不导入旧AI概念判断。R2先核对历史代码快照全部SHA，再通过独立Python -I -B进程运行其原验证器；旧包、计划、祖先和像素均不改写，禁止为迁就新增源码修改旧SHA。R2包保存当前源码快照、父证据与原图，后续检查父文件树是否变化。

## 目标与人体适用性审核

打开targets/index.html，编辑targets/targets.csv。保留身份列和行序，填写：

- person_status：absent/single/multiple/uncertain；target_status：selected/absent/ambiguous。
- selected时填原图整数框x1,y1,x2,y2，坐标为[x1,y1,x2,y2)，满足0≤x1<x2≤宽、0≤y1<y2≤高；其他状态框坐标留空。
- shoulders_visible、hands_visible、waist_visible：yes/no/uncertain，分别表示双肩、双手、腰部的直接可见性。
- target_description：可辨别服饰/外观身份，不写动作类别作为身份；reviewer、reviewer_origin=human或ai、带时区reviewed_at、当前图像evidence、有限非负elapsed_seconds。

固定12个不同视频的各1帧标记calibration=yes。AI校准不能冒充12帧人工校准；独立人工规则核验仍保留。目标审核与概念判断分开，不根据可见性字段自动填概念标签。

```powershell
# 完整填写目标表后执行；冻结后生成三条件输入并只开放native审核
& $R2Python -B $R2Script freeze --packet $R2Packet --view targets
if ($LASTEXITCODE -ne 0) { throw 'Target freeze failed' }
& $R2Python -B $R2Script encode --packet $R2Packet --device cuda
if ($LASTEXITCODE -ne 0) { throw 'R2 encode failed' }
```

编码可在标签前执行，但private分数不得提供给审核者。三条件全部重新编码并记录同一检查点、预处理和冻结前后哈希；缺ROI行不编码。每份包拒绝覆盖已有编码，代码或原证据变化需新版本。

## 三视图概念审核与人工复核

按native360→letterbox360→roi360逐步完成。每阶段查看原始输入和绿色框参考，固定target_id；只修改judgment、reason_code、reviewer、reviewer_origin、reviewed_at、evidence、elapsed_seconds。judgment为present/absent/uncertain；确定时reason_code=clear，不确定使用crop_missing/blur/occlusion/ambiguous_person/ground_unknown/other。没有固定目标、目标完全不在当前视图或ROI缺失时，只能填写uncertain；不得以其他人物替代目标。

```powershell
# 每次完成当前视图后执行一次，不要一次运行所有冻结命令
& $R2Python -B $R2Script freeze --packet $R2Packet --view native
if ($LASTEXITCODE -ne 0) { throw 'Review freeze failed' }
```

后续view依次为letterbox、roi；冻结后开放review-native、review-letterbox、review-roi各36项，由不同于目标/主审核者的独立审核者填写；最后处理adjudication实际分歧，空裁决也要显式冻结。不同名称不能证明真实独立；AI分支保留空白独立人工复核，不伪造第二人。

## 汇总与验证

```powershell
& $R2Python -B $R2Script status --packet $R2Packet
& $R2Python -B $R2Script summarize --packet $R2Packet
if ($LASTEXITCODE -ne 0) { throw 'R2 summarize failed' }
.\.venv\Scripts\python.exe -m pytest tests/test_e20r2_protocol.py tests/test_e20r2_workflow.py -q
```

报告新建于reports/e20r2-person-roi-report-*：e20r2-results.md、summary.json、target_quality.csv、18组全帧concept_alignment/concept_coverage、18组eligible_alignment、18组三输入配对paired_inputs、逐帧matched_scores、判断快照及provenance。主审核/编码不全时只报进度。冻结裁决才替换分析标签，双方原判断保留。

全帧覆盖与有效人体子集分开；单类AUROC/AP为NA；来源组bootstrap1000次，至少5来源/800有效才给区间。输入比较用同帧、同目标、两视图均可判定且标签一致的共同子集，同时单列标签冲突和新增/丢失可判定。

同一输入至少3概念同时满足70%可判定、5正5负各3视频、AUROC≥.70、较固定错配≥.05，并完成全人工目标/主审/独立复核/裁决，才达到探索性推进门槛。AI预审核和AI框只能支持探索；没有自动部署或溺水识别成绩。

data/e20r2_audits与data/e20r2_runs保存本地框、原图、特征和执行记录，reports/e20r2-*保存报告，均Git忽略。实现和执行验证见[E20R2_VERIFICATION](E20R2_VERIFICATION.md)。
