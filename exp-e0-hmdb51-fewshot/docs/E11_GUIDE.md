# E1.1：概念几何对照、固定融合与可视审核

状态：本机 E1.1 development 已完成；本文保留复现协议与启动方式。后续尺度／可见性诊断见[E1.2指南](E12_GUIDE.md)。沿用 E1 的 development 协议，最终测试封存。本轮不扩展概念、不引入 C2 关系模块、不重新训练 CLIP。

## 1. 要解决的问题

E1 中 C1 比 C0 好，但没有整体超过 V1 或 R1。现有随机单位方向 R1 与文本概念的方向几何不同，因此不足以区分概念语义、瓶颈压缩与方向相关性的作用。E1.1 补充三个诊断：

1. **概念画面是否对应？** 固定抽帧、导出原图与实际 CLIP 裁剪图，由人判断六种姿态是否可见。空白模板不算审核完成。
2. **对齐是否优于保留文本几何的随机方向？** 比较 C1 与 G1。G1 保留文本 Gram，改变其相对视觉特征的方向。
3. **概念能否提供视觉之外的补充信息？** 比较 VC1 与 V1，再比较 VC1 与相同维数、相同融合方式的 VG1。

**主分析预先固定 K=3；K=1/5 为学习曲线和稳定性检查。** 不选最好旋转、支持种子、类别或开发子集。当前 dev 已用于 E0/E1 诊断，本轮属于适应性开发分析，不能视作独立泛化证据。

## 2. 固定条件与新增变量

| 条件 | 表示 | 本机维数 | 新拟合数 | 作用 |
|---|---|---:|---:|---|
| V1 | 原视觉早/中/晚统计 | 2560 | 0，复用30 | 视觉参考 |
| C1 | 原16概念早/中/晚统计 | 80 | 0，复用30 | 概念参考 |
| G1 | E 经共同正交变换后，照 C1 汇总 | 80 | 150 | 文本几何对照 |
| VC1 | 等能量拼接 V1 与 C1 | 2640 | 30 | 概念是否有融合增量 |
| VG1 | 等能量拼接 V1 与 G1 | 2640 | 150 | 匹配融合对照 |

新增 330 次：11 种条件实例 × 3 个 K × 10 个支持种子。复用 60 次，报告合计 390 次；50 个开发视频会产生 19,500 条重复预测，不能当作 19,500 个独立视频。

固定变量：第二轮 protocol_id、来源分组、支持名单与开发名单、完整视频为 shot、类别 ID 0..4、16帧中点采样、冻结 OpenAI CLIP ViT-B/32、原生图像预处理、16个 E1 文本概念、6/5/5 时间分段、岭分类器 λ=1、K=1/3/5、支持种子0..9。来源分组继承既有 conservative 审核证据，不升级为人工 confirmed。

新增变量：`representation`=G1/VC1/VG1、`rotation_seed`=0..4。VC1 的旋转种子在 JSON 为 null、CSV 为空，与 support_seed 分开。配置固定在 `configs/e11.diagnostics.yaml`，偏离将报错；需要新假设时另建版本，不覆盖本轮。

## 3. 数学与实现约定

设原文本单位行向量 E∈R^(m×d)，逐帧视觉单位行向量 Z∈R^(16×d)。每个旋转种子用 NumPy PCG64 生成 d×d 标准高斯矩阵，做 QR 并修正 R 对角符号，得到共同的正交 Q。Q 属于 O(d)，可能含反射，不宣称只取 SO(d)。

`D = E @ Q`；所有概念、视频、K、支持集合共用该种子的同一个 Q。`D @ D.T ≈ E @ E.T`，绝对误差容忍 1e-6。D 直接转 float32，不再逐行归一化；Q 保存 float64。逐帧 `S_G = Z @ D.T`，不做 softmax、阈值或帧分数 L2；再用 E1 C1 的块归一化和全局归一化得到 G1。

**保留 Gram 不等于保留视觉分数的协方差。** 它不保证 `D Cov(Z) D.T`、边际分布、均值或有效维度等同原概念。报告额外保存分数相关性、首主成分占比和协方差参与率；这是更接近文本几何的对照，不能声称已经纯粹隔离语义的全部因素。

融合定义：`F(A,B) = concat(A/||A||, B/||B||) / sqrt(2)`，按视频计算。每块平方范数贡献 0.5，VC1=F(V1,C1)，VG1=F(V1,G1)。不按维数分配权重、不扫 α、不调 λ；原有 V1/C1 块已经单位化，但融合仍显式复核非零与有限性。其线性核相当于两块核各占一半，加入弱块会稀释视觉块，融合下降本身不证明语义绝对无用。

## 4. 启动前准备

- 使用项目现有 `.venv` 和 `configs/paths.local.yaml`。本轮特征派生、岭拟合用 CPU/NumPy，不需要重编码图像或文本，也不会加载模型权重。
- 必须有**完整成功的 E1 批次**，含特征库、概念快照、360个运行与完整60组旧参考。单独的 e1-results.md 不足以运行。
- `check` 验证现有 E1 的模型、预测、指标与证据，不创建 E1.1 缓存或拟合。
- 只有 `export-audit` 需要访问原视频并解码。按清单 `relative_path` 使用既有只读别名，核验原文件 SHA、16帧索引、解码器和 Pillow/torchvision 版本；不改名、不复制视频、不访问最终测试内容。
- 缓存与运行各复制完整 E1 证据快照。需要额外磁盘空间和较多小文件，均被 Git 忽略。全套软件测试使用临时合成数据，不产生正式研究结果。

## 5. 本机 PowerShell 执行顺序

在实验根目录执行。下列 E1 报告名是本机现有产物；新机器须换成自己的真实路径，不会随 Git 发布。

```powershell
Set-Location 'D:\desktop\Semantic-Driven-Compositional\exp-e0-hmdb51-fewshot'
$E11Python = '.\.venv\Scripts\python.exe'
$E1Summary = Get-Content -Raw -Encoding UTF8 'reports/e1-concept-report-20261118T120112876194Z-a600c588/summary.json' | ConvertFrom-Json
$E1Batch = $E1Summary.batch
& $E11Python scripts/10_e11_diagnostics.py check --e1-batch $E1Batch
if ($LASTEXITCODE -ne 0) { throw 'E1.1 preflight failed' }
```

### 5.1 导出审核材料

```powershell
$E11Audit = & $E11Python scripts/10_e11_diagnostics.py export-audit --e1-batch $E1Batch
if ($LASTEXITCODE -ne 0) { throw 'E1.1 audit export failed' }
Start-Process (Join-Path $E11Audit 'index.html')
```

抽样固定：在排序后的 development ID 中，以 PCG64 seed=20261118 无放回抽10段；不用类别、预测或正确率筛选，不保证类均衡。每段取已缓存16帧中的第3/8/14帧（零基2/7/13）；审核 `pose01/02/03/10/11/12`，共180行，每个概念30行。

产物：

- `index.html`：30组原图/CLIP裁剪图对照；`images/`为60张无损PNG。
- `judgments.csv`：填写 `judgment`=present/absent/uncertain，及 reviewer、带时区 reviewed_at、可见证据 evidence。以 CLIP 裁剪图为判断对象，原图仅提供上下文。多人身份不清、遮挡、模糊和被裁掉等情况记 uncertain 并说明。
- `judgments.template.csv`：冻结空表，不覆盖。
- `model_scores.after_review.csv`：相同帧与概念的原始余弦，建议人工判断完成后才看；余弦不是概率。
- `frames.csv` 与 `audit_manifest.json`：精确采样位置、视频/图片/空表 SHA、父证据摘要、配置。清单中 `human_review_completed=false` 表示导出时状态，不会因编辑CSV自动升级。

文件路径仍可能暴露动作类别，因此不是严格盲审。人工审核后另存版本、记录SHA；按概念报告 present/absent/uncertain 数量和两类可判定分数分布，uncertain 单列，不能默认算 absent。帧存在同视频相关性，180行不等于180个独立样本。**本版不自动完成人工审核，也不自动根据标签调阈值或改概念。** 拟合在软件上不依赖人工表，但概念可见性的研究结论必须等待真实审核。

### 5.2 派生 E1.1 缓存（不拟合）

```powershell
$E11Index = & $E11Python scripts/10_e11_diagnostics.py prepare --e1-batch $E1Batch
if ($LASTEXITCODE -ne 0) { throw 'E1.1 cache preparation failed' }
$E11Index
```

保存 `artifacts/e11_diagnostics/e11-diagnostic-features-*/index-development.json`。只用已验证 Z/E/C1；计算 Q/D/G1/V1/VC1/VG1，复制父证据。准备后改变任何源代码会令正式拟合拒绝旧缓存，应显式重新 prepare，不能手工改哈希绕过。

### 5.3 正式执行330次 development 拟合

```powershell
$E11Batch = & $E11Python scripts/10_e11_diagnostics.py run --e1-batch $E1Batch --feature-index $E11Index
if ($LASTEXITCODE -ne 0) { throw 'E1.1 fitting failed; inspect the new batch failure.log' }
$E11Batch
```

每次运行在新目录保存支持集、逐视频分数、岭模型、指标、配置和来源身份。不覆盖 E1、不自动续跑残缺批次；失败批次保留，修复后用新目录重跑。没有 final_test 入口。

### 5.4 验证并汇总

```powershell
$E11Report = & $E11Python scripts/10_e11_diagnostics.py summarize --batch $E11Batch
if ($LASTEXITCODE -ne 0) { throw 'E1.1 report validation failed' }
$E11Report
```

汇总新建 `reports/e11-diagnostic-report-*/e11-results.md`。跨终端时手工保存命令打印的三个路径 `$E11Audit/$E11Index/$E11Batch`，不要靠修改旧缓存或挑最近目录隐式决定输入。

## 6. 输出与统计检查

报告包含 `summary.json`、15行条件×K汇总、390行单次指标、150行支持集合指标、120行配对差值、12行配对汇总、30行逐旋转汇总、6行几何/分数分布诊断及全部逐视频预测。原始逐次5×5混淆矩阵在各运行 metrics.json 中保留。

G1/VG1：先在同一 K、support_seed 下平均5个旋转的**指标**，再对10个支持集合取均值和样本SD（ddof=1）。这不是预测集成。V1/C1/VC1每个支持集合各一次。全部比较先核对支持ID摘要相同。保留逐旋转结果，不选择最好种子；SD反映支持抽样敏感性，不是独立视频置信区间。

| 配对差值 | 正值所支持的有限结论 |
|---|---|
| C1 − G1 | 原概念相对共同正交方向对照更好，但输出统计仍可能不同 |
| VC1 − V1 | 固定等权融合对当前视觉基线有补充 |
| VC1 − VG1 | 原概念融合优于匹配维数/融合方式的几何对照 |
| VG1 − V1 | 随机几何瓶颈是否也能带来融合增益 |

先看 K=3 的差值均值、SD、正/平/负支持数，再检查 K=1/5 与逐类召回。不要只看个别改善类别。若 VC1 未超过 V1，不急着添加关系模块；若原概念在图中经常不可判定，先解决可见性/文本定义/时间表达问题，下一版另登记。若有稳定增量，再考虑独立划分验证和关系组合实验。

本轮只检验已见五类动作的视频级少样本分类；不能由此推断未见类别 zero-shot、组合泛化或真实溺水事件检测能力。

## 7. 文件边界

新增 `e11_features.py`（数值）、`e11_workflow.py`（缓存/拟合）、`e11_reporting.py`（验证/汇总）、`e11_audit.py`（画面审核）、`e11_cli.py`、脚本10、固定YAML和独立测试。旧 E0/E1 Python 不修改；完整输入证据与新源码均快照保留。实现验收见 [E11_VERIFICATION.md](E11_VERIFICATION.md)。
