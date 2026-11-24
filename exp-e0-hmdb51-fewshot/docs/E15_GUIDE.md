# E1.5：以类别文本为锚的实例缓存少样本校正

状态：实现及验证完成，全仓库185项测试通过；正式E1.5尚未准备或运行。本轮保留E1.4代码、配置和结果，复用其已验证的原始视频特征、类别文本与支持名单。最终测试继续封存。验收证据见[E1.5验证记录](E15_VERIFICATION.md)。

## 1. 研究问题与固定资源

E1.4的弱视觉原型校正A2得到57.49% Macro-F1，仍低于固定文本T0的59.44%。本轮不继续搜索原型融合权重，而检验：在相同支持视频预算下，保留每个支持实例，是否比先压缩成每类一个均值原型更有效地补充文本参照？

这是假设检验，不预先认定均值压缩就是E1.4失败的唯一原因。方法借鉴[Tip-Adapter](https://arxiv.org/abs/2207.09519)的免梯度键值缓存思路，但使用本项目的视频平均特征和固定评分协议，不宣称完整复现原论文或已经验证时间推理。

共同资源：

- 固定类别顺序clap/jump/run/walk/wave，ID为0—4。
- 原冻结CLIP ViT-B/32的512维视频表示q=unit(mean16Z)，原五条类别文本T。
- 每类3段、每组15段支持视频，support_seed为0—9；沿用原支持ID及来源阻断。
- 同一50段development；查询真值只用于核验与计算指标，不构建缓存、选择参数或生成标签。
- 不解码视频、不编码图像或文本、不训练编码器或缓存键、不增加标注。

## 2. 评分与匹配对照

对单位查询q和第i个缓存键k_i，按以下固定规则评分：

```text
cosine_i = q dot k_i
affinity_i = exp(-5.5 * (1 - cosine_i))
vote_i = affinity_i / 3
cache_score_c = sum(vote_i for cache_label_i == c)
text_score_c = q dot T_c
final_score_c = text_score_c + 0.1 * cache_score_c
```

每类始终3个缓存槽，除以3表示按类平均票数。它不是对全部键做softmax，也不将最终分数解释成概率。不做最终归一化、温度校准、伪标签更新、top-k截断或查询适配。

alpha=0.1、beta=5.5、按类平均是实现前固定的数值约定；beta参考既有缓存方法基线，alpha表示固定弱校正。它们不是当前开发集选择的最优参数，也不表示缓存分支与E1.4视觉分支已经完成尺度匹配。正式配置不扫描其他值、不新增支持集内拟合。

| 条件 | 缓存键 | 缓存标签 | 新评估次数 | 作用 |
|---|---|---|---:|---|
| IC1（主条件） | 原15段支持视频各自的q | 原正确标签 | 10 | 实例缓存是否补充文本 |
| IC2 | 将每个键替换为所属类的单位均值原型，仍保留15个槽 | 原正确标签 | 10 | 比较保留实例与单位均值压缩方案 |
| IC3 | 与IC1完全相同的15个键 | (原标签+1) mod 5 | 10 | 检验正确支持标签的作用 |
| T0（引用） | 无缓存 | 无支持 | 0 | 原文本参照，仅引用一次 |
| A2（引用） | 原E1.4弱视觉原型 | 原正确标签 | 0 | 与上一轮方案配对比较，引用10次 |

IC2与IC1共享槽数、标签、核函数、alpha、beta和分母，不能额外改变评分尺度。IC3的正确标签仍保存在原支持表和模型true_labels中；错配只改变cache_labels及其one-hot值，不改支持身份、键向量或原真值。

IC2的单位均值压缩也会改变键方向与查询相似度分布。因此IC1−IC2能比较这两种支持表示，不能排他地把差值归因于多模态信息损失，也不能声称控制了所有投影统计量。

新增30次支持评估，引用T0一次及A2十次，共41条评估记录、2050条重复视频预测。它们对应同一50段开发视频，不是2050段独立视频，也不是41个独立测试集。

## 3. 输入身份、快照与单次产物

输入是完整成功的E1.4运行目录，不能只提供e14-results.md。程序验证E1.4完整条件和父证据，再沿原父链获取q/T、支持表、来源清单和T0/A2预测。没有调用模型编码器的需要。

check为只读预检；prepare在新目录复制父证据并绑定固定配置、原协议、代码和NumPy身份，不执行30次正式评估。run使用准备好的快照，新建独立批次，不覆盖E1.4。代码改变后须重新prepare；不得手改SHA绕过身份检查。

每个IC运行保存：

- `cache_model.npz`：text_prototypes、原visual_prototypes、原support_keys、实际cache_keys、true_labels、cache_labels及one-hot cache_values。模型和支持表共同确定每一列缓存对应哪个原视频。
- `cache_evidence.npz`：`query_ids`、`support_ids`、`cosine_similarities`、`affinities`、`votes`。三个数值矩阵均为50×15，50行对应原evaluation_ids及query_ids顺序，15列对应`support_set.csv`及support_ids顺序。IC2的键虽已压缩，每个槽仍保留原支持ID；该ID不表示它使用了原实例键。
- `predictions.csv`：逐视频真实/预测类别、来源和支持种子，记录alpha/beta，以及五类`text_score_*`、`cache_score_*`、最终`score_*`。统一表头另保留`lambda`、`auxiliary_score_*`与`rotation_seed`，IC新条件这些不适用字段为空；引用A2保留其原lambda=0.1及原视觉auxiliary_score，T0不伪造支持种子、alpha/beta或缓存分数。
- `support_set.csv`、`resolved_config.json`、`metrics.json`、`provenance.json`与运行日志。

汇总前从父q/T及原支持身份重新构建模型、每个支持票数、各分支分数、预测和指标；同时检查文本分数与唯一T0参照一致。只修改产物并重写哈希，不能绕过数值重建。失败和缺失条件不能静默丢弃。

## 4. 本机PowerShell启动步骤

以下命令由用户准备正式运行时依次执行；本次代码修改不会自动执行prepare或run。使用现有`.venv`和`configs/paths.local.yaml`，不安装新依赖。示例从已完成E1.4报告读取真实父批次路径：

```powershell
Set-Location 'D:\desktop\Semantic-Driven-Compositional\exp-e0-hmdb51-fewshot'
$E15Python = '.\.venv\Scripts\python.exe'
$E14Summary = Get-Content -Raw -Encoding UTF8 'reports/e14-text-anchor-report-20261119T170721310859Z-ac5613e7/summary.json' | ConvertFrom-Json
$E14Batch = $E14Summary.batch
& $E15Python scripts/14_e15_instance_cache.py check --e14-batch $E14Batch
if ($LASTEXITCODE -ne 0) { throw 'E1.5 read-only preflight failed' }
```

### 4.1 准备独立输入快照

```powershell
$E15Index = & $E15Python scripts/14_e15_instance_cache.py prepare --e14-batch $E14Batch
if ($LASTEXITCODE -ne 0) { throw 'E1.5 input preparation failed' }
$E15Index
```

默认配置为`configs/e15.instance-cache.yaml`；输出在`artifacts/e15_instance_cache/`下的新输入目录，命令打印`index-development.json`路径。父证据树包含多轮快照，复制会产生较多小文件并占用磁盘；这些本地产物被Git忽略。

### 4.2 显式运行30次新增评估

```powershell
$E15Batch = & $E15Python scripts/14_e15_instance_cache.py run --e14-batch $E14Batch --feature-index $E15Index
if ($LASTEXITCODE -ne 0) { throw 'E1.5 evaluation failed; inspect failure.log' }
$E15Batch
```

输出新的`runs/e15-instance-cache-development-*`。仅支持development，没有最终测试入口。失败修复后应新建批次，保留原失败记录。

### 4.3 独立核验并生成报告

```powershell
$E15Report = & $E15Python scripts/14_e15_instance_cache.py summarize --batch $E15Batch
if ($LASTEXITCODE -ne 0) { throw 'E1.5 report verification failed' }
$E15Report
```

输出`reports/e15-instance-cache-report-*/e15-results.md`及CSV/JSON分析文件。正式CLI的check/prepare/run/summarize均拒绝合成或未验证特征；自动测试中的合成报告必须标为“仅程序验证”，不能作为实验成绩。

## 5. 主要比较、统计与解释

| 主比较 | 回答的问题 |
|---|---|
| IC1−T0 | 少量支持实例是否真正改善强文本参照 |
| IC1−IC2 | 收益是否依赖保留实例差异 |
| IC1−IC3 | 收益是否依赖正确支持标签 |

次要比较：IC1−A2、IC2−A2、IC2−T0、IC3−T0。不能只报告相对A2改善而忽略T0；A2与新方法使用原相同支持ID及SHA配对。

报告含5行条件汇总、41行单次指标、40行支持级指标、70行支持级配对差值、7行配对汇总、40行相对T0的错误转换和4行转换汇总。指标包括Macro-F1、Accuracy、各类召回、跨支持集合均值/样本SD、配对差值和正/平/负数量，转换包含changed/fixed/harmed/both_wrong/net_correct。T0只有一个固定参照，无支持集合SD；与10组支持条件配对不使它变成10次独立零样本运行。本轮没有随机旋转或预测集成。

探索性继续门槛：IC1−T0未四舍五入Macro-F1均值至少0.02、至少8/10组支持改善，且IC1平均高于IC2和IC3。该门槛用于研究决策，不是统计显著性检验。参考T0/A2没有新计算的缓存证据，合并预测中新缓存字段留空；不把A2原auxiliary_score伪装成cache_score。

- IC1超过IC2但不超过T0：实例保留有效果，整体少样本增量仍未成立。
- IC1超过T0却接近IC3：正确支持信息的作用仍需解释。
- IC1超过T0和两个匹配对照：形成实例缓存的开发信号，仍需后续独立评价。
- IC1仍低于T0：按事实报告，可收束当前静态视觉支持适配路线；不能据此否定所有少样本或语义方法。

当前50段开发视频已多轮参与诊断；支持集合SD不是独立视频或来源置信区间。本轮未验证概念画面对应性、时间建模、未见类别/组合、真实溺水误报或部署性能，也不自动开启封存测试。概念人工审核仍按[E1.2指南](E12_GUIDE.md)单独处理。
