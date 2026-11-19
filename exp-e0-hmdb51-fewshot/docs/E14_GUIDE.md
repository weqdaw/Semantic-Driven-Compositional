# E1.4：以类别文本为锚的少样本分数校正

状态：实现准备，正式E1.4尚未运行。复用已完成E1.3的q/T/Q、原3-shot支持名单和development；不重新编码文本或图像、不进行梯度训练。旧代码、提示与结果保留，最终测试封存。

## 1. 研究问题与控制变量

E1.3中正确文本改善视觉原型，但P1的45.80% Macro-F1仍低于固定文本T0的59.44%。本轮检验：保留强文本参照后，少量视觉支持能否提供额外校正？

E1.3的P1分数为：

`q @ unit(p_c + T_c) = (q@p_c + q@T_c) / sqrt(2 + 2*p_c@T_c)`

分母随类别及支持集合变化。本轮保留完全相同的q（16帧单位向量均值再单位化）、原支持原型p、五条类文本T，以及相同512维联合空间。只改变融合入口：

`S_c(q) = text_score_c + lambda * auxiliary_score_c`

文本分支始终为`q@T.T`；视觉原型来自原15段支持，每类3段。不使用查询真值构造或选择任何分支。不做合成权重逐行L2、不做最终分数归一化、softmax、温度校准、额外截断或伪标签更新。

## 2. 固定网格与数量

| 条件 | 辅助原型 | lambda | 新评估次数 | 作用 |
|---|---|---:|---:|---|
| A1 | p | 1.0 | 10 | 与原P1比较，诊断原型相加后的归一化影响 |
| A2（主条件） | p | 0.1 | 10 | 弱视觉校正能否真正超过T0 |
| A3 | p[[1,2,3,4,0]] | 0.1 | 10 | 错配支持类别，保持文本锚正确 |
| A4 | p@Q | 0.1 | 50 | 仅旋转视觉分支，5个固定种子 |

support_seed固定0..9，Q沿用原E1.3共同正交矩阵，rotation_seed为0..4。A3按clap/jump/run/walk/wave顺序分别使用jump/run/walk/wave/clap的视觉原型，不根据开发分数选择错配。A4的同一Q跨五类和视频共享；不得旋转T，不得每视频重新随机。

新增80次支持评估，引用原T0一次、P0/P1各10次共21条原评估记录；报告合计101组。T0没有支持视频与支持集合SD，不复制为十次独立零样本运行。真实50段开发视频形成5050条重复预测，不是5050段独立视频。

配置固定于`configs/e14.text-anchor.yaml`。lambda=0.1是主分析，lambda=1是诊断；本轮不扫描其他系数、不改提示、不扩展K=1/5。

## 3. 输入、准备与每次产物

输入是完整成功的E1.3运行批次，不能只提供e13-results.md。check会验证原生文本身份、原型/方向、保存预测和完整父链。E1.3的文本编码与tokenizer/BPE核验结果沿证据树继承，不再调用模型编码；本轮不需要加载检查点权重。

prepare只复制经过验证的完整父证据树并绑定新配置/代码/环境身份；不生成全局新特征、不计算新原型或评估。运行时按固定支持集合重建p。缓存与批次各保留父树，会占用额外磁盘空间并产生较多小文件，均被Git忽略。

每次新评估保存：

- `score_model.npz`：text_prototypes、visual_prototypes（正确原始p）、auxiliary_prototypes（实际受控p）与score_weights=`T+lambda*auxiliary`。合成权重不单位化，其行范数可以不为1。
- `predictions.csv`：最终score、text_score、auxiliary_score三组每类分数，真实/预测类、来源、支持种子、旋转与lambda。最终分数可能超出[-1,1]，不解释为余弦概率。
- `support_set.csv`、`resolved_config.json`、`metrics.json`、`provenance.json`与日志。

汇总从父q/T/Q与原支持ID独立重建四个模型数组和三组分数，复核真值、来源和指标；再核对每条text_score与原T0保存分数一致。修改文件后只重写SHA不能绕过重建校验。失败批次保留，不静默丢条件或续跑。

## 4. 本机PowerShell启动

示例使用本机已完成E1.3报告；新机器需换为自己实际成功批次。继续使用当前`.venv`及`configs/paths.local.yaml`。

```powershell
Set-Location 'D:\desktop\Semantic-Driven-Compositional\exp-e0-hmdb51-fewshot'
$E14Python = '.\.venv\Scripts\python.exe'
$E13Summary = Get-Content -Raw -Encoding UTF8 'reports/e13-class-text-report-20261119T080027665515Z-321bd075/summary.json' | ConvertFrom-Json
$E13Batch = $E13Summary.batch
& $E14Python scripts/13_e14_text_anchor.py check --e13-batch $E13Batch
if ($LASTEXITCODE -ne 0) { throw 'E1.4 preflight failed' }
```

### 4.1 准备独立输入快照

```powershell
$E14Index = & $E14Python scripts/13_e14_text_anchor.py prepare --e13-batch $E13Batch
if ($LASTEXITCODE -ne 0) { throw 'E1.4 input preparation failed' }
$E14Index
```

输出`artifacts/e14_text_anchor/e14-text-anchor-inputs-*/index-development.json`。之后若修改任何源码，必须显式重新prepare，不能手改哈希绕过。

### 4.2 显式运行80次评估

```powershell
$E14Batch = & $E14Python scripts/13_e14_text_anchor.py run --e13-batch $E13Batch --feature-index $E14Index
if ($LASTEXITCODE -ne 0) { throw 'E1.4 evaluation failed; inspect failure.log' }
$E14Batch
```

输出新的`runs/e14-text-anchor-development-*`，不改写E1.3。模型/分数输出完整后才能汇总；失败修复后新建批次。没有最终测试入口。

### 4.3 核验并汇总

```powershell
$E14Report = & $E14Python scripts/13_e14_text_anchor.py summarize --batch $E14Batch
if ($LASTEXITCODE -ne 0) { throw 'E1.4 report verification failed' }
$E14Report
```

输出新`reports/e14-text-anchor-report-*/e14-results.md`。正式CLI四阶段均拒绝合成特征。程序测试允许明确合成身份，并在报告显著标明“仅程序验证”。

## 5. 统计、主要比较与继续门槛

报告含7行条件汇总、101行单次指标、60行支持级指标、100行配对差值、10行配对汇总、5行逐旋转汇总、100行相对T0的逐次错误转换及6行转换汇总。原逐次混淆矩阵在各run的metrics.json。

A4先在同一支持集合内平均5个旋转的指标，再跨10支持计算样本SD；不是分数集成或50组独立支持集合。T0只一组固定预测；它与10支持条件配对不构成10次独立零样本测试。相对T0的改对/改错按同一视频ID匹配，控制组的转换汇总也先支持内平均旋转再跨支持汇总。

| 主比较 | 回答问题 |
|---|---|
| A2−T0 | 少量视觉支持是否提供真正增量 |
| A2−A3 | 增量是否依赖正确支持标签对应 |
| A2−A4 | 增量是否优于受扰动的视觉辅助方向 |

次要比较：A1−P1、A2−A1、A1−T0、A2−P1、A2−P0、P1−T0、P1−P0。

探索性继续门槛固定为：A2−T0未四舍五入的Macro-F1均值>=0.02、至少8组支持改善，且A2高于A3与A4均值。这不是统计显著性检验。原参考行没有新保存的分支分数，合并预测CSV的对应分支单元格留空，不伪造历史记录。

## 6. 结果解释与边界

- A1优于P1、仍低于T0：原型归一化可能贡献损失，但支持信息尚未改善强参照。
- A2优于T0及两个错误辅助对照：形成弱视觉校正的开发信号。
- A2回到T0却未超过：只说明少破坏了文本，不证明少样本增量。
- A2与错误校正相近：正确支持信息的作用仍不清楚。
- A1/A2都低于T0：可收束当前均值原型路线，再另立逐样本缓存等实验。

本轮没有构建Tip-Adapter缓存或可学习适配器，不将分数相加称为其完整复现。A4保视觉原型Gram，不匹配所有查询投影分布。原50段development已反复用于设计诊断，提升仍为开发证据；不能用支持SD代替独立视频/来源置信区间。

人工概念审核继续按[E1.2指南](E12_GUIDE.md)单独完成，不由本轮类别文本实验代替。未见组合、真实溺水误报、时序定位及部署效果仍未被验证。实现验收见[E14_VERIFICATION.md](E14_VERIFICATION.md)。
