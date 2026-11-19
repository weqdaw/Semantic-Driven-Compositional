# E1.3：类别文本参照与支持原型

本机E1.3 development已完成；本文保留复现协议。后续文本锚分数校正见[E1.4指南](E14_GUIDE.md)。输入为已成功完成的E1.2批次；沿用其原来源分组、支持集合与50段development，最终测试封存。人工概念审核继续使用脚本11，本轮不填写或使用人工标签。

## 研究问题与资源条件

E1.2尺度匹配没有恢复当前16姿态概念的稳定融合收益。本轮检验：冻结CLIP是否已经具备五个动作的类别语义？在少量支持视频之外，正确类名文本是否能改善支持原型估计？

这里使用类别名称，区别于旧姿态概念资源。T0不使用下游支持视频，不保证CLIP预训练未见这些动作；P组有每类3段支持，不能称zero-shot。本轮是轻量原型对照，不是CLIP-FSAR论文完整复现，也不直接证明可解释概念、关系推理或未见组合泛化。

## 固定协议

固定类别顺序clap/jump/run/walk/wave，提示分别是：

```text
a photo of a person clapping.
a photo of a person jumping.
a photo of a person running.
a photo of a person walking.
a photo of a person waving.
```

仅一个模板，不搜索提示或混合权重。类别顺序、描述、错配映射、种子等固定于`configs/e13.class-text.yaml`；改变将报错，应另立实验版本。

设每段原缓存Z为16×d的单位帧向量。定义：

- `q_i = unit(mean_t Z_it)`：每视频一个d维单位向量，float64累积。官方d=512，测试可使用显式合成小维度。
- `T_c = unit(E_c)`：同一原生CLIP检查点编码固定五条描述，E保留float32、T以float64归一化。
- `p_c = unit(mean(q_i for current class-c supports))`：只用当前支持集合中该类3个完整视频。开发视频及其真值不参与原型构造。
- 分类分数为`q @ prototypes.T`，直接argmax；保存余弦分数，不解释为概率。
- 任何均值或融合向量范数<=1e-6时报错并记录失败，不补零或静默剔除。

所有P条件共享同一d维视觉表示和分类规则。旧V1的5d维统计表示不是P0，不将跨维度／算法的差值视为单个文本模块的效果。

| 条件 | prototypes | 支持集合 | 评估次数 |
|---|---|---|---:|
| T0 | T | 无，k=0、support_seed=null | 1 |
| P0 | p | 原3-shot，种子0..9 | 10 |
| P1 | unit(p+T) | 同上 | 10 |
| P2 | unit(p+T[[1,2,3,4,0]]) | 同上 | 10 |
| P3 | unit(p+T@Q) | 同上，每组5个Q | 50 |

共81次评估，无优化器、无梯度训练。P2固定对应：clap使用jumping，jump使用running，run使用walking，walk使用waving，wave使用clapping；不根据查询真值或分数决定错配。P3用PCG64高斯矩阵QR与符号修正生成共同O(d)正交矩阵，种子0..4；同一Q用于全部类、视频和支持集合，可能含反射。保留类文本Gram，不保证视觉投影统计相同。

## 输入、缓存与身份

必须有完整E1.2批次，不仅是e12-results.md。校验沿E1.2→E1.1→E1祖先恢复原Z、编码器身份、清单、支持和原始模型证据。原始视频不解码、不移动；图像不重新编码。

- check：只读核验完整父链、原检查点SHA及原生文本依赖；不加载模型编码、不构建新评估。
- prepare：显式编码五条文本，验证图／文检查点、参数hash、双侧冻结及tokenizer/BPE与原E1祖先完全一致；计算q/T/Q/D和固定错配表，保存独立缓存。
- run：使用缓存构造支持原型，评估81次。不重新编码文本，不使用开发标签选择文本或参数。
- summarize：从原Z与已保存文本重建缓存，从原支持ID重建原型，再核验每条前向得分和指标；缺条件、错配身份、损坏数据报错。

新缓存绑定父证据树、代码、NumPy、图／文编码器、冻结核验、原生文本数组SHA、配置和全部数组文件SHA。改NPZ SHA无法绕过q/Q/D的重算或已编码文本的身份校验。旧源码不改；新代码改变后需显式重新prepare。

缓存与运行各复制完整E1.2证据树，磁盘空间和小文件数量会增加，均被Git忽略。每组保存support_set.csv、prototypes.npz、predictions.csv、metrics.json、provenance.json、resolved_config.json和日志；T0支持表只有表头，视觉原型数组为空，support_ids为空。不能为T0复制十份支持记录。

## 本机PowerShell启动

在实验根执行。示例E1.2报告是本机产物，新机器须替换自己的真实输入。当前环境继续使用现有`.venv`与`configs/paths.local.yaml`。prepare需要本地官方检查点及原生CLIP包；不会自动下载权重。默认cuda，可显式指定cpu，设备身份会记录。

```powershell
Set-Location 'D:\desktop\Semantic-Driven-Compositional\exp-e0-hmdb51-fewshot'
$E13Python = '.\.venv\Scripts\python.exe'
$E12Summary = Get-Content -Raw -Encoding UTF8 'reports/e12-support-scale-report-20261119T071125514498Z-73daa62c/summary.json' | ConvertFrom-Json
$E12Batch = $E12Summary.batch
& $E13Python scripts/12_e13_class_text.py check --e12-batch $E12Batch
if ($LASTEXITCODE -ne 0) { throw 'E1.3 preflight failed' }
```

### 准备五条类文本与缓存（尚未评估）

```powershell
$E13Index = & $E13Python scripts/12_e13_class_text.py prepare --e12-batch $E12Batch --device cuda
if ($LASTEXITCODE -ne 0) { throw 'E1.3 text preparation failed' }
$E13Index
```

输出`artifacts/e13_class_text/e13-class-text-features-*/index-development.json`。这一步使用真实模型编码文本；本次实现验收不自动执行它。

### 显式评估81次

```powershell
$E13Batch = & $E13Python scripts/12_e13_class_text.py run --e12-batch $E12Batch --feature-index $E13Index
if ($LASTEXITCODE -ne 0) { throw 'E1.3 evaluation failed; inspect failure.log' }
$E13Batch
```

输出新`runs/e13-class-text-development-*`目录，不覆盖父批次，不静默续跑或忽略失败。失败后保留记录，修复并使用新目录。

### 验证与汇总

```powershell
$E13Report = & $E13Python scripts/12_e13_class_text.py summarize --batch $E13Batch
if ($LASTEXITCODE -ne 0) { throw 'E1.3 report validation failed' }
$E13Report
```

输出`reports/e13-class-text-report-*/e13-results.md`及JSON/CSV。正式CLI所有阶段拒绝合成身份；Python测试API允许显式合成输入并在报告标明“仅程序验证”。没有final_test入口。

## 汇总与解释规则

输出81行单次指标、5行条件汇总、40行支持级指标、60行配对差值、6行配对汇总、5行逐旋转汇总，以及实际50段开发下的4050条重复预测。原始每次5×5混淆矩阵保留于run metrics.json。

T0只评一次，不报告支持集合SD；P3先在每组支持集合内平均5个旋转的指标，再跨10组支持求均值与样本SD（ddof=1）。不是平均分数的集成，也不是50组独立支持重复。T0在配对差值中作为固定参照，十个差值不代表十次独立zero-shot测试。

主比较：P1−P0（类文本增量）、P1−P2（正确对应关系）、P1−P3（相对共同正交类文本）。次要：P1−T0、P0−T0、P2−P0。不能挑最好旋转或支持集合。

- 类文本有效、旧姿态概念较弱：下一阶段优先概念落地与任务辨别性。
- P1稳定超过P0及错误文本对照：形成继续研究类别语义辅助支持原型的信号。
- P1接近T0：主要收益可能来自类文本先验，支持视频的额外作用有限。
- 类文本和辅助原型都弱：优先检查输入可见性与图像骨干对动作的表达。

本轮未自动定义显著性或继续门槛；报告全部均值、支持级差值及正／平／负数。开发集已反复使用，结果仍属开发证据。方法冻结后另立独立评价；未见组合与真实溺水检测仍需专门数据和任务设计。

旧180行人工审核继续按[E1.2指南](E12_GUIDE.md)完成，其结果与本轮类别文本参照分开解释。实现验收见[E13_VERIFICATION.md](E13_VERIFICATION.md)。
