# 第二轮补充消融实现验收

日期：2026-11-18。本记录验收代码与流程，不报告真实补充实验性能。

## 已交付

- 新增`ablation_features.py`：重复均值、确定性帧排列、条件配置及实现身份。
- 新增`ablation_experiment.py`：真实输入只读核验、独立控制缓存、完整180组拟合及代码／环境／父证据快照。
- 新增`ablation_reporting.py`：逐预测复算、完整配对、逐扰动结果、同支持集合内平均与跨支持集合样本SD、表格及图表。
- 新增`ablation_cli.py`、`scripts/08_round2_ablations.py`：四个显式子命令，只提供development。
- 新增`configs/e0.round2.ablations.yaml`，补齐[操作指南](ROUND2_ABLATION_GUIDE.md)、数据契约和导航。原0.1／0.2配置未修改。

## 测试证据

命令：`.venv/Scripts/python.exe -m pytest -q`。

最终结果：**116 passed in 53.31s**。包括原94项与新增22项。数值及流程测试先失败再实现；独立审查提出的环境快照检查缺口，以篡改environment.json导致测试`DID NOT RAISE`复现后修正，已纳入最终通过的全套测试。

新增测试使用pytest临时目录中的合成特征，弱信号输入产生非恒定的分类结果与支持／扰动差异。测试覆盖：

- 重复均值的五块结构、标准差相对权重、逆序不变性及输入约束。
- 帧排列确定性、不同视频／种子的独立规则、实际排列精确回放与全局统计保持。
- 合成80段训练范围缓存覆盖、NPZ及排列损坏／重算哈希后错误排列拒绝；真实350段另作只读核验。
- 默认180组新增运行，60组引用，共240组逐预测指标复核；120个主聚合单位、120项主要配对。
- 同支持集合内先平均扰动指标，再计算支持集合SD；各扰动原始结果均保留。
- 缺运行、支持ID错配、重算预测文件哈希后的错误真值、环境快照篡改拒绝。
- 生产CLI拒绝合成输入，最终测试阶段拒绝，帮助命令无拟合副作用。

合成学习曲线已可视检查，标题明确标为PROGRAM VALIDATION，四条件图例及误差条可读；合成分数不作为研究发现。

## 真实输入只读核验

使用`reports/round2_latest.json`指定的第二轮批次，执行：

```powershell
$AblationReference = (Get-Content -LiteralPath 'reports/round2_latest.json' -Raw -Encoding UTF8 | ConvertFrom-Json).batch
& ./.venv/Scripts/python.exe scripts/08_round2_ablations.py check --reference-batch $AblationReference
```

退出码0；返回`status=verified`，正式OpenAI CLIP ViT-B/32身份、350段训练范围缓存、50段开发、60组可引用基线、180组计划新增运行、扰动种子0—4、`fits_executed=0`、`final_test_enabled=false`。

数据协议：`fb75e9e07d306a6296c869a34b59b8c1ff6a95867acbe76dba002578b371d5e5`。

父特征身份：`34043c53e8ab272423b0e9cb38552768f14b59bef6b81396454b2e22ce3511df`。

## 保留性与审查

实现前对494个既有文件建立内容哈希及本地备份，位于`.ablation-backup/20261118T073649Z/`。实现后核验：全部原有源模块、脚本、测试、固定配置及受保护实验产物不变；既有文档仅补充导航／契约。备份和新生成缓存路径已加入`.gitignore`。

按requesting-code-review技能完成独立只读代码审查。未发现核心数值、支持／开发泄漏或聚合错误；唯一Important问题为环境文件未验证，已修正并通过回归。原来源边界仍为AI辅助保守审核，不升级为人工确认或完整独立来源认证。

真实`artifacts/round2_ablations/`尚未创建；`runs/e0-round2-ablations-*`数量为0；未执行真实`prepare`或`run`，没有改变最终测试封存状态。操作指南明确区分只读核验、特征准备与正式拟合。
