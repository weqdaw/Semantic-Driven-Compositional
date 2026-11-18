# 第二轮代码变更与验证记录

日期：2026-11-16。用户授权范围：实现来源分组与V1第二轮代码，暂不运行真实实验。

## 变更清单

| 文件 | 内容 |
| --- | --- |
| src/hmdb51_e0/source_groups.py | 从eligible训练清单生成跨类别来源候选；审核字段、覆盖、证据、占位符及同候选不拆分校验 |
| src/hmdb51_e0/round2_protocol.py | 独立0.2配置、父协议核验、确定性按组划分、重放验证、视频shot嵌套抽样和来源覆盖 |
| src/hmdb51_e0/temporal_features.py | 保持V0原算法；新增6/5/5分段V1，官方维度1024/2560 |
| src/hmdb51_e0/round2_features.py | 核验父缓存/原文件/冻结哈希；无编码导入Z，独立Z/V0/V1缓存及中断重试 |
| src/hmdb51_e0/round2_runner.py | 同划分同名单V0/V1共60组，保留逐视频预测、模型、来源计数和失败记录 |
| src/hmdb51_e0/round2_reporting.py | 全批次/配对核验、逐预测复算、差值统计、独立图表及程序验证标识 |
| src/hmdb51_e0/round2_cli.py、round2_io.py | 六个显式子命令、正式骨干身份要求、最终测试封存及输出边界 |
| scripts/07_round2.py、configs/e0.round2.example.yaml | 独立启动入口和第二轮配置 |
| tests/test_round2_protocol.py、test_round2_features.py、test_round2_runner.py、test_round2_cli.py | 55项第二轮自动测试，全部使用临时合成输入 |
| docs/ROUND2_V1_GUIDE.md | 完整PowerShell启动步骤、人工审核要求、结果判读及恢复方式 |
| README.md、docs/EXPERIMENT_GUIDE.md、docs/DATA_CONTRACTS.md、scripts/README.md | 增加第二轮入口和跨轮协议说明 |
| .gitignore | 忽略本机来源表、新缓存、真实输出及修改前备份 |

## 实际验证

新增第二轮测试命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_round2_protocol.py tests/test_round2_features.py tests/test_round2_runner.py tests/test_round2_cli.py -q
```

结果：55 passed in 28.65s。完整原有及新增测试的最终结果在本记录末尾追加。

主入口及audit-sources/freeze-split/sample-supports/prepare-features/run/summarize的`--help`共7项，均退出0。

另做一次独立只读代码审查，修正并通过回归测试的事项：未分配来源占位ID阻塞；官方冻结记录须有一致的before/after/初始参数哈希；失败或中断的同身份导入可核验后重试并留存旧尝试；每张合成图与结果CSV明确标识程序验证；整组重排并重算摘要也不能冒称固定种子划分。审查未执行真实实验。

## 保留与未执行项

- 真实第一轮批次记录的21个原有代码/项目文件SHA-256与修改后文件全部一致，原01—06实现保持原样。
- 修改前记录的2326个src/scripts/configs/data/artifacts/runs/reports既有文件中，大小/修改时间变化仅有预期的scripts/README.md；第一轮数据清单、支持集合、缓存、运行和报告均未改变。
- 未生成真实`data/source_groups`或`artifacts/round2_features`；真实第二轮运行目录和报告目录数量均为0。
- 原视频只读；最终测试未解码、未提特征、未运行评估；Research Board未修改。
- 修改前文档备份与原文件元数据清单在`.round2-backup/20261116T150649Z/`。此目录原本没有Git，本次未初始化Git、未提交、未创建分支。
- 尚待人工来源审核及真实第二轮执行。本记录不提供、也不推测第二轮准确率。

## 最终全量回归结果

实际执行`.\.venv\Scripts\python.exe -m pytest -q`：**92 passed in 38.60s**（原有37项＋新增55项），退出码0。此结果属于程序验证，未启动真实HMDB51第二轮实验。

## 2026-11-17补充验证与真实运行

用户明确委托AI完成来源审核。新增两个官方_l_后缀解析失败回归测试后修正候选提取器；完整套件实际结果为**94 passed in 42.04s**。原21个第一轮代码/项目文件SHA仍一致。

已完成60组V0/V1真实开发运行，3000条预测；独立按混淆矩阵公式复算一致，350段新缓存Z/V0与第一轮逐位一致。95个支持来源组与24个开发来源组无交集。审核身份明确为AI辅助保守审核、无人工签字。结果与证据见reports/round2_latest.json所指目录；150段最终测试不启用。
