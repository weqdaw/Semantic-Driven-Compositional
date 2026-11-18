# 六阶段入口

E1入口`09_e1_concepts.py`提供`check/prepare/run/summarize`，分别为只读核验、仅文本与特征准备、显式360组拟合、完整预测复核。没有隐式实验或最终测试入口，见[E1操作说明](../docs/E1_CONCEPT_GUIDE.md)。

每个脚本支持`--help`。完整命令见仓库README。

| 脚本 | 职责 |
| --- | --- |
| 01_audit_dataset.py | 官方/本地身份及训练解码资格，阻塞返回2 |
| 02_build_manifests.py | 成功审计后冻结bundle及角色名单 |
| 03_extract_features.py | 官方权重、原生预处理、smoke/development特征 |
| 04_sample_support_sets.py | 所有K/种子的嵌套支持名单 |
| 05_run_v0.py | 1个冒烟或30个开发运行 |
| 06_summarize_results.py | 完整批次复核后生成统计、图表和报告 |

默认development，smoke单独标记；没有开启final_test的选项。每次审计、清单、支持集合和运行生成唯一目录。

# 第二轮新增入口

补充消融使用`08_round2_ablations.py`，提供只读`check`、仅派生缓存的`prepare`、显式拟合180组的`run`和逐预测复核的`summarize`。详见[补充实验说明](../docs/ROUND2_ABLATION_GUIDE.md)。

`07_round2.py`提供`audit-sources`、`freeze-split`、`sample-supports`、`prepare-features`、`run`、`summarize`六个显式子命令。只运行`--help`不会启动实验；`run`会同时拟合V0/V1共60组开发结果。来源表未完成人工审核时，冻结会阻塞。详见[第二轮启动说明](../docs/ROUND2_V1_GUIDE.md)。01—06脚本保留第一轮行为。

