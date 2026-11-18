# 开发结果

见[e0-results.md](e0-results.md)、[summary.csv](summary.csv)、[学习曲线](learning_curve.png)和[混淆](confusion_matrices.png)。本机输入/输出路径见local_run.json，再编码核验见reproducibility_check.json，受控摘要见`docs/E0_RUN_RECORD.json`。

30个运行共享50段开发视频；跨支持集合使用样本标准差，混淆为10次预测计数之和。最终测试封存。报告复核每个种子、预测文件、支持身份、指标和特征条件；不能把合成验证标为CLIP实验。
