# 运行记录

每次调用生成唯一批次目录，不覆盖旧运行。真实批次位置见`reports/local_run.json`。

批次含batch.json、共享environment.json、manifest_snapshot.json、feature_index_snapshot.json、support_sets_snapshot.json；子运行含resolved_config.json、provenance.json、support_set.csv、predictions.csv、metrics.json、ridge.npz、run.log。代码快照与数据/特征身份在来源记录中登记。

失败明确标记failed并记录原因；报告拒绝不完整/失败批次。smoke和development不可混合，最终测试封存。大型产物忽略Git。
