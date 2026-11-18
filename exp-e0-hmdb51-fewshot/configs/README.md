# 配置

E1使用`e1.baseline.yaml`和`e1.concepts.v1.yaml`，固定16条姿态描述、5个投影种子及C0/C1/R0/R1规则；从第二轮成功批次继承来源划分、支持ID、K和岭分类设置。文本或资源变化须登记版本并生成新缓存，不能原位篡改历史身份。见[E1说明](../docs/E1_CONCEPT_GUIDE.md)。

`e0.round2.ablations.yaml`是第二轮补充条件配置，不重划分数据或重抽支持集。重复均值与帧序打乱使用固定归一化；打乱种子0—4独立于原支持集合种子0—9。具体K、支持ID和岭分类设置从指定的第二轮成功批次继承。见[操作说明](../docs/ROUND2_ABLATION_GUIDE.md)。

`e0.v0.example.yaml`是已实现的固定E0协议，拒绝静默修改类别、种子、λ或最终测试开关。复制`paths.local.example.yaml`为本机`paths.local.yaml`并填写真实标注和检查点；相对路径以仓库根为基准。本机路径不进入Git。

完整配置快照保存在批次和子运行。status不是实验通过证据，以运行状态、预测、指标和验收为准。
