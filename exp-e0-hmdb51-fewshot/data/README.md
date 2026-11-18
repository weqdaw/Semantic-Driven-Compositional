# 数据组织

原始视频只读引用`F:/Semantic-Driven-dataset/hmdb51`，不保存原视频副本。

- annotations/：官方153个划分文件、原RAR、来源时间/哈希；忽略Git。
- manifests/：全量审计、冻结bundle、类别映射和角色名单。
- support_sets/：种子0–9、K=1/3/5的嵌套支持身份。

五类1,165个视频全部对应官方列表，缺失/额外/字节级重复均为0。6个前导#差异经过官方内容验证，用显式只读别名处理，证明在`docs/provenance/verified_path_aliases.json`。详见DATA_INVENTORY和E0_RUN_RECORD。
