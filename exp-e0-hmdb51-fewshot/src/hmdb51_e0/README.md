# 可复用模块

E1模块独立为`e1_features`（概念余弦与匹配控制）、`e1_text`（原生冻结文本分支）、`e1_experiment`（核验／缓存／拟合）、`e1_reporting`（前向与指标复算、两层聚合）和`e1_cli`（显式入口）。旧Python模块不修改；详见[E1指南](../../docs/E1_CONCEPT_GUIDE.md)。

第二轮补充控制位于独立的`ablation_features`、`ablation_experiment`、`ablation_reporting`和`ablation_cli`模块，原V0/V1实现不变。提供重复均值、固定帧序打乱、来源／支持配对、完整证据校验和两层种子聚合；见[操作说明](../../docs/ROUND2_ABLATION_GUIDE.md)。

已实现：config（协议和路径）、inventory（官方/本地审计及训练资格）、manifest（开发留出）、sampling（PCG64嵌套支持）、video_io（OpenCV FFmpeg顺序解码）、clip_features（冻结原生编码）、feature_cache（NPZ/JSON完整性）、ridge（n×λ和无惩罚截距）、metrics（五类指标）、pipeline（编排和失败记录）、reporting（完整批次复核、统计和图表）、provenance（哈希、环境、代码身份）、cli（六阶段入口）。

编码器只接受RGB帧，不接受标签。拟合器仅接受当前支持特征和标签；开发标签只用于评价。生产CLI拒绝非官方CLIP特征，API测试注入的合成特征报告标为仅程序验证。
