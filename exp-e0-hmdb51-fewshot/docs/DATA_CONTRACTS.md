# E0数据与产物契约

当前实现遵循本规范；templates仍只作表头样例，真实产物位于data/artifacts/runs/reports。编码统一UTF-8。相对视频路径采用POSIX分隔符，始终相对外部dataset_root解析；禁止绝对路径逃逸和含..路径。仓库输出路径相对仓库根解析。

第二轮使用独立契约，见本文第8节和[ROUND2_V1_GUIDE.md](ROUND2_V1_GUIDE.md)；不改变第一轮字段和身份。

## 1. 视频身份与manifest

video_id固定为class_name/官方原视频文件名（含.avi）。不使用仅文件名作为跨类唯一键，不按随运行变化的行号标识。原名不自动重命名。

实现时发现本地6个clap文件名缺少官方原名的前导`#`。已从Serrelab官方发布的原始视频压缩包提取这6个参考文件，与本地文件逐一比较SHA-256，全部字节一致。为保持外部数据只读，允许**显式、经过内容验证的路径别名**：video_id仍用官方原名，relative_path记录实际本地名；记录path_alias_verified_sha256及alias_reference_sha256，别名证明及来源随审计保存。禁止模糊匹配、自动去字符或仅凭相似文件名认定同一视频。别名进入清单/协议哈希，不改变官方tag或样本集合。普通记录仍要求relative_path等于video_id。

| 字段 | 类型／约束 |
| --- | --- |
| video_id | 唯一字符串；class_name/filename.avi |
| relative_path | 默认与video_id一致；只有上述显式内容验证别名可不同；不得越出dataset_root或跨类别 |
| class_name/class_id | 固定五类及0–4映射 |
| official_split | 整数1；其他split作为新协议 |
| official_tag | 0/1/2；以原始标注为准 |
| experimental_role | support_pool/development/final_test/unused/excluded |
| bytes | 非负整数，实际文件大小 |
| sha256 | 内容哈希；计算前为空，不写猜测值 |
| frame_count | 实际有效帧数；未解码时为空 |
| duration_seconds | 已验证时长；未知为空，不用0替代未知 |
| decode_status | not_checked/ok/failed/too_short |
| exclusion_reason | 排除时必填，其余为空 |

official_tag、角色和排除状态需联合校验：tag=0只能unused或excluded；tag=2只能final_test或excluded；tag=1才可support_pool/development。失败记录不删除，形成全量审计与明确的eligible清单；实际训练只读eligible记录。

官方列表没有记录的本地额外视频不能编造official_tag，应放到独立annotation_audit差异记录，不强塞进上述已解析manifest。官方要求但缺失的视频同理记录，默认阻塞正式运行。

## 2. 划分与类别映射

class_mapping为固定五条记录，ID依次为clap=0/jump=1/run=2/walk=3/wave=4。所有得分矩阵列遵守此序。

split登记包含：protocol_id、official_split、官方标注哈希、开发种子、类序、实际视频ID、排除策略、清单哈希。support_pool/development/final_test的视频ID和已确认内容哈希集合必须互斥。开发只来自tag=1，最终测试不参与抽样。

## 3. 支持集合

字段：protocol_id、support_seed、k、class_id、video_id、rank_in_class。
唯一键：(protocol_id,support_seed,k,video_id)。
每个类恰好K条，video_id来自support_pool，rank从1起且不超过K。固定种子内小K是大K前缀，各集合不能有同一视频重复。实际支持集合重复时记录unique集合数量。

## 4. 特征缓存

每条记录以video_id连接，含：帧向量Z（16×d）、视频向量V0（2d）、实际帧索引、视频SHA-256、检查点SHA-256、预处理／解码／采样／汇总配置及特征dtype。

实现固定为每视频NPZ（Z/V0）与JSON侧车，缓存目录和文件名均由身份哈希确定。JSON包含完整配置身份、维度、dtype及NPZ字节哈希；索引记录本阶段成功/失败和冻结核验。d来自实际检查点，本次为512；Z/V0为float32，均值及总体标准差使用float64累积，防止常量帧舍入残差被归一化放大。原生预处理序列以稳定的结构化参数保存，不能使用带进程内存地址的对象repr。维度、有限值、身份及归一化全部通过后才可复用。标签不作为编码输入。

## 5. 单次预测

字段见templates/predictions.csv.example。每个(run_id,video_id)唯一，真值与预测ID均为0–4；得分列固定score_clap/score_jump/score_run/score_walk/score_wave，不是概率。预测为得分argmax，平局取最小ID。

evaluation_stage为smoke/development/final_test，不能混合聚合。support_seed/k/run_id必须与当前运行快照一致。每次开发运行应有相同视频ID集合且一视频一行。

## 6. 指标与聚合

metrics记录阶段、n_support、n_evaluated、类序、Accuracy、Macro-F1、五类召回和混淆矩阵；数值范围、矩阵形状和总计数都应校验。

summary按evaluation_stage和K分组，记录n_support_sets、n_unique_support_sets、评估视频数、均值和样本标准差；样例提供最低列集。有效运行数1时std=null；不把缺失运行当0分，也不静默丢弃失败。混合数据／权重／预处理身份拒绝聚合。

## 7. 运行身份

run_id包含实验ID、协议版本、阶段、K和种子，并保证不覆盖；时间戳用带时区ISO格式，不用它代替配置哈希。provenance包含：
- 输入清单、官方标注、支持ID、类别映射的哈希。
- 检查点身份、预处理和特征缓存身份。
- 环境、设备、依赖版本。
- 代码版本：Git提交存在时使用提交身份及脏状态；当前未初始化Git，未来可先用代码快照哈希。
- 开始／结束、成功／失败状态和排除记录位置。

scaffold_only不能作为成功运行。当前已有六阶段CLI及完整实现。批次目录保存共享environment.json、manifest_snapshot.json、feature_index_snapshot.json、support_sets_snapshot.json和batch.json；子运行保存resolved_config.json、provenance.json、support_set.csv、predictions.csv、metrics.json、ridge.npz和run.log。报告验证完整批次、全部种子及每个预测/指标文件，不静默丢弃失败。

## 8. 第二轮来源分组与V1增量契约

0.2配置与0.1分开。来源表以video_id一一覆盖父清单eligible训练视频；保留class_name、video_sha256、original_role、candidate_source_id，并要求source_group_id、verification_status、evidence、reviewer、带时区reviewed_at。unresolved阻塞；confirmed/conservative的证据边界在报告中保留。同候选前缀跨类同组；不同候选可人工合并，不自动推断独立。

分组bundle保存完整父manifest与审核快照。eligible官方训练仅变更实验角色并增加source_group_id/source_verification_status；父清单中的其他记录逐字段不变。支持/开发来源组互斥，各类开发≥1、支持≥5；目标开发10段允许偏差。protocol_id哈希整个分组payload，不把V0/V1表示名称作为不同划分。

第二轮support_sets.json包含records与source_coverage及整体sha256。records沿用原字段并增加source_group_id；coverage按K/seed/class记录视频数及来源组数；验证器重建确定性集合。video-shot定义保持不变。

特征根独立于旧缓存。每条NPZ保存Z(16×d)、V0(2d)、V1(5d)，三者float32；V1增加6/5/5连续分段均值。缓存身份包含新协议、父特征索引文件SHA、父特征身份、编码/解码/采样条件、表示规则、代码身份与NumPy版本；索引保存精确父索引快照、冻结核验与完整训练覆盖。复用前重新核验原训练文件字节、父缓存及新表示。正式CLI拒绝未验证/合成骨干身份；合成测试批次必须标为validation_only_unverified_features。

第二轮批次固定包含2表示×3K×10种子共60组。预测增加representation/source_group_id；provenance增加表示维度和实际支持来源数量。汇总检查每个V0/V1配对的协议、支持和开发身份，逐预测复算指标，输出配对差值与来源统计。缺项/失败/混合批次均报错；不覆盖第一轮汇总。

## 9. 第二轮补充消融契约

独立0.3条件配置，不改变0.2数据协议。V1_repeat=[μ,σ,μ,μ,μ]；V1_shuffle对原16帧按完整video_id和独立shuffle_seed固定排列后复用V1。二者均5d，保持原数值规则。默认扰动种子0—4；同视频排列跨K和支持种子复用，不读标签、角色或得分。

衍生缓存独立保存两项向量及int16排列，身份包含原protocol、原特征身份、原批次306个证据文件的摘要、编码器、条件配置、代码和NumPy版本。原始Z不复制、原特征不覆盖。排列CSV含零基排列、原始采样帧索引、排列后的采样帧索引。

新增批次默认180组（30重复均值＋150打乱）。shuffle_seed是独立字段：重复均值为JSON null／CSV空值，打乱为整数；不得与support_seed混用。预测字段在第二轮基础上增加shuffle_seed。每个run保存完整支持ID、开发ID、来源组、维度、条件、原始得分及文件哈希。reference/保存原60组的复算证据，code_snapshot/保存当前代码；仅development。

主汇总先对同K／support_seed的5次打乱指标平均，再跨10个支持集合取均值与样本SD；另保存逐扰动和全部逐视频结果。平均指标不是预测集成指标；50次扰动评价不等于50个独立支持集合。主要配对V1−V0、V1−repeat、V1−shuffle、repeat−V0均共享支持ID与开发ID。详见[补充实验说明](ROUND2_ABLATION_GUIDE.md)。

## 10. E1概念与随机瓶颈契约

E1条件版本1.0引用第二轮数据protocol_id，不改变来源划分。catalog保存稳定ID、中文名、英文描述、可见性说明及开发来源；源文件SHA与规范化内容摘要均冻结。默认m=16，文本E为m×512的单位行，逐帧C=ZE^T不做行归一化、softmax或阈值化。C0为2m、C1为5m，后者沿用6/5/5分段，其他数值规则与E0一致。

R0/R1按固定5种子0—4生成m×512单位行随机方向，跨视频/K/支持集合共用，分别与C0/C1同维、同汇总。projection_seed与support_seed分列，概念条件projection_seed为JSON null／CSV空值。不能用shuffle_seed代替或按每视频重抽方向。

缓存NPZ保存Z/E/P/C/C0/C1/R0/R1，身份绑定父证据摘要、编码器双侧冻结、tokenizer／BPE、概念快照、逐视频float32小端C序Z数组SHA、代码及NumPy。加载从Z/E及Z/P重算全部表示，拒绝仅更新NPZ哈希的错配。Z是已验证旧特征的副本，不是重新编码的图像。空白审核模板不表示概念真值，含类名路径不宣称严格盲审。

默认新增360组及引用60组；共享数据、支持与开发ID、λ=1。批次含完整特征库、概念库、代码／环境、原基线复核证据及360个岭模型。报告核对全部文件并重算每组前向得分和指标；R先在支持内平均5次指标，再跨10个支持集合统计。主配对150项、概念时间增量差30项，保留各投影结果及21000条重复预测。只提供development，详见[E1操作指南](E1_CONCEPT_GUIDE.md)。

## 11. E1.1 几何与融合诊断契约

独立脚本10与固定 e11.diagnostics.yaml，引用完整E1批次。原Python代码及旧特征不覆盖。缓存/批次各含完整parent/证据树，索引绑定逐文件SHA及整体摘要；新代码/环境在批次快照中保留。来源划分、支持/开发ID、K和λ继承原协议，最终测试封存。

features.npz包含Q(5,d,d) float64，D(5,m,d)、V1(n,5d)、G1(5,n,5m)、VC1(n,5d+5m)、VG1(5,n,5d+5m) float32。D=E@Q，不逐行重归一化；每种子Q跨概念与视频共享。加载从parent的Z/E/C1确定性重算全部数组；Gram误差绝对容忍1e-6。共同正交变换保文本Gram但不保证输出协方差一致。融合两块分别归一化后等能量拼接并最终归一化。

新增330组：G1/VG1各150、VC1为30。rotation_seed与support_seed分开；VC1的rotation_seed为null/CSV空值。每组保存预测/模型/指标/支持表与哈希，汇总核验完整网格、原真值/来源、特征库及保存ridge的前向得分。复用V1/C1各30，报告合计390组。随机旋转先在支持集合内平均指标，再跨10组支持统计；另保留每旋转与19,500条重复预测。

审核材料独立导出：固定10段dev视频×缓存位置2/7/13×六概念=180空白判断行；验证原视频SHA、别名、解码器、frame_indices及原生裁剪身份，保存原图/crop PNG。judgment只允许人工填写present/absent/uncertain；导出清单human_review_completed=false，不表示已审核。含类名路径不宣称严格盲审。完整操作与解释边界见[E1.1指南](E11_GUIDE.md)。

## 12. E1.2 支持集尺度匹配契约

独立脚本11与固定e12.support-scale.yaml，仅3-shot×10支持集合。新增240组：V1_lambda2/SV1/SC1/SVC1各10，SG1/SR1/SVG1/SVR1各50。除原V1_lambda2用lambda2外，其余lambda1；控制种子与支持种子分列。复用同一E1.1与E1祖先的原3-shot V1/C1/G1/R1/VC1/VG1共180组，报告420组。

prepare复制并绑定完整E1.1证据，不编码、不拟合尺度或分类器。每次run只用当前15段支持估计各块mean与中心化RMS范数scale；scale<=1e-6记录批次失败。scalers.npz保存float64的block_mean与block_scale；V1_lambda2为空参数表。scaling.json保存scaler_fit_ids、块维数、变换前后平均中心化平方范数，明确post_fusion_row_l2=false。融合concat/sqrt2，不使用开发数据拟合、不做逐维白化或再次逐行L2。变换依赖支持集合，不生成一个共用全局归一化库。

缓存与批次均保存parent/完整证据；新批次额外保存源代码/环境、240个模型及支持表、精确预测、配置、尺度和各文件SHA。汇总从父原特征与支持集合独立重算参数，再核验模型前向、真值和指标。随机控制仍先在支持内平均5次指标，再跨10支持计算样本SD。输出14条件汇总、140支持级指标、120配对差值、35逐控制汇总、350逐块尺度诊断；真实50视频共有21,000条重复预测。

audit读取E1.1既有180行材料，验证父证据/模板/图片/帧ID/分数；仅human字段允许修改。present/absent/uncertain必须带reviewer、时区时间与证据；未完成时不输出分数排序结论。人工表不进入拟合或概念选择；可选附加到模型报告并快照当时审核CSV、模板、导出清单。详见[E1.2指南](E12_GUIDE.md)。

## 13. E1.3 类别文本与原型契约

脚本12和固定e13.class-text.yaml。使用完整E1.2父批次的E1祖先Z及编码器身份；q=unit(mean16Z)，五条固定类文本E由同一原生CLIP编码并核验tokenizer/参数冻结，T=unit(E)。缓存保存q/E/T/Q/D/wrong_text_rows，E为float32、错配表int64，其余float64，绑定原文本数组SHA与父/代码/环境条件；加载重算q/T/Q/D，不重新编码文本。

T0仅一次k0/support_seed=null且无support，prototypes=T，visual_prototypes为空(0,d)。P0/P1/P2各10组原3-shot支持集合，P3五旋转×10共50组；总81评估、无梯度训练。P0=unit(每类3段支持q均值)，P1=unit(P0+T)，P2固定类文本行映射[1,2,3,4,0]，P3=unit(P0+T@Q)。全部同d维余弦分类，查询真值只计算指标。

每次保存原型、精确支持表、分数/指标、配置、来源/输入身份与哈希。完整批次快照原E1.2与新代码/环境/缓存，汇总重建模型和前向。缺失条件/错误参数/不匹配预测拒绝。输出81单次、5汇总、40支持级、60配对、5逐旋转，真实50视频为4050条重复预测。T0无支持SD；P3先支持内平均旋转，再跨支持SD；与T0配对不虚构零样本重复。详见[E1.3指南](E13_GUIDE.md)。

## 14. E1.4 文本锚与视觉分数校正契约

脚本13、固定e14.text-anchor.yaml，输入完整成功E1.3。原q/T/Q、类描述、3-shot/10support种子/开发ID不改，不重新编码。A1用qT+1*qp，主A2用qT+0.1*qp；A3只错配视觉p行[1,2,3,4,0]，A4只旋转视觉p@Q（5种子），文本T不动。不做合成权重／分数归一化、softmax或温度。新增80评估，引用T0一次与P0/P1各10，共101组。

prepare仅复制并绑定原E1.3证据树，不评估；run保存text_prototypes/visual_prototypes/auxiliary_prototypes/score_weights四个float64数组及text_score/auxiliary_score/score三组逐类分数，最终分数允许超出[-1,1]。记录实际lambda、支持来源/ID、配置、模型、指标和文件SHA。摘要前独立重建模型与各分支分数，核对text_score与原T0一致、保存分数的加法关系和原真值/指标；仅改SHA不能绕过。

报告7条件、101单次、60支持级、100配对、5逐旋转、100逐次T0错误转换及6组转换汇总；真实50段开发为5050重复预测。A4指标和转换计数先支持内平均再跨10support统计。T0不复制为10次独立零样本。参考行新增分支字段留空，保留原预测含义。探索门槛>=2pp/8正且优于两错误控制，只供研究决策。原代码和结果保留，最终测试封存。详见[E1.4指南](E14_GUIDE.md)。
