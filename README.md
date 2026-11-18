# Semantic-Driven Compositional

**语义驱动的少样本行为识别研究：从冻结视觉基线，到显式概念与时序组合。**

本项目研究在目标任务标注有限时，通用视觉语言预训练、可读概念及其时间结构能否改善行为识别。长期应用背景是受摄像头、场景和标注条件限制的防溺水项目；当前实验使用HMDB51公开动作视频验证方法，不代表已经完成溺水检测系统。

当前包含两个阶段：

- **E0：建立并审计冻结视觉基线。** 完成普通视频留出、保守来源分组、V0/V1配对及重复均值／帧序打乱消融。
- **E1：检验显式概念瓶颈。** 完成16个姿态概念、C0/C1及匹配随机投影R0/R1的开发实验。当前概念方案尚未体现整体语义增量。

## 研究问题

1. 同一冻结骨干下，少量支持视频是否足以进行可复核的任务适配？
2. 显式文本概念是否优于普通视觉表示及同维非语义瓶颈？
3. 概念时间扩展的增量，能否与一般统计、维度和容量变化区分？
4. 中间概念是否有真实画面依据，而不只是可读的名称？

本仓库保存实验实现、配置、测试、协议和阶段总结。当前不是训练完成的部署产品，也不是zero-shot或未见动词—对象组合泛化的验证。

## 方法与实验设置

公开预实验固定HMDB51 Split 1的五类：`clap / jump / run / walk / wave`。按完整视频定义shot，使用每类1、3、5个支持样本，支持种子0—9且跨K嵌套。

冻结OpenAI CLIP ViT-B/32，对每个完整视频取16帧；逐帧向量为512维。轻量多输出岭分类器使用固定λ=1，含不惩罚截距。均值和总体标准差以float64计算，汇总块及拼接向量归一化后保存float32。

| 条件 | 内容 | 维度 |
| --- | --- | ---: |
| V0 | 视觉均值＋总体标准差 | 1024 |
| V1 | V0＋早／中／晚6/5/5帧均值 | 2560 |
| V1_repeat | 均值、标准差、均值、均值、均值 | 2560 |
| V1_shuffle | 固定打乱16帧后计算V1，5个扰动种子 | 2560 |
| C0 | 16个概念余弦分数的均值＋标准差 | 32 |
| C1 | C0＋概念早／中／晚均值 | 80 |
| R0 / R1 | 5组固定随机单位方向，与C0/C1同维、同汇总 | 32 / 80 |

概念序列为`C = Z Eᵀ`，E为单位文本向量。C是余弦相似度，不是概念存在概率；不进行每帧概念向量归一化、softmax或阈值化。文本和图像分支使用同一检查点，并核对参数哈希与联合空间身份。

第二轮及E1共用300段支持候选、50段开发视频。支持／开发的审核来源ID互斥，涉及119个AI辅助保守来源阻断组；这不等于完整原始来源独立性认证。官方150段最终测试保持封存。

## 当前结果

下表为固定50段开发视频上，10个支持集合的平均Macro-F1。R组先在每个支持集合内平均5次随机投影指标。完整统计解释见[结果综述](exp-e0-hmdb51-fewshot/docs/RESULTS_OVERVIEW.md)。

| 方法 | 1-shot | 3-shot | 5-shot |
| --- | ---: | ---: | ---: |
| V0 | 33.62% | 42.28% | 46.97% |
| V1 | 33.69% | 42.41% | 46.75% |
| V1_repeat | 33.98% | 42.72% | 46.82% |
| V1_shuffle | 34.13% | 42.44% | 46.73% |
| C0 | 22.89% | 25.66% | 24.90% |
| C1 | 26.35% | 29.28% | 31.16% |
| R0 | 26.32% | 31.03% | 34.71% |
| R1 | 26.44% | 31.51% | 34.71% |

主要观察：

- E0的原序V1没有稳定优于重复均值或帧序打乱。
- E1中C1优于C0，但仍低于视觉基线和随机投影平均；正的概念时间增量不能替代语义增量证据。
- 概念画面质量尚未得到人工真值验证。当前优先方向是可见性审核、概念信息保留及更严格的匹配控制，C2关系模块尚未实现。

所有结果为探索性开发评价。重复预测不是独立视频，支持种子SD不是独立来源置信区间；不能外推为溺水安全效果或部署精度。

## 仓库结构

```text
Semantic-Driven-Compositional/
├── README.md
└── exp-e0-hmdb51-fewshot/
    ├── configs/        # E0、补充消融、E1条件及概念库
    ├── src/hmdb51_e0/  # 数据审计、冻结特征、模型、指标、证据校验
    ├── scripts/        # 01—09显式阶段入口
    ├── tests/          # 数值、身份、篡改阻断和合成端到端测试
    ├── docs/           # 协议、结果综述、各阶段指南和验收记录
    ├── environment/    # 已验证环境与固定依赖
    ├── references/     # 上游数据与模型来源
    └── templates/      # 输入／输出表头样例
```

数据、环境、检查点、缓存、备份、本机路径配置和详细运行产物不随代码发布。目录说明可以保留，但其中真实内容需要在本地准备或生成。

## 环境准备

已验证的模型实验环境：Windows、Python 3.11.9、PyTorch 2.7.1＋CUDA 12.8、RTX 4080。代码要求Python≥3.11；其他平台和硬件尚未单独验收，缓存身份会记录设备与依赖差异。

```powershell
git clone https://github.com/weqdaw/Semantic-Driven-Compositional.git
Set-Location Semantic-Driven-Compositional/exp-e0-hmdb51-fewshot
py -3.11 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install --upgrade pip
& ./.venv/Scripts/python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
& ./.venv/Scripts/python.exe -m pip install -r environment/requirements-win-py311-cu128.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
& ./.venv/Scripts/python.exe -m pip check
& ./.venv/Scripts/python.exe -m pytest -q
```

本机最近一次完整实现验收为**138项测试通过**。测试使用临时合成输入验证流程，不会启动正式实验或生成真实实验分数。

## 数据与配置

自行获取HMDB51视频、官方划分和OpenAI CLIP ViT-B/32检查点，遵守各自使用条款；本仓库不重新分发这些资源。

- [HMDB51官方页面](https://serre.lab.brown.edu/hmdb51.html)
- [Serrelab数据与官方划分镜像](https://huggingface.co/datasets/Serrelab/hmdb51)
- [OpenAI CLIP](https://github.com/openai/CLIP)

CLIP检查点SHA-256：`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`。加载时必须匹配，不静默替换模型。

```powershell
Copy-Item configs/paths.local.example.yaml configs/paths.local.yaml
```

修改本机配置中的数据、标注和权重路径。该文件被Git忽略。相对输出路径以实验目录为基准。资源获取、校验和解压步骤见[环境说明](exp-e0-hmdb51-fewshot/environment/README.md)。

## 实验流程

所有命令均在`exp-e0-hmdb51-fewshot/`下执行。先查看帮助：

```powershell
& ./.venv/Scripts/python.exe scripts/01_audit_dataset.py --help
& ./.venv/Scripts/python.exe scripts/07_round2.py --help
& ./.venv/Scripts/python.exe scripts/08_round2_ablations.py --help
& ./.venv/Scripts/python.exe scripts/09_e1_concepts.py --help
```

| 阶段 | 入口与顺序 | 说明 |
| --- | --- | --- |
| E0第一轮 | scripts/01—06 | 官方数据审计、清单、帧缓存、支持抽样、拟合、汇总 |
| E0来源分组 | 07_round2.py：audit-sources → freeze-split → sample-supports → prepare-features → run → summarize | 未解决的来源审核会阻断冻结 |
| E0补充消融 | 08_round2_ablations.py：check → prepare → run → summarize | 复用60组基线，新增180组 |
| E1概念基线 | 09_e1_concepts.py：check → prepare → run → summarize | 复用60组基线，新增360组 |

E1的`check`只读核验；`prepare`编码固定文本并派生缓存，无图像重编码或分类器拟合；`run`才执行任务拟合；`summarize`检查完整网格、支持与开发身份、文件哈希、保存模型前向输出并逐预测复算指标。没有隐式启动或最终测试入口。

**新机器需先生成自己的冻结输入。** 指南中的`reports/round2_latest.json`等便捷指针属于原本机运行产物，并不随Git发布；应替换为自己生成的真实路径。代码与协议公开不等于原批次已可一键逐位复现，尤其来源审核映射及完整缓存需自行准备。

## 文档导航

- [实验主规范](exp-e0-hmdb51-fewshot/docs/EXPERIMENT_GUIDE.md)
- [数据与产物契约](exp-e0-hmdb51-fewshot/docs/DATA_CONTRACTS.md)
- [第二轮来源分组与V1](exp-e0-hmdb51-fewshot/docs/ROUND2_V1_GUIDE.md)
- [重复均值与帧序打乱](exp-e0-hmdb51-fewshot/docs/ROUND2_ABLATION_GUIDE.md)
- [E0收束记录](exp-e0-hmdb51-fewshot/docs/E0_CLOSEOUT.md)
- [E1概念基线指南](exp-e0-hmdb51-fewshot/docs/E1_CONCEPT_GUIDE.md)
- [16个概念描述](exp-e0-hmdb51-fewshot/configs/e1.concepts.v1.yaml)
- [E1实现验收](exp-e0-hmdb51-fewshot/docs/E1_VERIFICATION.md)

详细原始报告、逐视频预测、审核材料及独立分析保存在本地项目与研究知识库；公开仓库提供不含本机路径记录的阶段摘要。数据与上游软件许可证以原发布方为准，本次未替项目另行指定开源许可证。
