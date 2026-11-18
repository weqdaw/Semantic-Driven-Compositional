# 仓库地图与文件职责

本图最初用于骨架交付。当前六阶段入口和各模块已实现，新增pipeline.py、cli.py、pyproject.toml及实际tests；运行方式见README，真实状态见E0_RUN_RECORD.json。原有空文件描述仅代表历史骨架。

## 结构

```text
exp-e0-hmdb51-fewshot/
├── README.md
├── AGENTS.md
├── .gitignore
├── configs/
│   ├── README.md
│   ├── paths.local.example.yaml
│   └── e0.v0.example.yaml
├── docs/
│   ├── EXPERIMENT_GUIDE.md
│   ├── REPOSITORY_MAP.md
│   ├── DATA_INVENTORY.md
│   ├── DATA_CONTRACTS.md
│   ├── ACCEPTANCE_CHECKLIST.md
│   ├── EXPERIMENT_LOG.md
│   ├── SCAFFOLD_VERIFICATION.md
│   └── superpowers/
│       ├── specs/2026-11-16-e0-scaffold-design.md
│       └── plans/2026-11-16-e0-implementation-plan.md
├── environment/README.md
├── references/README.md
├── templates/              # 仅表头的.csv.example格式样例
├── scripts/                # 6个空入口及说明
├── src/hmdb51_e0/          # 12个空模块及说明
├── tests/README.md
├── data/
│   ├── README.md
│   ├── annotations/
│   ├── manifests/
│   └── support_sets/
├── artifacts/
│   ├── README.md
│   ├── weights/
│   └── features/
├── runs/README.md
└── reports/
    ├── README.md
    ├── figures/
    └── tables/
```

## 数据流

外部原视频＋官方split → 清单／差异审计 → 固定开发与支持池 → 逐视频冻结特征 → 每种子K-shot训练 → 开发预测 → 聚合报告。

全流程未实现。annotations/manifests/support_sets/features/runs/figures/tables当前仅有目录占位，没有真实产物。

## 数据与版本控制

- 原始视频只引用外部F盘目录，仓库中不复制、链接或重新解压。
- 大文件和生成内容由.gitignore排除；.gitkeep用于保留目录。
- 以后需要公开复现的小型清单，先审查数据使用条件和路径信息，再选择单独受控版本化；本轮不自动发布。
- paths.local.yaml只在本机使用；路径样例可跟踪。
- 没有创建LICENSE、远端或Git提交，也没有假定用户的发布许可。
