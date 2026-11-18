# 来源与引用

- HMDB51官方数据与划分：https://serre.lab.brown.edu/hmdb51.html
- HMDB原始论文：H. Kuehne等，HMDB: A large video database for human motion recognition，ICCV 2011。
- CLIP论文：https://proceedings.mlr.press/v139/radford21a.html
- CLIP作者代码／权重说明：https://github.com/openai/CLIP
- Torchvision的HMDB51划分标签实现：https://github.com/pytorch/vision/blob/main/torchvision/datasets/hmdb51.py

实现阶段已取得[Serrelab官方划分](https://huggingface.co/datasets/Serrelab/hmdb51)，RAR MD5与[Torchvision作者实现](https://github.com/pytorch/vision/blob/main/torchvision/datasets/hmdb51.py)一致；已取得OpenAI官方ViT-B/32权重并核验SHA-256。来源、版本和哈希见E0_RUN_RECORD及environment/README。历史Brown下载链接已无法返回有效RAR，未将HTML错误页作为标注使用。数据条件以实际来源说明为准。

上游研究设计保存在作者本地研究知识库。E0文档自包含，不要求运行时访问Obsidian或Zotero。E0将原先本地水域二分类改为公开五类片段识别；不复用水域专用概念表或8秒固定窗。
