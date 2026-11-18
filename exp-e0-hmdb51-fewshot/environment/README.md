# 已核验环境

本次使用Windows、Python3.11.9独立`.venv`、RTX4080（16GB）、驱动576.02、PyTorch2.7.1+cu128、torchvision0.22.1+cu128、OpenCV FFmpeg4.12.0、NumPy2.2.6、Pillow11.3.0。冻结编码为float32、确定性算法、关闭TF32、batch_size=16；实际依赖见[锁文件](requirements-win-py311-cu128.lock.txt)。pip check无损坏依赖。

原规范建议Python3.12，本机注册的3.12无法启动，故使用3.11.9，未修改全局环境。其他硬件环境可能不逐位相同；设备和依赖纳入缓存身份。

本机已完成安装。新机器在仓库根目录执行PowerShell：

```powershell
py -3.11 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install --upgrade pip
& ./.venv/Scripts/python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
& ./.venv/Scripts/python.exe -m pip install -r environment/requirements-win-py311-cu128.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
& ./.venv/Scripts/python.exe -m pip check
& ./.venv/Scripts/python.exe -m pytest -q
```

CLIP固定作者提交`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`。权重来自[OpenAI作者发布](https://github.com/openai/CLIP)，ViT-B/32的SHA-256为`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`，加载前强制校验。

官方划分来自[Serrelab发布](https://huggingface.co/datasets/Serrelab/hmdb51)，RAR的SHA-256为`229c94f845720d01eb3946d39f39292ea962d50a18136484aa47c1eba251d2b7`，MD5为`15e67781e70dcfbdce2d7dbb9b3344b5`（与Torchvision作者实现一致）。下载并用本机实际可用7z解压：

```powershell
$ProgressPreference = 'SilentlyContinue'
Invoke-WebRequest -Uri 'https://huggingface.co/datasets/Serrelab/hmdb51/resolve/main/test_train_splits.rar' -OutFile data/annotations/test_train_splits.rar
Get-FileHash -Algorithm SHA256 data/annotations/test_train_splits.rar
7z x data/annotations/test_train_splits.rar '-odata/annotations' -y
Invoke-WebRequest -Uri 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt' -OutFile artifacts/weights/ViT-B-32.pt
Get-FileHash -Algorithm SHA256 artifacts/weights/ViT-B-32.pt
Copy-Item configs/paths.local.example.yaml configs/paths.local.yaml
```

将annotations_root填为`data/annotations/testTrainMulti_7030_splits`，weights_file填为`artifacts/weights/ViT-B-32.pt`，再按README运行。venv、权重、缓存和本机路径均忽略Git。
