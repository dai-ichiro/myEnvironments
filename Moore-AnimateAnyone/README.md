# Environtment
```text
Windows 11
CUDA 11.8
Python 3.10
```
# Installation
```text
pip install -r https://raw.githubusercontent.com/dai-ichiro/myEnvironments/main/Moore-AnimateAnyone/requirements.txt
```
# Download weights and scripts
```text
git clone https://github.com/MooreThreads/Moore-AnimateAnyone
python download_animateanyone_weights.py
python download_image_encoder.py
python download_vae.py
python download_sd15_pipeline.py
```
Download dwpose weights (`dw-ll_ucoco_384.onnx`, `yolox_l.onnx`) following [this](https://github.com/IDEA-Research/DWPose?tab=readme-ov-file#-dwpose-for-controlnet).
# Link to my blog
[【Moore-AnimateAnyone】1枚の画像とポーズ動画から動画を作成する](https://touch-sp.hatenablog.com/entry/2024/01/13/004949)
