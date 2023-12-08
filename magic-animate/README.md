### Environment
~~~
Ubuntu 22.04 on WSL2
Python 3.10
torch==2.0.1+cu118
~~~

### Installation
~~~
pip install -r https://raw.githubusercontent.com/dai-ichiro/myEnvironments/main/magic-animate/requirements_cu118.txt
~~~
#### option (for Detectron2-DensePose)
~~~
pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose
~~~

### Preparation
~~~
git clone https://github.com/magic-research/magic-animate
cd magic-animate
wget https://raw.githubusercontent.com/dai-ichiro/myEnvironments/main/magic-animate/download_model.py
python download_model.py
~~~

### How to make densepose motion video from original video
very simple!!
~~~
python video2denseposevideo.py --input dance26.mp4
~~~

### Link to my blog
https://touch-sp.hatenablog.com/entry/2023/12/06/121903

https://touch-sp.hatenablog.com/entry/2023/12/07/221215
