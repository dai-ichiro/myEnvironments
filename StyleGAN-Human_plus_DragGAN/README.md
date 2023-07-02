
### CUDA 11.7.1 + cuDNN 8.5.0

~~~
Ubuntu 22.04 on WSL2
Python 3.10
torch==2.0.1+cu117
~~~

~~~
pip install -U setuptools wheel
pip install -r https://raw.githubusercontent.com/dai-ichiro/myEnvironments/main/StyleGAN-Human_plus_DragGAN/requirements_cu117_torch201.txt
pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
git clone https://github.com/PaddlePaddle/PaddleSeg
cd PaddleSeg
git checkout release/2.5
pip install -v -e .
~~~
