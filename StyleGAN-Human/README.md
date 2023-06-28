
### CUDA 11.3

~~~
Ubuntu 20.04 on WSL2
Python 3.8
torch==1.12.1+cu113
~~~

~~~
pip install -r https://raw.githubusercontent.com/dai-ichiro/myEnvironments/main/StyleGAN-Human/requirements_cu113.txt
~~~


### CUDA 11.6

~~~
Ubuntu 20.04 on WSL2
Python 3.8
torch==1.12.1+cu116
~~~

~~~
pip install -r https://raw.githubusercontent.com/dai-ichiro/myEnvironments/main/StyleGAN-Human/requirements_cu116.txt
~~~
#### option

~~~
pip install paddlepaddle-gpu==2.4.2.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
git clone https://github.com/PaddlePaddle/PaddleSeg
cd PaddleSeg
git checkout release/2.5
pip install -U setuptools wheel
pip install -r requirements.txt
pip install -v -e .
pip install wandb==0.15.4
~~~

