### WSL2 + CUDA 11.8
~~~
Ubuntu 22.04 on WSL2
Python 3.10
torch==2.0.1+cu118
~~~

~~~
pip install -r https://raw.githubusercontent.com/dai-ichiro/myEnvironments/main/magic-animate/requirements_cu118.txt
~~~

~~~
git clone https://github.com/magic-research/magic-animate
cd magic-animate
wget https://raw.githubusercontent.com/dai-ichiro/myEnvironments/main/magic-animate/download_model.py
python download_model.py
~~~
