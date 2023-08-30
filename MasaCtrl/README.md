### CUDA 11.7 (Windows11  Python3.10  torch==2.0.1+cu117)
#### For SD 1.4 or 1.5
~~~
pip install -r https://raw.githubusercontent.com/dai-ichiro/myEnvironments/main/MasaCtrl/requirements.txt
~~~
#### For SDXL
~~~
pip install -r https://raw.githubusercontent.com/dai-ichiro/myEnvironments/main/MasaCtrl/requirements_sdxl.txt
~~~
### How to use
~~~
python masactrl_w_adapter_video.py ^
--which_cond openpose ^
--cond_path_src 0.png ^
--cond_path pose ^
--cond_inp_type openpose ^
--prompt_src "1boy, casual, outdoors, dancing" ^
--prompt "1boy, casual, outdoors, dancing" ^
--sd_ckpt models/sd-v1-4.ckpt ^
--resize_short_edge 512 ^
--cond_tau 1.0 ^
--cond_weight 1.0 ^
--n_samples 1 ^
--adapter_ckpt models/t2iadapter_openpose_sd14v1.pth
~~~

### Link to my blog
https://touch-sp.hatenablog.com/entry/2023/05/30/114132

https://touch-sp.hatenablog.com/entry/2023/05/31/125648
