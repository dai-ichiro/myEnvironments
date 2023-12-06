import os
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

os.makedirs("pretrained_models", exist_ok=False)

###stable-diffusion-v1-5
sd15_repo_id="runwayml/stable-diffusion-v1-5"
sd15_folder = os.path.join("pretrained_models", os.path.basename(sd15_repo_id))

hf_hub_download(
    repo_id=sd15_repo_id, 
    filename="model_index.json",
    local_dir=sd15_folder,
    local_dir_use_symlinks=False	
)

snapshot_download(
    repo_id=sd15_repo_id,
    allow_patterns=[
        #feature_extractor
        "feature_extractor/*",
        #safety_checker
        "safety_checker/config.json",
        "safety_checker/pytorch_model.bin",
        #scheduler
        "scheduler/*",
        #text_encoder
        "text_encoder/config.json",
        "text_encoder/pytorch_model.bin",
        #tokenizer
        "tokenizer/*",
        #unet
        "unet/config.json",
        "unet/diffusion_pytorch_model.bin",
        #vae
        "vae/config.json",
        "vae/diffusion_pytorch_model.bin"
    ],
    local_dir=sd15_folder,
    local_dir_use_symlinks=False
)

###sd-vae-ft-mse
vae_repo_id="stabilityai/sd-vae-ft-mse"
vae_folder = os.path.join("pretrained_models", os.path.basename(vae_repo_id))

snapshot_download(
    repo_id=vae_repo_id,
    allow_patterns=[
        "*.safetensors",
        "*.json"
    ],
    local_dir=vae_folder,
    local_dir_use_symlinks=False
)

###MagicAnimate
magicanimate_repo_id="zcxu-eric/MagicAnimate"
magicanimate_folder = os.path.join("pretrained_models", os.path.basename(magicanimate_repo_id))

snapshot_download(
    repo_id=magicanimate_repo_id,
    allow_patterns=[
        "appearance_encoder/*",
        "densepose_controlnet/*",
        "temporal_attention/*"
    ],
    local_dir=magicanimate_folder,
    local_dir_use_symlinks=False
)