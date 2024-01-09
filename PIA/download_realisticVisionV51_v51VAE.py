from huggingface_hub import hf_hub_download

repo_id="frankjoshua/realisticVisionV51_v51VAE"
folder ="."

hf_hub_download(
    repo_id=repo_id, 
    filename="realisticVisionV51_v51VAE.safetensors",
    local_dir=folder,
    local_dir_use_symlinks=False	
)
