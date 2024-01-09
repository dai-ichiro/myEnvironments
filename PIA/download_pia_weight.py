from huggingface_hub import hf_hub_download

repo_id="Leoxing/PIA"
folder ="."

hf_hub_download(
    repo_id=repo_id, 
    filename="pia.ckpt",
    local_dir=folder,
    local_dir_use_symlinks=False	
)
