from huggingface_hub import snapshot_download

repo_id="stabilityai/sd-vae-ft-mse"
folder = "Moore-AnimateAnyone/pretrained_weights/sd-vae-ft-mse"

snapshot_download(
    repo_id=repo_id,
    allow_patterns=[
        "config.json",
        "diffusion_pytorch_model.safetensors"
    ],
    local_dir=folder,
    local_dir_use_symlinks=False
)

