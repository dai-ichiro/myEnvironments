from huggingface_hub import snapshot_download

repo_id="lambdalabs/sd-image-variations-diffusers"
folder = "Moore-AnimateAnyone/pretrained_weights"

snapshot_download(
    repo_id=repo_id,
    allow_patterns=[
        "image_encoder/*"
    ],
    local_dir=folder,
    local_dir_use_symlinks=False
)

