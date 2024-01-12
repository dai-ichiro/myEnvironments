from huggingface_hub import snapshot_download

repo_id="patrolli/AnimateAnyone"
folder = "Moore-AnimateAnyone/pretrained_weights"

snapshot_download(
    repo_id=repo_id,
    allow_patterns=[
        "*.pth"
    ],
    local_dir=folder,
    local_dir_use_symlinks=False
)

