import os
from huggingface_hub import snapshot_download

repo_id="damo-vilab/i2vgen-xl"
folder = os.path.basename(repo_id)

snapshot_download(
    repo_id=repo_id,
    allow_patterns=["models/*"],
    local_dir=folder,
    local_dir_use_symlinks=False
)
