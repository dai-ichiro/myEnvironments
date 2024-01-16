from huggingface_hub import snapshot_download

repo_id="yzd-v/DWPose"
folder = "Moore-AnimateAnyone/pretrained_weights/DWPose"

snapshot_download(
    repo_id=repo_id,
    allow_patterns=[
        "dw-ll_ucoco_384.onnx",
        "yolox_l.onnx"
    ],
    local_dir=folder,
    local_dir_use_symlinks=False
)

