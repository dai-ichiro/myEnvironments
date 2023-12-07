import numpy as np
import cv2
import torch

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor
from densepose.structures import DensePoseDataRelative
from densepose.vis.base import MatrixVisualizer

cfg = get_cfg()
add_densepose_config(cfg)

cfg.merge_from_file("./detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml")
cfg.MODEL.WEIGHTS = "model_final_162be9.pkl"

predictor = DefaultPredictor(cfg)

cap = cv2.VideoCapture("trim512.mp4")

img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

img_size = 512
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter("mask.mp4", fourcc, 25.0, (img_size, img_size))

def predict(img, predictor):
    with torch.no_grad():
        outputs = predictor(img)["instances"]
        outputs = outputs.to("cpu")
    return outputs

val_scale = 255.0 / DensePoseDataRelative.N_PART_LABELS

mask_visualizer = MatrixVisualizer(
    inplace=True, cmap=cv2.COLORMAP_VIRIDIS, val_scale=val_scale, alpha=1.0
)

while True:

    ret, img = cap.read()
    if not ret:
        break

    outputs = predict(img, predictor)
    extractor = DensePoseResultExtractor()
    data = extractor(outputs)
    densepose_result, boxes_xywh = data

    matrix_scaled_8u = np.zeros((img_height, img_width), dtype=np.uint8)
    matrix_vis = cv2.applyColorMap(matrix_scaled_8u, cv2.COLORMAP_VIRIDIS)

    for i, result in enumerate(densepose_result):
        iuv_array = torch.cat(
            (result.labels[None].type(torch.float32), result.uv * 255.0)
        ).cpu().type(torch.uint8).cpu().numpy()

        bbox_xywh = boxes_xywh.cpu().numpy()[0]
        def _extract_i_from_iuvarr(iuv_arr):
            return iuv_arr[0, :, :]

        matrix = _extract_i_from_iuvarr(iuv_array)
        segm = _extract_i_from_iuvarr(iuv_array)
        mask = np.zeros(matrix.shape, dtype=np.uint8)
        mask[segm >= 0] = 1

        mask_visualizer.visualize(matrix_vis, mask, matrix, bbox_xywh)

        break

    height, width = matrix_vis.shape[:2]
    if height > width:
        pad = (height - width) // 2
        matrix_vis = np.pad(matrix_vis, ((0, 0), (pad, pad), (0, 0)), 'edge')
    elif width > height:
        pad = (width - height) // 2
        matrix_vis = np.pad(matrix_vis, ((pad, pad), (0, 0), (0, 0)), 'edge')

    matrix_vis = cv2.resize(matrix_vis, (img_size, img_size), cv2.INTER_NEAREST)
    writer.write(matrix_vis)

writer.release()
cap.release()