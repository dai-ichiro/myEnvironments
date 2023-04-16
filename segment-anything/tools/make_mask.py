import numpy as np
import cv2
from segment_anything import SamPredictor, sam_model_registry

image = cv2.imread('result2.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam = sam_model_registry['default'](checkpoint='checkpoints/default.pth')
sam.to(device='cuda')
predictor = SamPredictor(sam)
predictor.set_image(image)

height, width, _ = image.shape

input_point = np.array([[int(width/2), int(height/2)]])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

## masks:    type:numpy.ndarray, shape:(number_of_masks) x H x W
## scores:  type:numpy.ndarray, shape:(number_of_masks),
## logits:  type:numpy.ndarray, shape:(number_of_masks) x 255 x 255

from PIL import Image
for i, array in enumerate(masks):
    pil = Image.fromarray(array)
    pil.save(f'mask_{i}.png')