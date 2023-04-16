import cv2
import numpy as np
import argparse
from copy import deepcopy
from segment_anything import SamPredictor, sam_model_registry

def nothing(x):
    pass

# function to display the coordinates of
# of the points clicked on the image 
def click_event(event, x, y, flags, params):

    global img, input_points, input_labels
  
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        if cv2.getTrackbarPos('0:foreground 1:background','image') == 0:
            input_points.append([x, y])
            input_labels.append(1)

            cv2.circle(img,
            center=(x, y),
            radius=3,
            color=(255, 0, 0),
            thickness=-1)
            cv2.imshow('image', img)

        else:
            input_points.append([x, y])
            input_labels.append(0)
            
            cv2.circle(img,
            center=(x, y),
            radius=3,
            color=(0, 0, 255),
            thickness=-1)
            cv2.imshow('image', img)

def seg_anythings(image:np.ndarray, model:str, checkpoint:str, points:list, labels:list) -> None:
    sam = sam_model_registry[model](checkpoint=checkpoint)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=np.array(points),
        point_labels=np.array(labels),
        multimask_output=False)
    
    from PIL import Image
    pil = Image.fromarray(masks[0])
    pil.save('mask.png')
    
if __name__=="__main__":
  
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        help="original image"
    )
    parser.add_argument(
        "--model",
        type=str,
        default='default',
        choices=['default', 'vit_h', 'vit_l', 'vit_b'],
        help="the type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/default.pth",
        help="the path to the SAM checkpoint to use for mask generation."
    )
    opt = parser.parse_args()
    img_path = opt.image

    # reading the image
    img = cv2.imread(img_path)
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_points = []
    input_labels = []

    # displaying the image
    cv2.imshow('image', img)
    cv2.createTrackbar('0:foreground 1:background','image',0 , 1, nothing)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    # wait for a key to be pressed to exit
    while True:  
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    # close the window
    cv2.destroyAllWindows()
    print(input_points)
    print(input_labels)

    seg_anythings(
        image=original_img,
        model=opt.model,
        checkpoint=opt.checkpoint,
        points=input_points,
        labels=input_labels)