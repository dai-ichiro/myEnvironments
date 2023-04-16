import cv2
import argparse
from copy import deepcopy

def nothing(x):
    pass

# function to display the coordinates of
# of the points clicked on the image 
def click_event(event, x, y, flags, params):

    global img, input_point, input_label
  
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        if cv2.getTrackbarPos('0:foreground 1:background','image') == 0:
            input_point.append([x, y])
            input_label.append(1)

            cv2.circle(img,
            center=(x, y),
            radius=3,
            color=(255, 0, 0),
            thickness=-1)
            cv2.imshow('image', img)

        else:
            input_point.append([x, y])
            input_label.append(0)
            
            cv2.circle(img,
            center=(x, y),
            radius=3,
            color=(0, 0, 255),
            thickness=-1)
            cv2.imshow('image', img)

if __name__=="__main__":
  
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image',
        type=str,
        help='original image'
    )
    opt = parser.parse_args()
    img_path = opt.image

    # reading the image
    img = cv2.imread(img_path)
    original_img = deepcopy(img_path)

    input_point = []
    input_label = []

    # displaying the image
    cv2.imshow('image', img)
    #cv2.createTrackbar('0:foreground 1:background','image',0 , 1, nothing)

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
    print(input_point)
    print(input_label)