import cv2
import numpy as np
import argparse
import yaml

joints = ['neck', 'right_shoulder','right_elbow', 'right_hand', 'left_shoulder', 'left_elbow', 'left_hand', 'right_hip', 'right_knee', 'right_foot', 'left_hip', 'left_knee', 'left_foot']

# function to display the coordinates of
# of the points clicked on the image 
def click_event(event, x, y, flags, params):

    global img, joints_number, joints_dict
  
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        
        joints_dict[joints[joints_number]] = np.array((x, y))
        joints_number += 1

        cv2.circle(img,
        center=(x, y),
        radius=3,
        color=(255, 0, 0),
        thickness=-1)
        cv2.imshow('image', img)

        if joints_number < len(joints):
            print(f'select {joints[joints_number]}')
        else:
            print('finish')
    
if __name__=="__main__":
  
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        help="original image"
    )
    opt = parser.parse_args()
    img_path = opt.image

    joints_number = 0
    joints_dict  = {}

    # reading the image
    img = cv2.imread(img_path)
    
    height, width, _ = img.shape

    # displaying the image
    cv2.imshow('image', img)
    print(f'select {joints[joints_number]}')

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    # wait for a key to be pressed to exit
    while True:  
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if joints_number == len(joints):
            break
    # close the window
    cv2.destroyAllWindows()
    print(joints_dict)

    skeleton = []
    skeleton.append({'loc' : [round(x) for x in (joints_dict['right_hip']+joints_dict['left_hip'])/2          ], 'name': 'root'          , 'parent': None})
    skeleton.append({'loc' : [round(x) for x in (joints_dict['right_hip']+joints_dict['left_hip'])/2          ], 'name': 'hip'           , 'parent': 'root'})
    skeleton.append({'loc' : [round(x) for x in (joints_dict['right_shoulder']+joints_dict['left_shoulder'])/2], 'name': 'torso'         , 'parent': 'hip'})
    skeleton.append({'loc' : [round(x) for x in  joints_dict['neck']                                          ], 'name': 'neck'          , 'parent': 'torso'})
    skeleton.append({'loc' : [round(x) for x in  joints_dict['right_shoulder']                                ], 'name': 'right_shoulder', 'parent': 'torso'})
    skeleton.append({'loc' : [round(x) for x in  joints_dict['right_elbow']                                   ], 'name': 'right_elbow'   , 'parent': 'right_shoulder'})
    skeleton.append({'loc' : [round(x) for x in  joints_dict['right_hand']                                    ], 'name': 'right_hand'    , 'parent': 'right_elbow'})
    skeleton.append({'loc' : [round(x) for x in  joints_dict['left_shoulder']                                 ], 'name': 'left_shoulder' , 'parent': 'torso'})
    skeleton.append({'loc' : [round(x) for x in  joints_dict['left_elbow']                                    ], 'name': 'left_elbow'    , 'parent': 'left_shoulder'})
    skeleton.append({'loc' : [round(x) for x in  joints_dict['left_hand']                                     ], 'name': 'left_hand'     , 'parent': 'left_elbow'})
    skeleton.append({'loc' : [round(x) for x in  joints_dict['right_hip']                                     ], 'name': 'right_hip'     , 'parent': 'root'})
    skeleton.append({'loc' : [round(x) for x in  joints_dict['right_knee']                                    ], 'name': 'right_knee'    , 'parent': 'right_hip'})
    skeleton.append({'loc' : [round(x) for x in  joints_dict['right_foot']                                    ], 'name': 'right_foot'    , 'parent': 'right_knee'})
    skeleton.append({'loc' : [round(x) for x in  joints_dict['left_hip']                                      ], 'name': 'left_hip'      , 'parent': 'root'})
    skeleton.append({'loc' : [round(x) for x in  joints_dict['left_knee']                                     ], 'name': 'left_knee'     , 'parent': 'left_hip'})
    skeleton.append({'loc' : [round(x) for x in  joints_dict['left_foot']                                     ], 'name': 'left_foot'     , 'parent': 'left_knee'})   

    # create the character config dictionary
    char_cfg = {'skeleton': skeleton, 'height': height, 'width': width}

    # dump character config to yaml
    with open('char_cfg.yaml', 'w') as f:
        yaml.dump(char_cfg, f)
        
        
        
