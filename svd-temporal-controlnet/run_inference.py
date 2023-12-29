import os
import numpy as np
from PIL import Image
from pipeline.pipeline_stable_video_diffusion_controlnet import StableVideoDiffusionPipelineControlNet
from models.controlnet_sdv import ControlNetSDVModel
from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
import cv2
import re
from diffusers.utils import export_to_gif, load_image

def load_images_from_folder_to_pil(folder, target_size=(512, 512)):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    def frame_number(filename):
        matches = re.findall(r'\d+', filename)  # Find all sequences of digits in the filename
        if matches:
            if matches[-1] == '0000' and len(matches) > 1:
                return int(matches[-2])  # Return the second-to-last sequence if the last is '0000'
            return int(matches[-1])  # Otherwise, return the last sequence
        return float('inf')  # Return 'inf'

    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder), key=frame_number)

    # Load, resize, and convert images
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read image with original channels
            if img is not None:
                # Resize image
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

                # Convert to uint8 if necessary
                if img.dtype == np.uint16:
                    img = (img / 256).astype(np.uint8)

                # Ensure all images are in RGB format
                if len(img.shape) == 2:  # Grayscale image
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif len(img.shape) == 3 and img.shape[2] == 3:  # Color image in BGR format
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Convert the numpy array to a PIL image
                pil_img = Image.fromarray(img)
                images.append(pil_img)

    return images

if __name__ == "__main__":
    args = {
        "pretrained_model_name_or_path": "pre-trained/stable-video-diffusion-img2vid",
        "controlnet_name_or_path": "pre-trained/controlnet",
        "validation_control_folder": "./validation_demo/depth",
        "validation_image": "./validation_demo/chair.png",
        "output_dir": "./output",
        "height": 512,
        "width": 512,
        # cant be bothered to add the args in myself, just use notepad
    }

    # Load control images and validation image
    validation_control_images = load_images_from_folder_to_pil(args["validation_control_folder"])
    validation_image = load_image(args["validation_image"])

    # Load and set up the pipeline
    controlnet = controlnet = ControlNetSDVModel.from_pretrained(args["controlnet_name_or_path"])
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(args["pretrained_model_name_or_path"], subfolder="unet")
    
    pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(
        args["pretrained_model_name_or_path"],
        controlnet=controlnet,
        unet=unet
    )
    
    pipeline.enable_model_cpu_offload()

    video_frames = pipeline(
        validation_image,
        validation_control_images[:14],
        decode_chunk_size=8,
        num_frames=14,
        motion_bucket_id=100,
        controlnet_cond_scale=1.0
    ).frames[0]

    export_to_gif(video_frames, "generated.gif")
