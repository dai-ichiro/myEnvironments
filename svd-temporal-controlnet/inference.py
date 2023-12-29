from PIL import Image
from pipeline.pipeline_stable_video_diffusion_controlnet import StableVideoDiffusionPipelineControlNet
from models.controlnet_sdv import ControlNetSDVModel
from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
from diffusers.utils import export_to_gif, load_image

def gif2images(gif_filename):
    gif=Image.open(gif_filename)
    frames=[]
    for i in range(gif.n_frames):
        gif.seek(i)
        img = gif.copy()
        frames.append(img.convert("RGB").resize((1024, 576)))
    return frames

if __name__ == "__main__":
    args = {
        "pretrained_model_name_or_path": "pre-trained/stable-video-diffusion-img2vid",
        "controlnet_name_or_path": "pre-trained/controlnet",
        "controlnet-image": "controlnet-image/depth_leres++.gif",
        "validation_image": "./0.png",
        "output_dir": "./output",
    }

    validation_control_images = gif2images(args["controlnet-image"])
    validation_image = load_image(args["validation_image"]).resize((1024, 576))

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
