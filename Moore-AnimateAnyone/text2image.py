from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, AutoencoderKL
import torch

pipe = DiffusionPipeline.from_pretrained(
    "model/yabalMixTrue25D_v5",
    custom_pipeline="lpw_stable_diffusion",
    vae=AutoencoderKL.from_single_file("vae/vae-ft-mse-840000-ema-pruned.safetensors"),
    safety_checker=None
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True
    )
pipe.to("cuda")

prompt = "(masterpiece:1.2), (absurdres:1.2), (best quality:1.2), (looking at viewer), shiny skin, pantyhose, full body, skirt, outdoors, day, countryside, hand on hip, (white|black theme), medium breasts, happy face"
neg_prompt = "(worst quality), (low quality), (bad quality), (bad anatomy), (hat:1.2), (cap:1.2), (bag:1.2), greyscale"

generator = torch.manual_seed(2024)

image = pipe.text2img(
    prompt=prompt, 
    negative_prompt=neg_prompt,
    width=512,
    height=768,
    max_embeddings_multiples=3,
    generator=generator
).images[0]

image.save("girl.png")