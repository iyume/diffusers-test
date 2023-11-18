from typing import cast

from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained(
    "stablediffusionapi/anything-v5",
    use_safetensors=True,
    subfolder="unet",
    local_files_only=True,
)
unet = cast(UNet2DConditionModel, unet)

# from diffusers.loaders import CLIPTextModel
from transformers import CLIPTextModel

text_encoder = CLIPTextModel.from_pretrained(
    "stablediffusionapi/anything-v5",
    subfolder="text_encoder",
    use_safetensors=True,
    local_files_only=True,
)
text_encoder = cast(CLIPTextModel, text_encoder)

from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained(
    "stablediffusionapi/anything-v5",
    use_safetensors=True,
    subfolder="vae",
    local_files_only=True,
)
vae = cast(AutoencoderKL, vae)

from transformers import CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained(
    "stablediffusionapi/anything-v5",
    use_safetensors=True,
    subfolder="tokenizer",
    local_files_only=True,
)
tokenizer = cast(CLIPTokenizer, tokenizer)

# feature extractor is for the safety checker
from transformers import CLIPFeatureExtractor

feature_extractor = CLIPFeatureExtractor.from_pretrained(
    "stablediffusionapi/anything-v5",
    use_safetensors=True,
    subfolder="feature_extractor",
    local_files_only=True,
)
feature_extractor = cast(CLIPFeatureExtractor, feature_extractor)

from diffusers import PNDMScheduler

scheduler = PNDMScheduler.from_pretrained(
    "stablediffusionapi/anything-v5",  # type: ignore
    use_safetensors=True,
    subfolder="scheduler",
    local_files_only=True,
)
scheduler = cast(PNDMScheduler, scheduler)

# from diffusers import StableDiffusionPipeline

# # pipeline is composed of multiple submodels
# pipeline = StableDiffusionPipeline(
#     vae=vae,  # type: ignore
#     text_encoder=text_encoder,  # type: ignore
#     tokenizer=tokenizer,
#     unet=unet,  # type: ignore
#     scheduler=scheduler,
#     safety_checker=None,  # type: ignore  # TODO: fix this Optional
#     feature_extractor=feature_extractor,  # type: ignore
#     requires_safety_checker=False,
# )

# pipeline.to("cuda")

# output = pipeline("cute girl", num_inference_steps=20)
# output.images[0].save("output.png")  # type: ignore


# create own pipeline
torch_device = "cuda"
unet = unet.to(torch_device)
text_encoder = text_encoder.to(torch_device)  # type: ignore  # TODO: fix typing
vae = vae.to(torch_device)

import torch
from PIL import Image

sample_size = unet.config.sample_size  # type: ignore  # configuration_utils.FrozenDict  # TODO: fix typing
noise = torch.randn((1, 3, sample_size, sample_size)).to(torch_device)

prompt = ["cute girl"]
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 25  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance
generator = torch.manual_seed(1)  # Seed generator to create the inital latent noise
prompt_length = len(prompt)

text_input = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * prompt_length,
    padding="max_length",
    max_length=max_length,
    return_tensors="pt",
)
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

latents = torch.randn(
    (prompt_length, unet.config.in_channels, height // 8, width // 8),  # type: ignore
    generator=generator,
)
latents = latents.to(torch_device)

latents = latents * scheduler.init_noise_sigma

from tqdm.auto import tqdm

scheduler.set_timesteps(num_inference_steps)

for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)  # type: ignore

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample  # type: ignore

# scale and decode the image latents with vae
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample  # type: ignore

image = (image / 2 + 0.5).clamp(0, 1).squeeze()
image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
images = (image * 255).round().astype("uint8")
image = Image.fromarray(image)
image.save("output.png")
