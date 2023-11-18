from typing import cast

from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stablediffusionapi/anything-v5",
    use_safetensors=True,
    local_files_only=True,
    # requires_safety_checker=False,  # this parameter will be passed into pipeline __init__
)
print("Pipeline:", pipeline)
print("Model Type:", pipeline.__class__)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

pipeline = cast(StableDiffusionPipeline, pipeline)

print("Safety Checker:", pipeline.safety_checker.__class__)
pipeline.safety_checker = None  # disable NSFW checker
pipeline.to("cuda")
output = pipeline("cute girl, bare breast")

print("Output Type:", output.__class__)
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)

output = cast(StableDiffusionPipelineOutput, output)
output.images[0].save("output.png")

# try to change the scheduler
from diffusers import EulerDiscreteScheduler

output = pipeline("cute girl")
output.images[0].save("output-a.png")  # type: ignore

pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
output = pipeline("cute girl")
output.images[0].save("output-b.png")  # type: ignore
