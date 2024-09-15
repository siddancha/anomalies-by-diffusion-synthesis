
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained(
    'results/cityscapes_diffusers/empty_caption',
).to('cuda')

for _ in range(5):
    pipeline(prompt='', height=512, width=1024).images[0].show()
