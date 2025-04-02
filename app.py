from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import HEDdetector
from diffusers.utils import load_image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

# 1️⃣ ControlNet 모델 불러오기 (선 강조 + 깊이 감지)
controlnet_canny = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)
controlnet_depth = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
)

# 2️⃣ Stable Diffusion + ControlNet 로드
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "nitrosocke/Ghibli-Diffusion",
    controlnet=[controlnet_canny, controlnet_depth],
    torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.to("cuda")  

# image_path = "/kaggle/input/testtest/me_exmini.jpg"  # 변환할 이미지 경로
# image_path = "/kaggle/input/actor-jpg/AA.21463414.1.jpg"  # 변환할 이미지 경로g
image_path = "/kaggle/input/testsetst5/trump.jpg"  # 변환할 이미지 경로g
original_image = Image.open(image_path).convert("RGB")
original_image = original_image.resize((512, 512))  # 모델에 맞게 크기 조정
negative_prompt = (
    "realistic proportions, small eyes, sharp features, "
    "traditional clothing, historical costume, "  # 전통적/과거 의상 제거
    "formal wear, fancy dress, outdated fashion, "  # 격식있는 의상 제거
    "angular face, harsh lines, pointed features, "
    "photorealistic, small head, elongated face, "
    "distorted, ugly, deformed, low quality, blurry, "
    "oversaturated colors, harsh shadows"
)

import cv2
def apply_canny(image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)

control_image_canny = apply_canny(original_image)


prompt = (
    "Studio Ghibli style, high quality anime art, "
    "large cute round eyes, soft gentle eye expression, "  # 큰 눈
    "rounded cute face, soft circular features, "  # 둥근 얼굴
    "slightly enlarged head proportions, "  # 큰 머리
    "modern casual clothing, contemporary fashion, "  # 현대 의상 강조
    "urban casual style, comfortable modern outfit, "
    "clean line art, Miyazaki style character design, "
    "soft color palette, detailed cel shading"
)



# Ghibli 스타일 적용
result = pipe(
    prompt,
    image=[control_image_canny, original_image],  # ControlNet용 선화 입력
    num_inference_steps=30,
    guidance_scale=7.0,
    controlnet_conditioning_scale=[0.65, 0.35] ,
    negative_prompt=negative_prompt,
    # positive_prompt=positive_prompt
).images[0]

import matplotlib.pyplot as plt
plt.imshow(result)
plt.axis("off")  # 축 제거
plt.show()