from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import AutoImageProcessor
from huggingface_hub import login
import os

def download_models():
    # 모델 저장 경로 설정
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    print("모델 다운로드를 시작합니다...")
    
    # ControlNet 모델 다운로드
    print("ControlNet 모델 다운로드 중...")
    controlnet_canny = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        cache_dir=os.path.join(model_dir, "controlnet-canny")
    )
    
    # Stable Diffusion 모델 다운로드
    print("Stable Diffusion 모델 다운로드 중...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet_canny,
        cache_dir=os.path.join(model_dir, "stable-diffusion-v1-5")
    )
    
    # 이미지 프로세서 다운로드
    print("이미지 프로세서 다운로드 중...")
    processor = AutoImageProcessor.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        cache_dir=os.path.join(model_dir, "controlnet-canny-processor")
    )
    
    print("모든 모델 다운로드가 완료되었습니다!")

if __name__ == "__main__":
    # Hugging Face 토큰이 있다면 로그인
    if os.getenv("HUGGING_FACE_HUB_TOKEN"):
        login(token=os.getenv("HUGGING_FACE_HUB_TOKEN"))
    download_models() 