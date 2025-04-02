import streamlit as st
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import cv2
import numpy as np
import io
from controlnet_aux import CannyDetector
from huggingface_hub import login
import os
from datetime import datetime
import pathlib

# Hugging Face 토큰으로 로그인
if 'HUGGING_FACE_HUB_TOKEN' in st.secrets:
    login(token=st.secrets['HUGGING_FACE_HUB_TOKEN'])

# 페이지 설정
st.set_page_config(
    page_title="짭브리 스타일 변환기",
    page_icon="🎨",
    layout="wide"
)

# 제목
st.title("🎨 짭브리 스타일 이미지 변환기")
st.markdown("### 당신의 사진을 짭브리 스타일로 변환해보세요!")

# 모델 로드
@st.cache_resource
def load_models():
    # 운영체제 독립적인 경로 처리
    model_dir = pathlib.Path("models")
    if not model_dir.exists():
        st.error("모델이 다운로드되지 않았습니다. 먼저 download_models.py를 실행해주세요.")
        return None
    
    try:
        # ControlNet 모델 로드
        controlnet_canny = ControlNetModel.from_pretrained(
            str(model_dir / "controlnet-canny")
        )
        
        # Stable Diffusion 파이프라인 로드
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            str(model_dir / "stable-diffusion-v1-5"),
            controlnet=controlnet_canny,
            torch_dtype=torch.float16
        )
        
        # 스케줄러 설정
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        
        # GPU 사용 설정
        pipe = pipe.to("cuda")
        
        return pipe
    except Exception as e:
        st.error(f"모델 로드 중 오류가 발생했습니다: {str(e)}")
        return None

# 이미지 전처리 함수
def apply_canny(image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)

# 프롬프트 설정
negative_prompt = (
    "realistic proportions, small eyes, sharp features, "
    "traditional clothing, historical costume, "  # 전통적/과거 의상 제거
    "formal wear, fancy dress, outdated fashion, "  # 격식있는 의상 제거
    "angular face, harsh lines, pointed features,Wrinkles, "
    "photorealistic, small head, elongated face, "
    "distorted, angry, ugly, deformed, low quality, blurry, "
    "oversaturated colors, harsh shadows"
)

prompt = (
    "realistic proportions, small eyes, sharp features, "
    "traditional clothing, historical costume, "  # 전통적/과거 의상 제거
    "formal wear, fancy dress, outdated fashion, "  # 격식있는 의상 제거
    "angular face, harsh lines, pointed features,Wrinkles, "
    "photorealistic, small head, elongated face, "
    "distorted, angry, ugly, deformed, low quality, blurry, "
    "oversaturated colors, harsh shadows"
)

# 파일 업로더
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 원본 이미지 표시
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("원본 이미지")
        original_image = Image.open(uploaded_file).convert("RGB")
        original_image = original_image.resize((512, 512))
        st.image(original_image, use_column_width=True)
    
    # 변환 버튼
    if st.button("짭브리 스타일로 변환하기"):
        with st.spinner("이미지를 변환하는 중입니다..."):
            # 모델 로드
            pipe = load_models()
            
            # 이미지 전처리
            control_image_canny = apply_canny(original_image)
            
            # 이미지 생성
            result = pipe(
                prompt,
                image=[control_image_canny, original_image],
                num_inference_steps=30,
                guidance_scale=7.0,
                controlnet_conditioning_scale=[0.65, 0.35],
                negative_prompt=negative_prompt,
            ).images[0]
            
            # 결과 이미지 표시
            with col2:
                st.subheader("변환된 이미지")
                st.image(result, use_column_width=True)
                
                # 다운로드 버튼
                buf = io.BytesIO()
                result.save(buf, format='PNG')
                st.download_button(
                    label="변환된 이미지 다운로드",
                    data=buf.getvalue(),
                    file_name="ghibli_style.png",
                    mime="image/png"
                )

# 사용 방법 안내
with st.expander("사용 방법"):
    st.markdown("""
    1. '이미지 업로드' 버튼을 클릭하여 변환하고 싶은 이미지를 선택합니다.
    2. '짭브리 스타일로 변환하기' 버튼을 클릭합니다.
    3. 변환이 완료되면 결과 이미지를 확인하고 다운로드할 수 있습니다.
    
    **참고사항:**
    - 변환에는 몇 분 정도 소요될 수 있습니다.
    - 최적의 결과를 위해 얼굴이 잘 보이는 이미지를 사용해주세요.
    - 이미지는 자동으로 512x512 크기로 조정됩니다.
    """) 