import streamlit as st
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import cv2
import numpy as np
import io

# 페이지 설정
st.set_page_config(
    page_title="기블리 스타일 변환기",
    page_icon="🎨",
    layout="wide"
)

# 제목
st.title("🎨 기블리 스타일 이미지 변환기")
st.markdown("### 당신의 사진을 기블리 스타일로 변환해보세요!")

# 모델 로드
@st.cache_resource
def load_models():
    controlnet_canny = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
    )
    controlnet_depth = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
    )
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "nitrosocke/Ghibli-Diffusion",
        controlnet=[controlnet_canny, controlnet_depth],
        torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.to("cuda")
    return pipe

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
    "traditional clothing, historical costume, "
    "formal wear, fancy dress, outdated fashion, "
    "angular face, harsh lines, pointed features, "
    "photorealistic, small head, elongated face, "
    "distorted, ugly, deformed, low quality, blurry, "
    "oversaturated colors, harsh shadows"
)

prompt = (
    "Studio Ghibli style, high quality anime art, "
    "large cute round eyes, soft gentle eye expression, "
    "rounded cute face, soft circular features, "
    "slightly enlarged head proportions, "
    "modern casual clothing, contemporary fashion, "
    "urban casual style, comfortable modern outfit, "
    "clean line art, Miyazaki style character design, "
    "soft color palette, detailed cel shading"
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
    if st.button("기블리 스타일로 변환하기"):
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
    2. '기블리 스타일로 변환하기' 버튼을 클릭합니다.
    3. 변환이 완료되면 결과 이미지를 확인하고 다운로드할 수 있습니다.
    
    **참고사항:**
    - 변환에는 몇 분 정도 소요될 수 있습니다.
    - 최적의 결과를 위해 얼굴이 잘 보이는 이미지를 사용해주세요.
    - 이미지는 자동으로 512x512 크기로 조정됩니다.
    """) 