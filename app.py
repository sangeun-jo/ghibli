import streamlit as st
import os
from PIL import Image, ImageDraw
import sqlite3
from datetime import datetime
import openai
from dotenv import load_dotenv
import base64
import io

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# 데이터베이스 초기화
def init_db():
    conn = sqlite3.connect('ghibli.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  original_path TEXT,
                  transformed_path TEXT,
                  created_at TIMESTAMP,
                  likes INTEGER DEFAULT 0)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS comments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  image_id INTEGER,
                  comment TEXT,
                  created_at TIMESTAMP,
                  FOREIGN KEY (image_id) REFERENCES images (id))''')
    conn.commit()
    conn.close()

# 이미지 변환 함수
def transform_image(image):
    try:
        # 원본 이미지 로드
        original_image = Image.open(image)
        
        # 이미지를 1024x1024로 리사이즈
        resized_image = original_image.resize((1024, 1024), Image.Resampling.LANCZOS)
        
        # 리사이즈된 이미지를 바이트로 변환
        img_byte_arr = io.BytesIO()
        resized_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # 마스크 이미지 생성 (1024x1024 크기의 투명 이미지)
        mask = Image.new('RGBA', (1024, 1024), (0, 0, 0, 0))
        mask_byte_arr = io.BytesIO()
        mask.save(mask_byte_arr, format='PNG')
        mask_byte_arr = mask_byte_arr.getvalue()
        
        # 지브리 스타일 프롬프트 생성
        prompt = """Transform this image into a Studio Ghibli style character illustration, specifically following the style of Hayao Miyazaki's character designs:
        - Character should look like it's from Spirited Away, Howl's Moving Castle, or Princess Mononoke
        - Large, expressive eyes with detailed highlights and soft edges
        - Rounded, gentle facial features with small nose and mouth
        - Hair should have Ghibli's signature flowing style with soft edges
        - Clothing should have Ghibli's characteristic watercolor-like texture
        - Keep the same pose and expression but in Ghibli's style
        - Background should be simple and blurred
        - Overall style should match Ghibli's signature aesthetic with soft colors and gentle lighting"""
        
        # OpenAI API를 사용하여 이미지 편집
        response = openai.Image.create_edit(
            image=img_byte_arr,
            mask=mask_byte_arr,
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        return response['data'][0]['url']
    except Exception as e:
        st.error(f"이미지 변환 중 오류가 발생했습니다: {str(e)}")
        return None

# 메인 앱
def main():
    st.title("🎨 지브리 스타일 이미지 변환기")
    
    # 데이터베이스 초기화
    init_db()
    
    # 사이드바
    st.sidebar.title("메뉴")
    menu = st.sidebar.radio("선택하세요", ["이미지 변환", "갤러리"])
    
    if menu == "이미지 변환":
        st.header("이미지 업로드")
        uploaded_file = st.file_uploader("이미지를 선택하세요", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="원본 이미지", use_column_width=True)
            
            if st.button("지브리 스타일로 변환"):
                with st.spinner("변환 중..."):
                    transformed_url = transform_image(uploaded_file)
                    if transformed_url:
                        st.image(transformed_url, caption="변환된 이미지", use_column_width=True)
                        
                        # 데이터베이스에 저장
                        conn = sqlite3.connect('ghibli.db')
                        c = conn.cursor()
                        c.execute("INSERT INTO images (original_path, transformed_path, created_at) VALUES (?, ?, ?)",
                                (uploaded_file.name, transformed_url, datetime.now()))
                        conn.commit()
                        conn.close()
                        
                        st.success("이미지가 성공적으로 변환되었습니다!")
    
    else:  # 갤러리
        st.header("갤러리")
        
        # 데이터베이스에서 이미지 불러오기
        conn = sqlite3.connect('ghibli.db')
        c = conn.cursor()
        c.execute("SELECT * FROM images ORDER BY created_at DESC")
        images = c.fetchall()
        
        for img in images:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.image(img[2], use_column_width=True)
                with col2:
                    if st.button("❤️", key=f"like_{img[0]}"):
                        c.execute("UPDATE images SET likes = likes + 1 WHERE id = ?", (img[0],))
                        conn.commit()
                        st.rerun()
                    
                    st.write(f"좋아요: {img[4]}")
                    
                    # 댓글 섹션
                    st.text_area("댓글을 남겨주세요", key=f"comment_{img[0]}")
                    if st.button("댓글 작성", key=f"submit_{img[0]}"):
                        comment = st.session_state[f"comment_{img[0]}"]
                        if comment:
                            c.execute("INSERT INTO comments (image_id, comment, created_at) VALUES (?, ?, ?)",
                                    (img[0], comment, datetime.now()))
                            conn.commit()
                            st.rerun()
                    
                    # 댓글 표시
                    c.execute("SELECT comment, created_at FROM comments WHERE image_id = ? ORDER BY created_at DESC", (img[0],))
                    comments = c.fetchall()
                    for comment in comments:
                        st.text(f"💬 {comment[0]}")
        
        conn.close()

if __name__ == "__main__":
    main() 