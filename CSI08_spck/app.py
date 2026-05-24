import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import google.protobuf.runtime_version
google.protobuf.runtime_version.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import tensorflow as tf
import google.generativeai as genai

st.set_page_config(page_title="AI Math Drawer", page_icon="✍️", layout="wide")
st.title(" Vẽ Phương Trình - AI Giải Toán")

@st.cache_resource
def load_cnn_model():
    try:
        model = tf.keras.models.load_model('model_toan_hoc.h5')
        return model
    except:
        return None

cnn_model = load_cnn_model()

with st.sidebar:
    st.header("Cài đặt API")
    api_key = st.text_input("Nhập Gemini API Key:", type="password")
    
st.write("Vẽ phép tính của bạn vào bảng đen bên dưới:")
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=500,
    width=1200,
    drawing_mode="freedraw",
    key="canvas",
)

def extract_and_predict(image_data, model):
    gray = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)
    
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return ""
        
    rects = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 10]
    rects.sort(key=lambda b: b[0])
    
    equation_str = ""
    
    st.write(" **Ảnh đã được xử lý (Gửi cho CNN):**")
    cols = st.columns(len(rects))
    
    for idx, (x, y, w, h) in enumerate(rects):
        pad = 10
        roi = gray[max(0, y-pad):min(gray.shape[0], y+h+pad), 
                   max(0, x-pad):min(gray.shape[1], x+w+pad)]
        
        if roi.size == 0: continue
        
        side = max(roi.shape)
        square = np.zeros((side, side), dtype=np.uint8)
        
        x_offset = (side - roi.shape[1]) // 2
        y_offset = (side - roi.shape[0]) // 2
        square[y_offset:y_offset+roi.shape[0], x_offset:x_offset+roi.shape[1]] = roi
        
        resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        
        final_img = cv2.bitwise_not(resized)
        
        cols[idx].image(final_img, width=40)
        
        cnn_input = np.reshape(final_img, (1, 28, 28, 1))
        
        prediction = model.predict(cnn_input, verbose=0)
        class_id = np.argmax(prediction)
        
        labels = ['decimal val', 'div val', 'eight', 'equal val', 'five', 'four', 'minus val', 'nine', 'number', 'one', 'plus val', 'seven', 'sign', 'six', 'three', 'times val', 'two', 'zero']
        
        predicted_class = labels[class_id]
        
        symbol_map = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'plus val': '+', 'minus val': '-', 'times val': '*', 'div val': '/',
            'equal val': '=', 'decimal val': '.', 'number': '', 'sign': ''
        }
        
        final_char = symbol_map.get(predicted_class, predicted_class) 
        equation_str += final_char
        
    return equation_str

if st.button("Dịch & Giải toán"):
    if canvas_result.image_data is not None:
        with st.spinner("1. Đang dùng CNN để đọc nét vẽ..."):
            equation_text = extract_and_predict(canvas_result.image_data, cnn_model)
            
        # 🌟 VÁ LỖI MARKDOWN: Bọc kết quả bằng dấu backtick (`) thay vì dấu sao (**)
        st.info(f" CNN đọc được phép tính là: `{equation_text}`")
        
        if not api_key:
            st.error(" Nhập Gemini API Key ở thanh bên trái để giải tiếp!")
        else:
            with st.spinner("2. Đang gửi cho Gemini giải..."):
                genai.configure(api_key=api_key)
                best_model_name = None
                
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        best_model_name = m.name
                        if 'flash' in m.name.lower() or 'pro' in m.name.lower():
                            break 
                
                gemini_model = genai.GenerativeModel(best_model_name)
                
                prompt = f"""
                Hãy giải phép toán hoặc phương trình sau đây: {equation_text}
                """
                response = gemini_model.generate_content(prompt)
                
                st.success(f"Hoàn thành! (Đã dùng model: {best_model_name})")
                st.markdown(response.text)