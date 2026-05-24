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
        
    # Lấy tất cả các hộp bao quanh nét vẽ
    raw_rects = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 10]
    # Sắp xếp từ trái qua phải theo trục X
    raw_rects.sort(key=lambda b: b[0])
    
    # 🌟 THUẬT TOÁN GỘP NÉT VẼ (Khắc phục lỗi dấu = và dấu chia)
    rects = []
    for x, y, w, h in raw_rects:
        if not rects:
            rects.append([x, y, w, h])
        else:
            lx, ly, lw, lh = rects[-1] # Lấy thông số của hộp vẽ ngay trước đó
            
            # Nếu hộp hiện tại nằm đè lên hộp trước đó theo chiều ngang (cho phép sai số 10 pixel)
            if x <= lx + lw + 10:
                # Gộp 2 hộp lại bằng cách lấy điểm ngoài cùng của cả 2
                min_x = min(x, lx)
                min_y = min(y, ly)
                max_x = max(x + w, lx + lw)
                max_y = max(y + h, ly + lh)
                # Cập nhật lại hộp cuối cùng thành hộp đã gộp
                rects[-1] = [min_x, min_y, max_x - min_x, max_y - min_y]
            else:
                # Nếu không đè lên nhau, nó là một chữ số/dấu mới
                rects.append([x, y, w, h])
                
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
            'plus val': '+', 'minus val': '-', 'times val': 'x', 'div val': '/',
            'equal val': '=', 'decimal val': '.', 'number': '', 'sign': ''
        }
        
        final_char = symbol_map.get(predicted_class, predicted_class) 
        
        if idx > 0:

            lx, ly, lw, lh = rects[idx - 1] 
            
            prev_center_y = ly + (lh / 2)  
            curr_bottom_y = y + h         
            
            if curr_bottom_y < prev_center_y and final_char not in ['+', '-', '=', '/', '.']:
                equation_str += "^" 
        
        equation_str += final_char
        
    return equation_str

equation_text = ""

if canvas_result.image_data is not None:
    equation_text = extract_and_predict(canvas_result.image_data, cnn_model)
    

    if equation_text != "":
        st.info(f" CNN đang đọc được phép tính là: `{equation_text}`")


if st.button(" Giải toán"):
    # Chặn người dùng nếu chưa vẽ gì
    if equation_text == "":
        st.warning(" Bạn chưa vẽ phép tính nào lên bảng kìa")
        

    elif not api_key:
        st.error(" Nhập Gemini API Key ở thanh bên trái để giải tiếp")
        
    else:
        with st.spinner("Đang gửi cho Gemini giải..."):
            genai.configure(api_key=api_key)
            best_model_name = None
            
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    best_model_name = m.name
                    if 'flash' in m.name.lower() or 'pro' in m.name.lower():
                        break 
            
            gemini_model = genai.GenerativeModel(best_model_name)
            
            prompt = f"""
            Bạn là một gia sư Toán học xuất sắc. Hãy giải phương trình tìm x sau đây: {equation_text}
            
            Yêu cầu:
            - Trình bày lời giải từng bước thật chi tiết, dễ hiểu bằng tiếng Việt.
            - Phân tích rõ các quy tắc chuyển vế, đổi dấu (nếu có).
            - BẮT BUỘC dùng format LaTeX cho công thức toán học. Dùng 1 dấu $ cho công thức nằm cùng dòng (ví dụ: $x^2 = 4$) và 2 dấu $$ cho công thức đứng riêng 1 dòng (ví dụ: $$x = 2$$).
            - TUYỆT ĐỐI KHÔNG bọc công thức toán học trong dấu nháy đơn ngược (backtick).
            - Không được để khoảng trắng giữa dấu $ và công thức.
            - Cuối cùng, đưa ra kết luận rõ ràng: "Vậy x = ..."
            """
            response = gemini_model.generate_content(prompt)
            
            st.success(f"Hoàn thành! (Đã dùng model: {best_model_name})")
            st.markdown(response.text)
