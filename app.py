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

@st.cache_resource
def load_cnn_model():
    try:
        model = tf.keras.models.load_model('model_toan_hoc.h5')
        return model
    except:
        return None

cnn_model = load_cnn_model()



with st.sidebar:
    st.header("⚙️ Cài đặt hệ thống")
    api_key = st.text_input("Nhập Gemini API Key:", type="password")
    
    st.markdown("---")
    st.header("🔀 Chọn chế độ")
    app_mode = st.radio(
        "Bạn muốn làm gì hôm nay?",
        ("Tính toán cơ bản", "Giải phương trình (Tìm x)")
    )

# Đổi tiêu đề App tùy theo Page đang chọn
if app_mode == "Tính toán cơ bản":
    st.title("Tính toán cơ bản (Hỗ trợ cộng, trừ, nhân, chia, mũ)")
else:
    st.title("Giải phương trình (Tìm x)")

st.write("Vẽ phép toán của bạn vào bảng đen bên dưới:")
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=400,
    width=1000,
    drawing_mode="freedraw",
    key="canvas",
)


def extract_and_predict(image_data, model, current_mode):
    gray = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return ""
        
    raw_rects = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 10]
    raw_rects.sort(key=lambda b: b[0])
    

    rects = []
    for x, y, w, h in raw_rects:
        if not rects:
            rects.append([x, y, w, h])
        else:
            lx, ly, lw, lh = rects[-1] 
            if x <= lx + lw + 10:
                min_x = min(x, lx)
                min_y = min(y, ly)
                max_x = max(x + w, lx + lw)
                max_y = max(y + h, ly + lh)
                rects[-1] = [min_x, min_y, max_x - min_x, max_y - min_y]
            else:
                rects.append([x, y, w, h])
                
    equation_str = ""
    st.write("**Ảnh đã được xử lý (Gửi cho CNN):**")
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
        
        # TỪ ĐIỂN ĐỘNG (Thay đổi tùy theo Page)
        symbol_map = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'plus val': '+', 'minus val': '-', 'div val': '/', 'equal val': '=', 
            'decimal val': '.', 'number': '', 'sign': ''
        }
        
        if current_mode == "Giải phương trình (Tìm x)":
            symbol_map['times val'] = 'x'
        else:
            symbol_map['times val'] = '*'
            
        final_char = symbol_map.get(predicted_class, predicted_class) 
        
        # THUẬT TOÁN BẮT SỐ MŨ (Chỉ bật mạnh trong chế độ Tìm x)
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
    equation_text = extract_and_predict(canvas_result.image_data, cnn_model, app_mode)
    if equation_text != "":
        st.info(f" AI đọc được chuỗi là: `{equation_text}`")


if st.button(" Gửi bài giải"):
    if equation_text == "":
        st.warning("Bạn chưa vẽ gì lên bảng kìa!")
    elif not api_key:
        st.error("Nhập Gemini API Key ở thanh bên trái để giải tiếp!")
    else:
        with st.spinner("1. Máy tính đang phân tích đáp án..."):
            import sympy as sp
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
            
            ket_qua_python = ""
            try:
                transformations = (standard_transformations + (implicit_multiplication_application,))
                x = sp.Symbol('x')
                eq_str = equation_text.replace('^', '**')
                
                if '=' in eq_str:
                    left_str, right_str = eq_str.split('=')
                    lhs = parse_expr(left_str, transformations=transformations)
                    rhs = parse_expr(right_str, transformations=transformations)
                    equation = sp.Eq(lhs, rhs)
                    solutions = sp.solve(equation, x)
                    ket_qua_python = f"x \in {solutions}" if len(solutions) > 1 else f"x = {solutions[0]}"
                else:
                    expr = parse_expr(eq_str, transformations=transformations)
                    solutions = sp.simplify(expr)
                    if solutions.is_Float:
                        ket_qua_python = f"{float(solutions):g}"
                    else:
                        ket_qua_python = f"{solutions}"
                        
                st.success(f"Đáp án chuẩn của hệ thống: **{ket_qua_python}**")
            except Exception as e:
                st.error("Lỗi cú pháp! Hãy kiểm tra lại nét vẽ trên bảng.")
                ket_qua_python = "Không tính được"

        if ket_qua_python != "Không tính được":
            with st.spinner("2. AI đang soạn lời giải chi tiết..."):
                genai.configure(api_key=api_key)
                best_model_name = None
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        best_model_name = m.name
                        if 'flash' in m.name.lower() or 'pro' in m.name.lower():
                            break 
                
                gemini_model = genai.GenerativeModel(best_model_name)
                
                if app_mode == " Tính toán cơ bản":
                    prompt = f"""
                    Phép tính: {equation_text}
                    Kết quả đúng: {ket_qua_python}
                    Hãy giải thích ngắn gọn cách tính ra kết quả này (có thể nêu quy tắc nhân chia trước, cộng trừ sau). 
                    Sử dụng LaTeX (dấu $) cho các công thức số. Không dùng backtick bọc công thức.
                    """
                else:
                    prompt = f"""
                    Phương trình cần giải: {equation_text}
                    Nghiệm đúng: {ket_qua_python}
                    Bạn là một gia sư Toán. Hãy trình bày chi tiết các bước chuyển vế, đổi dấu, hoặc dùng công thức nghiệm để ra được kết quả trên.
                    BẮT BUỘC dùng LaTeX. Dùng 1 dấu $ cho công thức cùng dòng, và 2 dấu $$ cho công thức đứng riêng 1 dòng. 
                    TUYỆT ĐỐI KHÔNG dùng dấu nháy đơn ngược (backtick) để bọc công thức.
                    """
                    
                response = gemini_model.generate_content(prompt)
                st.markdown("---")
                st.markdown(response.text)