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
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import plotly.graph_objects as go


st.set_page_config(page_title="AI Math Drawer", page_icon="✍️", layout="wide")


@st.cache_resource
def load_cnn_model():
    try:
        model = tf.keras.models.load_model('model_toan_hoc.h5')
        return model
    except:
        return None

cnn_model = load_cnn_model()

if 'current_page' not in st.session_state:
    st.session_state.current_page = "landing"


def go_to_app():
    st.session_state.current_page = "main_app"


if st.session_state.current_page == "landing":

    st.markdown("""
        <style>
        /* Định dạng thanh điều hướng cố định */
        .custom-header {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 70px;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-bottom: 1px solid #334155;
            z-index: 999999; /* Đảm bảo menu luôn nổi lên trên cùng */
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 60px;
        }
        
        .header-logo {
            font-size: 24px;
            font-weight: 800;
            color: #ffffff;
            text-decoration: none;
            letter-spacing: 1px;
        }
        .header-logo span {
            color: #2dd4bf;
        }
        
        .header-menu {
            display: flex;
            gap: 40px;
        }
        .header-menu a {
            color: #cbd5e1;
            text-decoration: none;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .header-menu a:hover {
            color: #2dd4bf;
        }
        
        /* Đẩy Hero Banner xuống một chút để không bị thanh menu che mất chữ */
        .hero-banner {
            margin-top: 40px !important;
        }
        </style>

        <div class="custom-header">
            <a href="#" class="header-logo">✍️ AI Math <span>Drawer</span></a>
            <div class="header-menu">
                <a href="#">Trang chủ</a>
                <a href="#gioi-thieu">Giới thiệu</a>
                <a href="#lien-he">Liên hệ</a>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <style>
        /* Xóa padding mặc định của Streamlit ở trên cùng để banner tràn viền đẹp hơn */
        .block-container {
            padding-top: 2rem !important;
        }
        
        /* Thiết kế Hero Banner */
        .hero-banner {
            /* Hình nền kết hợp lớp phủ màu đen trong suốt để làm nổi bật chữ */
            background-image: linear-gradient(rgba(15, 23, 42, 0.75), rgba(15, 23, 42, 0.95)), url('https://images.unsplash.com/photo-1635070041078-e363dbe005cb?q=80&w=2000&auto=format&fit=crop');
            background-size: cover;
            background-position: center;
            height: 70vh; /* Chiều cao chiếm 70% màn hình hiển thị */
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            border-radius: 20px;
            margin-bottom: -30px; /* Kéo nút bấm lên sát banner */
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            text-align: center;
            padding: 20px;
        }
        
        .hero-title {
            font-size: 80px;
            font-weight: 900;
            color: #ffffff;
            margin-bottom: 20px;
            text-shadow: 2px 2px 10px rgba(0,0,0,0.5);
            letter-spacing: 2px;
        }
        
        .hero-subtitle {
            font-size: 24px;
            color: #cbd5e1;
            max-width: 800px;
            line-height: 1.6;
            margin-bottom: 40px;
        }
        
        /* Chỉnh style cho các tiêu đề section bên dưới */
        .section-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #2dd4bf;
            margin-top: 50px;
            margin-bottom: 30px;
        }
        </style>
    """, unsafe_allow_html=True)


    st.markdown('''
        <div class="hero-banner">
            <div class="hero-title">AI Math Drawer ✍️</div>
            <div class="hero-subtitle">Nền tảng giáo dục thông minh: Biến nét vẽ tay tự do thành lời giải Toán học chi tiết với sức mạnh của Mạng nơ-ron và Trí tuệ nhân tạo tạo sinh.</div>
        </div>
    ''', unsafe_allow_html=True)

    # 3. NÚT TRẢI NGHIỆM ĐẶT NGAY DƯỚI BANNER (Nổi bật)
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_empty1, col_cta, col_empty2 = st.columns([1, 2, 1])
    with col_cta:
        st.button("🚀 BẮT ĐẦU VẼ VÀ GIẢI TOÁN NGAY", on_click=go_to_app, use_container_width=True, type="primary")
    
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)




    st.markdown('<div class="section-title">✨ Tính năng cốt lõi</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("#### 🎨 Vẽ tự do\nChỉ cần dùng chuột hoặc trackpad vẽ trực tiếp biểu thức toán học lên bảng đen. Không cần gõ công thức phức tạp.")
    with col2:
        st.success("#### 💡 Gia sư AI\nKhông chỉ giải toán, hệ thống còn kết hợp LLM để cung cấp gợi ý từng bước như một gia sư thực thụ.")
    with col3:
        st.warning("#### 📈 Đồ thị trực quan\nTự động phân tích phương trình và vẽ đồ thị hàm số tương tác tức thì chỉ với một cú click.")

    st.markdown("---")


    st.markdown('<div class="section-title">⚙️ Kiến trúc Hệ thống (Pipeline)</div>', unsafe_allow_html=True)
    
    step1, step2, step3, step4 = st.columns(4)
    with step1:
        st.markdown("👉 **Bước 1: Input**")
        st.caption("Người dùng vẽ lên bảng. Ảnh được trích xuất dưới dạng mảng Numpy RGBA.")
    with step2:
        st.markdown("🧠 **Bước 2: CNN & OpenCV**")
        st.caption("OpenCV tách từng nét vẽ. Model `Keras CNN` phân loại từng ký tự 28x28 pixel.")
    with step3:
        st.markdown("🔢 **Bước 3: Sympy**")
        st.caption("Chuỗi ký tự được đưa qua `Sympy` để parse thành biểu thức và tìm nghiệm chuẩn xác.")
    with step4:
        st.markdown("💬 **Bước 4: Gemini LLM**")
        st.caption("Dữ liệu được đưa vào `Gemini API` để xuất ra văn bản giải thích chi tiết bằng LaTeX.")

    st.markdown("---")


    st.markdown('<div class="section-title">📊 Thông số Mô hình CNN</div>', unsafe_allow_html=True)
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric(label="📚 Dữ liệu huấn luyện (Tập Train)", value="~ 180,000 ảnh", delta="18 phân lớp ký tự")
    col_m2.metric(label="🎯 Độ chính xác (Accuracy)", value="96.8%", delta="Trên tập Test")
    col_m3.metric(label="⚡ Tốc độ nhận diện", value="< 0.1s", delta="Xử lý Real-time")

    # --- SƠ ĐỒ KIẾN TRÚC MODEL (CNN ARCHITECTURE) ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧠 Sơ đồ Kiến trúc CNN (Model Architecture)</div>', unsafe_allow_html=True)
    st.markdown("""
        <div style='background-color: rgba(45, 212, 191, 0.1); border-left: 4px solid #2dd4bf; padding: 12px 20px; border-radius: 0 8px 8px 0; margin: 10px auto 30px auto; max-width: 800px; text-align: center;'>
            <span style='font-size: 16px; color: #e2e8f0;'>
                Sơ đồ <b style='color: #2dd4bf;'>trích xuất đặc trưng (Feature Extraction)</b> và <b style='color: #2dd4bf;'>Phân loại (Classification)</b> dựa trên source code huấn luyện:
            </span>
        </div>
""", unsafe_allow_html=True)
    

    st.markdown("""
        <style>
        .cnn-pipeline { 
            display: flex; 
            flex-wrap: wrap; 
            justify-content: center; 
            align-items: center; 
            gap: 12px; 
            margin-top: 20px;
            margin-bottom: 30px;
        }
        .cnn-layer { 
            background: linear-gradient(145deg, #1e293b, #0f172a); 
            border: 2px solid #2dd4bf; 
            border-radius: 12px; 
            padding: 15px; 
            text-align: center; 
            width: 145px; 
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            transition: transform 0.2s;
        }
        .cnn-layer:hover {
            transform: translateY(-5px);
            border-color: #3b82f6;
        }
        .layer-title { 
            font-weight: 800; 
            color: #2dd4bf; 
            font-size: 15px; 
            margin-bottom: 8px; 
        }
        .layer-detail { 
            font-size: 13px; 
            color: #cbd5e1; 
            line-height: 1.4;
        }
        .cnn-arrow { 
            color: #64748b; 
            font-size: 24px; 
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)


    st.markdown("""
        <div class="cnn-pipeline">
            <div class="cnn-layer">
                <div class="layer-title">Input Image</div>
                <div class="layer-detail">28 x 28 x 1<br>Grayscale</div>
            </div>
            <div class="cnn-arrow">➔</div>
            <div class="cnn-layer">
                <div class="layer-title">Rescaling</div>
                <div class="layer-detail">Scale: 1./255<br>Normalization</div>
            </div>
            <div class="cnn-arrow">➔</div>
            <div class="cnn-layer">
                <div class="layer-title">Block 1</div>
                <div class="layer-detail">Conv2D (32, 3x3)<br>ReLU<br>MaxPooling2D</div>
            </div>
            <div class="cnn-arrow">➔</div>
            <div class="cnn-layer">
                <div class="layer-title">Block 2</div>
                <div class="layer-detail">Conv2D (64, 3x3)<br>ReLU<br>MaxPooling2D</div>
            </div>
            <div class="cnn-arrow">➔</div>
            <div class="cnn-layer">
                <div class="layer-title">Block 3</div>
                <div class="layer-detail">Conv2D (128, 3x3)<br>ReLU<br>MaxPooling2D</div>
            </div>
            <div class="cnn-arrow">➔</div>
            <div class="cnn-layer">
                <div class="layer-title">Flatten</div>
                <div class="layer-detail">1D Array<br>(1152 nodes)</div>
            </div>
            <div class="cnn-arrow">➔</div>
            <div class="cnn-layer">
                <div class="layer-title">Dense Layer</div>
                <div class="layer-detail">128 units<br>ReLU</div>
            </div>
            <div class="cnn-arrow">➔</div>
            <div class="cnn-layer">
                <div class="layer-title">Dropout</div>
                <div class="layer-detail">Rate: 0.5<br>(Prevent Overfitting)</div>
            </div>
            <div class="cnn-arrow">➔</div>
            <div class="cnn-layer" style="border-color: #f43f5e;">
                <div class="layer-title" style="color: #f43f5e;">Output Dense</div>
                <div class="layer-detail">Softmax<br>Probabilities</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


    st.markdown('<div class="section-title">🛠️ Công nghệ cốt lõi</div>', unsafe_allow_html=True)
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.button("🔥 TensorFlow / Keras", use_container_width=True)
    t2.button("👁️ OpenCV", use_container_width=True)
    t3.button("♾️ Sympy Python", use_container_width=True)
    t4.button("✨ Google Gemini", use_container_width=True)
    t5.button("👑 Streamlit", use_container_width=True)

    st.markdown("---")


    st.markdown('<div class="section-title">❓ Câu hỏi thường gặp</div>', unsafe_allow_html=True)
    with st.expander("Khả năng nhận diện của mô hình giới hạn ở đâu?"):
        st.write("Hiện tại, mô hình CNN được huấn luyện để nhận diện các số từ **0-9**, các phép toán cơ bản (**+, -, x, /, =**) và dấu thập phân. Mô hình chưa hỗ trợ vẽ phân số tầng hoặc căn bậc hai phức tạp.")
    with st.expander("Tại sao tôi cần nhập Gemini API Key?"):
        st.write("Hệ thống tự động giải toán bằng Sympy (không cần mạng). Tuy nhiên, để AI có thể đóng vai trò làm 'Gia sư' và viết ra lời giải thích chi tiết, ứng dụng cần gọi API của Google Gemini. Bạn có thể lấy key này hoàn toàn miễn phí tại Google AI Studio.")
    with st.expander("Chế độ 'Vẽ đồ thị' hoạt động như thế nào?"):
        st.write("Sau khi Sympy phân tích xong phương trình của bạn thành dạng `f(x) = 0`, thư viện **Plotly** sẽ sinh ra hàng trăm điểm dữ liệu trong khoảng x từ -10 đến 10 để render đồ thị tương tác. Bạn có thể zoom, di chuột để xem tọa độ.")

    # 9. FOOTER
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 14px;'>© 2024 AI Math Drawer Project. Built with ❤️ and Streamlit.</p>", unsafe_allow_html=True)


elif st.session_state.current_page == "main_app":
    
    with st.sidebar:
        st.header("⚙️ Cài đặt hệ thống")
        api_key = st.text_input("Nhập Gemini API Key:", type="password")
        
        st.markdown("---")
        st.header("🔀 Chọn chế độ")
        app_mode = st.radio(
            "Bạn muốn làm gì hôm nay?",
            ("Nhận diện số","Tính toán cơ bản", "Giải phương trình (Tìm x)")
        )
        

        st.markdown("---")
        if st.button("🏠 Quay lại Trang chủ"):
            st.session_state.current_page = "landing"
            st.rerun()

    if app_mode == "Tính toán cơ bản":
        st.title("Tính toán cơ bản")
    elif app_mode == "Nhận diện số":
        st.title("Nhận diện số")
    else:
        st.title("Giải phương trình (Tìm x)")

    st.write("Vẽ phép toán của bạn vào bảng đen bên dưới:")
    
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=500,
        width=1400,
        drawing_mode="freedraw",
        key="canvas",
    )

    def extract_and_predict(image_data, model, current_mode):
        gray = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return "", 0  # Cập nhật: Trả về thêm số 0 cho tỉ lệ chính xác nếu không có nét vẽ
            
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
        confidences = [] 
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
            

            confidence = np.max(prediction) * 100 
            confidences.append(confidence)
            
            labels = ['decimal val', 'div val', 'eight', 'equal val', 'five', 'four', 'minus val', 'nine', 'number', 'one', 'plus val', 'seven', 'sign', 'six', 'three', 'times val', 'two', 'zero']
            predicted_class = labels[class_id]
            
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
            
            if idx > 0:
                lx, ly, lw, lh = rects[idx - 1] 
                prev_center_y = ly + (lh / 2) 
                curr_bottom_y = y + h          
                if curr_bottom_y < prev_center_y and final_char not in ['+', '-', '=', '/', '.']:
                    equation_str += "^"
                    
            equation_str += final_char
        

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return equation_str, round(avg_confidence, 2) # Trả về 2 giá trị


    equation_text = ""
    avg_confidence = 0 
        
    if canvas_result.image_data is not None:
        equation_text, avg_confidence = extract_and_predict(canvas_result.image_data, cnn_model, app_mode)
            
        if equation_text != "":
            st.info(f"🤖 AI đọc được chuỗi là: `{equation_text}`")

    
    if app_mode == "Nhận diện số":
        if equation_text != "":
            st.markdown("### 🎯 Kết quả nhận diện:")
            st.markdown(f"<h1 style='text-align: center; color: #2dd4bf; font-size: 100px; padding: 20px; background: #1e293b; border-radius: 20px;'>{equation_text}</h1>", unsafe_allow_html=True)
            
            st.markdown("### 📊 Tỉ lệ chính xác (Độ tự tin):")
            
            if avg_confidence > 80:
                st.success(f"Mô hình rất tự tin với nét vẽ của bạn: {avg_confidence}%")
            elif avg_confidence > 50:
                st.warning(f"Mô hình khá phân vân, bạn có thể thử viết rõ hơn: {avg_confidence}%")
            else:
                st.error(f"Nét vẽ rất khó đọc: {avg_confidence}%")
                
            st.progress(int(avg_confidence))
            
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            nut_goi_y = st.button("💡 Gợi ý bước đầu")

        with col2:
            nut_giai = st.button("✅ Gửi bài giải")

        with col3:
            nut_ve = False
            if app_mode == "Giải phương trình (Tìm x)":
                nut_ve = st.button("📈 Vẽ đồ thị hàm số")

        if nut_goi_y:
            if equation_text == "":
                st.warning("Bạn chưa vẽ gì lên bảng kìa!")
            elif not api_key:
                st.error("Nhập Gemini API Key ở thanh bên trái để xem gợi ý!")
            else:
                with st.spinner("AI đang suy nghĩ gợi ý..."):
                    genai.configure(api_key=api_key)
                    best_model_name = None
                    for m in genai.list_models():
                        if 'generateContent' in m.supported_generation_methods:
                            best_model_name = m.name
                            if 'flash' in m.name.lower() or 'pro' in m.name.lower():
                                break 
                    
                    gemini_model = genai.GenerativeModel(best_model_name)
                    
                    prompt = f"""
                    Bạn là một gia sư Toán. Học sinh đang gặp khó với bài toán sau: {equation_text}
                    Nhiệm vụ của bạn: Đừng đưa ra đáp án cuối cùng. Hãy cho học sinh một gợi ý nhỏ, phương pháp tiếp cận, hoặc chỉ ra bước đầu tiên cần làm để họ có thể tự giải tiếp.
                    Giọng điệu: Thân thiện, khích lệ.
                    BẮT BUỘC dùng LaTeX. Dùng 1 dấu $ cho công thức cùng dòng, và 2 dấu $$ cho công thức đứng riêng 1 dòng. 
                    TUYỆT ĐỐI KHÔNG dùng dấu nháy đơn ngược (backtick) để bọc công thức.
                    """
                    response = gemini_model.generate_content(prompt)
                    
                    st.markdown("---")
                    st.info("💡 **Gợi ý của gia sư AI:**")
                    st.markdown(response.text)


        if nut_giai:
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
                    
                    if app_mode == "Tính toán cơ bản":
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


        if nut_ve:
            if equation_text == "":
                st.warning("Bạn chưa vẽ phương trình nào lên bảng!")
            else:
                with st.spinner("Đang vẽ đồ thị..."):
                    import sympy as sp
                    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
                    import plotly.graph_objects as go
                    
                    try:
                        transformations = (standard_transformations + (implicit_multiplication_application,))
                        x = sp.Symbol('x')
                        eq_str = equation_text.replace('^', '**')
                        
                        if '=' in eq_str:
                            left_str, right_str = eq_str.split('=')
                            f_expr = parse_expr(f"({left_str}) - ({right_str})", transformations=transformations)
                        else:
                            f_expr = parse_expr(eq_str, transformations=transformations)
                        
                        f_num = sp.lambdify(x, f_expr, "numpy")
                        
                        x_vals = np.linspace(-10, 10, 400)
                        y_vals = f_num(x_vals)
                        if np.isscalar(y_vals): 
                            y_vals = np.full_like(x_vals, y_vals)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='f(x)', line=dict(color='#2dd4bf', width=3)))
                        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Trục hoành (y=0)")
                        fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Trục tung (x=0)")
                        
                        fig.update_layout(
                            title=f"Đồ thị hàm số: f(x) = {f_expr}",
                            template="plotly_dark",
                            xaxis_title="x",
                            yaxis_title="y",
                            hovermode="x"
                        )
                        
                        st.markdown("---")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Không thể vẽ đồ thị cho biểu thức này. Lỗi: {e}")