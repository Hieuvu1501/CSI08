import os
# Ép thư viện Protobuf chạy bằng Python thuần để tránh lỗi xung đột phiên bản C++ của TensorFlow
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import google.protobuf.runtime_version
# Vô hiệu hóa hàm kiểm tra phiên bản của Protobuf để chặn cảnh báo đỏ (Incompatible Gencode)
google.protobuf.runtime_version.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None

import streamlit as st                      # Thư viện tạo giao diện web nhanh
from streamlit_drawable_canvas import st_canvas # Thư viện tạo bảng vẽ trên web
import cv2                                  # OpenCV: Thư viện xử lý ảnh (cắt, ghép, tìm viền)
import numpy as np                          # Thư viện xử lý mảng và ma trận số
import tensorflow as tf                     # Thư viện AI chứa mạng nơ-ron CNN
import google.generativeai as genai         # Thư viện gọi API của Google Gemini

# --- 1. CẤU HÌNH GIAO DIỆN WEB ---
# Đặt tiêu đề tab trình duyệt, icon và chế độ hiển thị toàn màn hình (wide)
st.set_page_config(page_title="AI Math Drawer", page_icon="✍️", layout="wide")
st.title("✍️ Vẽ Phương Trình - AI Giải Toán")

# --- 2. TẢI MÔ HÌNH NHẬN DIỆN (CNN) ---
# Dùng @st.cache_resource để lưu mô hình vào bộ nhớ đệm, giúp web không phải load lại file .h5 mỗi khi F5
@st.cache_resource
def load_cnn_model():
    try:
        model = tf.keras.models.load_model('model_toan_hoc.h5') # Tải "bộ não" đã huấn luyện
        return model
    except:
        return None

cnn_model = load_cnn_model()

# --- 3. CÀI ĐẶT THANH BÊN (SIDEBAR) ---
with st.sidebar:
    st.header("Cài đặt API")
    # Ô nhập mật khẩu API Key (ẩn các ký tự khi gõ)
    api_key = st.text_input("Nhập Gemini API Key:", type="password")
    
# --- 4. KHU VỰC BẢNG VẼ ---
st.write("Vẽ phép tính của bạn vào bảng đen bên dưới:")
canvas_result = st_canvas(
    fill_color="#000000",       # Màu tô mặc định
    stroke_width=15,            # Độ dày nét bút (rất quan trọng để AI dễ nhìn)
    stroke_color="#FFFFFF",     # Màu nét bút là Trắng
    background_color="#000000", # Màu nền bảng là Đen
    height=200,                 # Chiều cao bảng
    width=700,                  # Chiều rộng bảng
    drawing_mode="freedraw",    # Chế độ vẽ tự do
    key="canvas",
)

# --- 5. HÀM XỬ LÝ ẢNH VÀ NHẬN DIỆN KÝ TỰ ---
def extract_and_predict(image_data, model):
    # Chuyển ảnh màu (RGBA) từ canvas sang ảnh thang độ xám (Grayscale)
    gray = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)
    
    # Tìm các đường viền (contours) bao quanh các nét vẽ trên bảng
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "" # Nếu bảng trống thì trả về chuỗi rỗng
        
    # Tạo các hộp chữ nhật (Bounding Box) bao quanh các nét vẽ, loại bỏ các đốm nhiễu nhỏ (diện tích < 10)
    rects = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 10]
    
    # Sắp xếp các hộp chữ nhật theo tọa độ x (từ trái sang phải) để đọc phương trình đúng thứ tự
    rects.sort(key=lambda b: b[0])
    
    equation_str = ""
    
    st.write(" **Ảnh đã được xử lý (Gửi cho CNN):**")
    cols = st.columns(len(rects)) # Tạo các cột trên giao diện để hiển thị ảnh debug
    
    # Duyệt qua từng ký tự đã cắt được
    for idx, (x, y, w, h) in enumerate(rects):
        pad = 10 # Thêm một chút lề (padding) xung quanh chữ
        
        # Cắt ảnh vùng ký tự (ROI - Region of Interest) ra khỏi bảng lớn
        roi = gray[max(0, y-pad):min(gray.shape[0], y+h+pad), 
                   max(0, x-pad):min(gray.shape[1], x+w+pad)]
        
        if roi.size == 0: continue
        
        # --- THUẬT TOÁN CHỐNG MÉO HÌNH ---
        side = max(roi.shape) # Tìm cạnh dài nhất của vùng cắt
        square = np.zeros((side, side), dtype=np.uint8) # Tạo một hình vuông đen xì có cạnh = side
        
        # Tính toán tọa độ để dán chữ số vào chính giữa hình vuông đen vừa tạo
        x_offset = (side - roi.shape[1]) // 2
        y_offset = (side - roi.shape[0]) // 2
        square[y_offset:y_offset+roi.shape[0], x_offset:x_offset+roi.shape[1]] = roi
        
        # Thu nhỏ/phóng to hình vuông đó về đúng chuẩn 28x28 pixels mà AI yêu cầu
        resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Đảo ngược màu: Nền đen chữ trắng -> Nền trắng chữ đen (Để khớp với Dataset ClarenceZhao)
        final_img = cv2.bitwise_not(resized)
        
        # Hiển thị ảnh nhỏ đã xử lý lên web cho người dùng xem
        cols[idx].image(final_img, width=40)
        
        # Thay đổi hình dáng mảng từ 2D (28,28) sang 4D (1, 28, 28, 1) để đưa vào mạng CNN. 
        # (Không chia 255 ở đây vì Rescaling layer trong Model đã làm việc đó).
        cnn_input = np.reshape(final_img, (1, 28, 28, 1))
        
        # Model dự đoán ảnh, trả về mảng xác suất, np.argmax sẽ lấy ra vị trí có xác suất cao nhất
        prediction = model.predict(cnn_input, verbose=0)
        class_id = np.argmax(prediction)
        
        # Danh sách các nhãn (labels) do quá trình Train sinh ra (Theo thứ tự Alphabet)
        labels = ['decimal val', 'div val', 'eight', 'equal val', 'five', 'four', 'minus val', 'nine', 'number', 'one', 'plus val', 'seven', 'sign', 'six', 'three', 'times val', 'two', 'zero']
        
        predicted_class = labels[class_id] # Lấy ra tên tiếng Anh của ký tự
        
        # Từ điển dịch tên tiếng Anh của tác giả Dataset sang ký hiệu Toán học chuẩn
        symbol_map = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'plus val': '+', 'minus val': '-', 'times val': '*', 'div val': '/',
            'equal val': '=', 'decimal val': '.', 'number': '', 'sign': ''
        }
        
        # Áp dụng từ điển để lấy ra ký tự cuối cùng (Ví dụ 'two' -> '2')
        final_char = symbol_map.get(predicted_class, predicted_class) 
        equation_str += final_char # Ghép ký tự vừa nhận diện vào chuỗi phương trình tổng
        
    return equation_str # Trả về phương trình hoàn chỉnh (VD: "1+2")

# --- 6. LOGIC XỬ LÝ KHI BẤM NÚT "Dịch & Giải toán" ---
if st.button("Dịch & Giải toán"):
    # Kiểm tra xem người dùng đã vẽ gì chưa
    if canvas_result.image_data is not None:
        with st.spinner("1. Đang dùng CNN để đọc nét vẽ..."):
            # Gọi hàm xử lý ảnh và nhận diện phương trình
            equation_text = extract_and_predict(canvas_result.image_data, cnn_model)
            
        # In phương trình vừa đọc được ra màn hình
        st.info(f" CNN đọc được phép tính là: `{equation_text}`")
        
        # Kiểm tra xem đã có API Key chưa
        if not api_key:
            st.error(" Nhập Gemini API Key ở thanh bên trái để giải tiếp!")
        else:
            with st.spinner("2. Đang gửi cho Gemini giải..."):
                # Cấu hình API Key cho Google Gemini
                genai.configure(api_key=api_key)
                best_model_name = None
                
                # Quét tự động để tìm model Gemini xịn nhất đang khả dụng trong tài khoản
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        best_model_name = m.name
                        if 'flash' in m.name.lower() or 'pro' in m.name.lower():
                            break 
                
                # Khởi tạo AI với model vừa tìm được
                gemini_model = genai.GenerativeModel(best_model_name)
                
                # Tạo lời nhắc (Prompt) yêu cầu AI giải bài toán
                prompt = f"""
                Hãy giải phép toán hoặc phương trình sau đây: {equation_text}
                """
                
                # Gửi yêu cầu lên máy chủ Google và nhận kết quả
                response = gemini_model.generate_content(prompt)
                
                # In ra thông báo thành công và hiển thị lời giải lên màn hình
                st.success(f"Hoàn thành! (Đã dùng model: {best_model_name})")
                st.markdown(response.text)