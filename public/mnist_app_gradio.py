import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Tải mô hình mnist_model.h5 đã huấn luyện
# Đảm bảo file mô hình nằm cùng thư mục
try:
    model = tf.keras.models.load_model('mnist_model.h5')
    print("Đã tải mô hình thành công!")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    print("Hãy đảm bảo bạn đã chạy script huấn luyện trước!")

def predict_digit(sketchpad):
    """
    Nhận đầu vào từ sketchpad, tiền xử lý và trả về dự đoán.
    """
    if sketchpad is None:
        return "Không phát hiện nét vẽ", {}

    # Trích xuất dữ liệu hình ảnh
    if isinstance(sketchpad, dict):
        img = sketchpad['composite']
    else:
        img = sketchpad

    # Tiền xử lý:
    # 1. Chuyển sang ảnh xám (L)
    img_pil = Image.fromarray(img.astype('uint8')).convert('L')
    
    # 2. Thay đổi kích thước thành 28x28 (chuẩn MNIST)
    img_28x28 = img_pil.resize((28, 28))
    
    # 3. Chuyển thành mảng numpy và chuẩn hóa về 0-1
    img_array = np.array(img_28x28).astype('float32') / 255.0
    
    # 4. Làm phẳng thành vector 784 (1D)
    img_flattened = img_array.reshape(1, 784)
    
    # 5. Dự đoán xác suất
    preds = model.predict(img_flattened)[0]
    
    # Tạo từ điển kết quả cho nhãn và điểm số
    scores = {str(i): float(preds[i]) for i in range(10)}
    predicted_digit = int(np.argmax(preds))
    
    return f"Chữ số dự đoán: {predicted_digit}", scores

# 2. Tạo giao diện Gradio
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(label="Vẽ một chữ số (0-9)", type="numpy"),
    outputs=[
        gr.Textbox(label="Dự đoán"),
        gr.Label(label="Xác suất chi tiết", num_top_classes=10)
    ],
    title="Nhận Diện Chữ Số Viết Tay (MNIST)",
    description="Vẽ một chữ số duy nhất lên canvas và nhấn dự đoán.",
    live=True # Bật dự đoán thời gian thực
)

if __name__ == "__main__":
    interface.launch()
