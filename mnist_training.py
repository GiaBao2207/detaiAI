import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

def train_mnist_model():
    print("Đang tải tập dữ liệu MNIST...")
    # 1. Tải tập dữ liệu MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 2. Tiền xử lý dữ liệu
    # Chuyển đổi ma trận 28x28 thành vector 1D gồm 784 pixel (Làm phẳng - Flattening)
    x_train = x_train.reshape((-1, 784))
    x_test = x_test.reshape((-1, 784))

    # Chuẩn hóa giá trị pixel về khoảng 0-1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # 3. Xây dựng kiến trúc mạng nơ-ron
    model = models.Sequential([
        # Lớp ẩn (Dense) với 128 nơ-ron và hàm kích hoạt ReLU
        layers.Dense(128, activation='relu', input_shape=(784,)),
        
        # Lớp đầu ra (Dense) với 10 nơ-ron và hàm kích hoạt Softmax
        layers.Dense(10, activation='softmax')
    ])

    # 4. Biên dịch mô hình
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Bắt đầu huấn luyện...")
    # 5. Huấn luyện mô hình
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # 6. Đánh giá độ chính xác
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nĐộ chính xác trên tập kiểm tra: {test_acc:.4f}")

    # 7. Lưu mô hình đã huấn luyện
    model.save('mnist_model.h5')
    print("Đã lưu mô hình thành mnist_model.h5")

if __name__ == "__main__":
    train_mnist_model()
