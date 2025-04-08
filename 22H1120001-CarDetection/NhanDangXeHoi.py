import cv2

# Load bộ phân loại xe hơi
car_cascade = cv2.CascadeClassifier('cars.xml')

# Đọc hai ảnh đầu vào
image_paths = [
    "car3.jpg",
    "car5.jpg"  # Thêm một ảnh khác
]

# Tên sinh viên
student_name = "Ha Duc An"

for img_path in image_paths:
    # Đọc ảnh màu
    img = cv2.imread(img_path)

    # Chuyển ảnh sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện xe
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)

    # Đánh số và vẽ hình chữ nhật quanh xe
    for i, (x, y, w, h) in enumerate(cars, start=1):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Vẽ hình chữ nhật màu xanh lá
        cv2.putText(img, str(i), (x + w // 2, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)  # Hiển thị số thứ tự màu đỏ ở giữa xe

    # Chèn thông tin sinh viên ở góc trái dưới cùng
    height, width, _ = img.shape
    cv2.rectangle(img, (10, height - 50), (250, height - 10), (0, 255, 255), -1)  # Hình chữ nhật màu vàng
    cv2.putText(img, student_name, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2, cv2.LINE_AA)  # Chữ màu đỏ

    # Hiển thị ảnh
    cv2.imshow(f"Detected Cars - {img_path}", img)

# Chờ phím bất kỳ để thoát
cv2.waitKey(0)
cv2.destroyAllWindows()
