import cv2
import pytesseract
import imutils
from matplotlib import pyplot as plt

# Đọc ảnh
image = cv2.imread('bienso1.jpg')
image = imutils.resize(image, width=300)  # Resize ảnh cho dễ xử lý

# Chuyển ảnh sang dạng RGB để hiển thị
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# Chuyển ảnh sang dạng xám (Gray)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image, cmap="gray")
plt.show()

# Làm mịn ảnh để giảm nhiễu
gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
plt.imshow(gray_image, cmap="gray")
plt.show()

# Phát hiện biên ảnh
edged = cv2.Canny(gray_image, 30, 200)
plt.imshow(edged, cmap="gray")
plt.axis("off")
plt.show()

# Tìm các đường viền
cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

# Tìm và vẽ hình chữ nhật bao quanh biển số xe
screenCnt = None
for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
    if len(approx) == 4:  # Nếu tìm được 4 điểm (hình chữ nhật)
        screenCnt = approx
        x, y, w, h = cv2.boundingRect(c)  # Lấy toạ độ của biển số
        new_img = image[y:y + h, x:x + w]
        plt.imshow(new_img)
        plt.show()
        break

# Vẽ khung chữ nhật lên ảnh gốc
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

# Nhận diện văn bản biển số
# Dùng pytesseract để nhận diện biển số xe
custom_config = r'--oem 3 --psm 6'
license_plate_text = pytesseract.image_to_string(new_img, config=custom_config)
print(f"Biển số xe: {license_plate_text}")

# Hiển thị thông tin biển số lên ảnh
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, license_plate_text, (x, y - 10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
plt.imshow(image)
plt.show()

