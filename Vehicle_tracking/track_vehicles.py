import cv2
import numpy as np

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def diffUpDown(img):
    height, width, _ = img.shape
    half = int(height / 2)
    top = img[0:half, 0:width]
    bottom = img[half:half+half, 0:width]

    top = cv2.flip(top, 1)
    bottom = cv2.resize(bottom, (32, 64))
    top = cv2.resize(top, (32, 64))
    return mse(top, bottom)

def diffLeftRight(img):
    height, width, _ = img.shape
    half = int(width / 2)
    left = img[0:height, 0:half]
    right = img[0:height, half:half + half - 1]

    right = cv2.flip(right, 1)
    left = cv2.resize(left, (32, 64))
    right = cv2.resize(right, (32, 64))
    return mse(left, right)

def isNewRoi(rx, ry, rw, rh, rectangles):
    for r in rectangles:
        if abs(r[0] - rx) < 40 and abs(r[1] - ry) < 40:
            return False
    return True

def detectRegionsOfInterest(frame, cascade):
    scaleDown = 2
    frame = cv2.resize(frame, (frame.shape[1] // scaleDown, frame.shape[0] // scaleDown))
    frameHeight, frameWidth, _ = frame.shape
    cars = cascade.detectMultiScale(frame, 1.2, 1)

    newRegions = []
    minY = int(frameHeight * 0.3)

    for (x, y, w, h) in cars:
        roiImage = frame[y:y+h, x:x+w]
        if y > minY:
            diffX = diffLeftRight(roiImage)
            diffY = round(diffUpDown(roiImage))
            if 1600 < diffX < 3000 and diffY > 12000:
                newRegions.append([x*scaleDown, y*scaleDown, w*scaleDown, h*scaleDown])

    return newRegions

def detectCars(filename):
    rectangles = []
    cascade = cv2.CascadeClassifier('cars.xml')
    vc = cv2.VideoCapture(filename)

    if not vc.isOpened():
        print("Không thể mở video.")
        return

    frameCount = 0

    while True:
        rval, frame = vc.read()
        if not rval:
            break

        newRegions = detectRegionsOfInterest(frame, cascade)
        for region in newRegions:
            if isNewRoi(region[0], region[1], region[2], region[3], rectangles):
                rectangles.append(region)

        for r in rectangles:
            cv2.rectangle(frame, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (0, 0, 255), 3)

        cv2.putText(frame, f"So xe phat hieen: {len(rectangles)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Keet qua", frame)

        frameCount += 1
        if frameCount > 30:
            frameCount = 0
            rectangles = []

        if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
            break

    vc.release()
    cv2.destroyAllWindows()

# Gọi hàm chính
detectCars('road.mp4')
