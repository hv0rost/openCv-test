import cv2

capture = cv2.VideoCapture('1.mp4')
cascade = cv2.CascadeClassifier('cascade.xml')

while True:
    ret, img = capture.read()

    subject = cascade.detectMultiScale(img, scaleFactor = 1.1, minNeighbors = 10, minSize = (80, 80))

    for (x, y, w, h) in subject:
        cv2.rectangle(img, (x, y), (x+w, y + h), (0, 0, 0), 2)
    cv2.imshow('From video', img)

    k = cv2.waitKey(30) & 0xFF

    if k == 27:
        break

capture.release()
cv2.destroyAllWindows


