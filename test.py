import cv2 as cv
import time

Conf_threshold = 0.3
NMS_threshold = 0.3

class_name = 'tank'

net = cv.dnn.readNet('tanks_final.weights', 'tanks.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

cap = cv.VideoCapture('assets/4.mp4')

while True:
    ret, frame = cap.read()

    if ret == False:
        break
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = (0, 0, 0)
        label = "%s : %f" % (class_name, score)
        cv.rectangle(frame, box, color, 1)
        cv.putText(frame, label, (box[0], box[1] - 10),
                   cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)

    cv.imshow('frame', frame)
    key = cv.waitKey(1)

    if key == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
