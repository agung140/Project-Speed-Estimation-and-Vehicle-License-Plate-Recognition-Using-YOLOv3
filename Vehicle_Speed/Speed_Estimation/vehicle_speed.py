# Import necessary packages

import cv2
import csv
import collections
import numpy as np
from tracker2 import *
import time
import datetime

end = 0
ids_lst = []
spd_lst = []
frameIndex = 0

# Initialize Tracker
tracker = EuclideanDistTracker()

# Initialize the videocapture object
cap = cv2.VideoCapture('input/MVI_2923.mp4')
input_size = 416
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frame_count = 0

# Detection confidence threshold
confThreshold = 0.3
nmsThreshold = 0.2

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# Store Coco Names in a list
classesFile = "yolo-coco/skripsi.names"
classNames = open(classesFile).read().strip().split('\n')

# class index for our required detection classes
required_class_index = [0, 1, 3, 4]
detected_classNames = []

# Model Files
modelConfiguration = 'yolo-coco/yolov3-training.cfg'
modelWeigheights = 'yolo-coco/yolov3-training_last.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Configure the network backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


# Function for speed vehicle
def speed_vehicle(box_id, img):
    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center

    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)

    # get speed
    if tracker.getsp(id) < tracker.limit():
        cv2.putText(img, str(tracker.getsp(id)) + " km/h ", (center), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    0.5, (255, 255, 0), 1)

    if tracker.getsp(id) > tracker.limit():
        cv2.putText(img, str(tracker.getsp(id)) + " km/h ", (center), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                    (0, 0, 255), 1)

    s = tracker.getsp(id)
    if tracker.f[id] == 1 and s != 0:
        tracker.capture(img, x, y, h, w, s, id)


# Function for finding the detected objects from the network output
def postProcess(outputs, img):
    global detected_classNames
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w, h = int(det[2] * width), int(det[3] * height)
                    x, y = int((det[0] * width) - w / 2), int((det[1] * height) - h / 2)
                    boxes.append([x, y, w, h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)

    # print(classIds)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

            color = [int(c) for c in colors[classIds[i]]]
            name = classNames[classIds[i]]
            detected_classNames.append(name)
            # Draw classname and confidence score
            cv2.putText(img, f'{name.lower()} {int(confidence_scores[i] * 100)}%', (x, y - 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1)

            # Draw bounding rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            detection.append([x, y, w, h, required_class_index.index(classIds[i])])

        # Update the tracker for each object
        boxes_ids = tracker.update(detection)
        for box_id in boxes_ids:
            speed_vehicle(box_id, img)


while True:
    (success, img) = cap.read()

    if success == True:
        frame_count += 1

    percent = 90
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)

    cv2.resize(img, dim, fx=0.5, fy=0.5)
    ih, iw, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

    # Set the input of the network
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    # Feed data to the network
    outputs = net.forward(outputNames)

    # Find the objects from the network output
    postProcess(outputs, img)

    cv2.line(img, (0, 235), (2000, 235), (0, 255, 45), 2)  # start 2
    cv2.line(img, (0, 255), (2000, 255), (0, 255, 255), 2)  # start 1

    cv2.line(img, (0, 135), (2000, 135), (0, 255, 45), 2)  # finish 2
    cv2.line(img, (0, 155), (2000, 155), (0, 255, 255), 2)  # finish 1

    # Draw counting texts in the frame
    cv2.putText(img, "Jumlah Kendaraan Yang Melintas: " + str(tracker.count), (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                font_color,
                font_thickness)

    cv2.putText(img, "Jumlah Kendaraan Yang Melanggar: " + str(tracker.exceeded), (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                font_color,
                font_thickness)

    # DISPLAY DATE, TIME, FPS & CURRENT FRAME
    cv2.line(img, (0, 10), (2000, 10), (79, 79, 47), 30)
    d = str(datetime.datetime.now().strftime("%d-%m-%y"))
    t = str(datetime.datetime.now().strftime("%H-%M-%S"))
    cv2.putText(img, f'DATE: {d} |', (25, 19), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 2)
    cv2.putText(img, f'TIME: {t} |', (209, 19), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 2)
    cv2.putText(img, f'FPS: {fps} |', (393, 19), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 2)
    cv2.putText(img, f'FRAMES: {frame_count} of {total_frames} ', (510, 19), cv2.FONT_HERSHEY_PLAIN, 1.1,
                (255, 255, 255), 2)
    cv2.line(img, (0, 26), (2000, 26), (255, 255, 255), 2)

    cv2.imwrite("output/frame_detection/frame-{}.png".format(frameIndex), img)

    # DATA ALLOCATION
    ids_lst, spd_lst = tracker.dataset()

    # Show the frames
    cv2.imshow('Output', img)
    frameIndex += 1

    if cv2.waitKey(1) == ord('q'):
        tracker.end()
        tracker.datavis(ids_lst, spd_lst)
        end = 1
        break

if end != 1:
    tracker.end()

    # Finally realese the capture object and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()


