import cv2 as cv
import time
import csv
import os

from data_gen.constants import TIME_LIMIT, FRAME_OFFSET

# What should we label these images as?
LABEL = 'radical'

# Read frames from webcam while user does hand gestures
cap = cv.VideoCapture(0)
start, diff = time.process_time(), 0
frames, counter = [], 0

while diff < TIME_LIMIT:
    ret, frame = cap.read()

    diff = time.process_time() - start
    cv.imshow('OUTPUT', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    if counter + 1 == FRAME_OFFSET:
        frames.append(frame)
    counter = (counter + 1) % FRAME_OFFSET

cap.release()

# Ask the user to draw bounding boxes around hand
boxes, current = [], 0
print('TOTAL: ', len(frames))
for frame in frames:
    bbox = cv.selectROI("Image", frame)
    cv.waitKey(0)
    boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
    print('DONE: ', current)
    current += 1

# Save the boxes in a .csv file and frames to folder
filename = './{}/{}.csv'.format(LABEL, LABEL)
file_exists = os.path.isfile(filename)

FRAME_NUM = 0
with open(filename, 'w') as csv_file:
    headers = ['image_name', 'box']
    writer = csv.DictWriter(csv_file, fieldnames=headers)

    if not file_exists:
        writer.writeheader()

    for frame, box in zip(frames, boxes):
        frame_name = 'frame{}.jpg'.format(FRAME_NUM)
        cv.imwrite('./{}/{}'.format(LABEL, frame_name), frame)

        writer.writerow({
            'image_name': frame_name,
            'box': box
        })
        FRAME_NUM += 1
