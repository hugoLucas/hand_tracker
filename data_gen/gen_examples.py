import cv2 as cv
import time
import csv
import os

from data_gen.constants import TIME_LIMIT, FRAME_OFFSET

# What should we label these images as?
LABEL = 'ok_sign'

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
    bbox = cv.selectROI("Image", frame, False, False)
    if cv.waitKey(1) & 0xFF == ord('q'):
        print('Skip!')
    else:
        boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        print('DONE: ', current+1)
        current += 1

# Save the boxes in a .csv file and frames to folder
root_directory = './images'
if not os.path.exists(root_directory):
    os.mkdir(root_directory)

image_directory = os.path.join(root_directory, LABEL)
if not os.path.exists(image_directory):
    os.mkdir(image_directory)

csv_path = os.path.join(image_directory, '{}.csv'.format(LABEL))
file_exists = os.path.exists(csv_path)

FRAME_NUM = sum(1 for line in open(csv_path, 'a+'))
with open(csv_path, 'a') as csv_file:

    headers = ['image_name', 'box']
    writer = csv.DictWriter(csv_file, fieldnames=headers)

    if not file_exists:
        writer.writeheader()

    for frame, box in zip(frames, boxes):
        frame_name = 'frame_{}.jpg'.format(FRAME_NUM)
        cv.imwrite(os.path.join(image_directory, frame_name), frame)

        writer.writerow({
            'image_name': frame_name,
            'box': box
        })
        FRAME_NUM += 1
