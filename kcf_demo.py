import cv2 as cv

tracker = cv.Tracker_create('MIL')

cap = cv.VideoCapture(0)
_, frame = cap.read()

rows, columns, channels = frame.shape
y1, x1 = int(rows/2), int(columns/2)
y2, x2 = int(rows * .40), int(columns * .40)

bbox = (x1, y1, x2, y2)
ok = tracker.init(frame, bbox)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    cv.rectangle(img=frame, pt1=(x1, y1), pt2=(x1 + x2, y1 + y2), color=(0, 256, 0), thickness=1)

    ok, bbox = tracker.update(frame)
    print(ok, bbox)

    if ok:
        p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv.rectangle(frame, p1, p2, (200, 0, 0), 2, 1)

    cv.imshow('window', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
