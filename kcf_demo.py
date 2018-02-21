import cv2 as cv

tracker = cv.Tracker_create('MIL')

cap = cv.VideoCapture(0)
fgbg = cv.createBackgroundSubtractorMOG2()

while True:
    ok, frame = cap.read()
    fgmask = fgbg.apply(frame)
    blur = cv.GaussianBlur(fgmask, (5, 5), 0)
    _, thresh1 = cv.threshold(blur, 100, 255, cv.THRESH_BINARY)

    res = cv.bitwise_and(frame, frame, mask=thresh1)
    cv.imshow('window', res)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
