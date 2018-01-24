import cv2 as cv
from os import path

output_directory = '/home/hugolucas/ml_data/custom_examples'

camera = cv.VideoCapture(0)

counter = 0
while True:
    ret, frame = camera.read()
    cv.imshow('Output', frame)

    response = cv.waitKey(1) & 0xFF
    if response == ord('q'):
        break
    elif response == ord('c'):
        cv.imwrite(path.join(output_directory, 'frame_{}.jpg'.format(counter)), frame)
        print('Image {} created!'.format(counter))
        counter += 1

camera.release()
cv.destroyAllWindows()
