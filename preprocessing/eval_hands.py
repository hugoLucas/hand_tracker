from os import path, listdir
from scipy.io import loadmat
from cv2 import imread, imshow, waitKey, rectangle


def load_coordinates(mat_file):
    coordinate_list = []
    for coordinate_set in mat_file:
        coordinate_array = coordinate_set[0][0]
        n_hands = int(len(coordinate_array) / 4)
        for i in range(0, n_hands):
            start_index = 4 * i
            p1, p2, p3, p4 = coordinate_array[start_index][0], coordinate_array[start_index + 1][0], \
                             coordinate_array[start_index + 2][0], coordinate_array[start_index + 3][0]
            y_min, x_min = min(p1[0], p2[0], p3[0], p4[0]), min(p1[1], p2[1], p3[1], p4[1])
            y_max, x_max = max(p1[0], p2[0], p3[0], p4[0]), max(p1[1], p2[1], p3[1], p4[1])
            coordinate_list.append([x_min, y_min, x_max, y_max])
    return coordinate_list


def process_mat(mat_file_name):
    mat_path = path.join('/home/hugolucas/ml_data/hand_dataset/training_dataset/training_data/annotations',
                         mat_file_name)
    mat_file = loadmat(mat_path)
    return load_coordinates(mat_file['boxes'][0])


DIRECTORY = '/home/hugolucas/ml_data/hand_dataset'
IMAGES = 'training_dataset/training_data/images'
COLOR = (0, 255, 0)

IMG_DIR = path.join(DIRECTORY, IMAGES)
for img_filename in listdir(IMG_DIR):
    mat_filename = img_filename.replace('.jpg', '.mat')
    hands = process_mat(mat_filename)

    img = imread(path.join(IMG_DIR, img_filename))
    height, width, channels = img.shape
    for x_mi, y_mi, x_ma, y_ma in hands:
        if x_mi > width or y_mi > height or y_ma > height or x_ma > width:
            print(x_mi, y_mi, x_ma, y_ma)
            COLOR = (255, 0, 0)

        rectangle(img, (int(x_mi), int(y_mi)), (int(x_ma), int(y_ma)), COLOR, thickness=3)
        COLOR = (0, 255, 0)

    imshow('window', img)
    waitKey(0)

