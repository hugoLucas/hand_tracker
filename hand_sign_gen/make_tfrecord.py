import numpy as np
import cv2 as cv

from preprocessing.utils import bytes_feature, int64_feature
from tensorflow import python_io, compat, train
from os import listdir, path
from csv import DictReader
from random import random
from sys import stdout

from hand_sign_gen.constants import IMG_ROOT_DIR, HEADER_KEY, HEADER_VALUE, TRAIN_THRESH


def process_img_directories():
    test_writer = python_io.TFRecordWriter('./test.tfrecord')
    train_writer = python_io.TFRecordWriter('./train.tfrecord')

    folder_label, label_map = 0, {}
    ex_tst, ex_trn = 0, 0
    for folder in listdir(IMG_ROOT_DIR):
        folder_path = path.join(IMG_ROOT_DIR, folder)
        csv_data = load_csv(folder, folder_path)

        n_test, n_train = process_folder(folder_path, csv_data, folder_label, test_writer, train_writer)
        ex_tst, ex_trn = ex_tst + n_test, ex_trn + n_train

        label_map[folder] = folder_label
        folder_label += 1

    test_writer.close()
    train_writer.close()
    stdout.flush()

    print('-' * 80)
    print(label_map)
    print('-' * 80)
    print('Number of training examples: ', ex_trn)
    print('Number of testing examples: ', ex_tst)


def process_folder(folder_path, folder_csv, img_label, test_writer, train_writer):
    folder_files = filter(lambda f: f.endswith('.jpg'), listdir(folder_path))

    ex_test, ex_train = 0, 0
    for img_name in folder_files:
        img_data = folder_csv.get(img_name, None)
        if img_data is None:
            print('Unable to retrieve data for {}'.format(img_name))
        else:
            img = load_img(path.join(folder_path, img_name), img_data)
            if img is not None:
                assert img.shape == (300, 300, 3)
                feature = {
                    'label': int64_feature(img_label),
                    'image': bytes_feature(img.tostring())
                }

                example = train.Example(features=train.Features(feature=feature))
                if put_in_training_set():
                    train_writer.write(example.SerializeToString())
                    ex_train += 1
                else:
                    test_writer.write(example.SerializeToString())
                    ex_test += 1
    return ex_test, ex_train

def load_img(img_path, img_data):
    img = cv.imread(img_path)
    rows, columns, channels = img.shape
    clean_box_data(img_data, rows, columns)

    try:
        img = img[int(img_data[1]):int(img_data[3]), int(img_data[0]):int(img_data[2])]
        img = cv.resize(img, (300, 300), interpolation=cv.INTER_CUBIC)
        img = img.astype(np.float32)
        return img
    except cv.error:
        print("Unable to process img {}, skipped.".format(img_path))
        return None


def clean_box_data(img_data, img_height, img_width):
    for i in range(0, len(img_data)):
        point = img_data[i]
        if point < 0:
            img_data[i] = 0
        elif i == 0 or i == 2:
            if point > img_width:
                img_data[i] = img_width
        else:
            if point > img_height:
                img_data[i] = img_height


def load_csv(csv_label, csv_path):
    csv_data = {}
    total_path = path.join(csv_path, csv_label + '.csv')
    with open(total_path, 'r') as data_file:
        reader = DictReader(data_file)
        for row in reader:
            row_key = row[HEADER_KEY]
            row_data = list(map(lambda v: float(v), row[HEADER_VALUE][1:-1].split(', ')))
            csv_data[row_key] = row_data
    return csv_data

def put_in_training_set():
    return random() < TRAIN_THRESH

process_img_directories()
