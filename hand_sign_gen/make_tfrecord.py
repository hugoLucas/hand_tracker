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

    for folder in listdir(IMG_ROOT_DIR):
        folder_path = path.join(IMG_ROOT_DIR, folder)
        csv_data = load_csv(folder, folder_path)

        writer = train_writer if random() < TRAIN_THRESH else test_writer
        process_folder(folder_path, csv_data, folder, writer)

    test_writer.close()
    train_writer.close()
    stdout.flush()


def process_folder(folder_path, folder_csv, img_label, tf_writer):
    folder_files = filter(lambda f: f.endswith('.jpg'), listdir(folder_path))
    for img_name in folder_files:
        img_data = folder_csv.get(img_name, None)
        if img_data is None:
            print('Unable to retrieve data for {}'.format(img_name))
        else:
            img = load_img(path.join(folder_path, img_name), img_data)
            feature = {
                'label': bytes_feature(compat.as_bytes(img_label)),
                'image': bytes_feature(compat.as_bytes(img.tostring()))
            }

            example = train.Example(features=train.Features(feature=feature))
            tf_writer.write(example.SerializeToString())


def load_img(img_path, img_data):
    img = cv.imread(img_path)
    rows, columns, channels = img.shape
    clean_box_data(img_data, rows, columns)

    img = img[int(img_data[1]):int(img_data[3]), int(img_data[0]):int(img_data[2])]
    img = cv.resize(img, (300, 300), interpolation=cv.INTER_CUBIC)
    img = img.astype(np.float32)
    return img


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


process_img_directories()
