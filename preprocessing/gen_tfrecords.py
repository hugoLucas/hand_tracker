import tensorflow as tf
import numpy as np
import cv2 as cv
import argparse
import csv
import sys
import os

import preprocessing.constants as const


class ReadableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))


def iterate_over_directory(directory):
    file_name = 'train.tfrecords' if 'train/' in directory else 'test.tfrecords'
    writer = tf.python_io.TFRecordWriter(directory + file_name)

    csv_map = load_csv_file(directory)
    counter = 1
    for file in os.listdir(directory):
        img, img_shape = load_image(directory + file)
        if img is not None:
            img_features = csv_map[file]

            for box in img_features:
                example = construct_feature(box, img, img_shape, file)
                writer.write(example.SerializeToString())

            if counter % 50 == 0:
                print(file, '{}: complete!'.format(counter))
            counter += 1

    writer.close()
    sys.stdout.flush()


def load_image(file_path):
    img = cv.imread(file_path)

    try:
        img = cv.resize(img, const.NEW_IMG_SIZE, interpolation=cv.INTER_CUBIC)
        shape = img.shape
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        return img, shape
    except cv.error:
        return None, None


def load_csv_file(directory):
    csv_file, csv_data = directory + const.CSV, {}
    with open(csv_file, 'r') as data_file:
        reader = csv.DictReader(data_file)
        for row in reader:
            row_key = row['filename']
            row_data = [parse_data(row['hand_1']), parse_data(row['hand_2']), parse_data(row['hand_3']),
                        parse_data(row['hand_4'])]
            row_data = list(filter(lambda x: x is not None, row_data))
            csv_data[row_key] = row_data
    return csv_data


def parse_data(data_str):
    if len(data_str) > 0:
        data_str = data_str[1:-1]
        return [int(s) for s in data_str.split(',')]


def construct_feature(box, encoded_img, encoded_img_shape, img_name):
    height, width, channels = encoded_img_shape
    encoded_img_format = b'jpg'
    encoded_img_name = img_name.encode('utf-8')
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_img_name])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_img_name])),
        'image/encoded':  tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[tf.compat.as_bytes(encoded_img.tostring())])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_img_format])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=[box[0]])),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=[box[2]])),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=[box[1]])),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=[box[3]])),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'hand'])),
        'image/object/class/label':  tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
    }))
    return example


def scale_boxes(scale_x, scale_y, box):
    return [int(box[0] * scale_x), int(box[1] * scale_y), int(box[2] * scale_x), int(box[3] * scale_y)]


def verify_scaling(path, features):
    img = cv.imread(path)
    img = cv.resize(img, const.NEW_IMG_SIZE, interpolation=cv.INTER_CUBIC)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    s_y, s_x = const.NEW_IMG_SIZE[0] / const.OLD_IMG_SIZE[0], const.NEW_IMG_SIZE[1] / const.OLD_IMG_SIZE[1]
    for box in features:
        n_f = scale_boxes(s_x, s_y, box)
        cv.rectangle(img, (n_f[0], n_f[1]), (n_f[2], n_f[3]), (0, 256, 0))
        cv.imshow('window', img)
        cv.waitKey(0)


script_parser = argparse.ArgumentParser(description='Allows user to visualize the results of gen_data.py')
script_parser.add_argument(dest='root_dir', metavar='DIR', help='Root directory of extracted egohands_data.',
                           action=ReadableDir)
args = script_parser.parse_args()

root_dir = args.root_dir
if root_dir[-1] != '/':
    root_dir += '/'

train_directory = root_dir + const.TRAIN_DIRECTORY
test_directory = root_dir + const.TEST_DIRECTORY

iterate_over_directory(train_directory)
iterate_over_directory(test_directory)
