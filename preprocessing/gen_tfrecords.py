import tensorflow as tf
from PIL import Image
import numpy as np
import cv2 as cv
import argparse
import csv
import sys
import os
import io

import preprocessing.constants as const


# Taken from tensor-flow object detection tutorial #####################################################################
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
########################################################################################################################


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
        if 'jpg' in file:

            with tf.gfile.GFile(directory + file, 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            img = Image.open(encoded_jpg_io)

            if img is not None:
                img_features = csv_map[file]
                width, height = img.size

                example = construct_feature(img_features, encoded_jpg, height, width, img_name=file)
                writer.write(example.SerializeToString())

                if counter % 50 == 0:
                    print(file, '{}: complete!'.format(counter))
                counter += 1

    writer.close()
    sys.stdout.flush()


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


# tf.compat.as_bytes() .tostring()
def construct_feature(boxes, encoded_img, encoded_img_height, encoded_img_width, img_name):
    encoded_img_format, encoded_img_label, encoded_img_label = b'jpg', b'hand', 1
    encoded_img_name = img_name.encode('utf-8')

    x_min, x_max, y_min, y_max = [], [], [], []
    classes, labels = [], []
    for box in boxes:
        x_min.append(box[0] / encoded_img_width)
        x_max.append(box[2] / encoded_img_width)
        y_min.append(box[1] / encoded_img_height)
        y_max.append(box[3] / encoded_img_height)
        classes.append(encoded_img_label)
        labels.append(encoded_img_label)

    example = tf.train.Example(features=tf.train.Features(feature={
        const.HEIGHT_KEY: int64_feature(encoded_img_height),
        const.WIDTH_KEY: int64_feature(encoded_img_width),
        const.FILENAME_KEY: bytes_feature(encoded_img_name),
        const.SOURCE_KEY: bytes_feature(encoded_img_name),
        const.ENCODED_IMAGE_KEY: bytes_feature(encoded_img),
        const.FORMAT_KEY: bytes_feature(encoded_img_format),
        const.XMIN_KEY: float_list_feature(x_min),
        const.XMAX_KEY: float_list_feature(x_max),
        const.YMIN_KEY: float_list_feature(y_min),
        const.YMAX_KEY: float_list_feature(y_max),
        const.CLASS_KEY: bytes_list_feature(classes),
        const.LABEL_KEY: int64_list_feature(labels)
    }))
    return example


def scale_boxes(scale_x, scale_y, box):
    return [int(box[0] * scale_x), int(box[1] * scale_y), int(box[2] * scale_x), int(box[3] * scale_y)]


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


# example = tf.train.Example(features=tf.train.Features(feature={
#     const.HEIGHT_KEY: int64_feature(encoded_img_height),
#     const.WIDTH_KEY: int64_feature(encoded_img_width),
#     const.FILENAME_KEY: bytes_feature(encoded_img_name),
#     const.SOURCE_KEY: bytes_feature(encoded_img_name),
#     const.ENCODED_IMAGE_KEY:  bytes_feature(encoded_img),
#     const.FORMAT_KEY: bytes_feature(encoded_img_format),
#     const.XMIN_KEY: float_list_feature([box[0] / encoded_img_width]),
#     const.XMAX_KEY: float_list_feature([box[2] / encoded_img_width]),
#     const.YMIN_KEY: float_list_feature([box[1] / encoded_img_height]),
#     const.YMAX_KEY: float_list_feature([box[3] / encoded_img_height]),
#     const.CLASS_KEY: bytes_list_feature([b'hand']),
#     const.LABEL_KEY:  int64_list_feature([1])
# }))
