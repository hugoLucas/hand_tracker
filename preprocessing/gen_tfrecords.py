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
        img = load_image(directory + file)
        if img is not None:
            img_features = csv_map[file]

            example = construct_feature(img_features, img)
            writer.write(example.SerializeToString())
            print(file, '{}: complete!'.format(counter))
            counter += 1

    writer.close()
    sys.stdout.flush()


def load_image(file_path):
    img = cv.imread(file_path)

    try:
        img = cv.resize(img, const.IMG_SIZE, interpolation=cv.INTER_CUBIC)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        return img
    except cv.error:
        return None


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


def construct_feature(img_features, img):
    feature = {
        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(img.tostring())]))
    }
    for i in range(0, len(img_features)):
        feature['hand_{}'.format(i+1)] = tf.train.Feature(int64_list=tf.train.Int64List(value=img_features[i]))

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example

script_parser = argparse.ArgumentParser(description='Allows user to visualize the results of gen_data.py')
script_parser.add_argument(dest='root_dir', metavar='DIR', help='Root directory of extracted egohands_data.',
                           action=ReadableDir)
args = script_parser.parse_args()

root_dir = args.root_dir
if root_dir[-1] != '/':
    root_dir += '/'

train_directory = root_dir + const.TRAIN_DIRECTORY
test_directory = root_dir + const.TEST_DIRECTORY

# iterate_over_directory(train_directory)
iterate_over_directory(test_directory)



