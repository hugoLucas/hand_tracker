import random as ran
import cv2 as cv
import argparse
import csv
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


def verify_test_train_set_existence(parser, root):
    if not os.path.isdir(root + const.EGO_TRAIN_DIRECTORY) or not os.path.isdir(root + const.EGO_TEST_DIRECTORY):
        parser.error('Please run gen_data script first to generate a test and train set.')


def visualize_results(root):
    while True:
        directories = [root + const.EGO_TEST_DIRECTORY, root + const.EGO_TRAIN_DIRECTORY]

        random_directory = directories[ran.randint(0, len(directories) - 1)]
        image_name, image = gen_random_image(random_directory)
        boxes = get_image_boxes(random_directory, image_name)

        if boxes is not None:
            for box in boxes:
                if box is not None:
                    x_min, y_min, x_max, y_max = box
                    cv.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 256, 0), thickness=3)
            cv.imshow(random_directory + image_name, image)
            key_val = cv.waitKey(0)

            # Value of q
            if key_val == 1048689:
                break


def gen_random_image(directory):
    img_files = os.listdir(directory)
    ran_image = img_files[ran.randint(0, len(img_files) - 1)]

    return ran_image, cv.imread(directory + ran_image)


def get_image_boxes(directory, img_name):
    csv_file = directory + const.CSV
    with open(csv_file, 'r') as data_file:
        reader = csv.DictReader(data_file)
        for row in reader:
            if row['filename'] == img_name:
                return [parse_data(row['hand_1']), parse_data(row['hand_2']), parse_data(row['hand_3']),
                        parse_data(row['hand_4'])]


def parse_data(data_str):
    if len(data_str) > 0:
        data_str = data_str[1:-1]
        return [int(s) for s in data_str.split(',')]

script_parser = argparse.ArgumentParser(description='Allows user to visualize the results of gen_directories.py')
script_parser.add_argument(dest='root_dir', metavar='DIR', help='Root directory of extracted egohands_data.',
                           action=ReadableDir)
args = script_parser.parse_args()

root_dir = args.root_dir
if root_dir[-1] != '/':
    root_dir += '/'

verify_test_train_set_existence(script_parser, root_dir)

visualize_results(root_dir)
