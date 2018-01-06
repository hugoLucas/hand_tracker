import scipy.io
import argparse
import numpy as np
import cv2 as cv
import os


IMG_ROOT_FOLDER = '_LABELLED_SAMPLES/'
TEST_DIRECTORY = 'test/'
TRAIN_DIRECTORY = 'train/'
LABELS_FILE = 'polygons.mat'


class ReadableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))


def get_image_root_directory(root_dir):
    if root_dir[-1] != '/':
        root_dir += '/'
    return root_dir + IMG_ROOT_FOLDER


def process_images(root_dir):
    img_root = get_image_root_directory(root_dir)

    for img_folder in sorted(os.listdir(img_root)):
        folder_dir = img_root + img_folder + '/'
        polygons, row_number = load_labels(folder_dir), 0

        for img_file in sorted(os.listdir(folder_dir)):
            if img_file != LABELS_FILE:
                current_img = cv.imread(folder_dir + img_file)
                bounding_boxes = gen_bounding_boxes(polygons[row_number])
                row_number += 1
                break
        break


def create_or_delete_directory(directory_path):
    if os.path.isdir(directory_path):
        # Delete old contents and start again
        pass
    else:
        # Create an empty directory
        pass


def make_directories(root_dir):
    test_dir, train_dir = root_dir + TEST_DIRECTORY, root_dir + TRAIN_DIRECTORY
    create_or_delete_directory(test_dir)
    create_or_delete_directory(train_dir)


def load_labels(folder):
    labels_path = folder + LABELS_FILE

    mat = scipy.io.loadmat(labels_path)
    polygons = mat['polygons'][0]

    return polygons


def gen_bounding_boxes(polygon_row):
    boxes = []
    for hand_pixels in polygon_row:
        if hand_pixels.shape[0] > 1:
            x_min, y_min = hand_pixels[0]
            x_max, y_max = hand_pixels[0]

            for x_px, y_px in hand_pixels:
                if x_px < x_min:
                    x_min = x_px
                elif x_px > x_max:
                    x_max = x_px
                if y_px < y_min:
                    y_min = y_px
                elif y_px > y_max:
                    y_max = y_px

            boxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
    return boxes


script_parser = argparse.ArgumentParser(description='Takes the EgoHands Labeled Data zip file and creates a train and '
                                                    'test set. Creates .csv files with bounding boxes for both data '
                                                    'sets.')
script_parser.add_argument(dest='root_dir', metavar='DIR', help='Root directory of extracted egohands_data.',
                           action=ReadableDir)
args = script_parser.parse_args()
process_images(args.root_dir)
