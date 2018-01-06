import random as ran
import cv2 as cv
import scipy.io
import argparse
import shutil
import csv
import os


IMG_ROOT_FOLDER = '_LABELLED_SAMPLES/'
LABELS_FILE = 'polygons.mat'
TRAIN_DIRECTORY = 'train/'
TEST_DIRECTORY = 'test/'

TRAIN_PROB = 0.80
TRAIN_SET = {}

TEST_PROB = 1 - TRAIN_PROB
TEST_SET = {}


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
    test_path, train_path = make_directories(root_dir)
    counter = 0

    for img_folder in sorted(os.listdir(img_root)):
        folder_dir = img_root + img_folder + '/'
        polygons, row_number = load_labels(folder_dir), 0

        for img_file in sorted(os.listdir(folder_dir)):
            if img_file != LABELS_FILE:
                bounding_boxes = gen_bounding_boxes(polygons[row_number])
                assign_set(folder_path=folder_dir, img_name=img_file, bounding_boxes=bounding_boxes, counter=counter,
                           test_path=test_path, train_path=train_path)
                row_number += 1
                counter += 1
        break
    gen_csv_files(test_path, train_path)


def create_or_delete_directory(directory_path):
    if os.path.isdir(directory_path):
        shutil.rmtree(directory_path)
    os.mkdir(directory_path)

    return directory_path


def make_directories(root_dir):
    test_path = create_or_delete_directory(root_dir + TEST_DIRECTORY)
    train_path = create_or_delete_directory(root_dir + TRAIN_DIRECTORY)
    return test_path, train_path


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


def assign_set(folder_path, img_name, bounding_boxes, counter, test_path, train_path):
    ran_num = ran.random()

    new_filename = 'img_{}.jpg'.format(counter)
    if ran_num <= TRAIN_PROB:
        shutil.copy(src=folder_path + img_name, dst=train_path + new_filename)
        TRAIN_SET[new_filename] = bounding_boxes
    else:
        shutil.copy(src=folder_path + img_name, dst=test_path + new_filename)
        TEST_SET[new_filename] = bounding_boxes


def visualize_results(img_path, boxes):
    img = cv.imread(img_path)
    for box in boxes:
        cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(256, 0, 0))
    cv.imshow('window', img)
    cv.waitKey(0)


def gen_csv_files(test_path, train_path):
    populate_csv_file(test_path, TEST_SET)
    populate_csv_file(train_path, TRAIN_SET)


def populate_csv_file(file_path, csv_data):
    headers = ['filename', 'hand_1', 'hand_2', 'hand_3', 'hand_4']
    with open(file_path + 'test.csv', 'a') as test_file:
        writer = csv.DictWriter(test_file, fieldnames=headers)

        writer.writeheader()
        for key in csv_data.keys():
            data = csv_data[key]
            while len(data) < 4:
                data.append(None)

            writer.writerow({'filename': key, 'hand_1': data[0], 'hand_2': data[1], 'hand_3': data[2],
                             'hand_4': data[3]})

script_parser = argparse.ArgumentParser(description='Takes the EgoHands Labeled Data zip file and creates a train and '
                                                    'test set. Creates .csv files with bounding boxes for both data '
                                                    'sets.')
script_parser.add_argument(dest='root_dir', metavar='DIR', help='Root directory of extracted egohands_data.',
                           action=ReadableDir)
args = script_parser.parse_args()
process_images(args.root_dir)
