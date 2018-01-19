import preprocessing.constants as const
import tensorflow as tf

from preprocessing.utils import construct_feature, load_encoded_image, process_mat, load_csv_file
from preprocessing.reachable_dir import ReadableDir
from argparse import ArgumentParser
from os import path, listdir
from sys import stdout


def process_data(ego_directory, hand_directory, out_directory, training=True):
    """
    Creates a training or test .tfrecord file by iterating over the training/test sub-folders of both data sets.

    :param training: True if function is being used to generate a training tf record file
    :param ego_directory: directory of the Training sub-folder created by gen_directories.py
    :param hand_directory: directory of the Training sub-folder that comes with the HANDS data set
    :param out_directory: output directory to store tfrecord in
    :return: None
    """
    if training:
        writer = tf.python_io.TFRecordWriter(path.join(out_directory + 'train.tfrecords'))
    else:
        writer = tf.python_io.TFRecordWriter(path.join(out_directory + 'test.tfrecords'))

    process_hands(hand_directory, writer)
    process_ego_hands(ego_directory, writer)
    writer.close()
    stdout.flush()


def process_hands(directory, writer):
    """
    Iterate over a given directory in the HANDS data set and write its data to tf record writer

    :param directory: a directory (training or test) in the extracted HANDS folder
    :param writer: a tf.python_io.TFRecordWriter object
    :return: None
    """
    for img_filename in listdir(directory):
        mat_filename = img_filename.replace('.jpg', '.mat')
        hand_coordinates = process_mat(directory, mat_filename)
        encoded_jpg, height, width = load_encoded_image(path.join(directory, img_filename))

        if encoded_jpg is not None:
            example = construct_feature(hand_coordinates, encoded_jpg, height, width, img_name=img_filename)
            writer.write(example.SerializeToString())


def process_ego_hands(directory, writer):
    """
    Iterate over a given directory in the Ego Hands data set and write its data to tf record writer

    :param directory: a directory (training or test) in the extracted Ego Hands folder
    :param writer: a tf.python_io.TFRecordWriter object
    :return: None
    """
    csv_map = load_csv_file(directory)
    for file in listdir(directory):
        if 'jpg' in file:

            encoded_jpg, height, width = load_encoded_image(path.join(directory, file))
            if encoded_jpg is not None:
                img_features = csv_map[file]
                example = construct_feature(img_features, encoded_jpg, height, width, img_name=file)
                writer.write(example.SerializeToString())


script_parser = ArgumentParser(description='Allows user to visualize the results of gen_directories.py')
script_parser.add_argument(dest='ego_dir', metavar='E_DIR', help='Root directory of extracted egohands_data.',
                           action=ReadableDir)
script_parser.add_argument(dest='hand_dir', metavar='H_DIR', help='Root directory of extracted hand_dataset.',
                           action=ReadableDir)
script_parser.add_argument(dest='out_dir', metavar='O_DIR', help='Output directory.',
                           action=ReadableDir)

args = script_parser.parse_args()
ego_dir, hand_dir, out_dir = args.ego_dir, args.hand_dir, args.out_dir

ego_train_directory, ego_test_directory = path.join(ego_dir, const.EGO_TRAIN_DIRECTORY), \
                                          path.join(ego_dir, const.EGO_TEST_DIRECTORY)

hands_train_directory, hand_test_directory = path.join(hand_dir, const.HAND_TRAIN_DIRECTORY), \
                                             path.join(hand_dir, const.HAND_TEST_DIRECTORY)

# process_training_data(ego_train_directory, hands_train_directory, out_dir)
# process_training_data(ego_test_directory, hands_test_directory, out_dir, training=False)
