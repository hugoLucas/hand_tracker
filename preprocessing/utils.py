import preprocessing.constants as const
import tensorflow as tf

from scipy.io import loadmat
from csv import DictReader
from io import BytesIO
from PIL import Image
from os import path


# ################################### Taken from tensor-flow object detection tutorial ############################### #
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
# #################################################################################################################### #


def construct_feature(boxes, encoded_img, encoded_img_height, encoded_img_width, img_name):
    encoded_img_format, encoded_img_class, encoded_img_label = b'jpg', b'hand', 1
    encoded_img_name = img_name.encode('utf-8')

    x_min, x_max, y_min, y_max = [], [], [], []
    classes, labels = [], []
    for box in boxes:
        x_min.append(box[0] / encoded_img_width)
        x_max.append(box[2] / encoded_img_width)
        y_min.append(box[1] / encoded_img_height)
        y_max.append(box[3] / encoded_img_height)
        classes.append(encoded_img_class)
        labels.append(encoded_img_label)

    if len(x_min) == len(x_max) == len(y_min) == len(y_max):
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
    else:
        raise ValueError(img_name)


def parse_data(data_str):
    if len(data_str) > 0:
        data_str = data_str[1:-1]
        return [int(s) for s in data_str.split(',')]


def load_encoded_image(image_path):
    """
    Loads an image in a format suitable for encoding into a tf record file.

    :param image_path: the full path to a .jpg image
    :return: the encoded image, its height, its width
    """
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = BytesIO(encoded_jpg)
    img = Image.open(encoded_jpg_io)

    height, width = img.size
    return encoded_jpg, height, width


def load_coordinates(mat_data):
    """
    Iterates through a .mat file and finds the coordinates for all hands in a given image.

    :param mat_data: the data from a .mat file that has been indexed to access the 'boxes' structure in the file
    :return: returns a list of hand coordinates in form [upper lhs x, upper lhs y, lower rhs x, lower rhs y]
    """
    coordinate_list = []
    for coordinate_set in mat_data:
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


def process_mat(base_dir, mat_file_name):
    """
    Opens the corresponding .mat file for a given image and retrieves the images hand coordinates

    :param base_dir: the directory passed to the script as input
    :param mat_file_name: the file name of the .mat file for the image
    :return: a list of hand coordinates
    """

    try:
        mat_path = path.join(base_dir, mat_file_name)
        mat_file = loadmat(mat_path)['boxes'][0]
        return load_coordinates(mat_file)
    except MemoryError:
        print('Memory Error encountered for file: {}'.format(mat_file_name))
        return None


def load_csv_file(directory):
    csv_file, csv_data = directory + const.CSV, {}
    with open(csv_file, 'r') as data_file:
        reader = DictReader(data_file)
        for row in reader:
            row_key = row['filename']
            row_data = [parse_data(row['hand_1']), parse_data(row['hand_2']), parse_data(row['hand_3']),
                        parse_data(row['hand_4'])]
            row_data = list(filter(lambda x: x is not None, row_data))
            csv_data[row_key] = row_data
    return csv_data

