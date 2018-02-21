from os import listdir, path
from csv import DictReader

from hand_sign_gen.constants import IMG_ROOT_DIR, HEADER_KEY, HEADER_VALUE


def process_img_directories():
    for folder in listdir(IMG_ROOT_DIR):
        csv_data = load_csv(folder, path.join(IMG_ROOT_DIR, folder))


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
