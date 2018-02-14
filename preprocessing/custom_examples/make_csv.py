from os import path, listdir
from csv import DictWriter
import xml.etree.ElementTree as elTree


DATA = '/home/hugolucas/ml_data/custom_examples/'


def write_to_csv():
    data = process_xml_files()
    headers = ['filename', 'hand_1', 'hand_2']

    with open(path.join(DATA, 'data.csv'), 'a') as test_file:
        writer = DictWriter(test_file, fieldnames=headers)
        writer.writeheader()

        for key in data.keys():
            hand_set = data[key]
            writer.writerow({'filename': key,
                             'hand_1': hand_set[0],
                             'hand_2': hand_set[1] if len(hand_set) > 1 else None})


def process_xml_files():
    valid_files = get_file_list()

    results = {}
    for file in valid_files:
        xml = elTree.parse(path.join(DATA, file))

        file_hands = []
        for hand in xml.getroot().iter('object'):
            for bbox in hand.iter('bndbox'):
                file_hands.append([int(bbox[0].text), int(bbox[1].text), int(bbox[2].text), int(bbox[3].text)])
        results[file[:-4] + ".jpg"] = file_hands
    return results


def get_file_list():
    directory_files = listdir(DATA)
    xml_files = list(filter(lambda f: f[-4:] == '.xml', directory_files))
    img_files = find_image_files(directory_files)
    valid_files = filter_xml_files(xml_files, img_files)

    return valid_files


def find_image_files(directory_files):
    file_map = {}
    for file in directory_files:
        if file.endswith('.jpg'):
            file_map[file] = True
    return file_map


def filter_xml_files(xml_files, img_files):
    for xml in xml_files:
        image_name = xml[:-4] + '.jpg'

        if not img_files.get(image_name, False):
            xml_files.remove(xml)
            print('{} does not have an accompanying image.'.format(xml))
    return xml_files

write_to_csv()
