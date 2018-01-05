import scipy.io as sio
import argparse

parser = argparse.ArgumentParser(description='Takes the EgoHands Labeled Data zip file and creates a train and'
                                             'test set. Creates .csv files with bounding boxes for both data '
                                             'sets.')
parser.add_argument(name='root_dir', metavar='R', type=str)