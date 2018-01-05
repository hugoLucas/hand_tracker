import argparse
import os


class ReadableDir(argparse.Action):
    def __call__(self, prser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))


parser = argparse.ArgumentParser(description='Takes the EgoHands Labeled Data zip file and creates a train and'
                                             'test set. Creates .csv files with bounding boxes for both data '
                                             'sets.')
parser.add_argument(dest='root_dir', metavar='DIR', help='Root directory of extracted egohands_data.',
                    action=ReadableDir)

args = parser.parse_args()
print(args.root_dir)
