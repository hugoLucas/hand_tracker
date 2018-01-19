from argparse import Action, ArgumentTypeError
from os import path, access, R_OK


class ReadableDir(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not path.isdir(prospective_dir):
            raise ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
        if access(prospective_dir, R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))
