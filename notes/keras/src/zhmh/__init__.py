import os


def make_dump(dump_path):
    if os.path.exists(dump_path):
        if not os.path.isdir(dump_path):
            print(dump_path, "is not a valid path.")
            exit(1)
    else:
        os.mkdir(dump_path)
