import pandas as pd
from string import ascii_letters
from utils import Camera

camera = Camera()


def get_camera_list():
    return camera.get_camera


def check_camera(camera_index):
    Camera.test_camera(camera_index=camera_index)


if __name__ == "__main__":
    import argparse
    from tabulate import tabulate

    parser = argparse.ArgumentParser(description="Camera params")
    parser.add_argument("ls", nargs="?", help="list all cameras")
    parser.add_argument("-c", "--test", type=int, help="Check camera")
    args = parser.parse_args()
    if args.ls:
        camera_list = get_camera_list()
        df = pd.DataFrame(camera_list)
        df.columns = df.columns.str.title()
        df = df.to_dict(orient="list")
        print("\n\nAvailable video capture devices (cameras)")
        print(tabulate(df, headers="keys", tablefmt="grid"))
        print("\n")
    if args.test is not None:
        check_camera(args.test)
