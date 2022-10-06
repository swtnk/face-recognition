import argparse
from config import Config
from utils import Recognizer, setup_project, clean_project

CAMERA_INDEX = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition")
    parser.add_argument(
        "action",
        nargs="?",
        help="Actions: setup, clean, add, train, recognize, add_train_recognize, add_train, train_recognize",
    )
    parser.add_argument("-c", "--camera", type=int, default=0, help="camera index")
    parser.add_argument("-v", "--version", action=argparse.BooleanOptionalAction, help="version info")
    args = parser.parse_args()

    if args.camera is not None:
        CAMERA_INDEX = args.camera

    if args.version:
        with open(Config.BASE_DIR / 'VERSION', 'r') as fread:
            print(f'{fread.read()}')

    recognizer = Recognizer(camera_index=CAMERA_INDEX)

    match args.action:
        case "setup":
            setup_project()
        case "clean":
            clean_project()
        case "reset":
            clean_project()
            setup_project()
        case "add":
            recognizer.add_face()
        case "train":
            recognizer.train()
        case "recognize":
            recognizer.recognize()
        case "add_train_recognize":
            recognizer.add_face()
            recognizer.train()
            recognizer.recognize()
        case "add_train":
            recognizer.add_face()
            recognizer.train()
        case "train_recognize":
            recognizer.train()
            recognizer.recognize()
