import argparse
from utils import Recognizer

CAMERA_INDEX = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition")
    parser.add_argument("action", nargs="?", help="Actions: add, train, recognize")
    parser.add_argument("-c", "--camera", type=int, default=0, help="camera index")
    args = parser.parse_args()

    if args.camera is not None:
        CAMERA_INDEX = args.camera

    recognizer = Recognizer(camera_index=CAMERA_INDEX)

    match args.action:
        case "add":
            recognizer.add_face()
        case "train":
            recognizer.train()
        case "recognize":
            recognizer.recognize()
