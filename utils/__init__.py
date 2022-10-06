import typing
import platform
import asyncio
import subprocess
import pip
import numpy as np
import cv2
import json
import shutil
from config import Config
from PIL import Image
from pathlib import Path
from tqdm import tqdm

DIRECTORIES = Config.DIRECTORIES
FACE_MAP_JSON = Config.FACE_MAP_JSON


def install_package(package_name: str) -> bool:
    try:
        pip.main(["install", package_name])
    except Exception:
        return False
    return True


if platform.system().lower() == "windows":
    try:
        import winsdk.windows.devices.enumeration as windows_devices
    except ImportError:
        install_package("winsdk")
        import winsdk.windows.devices.enumeration as windows_devices


def setup_project() -> None:
    for directory in DIRECTORIES:
        Path(directory).mkdir(parents=True, exist_ok=True)


def clean_project() -> None:
    for directory in DIRECTORIES:
        shutil.rmtree(directory, ignore_errors=True)
    Path(FACE_MAP_JSON).unlink(missing_ok=True)


class Camera:
    def __init__(self) -> None:
        self.cameras = list()

    async def get_camera_information_for_windows(self, video_devices: int = 4):
        return await windows_devices.DeviceInformation.find_all_async(video_devices)

    def add_camera_information(self, camera_indices: typing.List) -> typing.List:
        platform_name = platform.system().lower()
        cameras = []
        match platform_name:
            case "windows":
                cameras_info_windows = asyncio.run(
                    self.get_camera_information_for_windows()
                )

                for index, camera in enumerate(cameras_info_windows):
                    cameras.append(
                        {
                            "index": index,
                            "name": camera.name,
                            "is_enabled": camera.is_enabled,
                        }
                    )
            case "linux":
                for camera_index in camera_indices:
                    camera_name = subprocess.run(
                        [
                            "cat",
                            "/sys/class/video4linux/video{}/name".format(camera_index),
                        ],
                        stdout=subprocess.PIPE,
                    ).stdout.decode("utf-8")
                    camera_name = camera_name.replace("\n", "")
                    cameras.append({"index": camera_index, "name": camera_name})
        return cameras

    def get_camera_indices(self, max_camera_to_check: int = 5) -> typing.List:
        camera_indices: typing.List = list()
        for index in range(0, max_camera_to_check):
            capture = cv2.VideoCapture(index)
            if capture.read()[0]:
                camera_indices.append(index)
                capture.release()
        return camera_indices

    @property
    def get_camera(self) -> typing.List:
        camera_indices = self.get_camera_indices()
        if len(camera_indices) == 0:
            return self.cameras
        self.cameras = self.add_camera_information(camera_indices)

        return self.cameras

    @classmethod
    def test_camera(cls, camera_index: int = 0) -> None:
        with VideoCaptureManager(camera_index=camera_index) as vcm:
            while True:
                _, frame = vcm.camera.read()
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                cv2.imshow("frame", frame)
                cv2.imshow("gray", gray)

                k = cv2.waitKey(30) & 0xFF

                if k == 27:
                    break


class VideoCaptureManager:
    def __init__(
        self,
        camera_index: int = 0,
        width: typing.Iterable = (3, 640),
        height: typing.Iterable = (4, 480),
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height

    def __enter__(self):
        self.camera = cv2.VideoCapture(self.camera_index)
        self.camera.set(*self.width)
        self.camera.set(*self.height)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.camera.release()
        cv2.destroyAllWindows()


class Recognizer:
    def __init__(
        self,
        camera_index: int = 0,
        dataset_path: str = "dataset",
        detection_model: Path = Config.CASCASE_MODEL,
    ) -> None:
        self.camera_index = camera_index
        self.detection_model = Config.BASE_DIR / detection_model
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_detector = cv2.CascadeClassifier(str(self.detection_model))
        self.dataset_path = Config.BASE_DIR / dataset_path
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def add_face(self):
        face_id = 1
        face_name = input("Enter Name: ")
        data = list()
        if not FACE_MAP_JSON.exists():
            with open(FACE_MAP_JSON, "w+") as f:
                json.dump({"count": 1, "faces": {face_id: face_name}}, f)
        else:
            with open(FACE_MAP_JSON, "r") as fread:
                data = json.load(fread)

            with open(FACE_MAP_JSON, "w") as fwrite:
                data["count"] += 1
                face_id = data["count"]
                data["faces"][face_id] = face_name
                json.dump(data, fwrite)

        print("[INFO] Initializing face capture. Look into camera and wait ...")
        count: int = 0
        with VideoCaptureManager(camera_index=self.camera_index) as vcm:
            progress_bar = tqdm(
                total=Config.IMAGE_COUNT_PER_ID,
                ascii="░▒█",
                bar_format="Observing face: {desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}",
            )
            while True:
                _, image = vcm.camera.read()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    count += 1
                    progress_bar.update(1)
                    cv2.imwrite(
                        f"{Config.DATASET_DIR}/User.{face_id}.{count}.jpg",
                        gray[y : y + h, x : x + w],
                    )
                    cv2.imshow(Config.RECOGNIZER_FRAME_NAME, image)

                k = cv2.waitKey(100) & 0xFF
                if k == 27 or count >= Config.IMAGE_COUNT_PER_ID:
                    break
            progress_bar.close()

    def __get_images_and_labels(self):

        image_paths = [f for f in self.dataset_path.iterdir()]
        face_samples = list()
        ids = list()

        for image_path in image_paths:
            pil_image = Image.open(image_path).convert("L")
            image_numpy = np.array(pil_image, "uint8")
            _id = int(image_path.name.split(".")[1])
            faces = self.face_detector.detectMultiScale(image_numpy)

            for (x, y, w, h) in faces:
                face_samples.append(image_numpy[y : y + h, x : x + w])
                ids.append(_id)

        return face_samples, ids

    def train(self):
        print("Memorizing faces...")
        faces, ids = self.__get_images_and_labels()
        self.face_recognizer.train(faces, np.array(ids))
        self.face_recognizer.write(f"{Config.TRAINER_YAML}")
        print(f"{len(np.unique(ids))} faces memorized.")

    def recognize(self):
        print(f"[INFO] Starting face recognition.")
        self.face_recognizer.read(f"{Config.TRAINER_YAML}")
        names = dict()
        with open("face_map.json", "r") as f:
            names = json.load(f)
        with VideoCaptureManager(camera_index=self.camera_index) as vcm:
            min_w = 0.1 * vcm.camera.get(3)
            min_h = 0.1 * vcm.camera.get(4)
            detected_faces = set()
            while True:
                _, image = vcm.camera.read()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(int(min_w), int(min_h)),
                )

                for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    _id, confidence = self.face_recognizer.predict(
                        gray[y : y + h, x : x + w]
                    )

                    if confidence > Config.CONFIDENCE:
                        _id = names["faces"].get(str(_id), Config.UNKNOWN_FACE_NAME)
                        confidence = f"{round(100 - confidence)}%"
                    else:
                        _id = Config.UNKNOWN_FACE_NAME
                        confidence = f"{round(100 - confidence)}%"
                    if _id not in detected_faces:
                        detected_faces.add(_id)
                        print(f"Detected: {_id}")
                    cv2.putText(
                        image,
                        str(_id),
                        (x + 5, y - 5),
                        self.font,
                        1,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        image,
                        str(confidence),
                        (x + 5, y + h - 5),
                        self.font,
                        1,
                        (255, 255, 0),
                        1,
                    )

                cv2.imshow(Config.RECOGNIZER_FRAME_NAME, image)
                k = cv2.waitKey(10) & 0xFF
                if k == 27:
                    break
