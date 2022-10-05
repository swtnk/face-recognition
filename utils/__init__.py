import typing
import platform
import asyncio
import subprocess
import pip
import numpy as np
import cv2
import json
from PIL import Image
from pathlib import Path


def install_package(package_name: str) -> bool:
    pip.main(["install", package_name])


if platform.system().lower() == "windows":
    try:
        import winsdk.windows.devices.enumeration as windows_devices
    except ImportError:
        install_package("winsdk")
        import winsdk.windows.devices.enumeration as windows_devices


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
    def test_camera(cls, camera_index: int = 0):
        cap = cv2.VideoCapture(camera_index)
        cap.set(3, 640)
        cap.set(4, 480)

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow("frame", frame)
            cv2.imshow("gray", gray)

            k = cv2.waitKey(30) & 0xFF

            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


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

    def __exit__(self, exc_type, exc_value, traceback):
        self.camera.release()
        cv2.destroyAllWindows()


class Recognizer:
    def __init__(
        self,
        camera_index: int = 0,
        dataset_path: str = "dataset",
        detection_model: str = "./Cascades/haarcascades_cuda/haarcascade_frontalface_default.xml",
    ) -> None:
        self.camera_index = camera_index
        self.detection_model = detection_model
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_detector = cv2.CascadeClassifier(self.detection_model)
        self.dataset_path = Path(dataset_path)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def add_face(self):
        face_id = 1
        face_name = input("Enter Name: ")
        data = list()
        FACE_MAP_PATH = Path("./face_map.json")
        if not FACE_MAP_PATH.exists():
            with open(FACE_MAP_PATH, "w+") as f:
                json.dump({"count": 1, "faces": {face_id: face_name}}, f)
        else:
            with open(FACE_MAP_PATH, "r") as fread:
                data = json.load(fread)

            with open(FACE_MAP_PATH, "w") as fwrite:
                data["count"] += 1
                face_id = data["count"]
                data["faces"][face_id] = face_name
                json.dump(data, fwrite)

        print("[INFO] Initializing face capture. Look the camera and wait ...")
        count: int = 0
        with VideoCaptureManager(camera_index=self.camera_index) as vcm:
            while True:
                ret, image = vcm.camera.read()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    count += 1
                    cv2.imwrite(
                        f"./dataset/User.{face_id}.{count}.jpg",
                        gray[y : y + h, x : x + w],
                    )
                    cv2.imshow("image", image)

                k = cv2.waitKey(100) & 0xFF
                if k == 27 or count >= 30:
                    break

    def __get_images_and_labels(self):
        import os

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
        print("Training started...")
        faces, ids = self.__get_images_and_labels()
        self.face_recognizer.train(faces, np.array(ids))
        self.face_recognizer.write("trainer/trainer.yml")
        print(f"\n [INFO] {len(np.unique(ids))} faces trained. Exiting Program")

    def recognize(self):
        self.face_recognizer.read("trainer/trainer.yml")
        names = dict()
        with open("face_map.json", "r") as f:
            names = json.load(f)
        with VideoCaptureManager(camera_index=self.camera_index) as vcm:
            min_w = 0.1 * vcm.camera.get(3)
            min_h = 0.1 * vcm.camera.get(4)
            while True:
                ret, img = vcm.camera.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(int(min_w), int(min_h)),
                )

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    _id, confidence = self.face_recognizer.predict(
                        gray[y : y + h, x : x + w]
                    )

                    if confidence < 100:
                        _id = names["faces"].get(str(_id), "Unknown")
                        confidence = "  {0}%".format(round(100 - confidence))
                    else:
                        _id = "unknown"
                        confidence = "  {0}%".format(round(100 - confidence))

                    cv2.putText(
                        img, str(_id), (x + 5, y - 5), self.font, 1, (255, 255, 255), 2
                    )
                    cv2.putText(
                        img,
                        str(confidence),
                        (x + 5, y + h - 5),
                        self.font,
                        1,
                        (255, 255, 0),
                        1,
                    )

                cv2.imshow("camera", img)
                k = cv2.waitKey(10) & 0xFF
                if k == 27:
                    break
