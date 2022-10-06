from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASET_DIR = BASE_DIR / "dataset"
    IMAGE_COUNT_PER_ID = 50
    TRAINER_DIR = BASE_DIR / "trainer"
    DIRECTORIES = (DATASET_DIR, TRAINER_DIR)
    CASCASE_MODEL = (
        BASE_DIR / "Cascades/haarcascades_cuda/haarcascade_frontalface_default.xml"
    )
    TRAINER_YAML = TRAINER_DIR / "trainer.yml"
    FACE_MAP_JSON = BASE_DIR / "face_map.json"
    UNKNOWN_FACE_NAME = "Unknown"
    CONFIDENCE = 25
    RECOGNIZER_FRAME_NAME = "View Finder"
