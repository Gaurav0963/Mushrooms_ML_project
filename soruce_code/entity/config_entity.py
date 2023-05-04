import os
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_file_path: str = os.path.join('artifacts', "train.csv")
    test_file_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    MISSING_VALUE_THRESHOLD = 30
