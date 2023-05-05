import os
import sys
from datetime import datetime
from soruce_code.exception import CustomException
from dataclasses import dataclass

ARTIFACTS = 'artifacts'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
RAW_DATA_FILE = 'raw_data.csv'


# class TrainingPipelineConfig:
#     def __init__(self):
#         try:
#             self.artifact_dir = os.path.join(ARTIFACTS, f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
#         except Exception as e:
#             raise CustomException(e, sys)


@dataclass
class DataIngestionConfig:
    train_file_path: str = os.path.join(ARTIFACTS, TRAIN_FILE)
    test_file_path: str = os.path.join(ARTIFACTS, TEST_FILE)
    raw_data_path: str = os.path.join(ARTIFACTS, RAW_DATA_FILE)
    MISSING_VALUE_THRESHOLD = 30
