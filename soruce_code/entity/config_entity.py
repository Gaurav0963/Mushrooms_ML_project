import os
from datetime import datetime
from dataclasses import dataclass


ARTIFACTS = 'artifacts'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
PREPROCESSOR = 'preprocessor.pkl'
BASE_DATA_FILE = 'base_data.csv'
RAW_DATA_FILE = 'raw_data.csv'
TARGET_COLUMN = 'class'

ARTIFACTS_DIR = os.path.join(os.getcwd(), ARTIFACTS, f"{datetime.now().strftime('%d_%m_%Y-%H_%M_%S')}")


@dataclass
class DataIngestionConfig:
    base_data_path: str = os.path.join(ARTIFACTS_DIR, BASE_DATA_FILE)
    train_file_path: str = os.path.join(ARTIFACTS_DIR, TRAIN_FILE)
    test_file_path: str = os.path.join(ARTIFACTS_DIR, TEST_FILE)
    TEST_DATA_SIZE = 0.2  # test data is set 20% of base data
    MISSING_VALUE_THRESHOLD = 30


@dataclass
class DataTransformationConfig:
    DATA_TRANSFORMATION_DIR = os.path.join(ARTIFACTS_DIR, 'Data_Transformation')
    preprocessor_obj_path = os.path.join(DATA_TRANSFORMATION_DIR, PREPROCESSOR)
    train_np_arr_path = os.path.join(DATA_TRANSFORMATION_DIR, TRAIN_FILE.replace('csv', 'npz'))
    test_np_arr_path = os.path.join(DATA_TRANSFORMATION_DIR, TEST_FILE.replace('csv', 'npz'))
    # Columns that are most correlated with TARGET COLUMN, including TARGET COLUMN
    IMP_COLUMNS = ['class', 'odor', 'spore-print-color', 'gill-color']
    CORRELATION_THRESHOLD_VALUE = 20  # This value is a percentage
