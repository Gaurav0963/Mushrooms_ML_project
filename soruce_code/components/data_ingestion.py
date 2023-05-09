import os
import sys
import pandas as pd
import numpy as np

from soruce_code.logger import logging
from soruce_code.exception import CustomException
from soruce_code.utils import get_df_from_mongo, drop_missing_val_columns
from soruce_code import utils
from soruce_code.entity.config_entity import DataIngestionConfig
from soruce_code.entity.artifact_entity import DataIngestionArtifact

from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self):
        logging.info(f"{'--' * 20} Data Ingestion {'--' * 20}")
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        print('in data Ingestion')
        logging.info("Initiating Data-Ingestion")
        try:
            df = get_df_from_mongo('mushrooms_data', 'mushroom')
            logging.info('Successfully read Mushrooms Dataset')

            logging.info(f"stalk-root: {df['stalk-root'].unique()}")
            logging.info("Replacing '?' in 'stalk-root' with np.nan")
            df = df.replace(to_replace='?', value=np.nan)
            logging.info(f"stalk-root: {df['stalk-root'].unique()}")

            threshold = self.data_ingestion_config.MISSING_VALUE_THRESHOLD
            logging.info(f'Checking for columns with more than {threshold}% missing values')
            missing_val_col_list = drop_missing_val_columns(df)
            logging.info(f'Columns found: {missing_val_col_list}')
            logging.warn(f'Dropping columns found with more than {threshold}% missing values')
            df = df.drop(missing_val_col_list, axis=1)

            logging.info('Checking for columns with single unique value')
            single_unique_val_col_list = utils.drop_single_unique_val_columns(df)
            logging.info(f'Columns found: {single_unique_val_col_list}')
            logging.warn('Dropping columns found with single unique value')
            df.drop(single_unique_val_col_list, axis=1, inplace=True)

            # Making directory to store all paths. Test & raw_data paths are also stored in same dir
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_file_path), exist_ok=True)

            logging.info('Saving base_data_file to artifacts folder')
            df.to_csv(self.data_ingestion_config.base_data_path, index=False, header=True)

            logging.info("Splitting base data into train & test set")
            test_size = self.data_ingestion_config.TEST_DATA_SIZE
            logging.info(f"test data size : {test_size*100}% of base data")
            train_set, test_set = train_test_split(df, test_size=test_size, random_state=42)

            logging.info('Saving train-set to artifacts folder')
            train_set.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)

            logging.info('Saving test-set to artifacts folder')
            test_set.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)

            logging.info("Data-Ingestion completed")

            data_ingestion_artifact = DataIngestionArtifact(
                base_data_path=self.data_ingestion_config.base_data_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)
