import sys
from typing import Optional, Any

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from soruce_code.entity import config_entity, artifact_entity
from soruce_code.entity.config_entity import TARGET_COLUMN, DataTransformationConfig
from soruce_code.exception import CustomException
from soruce_code.logger import logging
from soruce_code.utils import save_numpy_array_data, save_object, cramers_V


class ModifiedLabelEncoder(LabelEncoder):
    """
    This class modifies LabelEncoder class to work properly with scikit-learn Pipeline.
    """
    try:
        def fit_transform(self, y, *args, **kwargs):
            return super().fit_transform(y).reshape(-1, 1)

        def transform(self, y, *args, **kwargs):
            return super().transform(y).reshape(-1, 1)

    except Exception as e:
        logging.warning('ModifiedLabelEncoder malfunctioned')
        raise CustomException(e, sys)


class DataTransformation:
    print("in DataTransformation...")

    def __init__(self, data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        logging.info(f"{'--' * 20} Data Transformation {'--' * 20}")
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_config = DataTransformationConfig()

    @staticmethod
    def get_data_transformation_object(column_list: list):
        try:
            dependent_pipe = Pipeline(steps=[("Imputer", SimpleImputer(strategy="most_frequent")),
                                             ("OHE", OneHotEncoder(sparse_output=False)),
                                             ("std_scaler", StandardScaler(with_mean=False))])

            target_pipe = Pipeline(steps=[("le", ModifiedLabelEncoder())])

            transformer = ColumnTransformer([('pipe_1', dependent_pipe, column_list),
                                             ('target_encoding', target_pipe, TARGET_COLUMN)],
                                            remainder='drop')
            return transformer

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        logging.info('Data Transformation initiated')
        try:
            logging.info('Reading Base Data')
            base_df = pd.read_csv(self.data_ingestion_artifact.base_data_path)

            logging.info('Reading Train and Test data')
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # logging.info('Checking if dataframe has required columns')
            corr_threshold = config_entity.DataTransformationConfig.CORRELATION_THRESHOLD_VALUE

            logging.info(f"Getting input features which are at-least {corr_threshold}% associated with Target feature")
            associated_columns = get_associated_columns(base_df)

            preprocess_obj = self.get_data_transformation_object(associated_columns)

            logging.info("Pre-processing Train and Test DataFrame")
            train_arr = preprocess_obj.fit_transform(train_df)
            test_arr = preprocess_obj.transform(test_df)

            logging.info("Pre-processing done: got Train & Test Numpy arrays")

            # Saving train_arr and test_arr numpy arrays
            logging.info("Saving Train & Test Numpy array")
            save_numpy_array_data(path=self.data_transformation_config.train_np_arr_path, array=train_arr)
            save_numpy_array_data(path=self.data_transformation_config.test_np_arr_path, array=test_arr)

            # Saving pre-processor object path
            save_object(path=self.data_transformation_config.preprocessor_obj_path, obj=preprocess_obj)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                preprocessor_object_path=self.data_transformation_config.preprocessor_obj_path,
                train_arr_path=self.data_transformation_config.train_np_arr_path,
                test_arr_path=self.data_transformation_config.test_np_arr_path
            )

            logging.info("Data Transformation Completed.")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)


def get_associated_columns(dataframe: pd.DataFrame) -> Optional[list]:
    """
    This function checks for association/correlation of input features with target feature using Cramer's V rule.
    :param dataframe: base dataframe
    :return: A list of most associated columns
    """
    try:
        logging.info("Checking for Association (using CRAMER'S-V rule)")
        associations = dict()
        associated_columns = list()
        corr_threshold = config_entity.DataTransformationConfig.CORRELATION_THRESHOLD_VALUE
        for col_name in dataframe.columns:
            res = cramers_V(dataframe[TARGET_COLUMN], dataframe[col_name])
            associations[col_name] = round(res * 100, 2)

        for col_name, corr_value in associations.items():
            if corr_value >= corr_threshold:
                associated_columns.append(col_name)

        if TARGET_COLUMN in associated_columns:
            associated_columns.remove(TARGET_COLUMN)

        if len(associated_columns) > 0:  # there is always going to be TARGET_COLUMN in this list
            logging.info(f"Most associated columns with '{TARGET_COLUMN}' are : {associated_columns}")
            return associated_columns

        logging.info('NO ASSOCIATED COLUMNS FOUND.')
        return None

    except Exception as e:
        raise CustomException(e, sys)


def required_columns_check(dataframe: pd.DataFrame) -> bool | list[Any] | Any:
    """This function checks if the most correlated features of the dataset are present or not."""
    try:
        check_list = list()
        missing_list = list()
        imp_cols = config_entity.DataTransformationConfig.IMP_COLUMNS
        for col in imp_cols:
            if col in dataframe.columns:
                check_list.append(col)
            else:
                missing_list.append(col)

        if len(check_list) == len(imp_cols):
            logging.info(f'All important columns are present: {check_list}')
            return True

        logging.info(f'Important columns are missing : {missing_list}')
        return False

    except Exception as e:
        raise CustomException(e, sys)
