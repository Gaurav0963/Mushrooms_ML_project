import os
import pymongo
import pandas as pd
import numpy as np
from typing import List

from soruce_code.entity.config_entity import DataIngestionConfig

MONGO_DB_URL = os.environ.get('MONGO_DB_URL')

mongo_client = pymongo.MongoClient(MONGO_DB_URL)


def get_df_from_mongo(db_name: str, collection_name: str) -> pd.DataFrame:
    dataframe = pd.DataFrame(list(mongo_client[db_name][collection_name].find()))
    if "_id" in dataframe.columns:
        dataframe = dataframe.drop("_id", axis=1)
    return dataframe


def missing_val_columns(df: pd.DataFrame) -> List[str]:
    obj = DataIngestionConfig()
    # df = df.replace(to_replace='?', value=np.nan)
    missing_data_dict = dict()
    drop_col_list = list()
    for col_name in df.columns:
        missing_data_dict[col_name] = (df[col_name].isna().sum() / len(df)) * 100

    for col_name, missing_value in missing_data_dict.items():
        if missing_value > obj.MISSING_VALUE_THRESHOLD:
            drop_col_list.append(col_name)

    return drop_col_list


def single_unique_val_columns(df: pd.DataFrame) -> List[str]:
    column_name_with_unique_val_count = dict()
    unique_val_column_list = list()

    for col_name in df.columns:
        count = 0
        for _ in df[col_name].unique():
            count += 1
        column_name_with_unique_val_count[col_name] = count

    for col_name, unique_val in column_name_with_unique_val_count.items():
        if unique_val == 1:
            unique_val_column_list.append(col_name)

    return unique_val_column_list


if __name__ == "__main__":
    df = get_df_from_mongo('mushrooms_data', 'mushroom')
    print(df.shape)
    dcl = missing_val_columns(df)
    print(dcl)
