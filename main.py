from soruce_code.components.data_ingestion import DataIngestion
from soruce_code.components.data_transformation import DataTransformation

import pandas as pd

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
    print(data_ingestion_artifact.train_file_path)

    train = pd.read_csv(data_ingestion_artifact.train_file_path)
    print(train['odor'][0])

    data_transform = DataTransformation(data_ingestion_artifact=data_ingestion_artifact)
    train_data = data_transform.initiate_data_tranformation()
    print(train_data['odor'][0])

