import pandas as pd
from soruce_code.components import data_ingestion as di
from soruce_code.entity import config_entity, artifact_entity


class DataTransformation:
    print("in DataTransformation...")

    def __init__(self, data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        self.data_ingestion_artifact = data_ingestion_artifact

    def initiate_data_tranformation(self):
        # Reading train and test dataset
        print('initiate_data_tranformation')
        print(self.data_ingestion_artifact.train_file_path)
        train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
        return train_df


if __name__ == '__main__':
    di_obj = di.DataIngestion()
    di_artifact = di_obj.initiate_data_ingestion()
    obj = DataTransformation(di_artifact.train_file_path)
    train = obj.initiate_data_tranformation()
    print(train['odor'][0])
