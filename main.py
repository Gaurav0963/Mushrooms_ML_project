from soruce_code.components.data_ingestion import DataIngestion
from soruce_code.components.data_transformation import DataTransformation, get_associated_columns, required_columns_check

if __name__ == "__main__":

    data_ingestion = DataIngestion()
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
    print(data_ingestion_artifact.base_data_path)
    print(data_ingestion_artifact.train_file_path)
    print(data_ingestion_artifact.test_file_path)

    # train = pd.read_csv(data_ingestion_artifact.train_file_path)
    # print(train['odor'][0])

    data_transform = DataTransformation(data_ingestion_artifact=data_ingestion_artifact)
    obj = data_transform.initiate_data_transformation()
    # obj, base_data = data_transform.initiate_data_transformation()
    # get_corr = get_associated_columns(base_data)
    # print(required_columns_check(base_data))
    # print(base_data.columns)
    # print(get_corr)

