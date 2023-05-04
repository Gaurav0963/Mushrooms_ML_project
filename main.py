from soruce_code.components import data_ingestion as di

if __name__ == "__main__":
    di_obj = di.DataIngestion()
    train_file_path, test_file_path = di_obj.initiate_data_ingestion()
    print(train_file_path)
