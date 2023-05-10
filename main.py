from source_code.components.data_ingestion import DataIngestion
from source_code.components.data_transformation import DataTransformation
from source_code.components.model_trainer import ModelTrainer

if __name__ == "__main__":

    # Data-Ingestion
    data_ingestion = DataIngestion()
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

    # Data-Transformation
    data_transform = DataTransformation(data_ingestion_artifact=data_ingestion_artifact)
    data_transformation_artifacts = data_transform.initiate_data_transformation()

    # Model-Training
    model_trainer = ModelTrainer(data_transformation_artifacts=data_transformation_artifacts)
    model_trainer_artifact = model_trainer.initiate_model_training()