import os
import sys
from source_code.components.data_ingestion import DataIngestion
from source_code.components.data_transformation import DataTransformation
from source_code.components.model_trainer import ModelTrainer
from source_code.exception import CustomException
from source_code.entity.config_entity import ARTIFACTS, MODEL_NAME, PREPROCESSOR


class PathResolver:
    def __int__(self):
        self.data_ingestion = DataIngestion()

    def start_training_pipeline(self):
        try:
            # Data-Ingestion
            # data_ingestion = DataIngestion()
            data_ingestion_artifact = self.data_ingestion.initiate_data_ingestion()

            # Data-Transformation
            data_transform = DataTransformation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifacts = data_transform.initiate_data_transformation()

            # Model-Training
            model_trainer = ModelTrainer(data_transformation_artifacts=data_transformation_artifacts)
            model_trainer_artifact = model_trainer.initiate_model_training()

            model_obj = model_trainer_artifact.trained_model_path
            preprocessor_obj = data_transformation_artifacts.preprocessor_object_path

            return model_obj, preprocessor_obj

        except Exception as e:
            raise CustomException(e, sys)

    def latest_model_path(self):
        try:
            if ARTIFACTS in os.listdir(os.getcwd()):
                if len(os.listdir(ARTIFACTS)) == 0:
                    return None

                latest_dir = str(os.listdir(ARTIFACTS)[-1])
                latest_dir_path = os.path.join(ARTIFACTS, latest_dir)
                model = str(os.listdir(latest_dir_path)[-1])
                model_obj = os.path.join(ARTIFACTS, latest_dir, model, MODEL_NAME)
                return model_obj
            else:
                model_obj, _ = self.start_training_pipeline()
                return model_obj
        except Exception as e:
            raise CustomException(e, sys)

    def latest_preprocessor_path(self):
        try:
            if ARTIFACTS in os.listdir(os.getcwd()):
                if len(os.listdir(ARTIFACTS)) == 0:
                    return None

                latest_dir = str(os.listdir(ARTIFACTS)[-1])
                file = os.path.join('artifacts', latest_dir)
                preprocessor = str(os.listdir(file)[-2])
                preprocessor_obj = os.path.join(ARTIFACTS, latest_dir, preprocessor, PREPROCESSOR)
                return preprocessor_obj
            else:
                _, preprocessor = self.start_training_pipeline()
                return preprocessor
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    latest_path = PathResolver()
    model = latest_path.latest_model_path()
    print(model)
    print(latest_path.latest_preprocessor_path())