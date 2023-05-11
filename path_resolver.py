import os
import sys
from source_code.logger import logging
from source_code.components.data_ingestion import DataIngestion
from source_code.components.data_transformation import DataTransformation
from source_code.components.model_trainer import ModelTrainer
from source_code.exception import CustomException
from source_code.entity.config_entity import ARTIFACTS, MODEL_NAME, PREPROCESSOR


logging.info(f"{'--' * 20} Path Resolver {'--' * 20}")


def start_training_pipeline():
    """
    This function is triggered if saved model object or saved pre-processor object is not found in working directory
    :return: latest trained model object path & pre-processor path
    """
    try:
        logging.info("Training Pipeline has started")
        # Data-Ingestion
        data_ingestion = DataIngestion()
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        # Data-Transformation
        data_transform = DataTransformation(data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifacts = data_transform.initiate_data_transformation()

        # Model-Training
        model_trainer = ModelTrainer(data_transformation_artifacts=data_transformation_artifacts)
        model_trainer_artifact = model_trainer.initiate_model_training()

        model_obj_path = model_trainer_artifact.trained_model_path
        preprocessor_obj_path = data_transformation_artifacts.preprocessor_object_path

        return model_obj_path, preprocessor_obj_path

    except Exception as e:
        raise CustomException(e, sys)


def latest_model_path():
    """
    :return: Latest saved (in artifacts folder) Trained Model Object
    """
    try:
        logging.info("Looking for Latest Trained Model Object Path")

        logging.info(f"Looking for {ARTIFACTS} folder in current working directory: {os.getcwd()}")
        if ARTIFACTS in os.listdir(os.getcwd()):
            logging.info(f"{ARTIFACTS} folder found")

            if len(os.listdir(ARTIFACTS)) == 0:
                logging.info(f"{ARTIFACTS} folder is empty")
                return None

            latest_dir = str(os.listdir(ARTIFACTS)[-1])
            latest_dir_path = os.path.join(ARTIFACTS, latest_dir)
            model = str(os.listdir(latest_dir_path)[-1])
            model_obj_path = os.path.join(ARTIFACTS, latest_dir, model, MODEL_NAME)
            return model_obj_path
        else:
            logging.info(f"{ARTIFACTS} folder NOT found!")
            model_obj_path, _ = start_training_pipeline()
            return model_obj_path
    except Exception as e:
        raise CustomException(e, sys)


def latest_preprocessor_path():
    """
    :return: Latest saved Pre-Processor object path
    """
    try:
        logging.info("Looking for Latest Pre-processor Object Path")

        logging.info(f"Looking for {ARTIFACTS} folder in current working directory: {os.getcwd()}")
        if ARTIFACTS in os.listdir(os.getcwd()):
            logging.info(f"{ARTIFACTS} folder found")

            if len(os.listdir(ARTIFACTS)) == 0:
                logging.info(f"{ARTIFACTS} folder is empty")
                return None

            latest_dir = str(os.listdir(ARTIFACTS)[-1])
            file = os.path.join('artifacts', latest_dir)
            preprocessor = str(os.listdir(file)[-2])
            preprocessor_obj = os.path.join(ARTIFACTS, latest_dir, preprocessor, PREPROCESSOR)
            return preprocessor_obj
        else:
            logging.info(f"{ARTIFACTS} folder NOT found!")
            _, preprocessor = start_training_pipeline()
            return preprocessor
    except Exception as e:
        raise CustomException(e, sys)
