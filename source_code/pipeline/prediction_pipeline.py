import sys
import pandas as pd
from source_code.logger import logging
from path_resolver import PathResolver
from source_code.utils import load_object
from source_code.exception import CustomException
from source_code.components.model_trainer import ModelTrainer
from source_code.components.data_ingestion import DataIngestion
from source_code.components.data_transformation import DataTransformation


class PredictPipeline:
    def __int__(self):
        self.path_resolver = PathResolver()

    @staticmethod
    def decode_prediction(prediction):
        try:
            if prediction == 1:
                return "Edible"
            elif prediction == 0:
                return "Poisonous"

        except Exception as e:
            raise CustomException(e, sys)

    def model_predict(self, features: pd.DataFrame):
        try:
            logging.info('Loading Pre-Processor and Model objects')

            pre_processor_obj_path = self.path_resolver.latest_preprocessor_path()
            logging.info(f"Preprocessor object path: {pre_processor_obj_path}")

            trained_model_path = self.path_resolver.latest_model_path()
            logging.info(f"Trained Model Path: {trained_model_path}")

            logging.info('Loading pre_processor and model objects')
            pre_processor = load_object(file_path=pre_processor_obj_path)
            model = load_object(file_path=trained_model_path)

            print(features.shape)
            print(features.columns)

            logging.info('Pre-processing input from user')
            scaled_features = pre_processor.transform(features)

            logging.info('Making Predictions')
            predictions_val = model.predict(scaled_features)
            predictions = self.decode_prediction(predictions_val)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, gill_size,
                 gill_color, stalk_shape, stalk_root, stalk_surface_above_ring, stalk_surface_below_ring,
                 stalk_color_above_ring, stalk_color_below_ring, veil_type, veil_color, ring_number, ring_type,
                 spore_print_color, population, habitat):
        self.cap_shape = cap_shape
        self.cap_surface = cap_surface
        self.cap_color = cap_color
        self.bruises = bruises
        self.odor = odor
        self.gill_attachment = gill_attachment
        self.gill_spacing = gill_spacing
        self.gill_size = gill_size
        self.gill_color = gill_color
        self.stalk_shape = stalk_shape
        self.stalk_root = stalk_root
        self.stalk_surface_above_ring = stalk_surface_above_ring
        self.stalk_surface_below_ring = stalk_surface_below_ring
        self.stalk_color_above_ring = stalk_color_above_ring
        self.stalk_color_below_ring = stalk_color_below_ring
        self.veil_type = veil_type
        self.veil_color = veil_color
        self.ring_number = ring_number
        self.ring_type = ring_type
        self.spore_print_color = spore_print_color
        self.population = population
        self.habitat = habitat

    def get_data_as_dataframe(self):
        try:
            custom_data_dict = {
                "cap-shape": [self.cap_shape],
                "cap-surface": [self.cap_surface],
                "cap-color": [self.cap_color],
                "bruises": [self.bruises],
                "odor": [self.odor],
                "gill-attachment": [self.gill_attachment],
                "gill-spacing": [self.gill_spacing],
                "gill-size": [self.gill_size],
                "gill-color": [self.gill_color],
                "stalk-shape": [self.stalk_shape],
                "stalk-root": [self.stalk_root],
                "stalk-surface-above-ring": [self.stalk_surface_above_ring],
                "stalk-surface-below-ring": [self.stalk_surface_below_ring],
                "stalk-color-above-ring": [self.stalk_color_above_ring],
                "stalk-color-below-ring": [self.stalk_color_below_ring],
                "veil-type": [self.veil_type],
                "veil-color": [self.veil_color],
                "ring-number": [self.ring_number],
                "ring-type": [self.ring_type],
                "spore-print-color": [self.spore_print_color],
                "population": [self.population],
                "habitat": [self.habitat]
            }

            return pd.DataFrame(custom_data_dict)

        except Exception as e:
            raise CustomException(e, sys)
