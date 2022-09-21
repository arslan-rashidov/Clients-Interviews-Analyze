import pickle
import pandas as pd
import numpy as np


class MLModel:
    def __init__(self, model_path, one_hot_encoder_path):
        loaded_model = pickle.load(open(model_path, 'rb'))
        self.model = loaded_model
        self.feature_transformer = FeaturesTransformer(one_hot_encoder_path)

    def make_prediction(self, data):
        features = self.feature_transformer.transform_features(data=data)
        return self.model.predict_proba(features)

    def get_main_technologies(self):
        main_technologies = self.feature_transformer.one_hot_encoder.categories_[0]
        return main_technologies

    def get_seniority_levels(self):
        seniority_levels = self.feature_transformer.one_hot_encoder.categories_[1]
        return seniority_levels

    def get_english_levels(self):
        english_levels = self.feature_transformer.one_hot_encoder.categories_[2]
        return english_levels

    def get_location_ids(self):
        location_ids = self.feature_transformer.one_hot_encoder.categories_[3]
        return location_ids


class FeaturesTransformer:
    def __init__(self, one_hot_encoder_path):
        loaded_one_hot_encoder = pickle.load(open(one_hot_encoder_path, 'rb'))
        self.one_hot_encoder = loaded_one_hot_encoder

    def transform_features(self, data):
        features = pd.DataFrame(data=data)
        transformed_features = self.one_hot_encoder.transform(features).toarray()
        transformed_labels = np.array(self.one_hot_encoder.get_feature_names_out()).ravel()

        encoded_features = pd.DataFrame(transformed_features, columns=transformed_labels)
        return encoded_features
