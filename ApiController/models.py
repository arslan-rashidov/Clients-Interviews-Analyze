import pickle
import pandas as pd
import numpy as np


class MLModel:
    def __init__(self, model_path, one_hot_encoder_path, min_max_scaler_path):
        loaded_model = pickle.load(open(model_path, 'rb'))
        self.model = loaded_model
        self.features_transformer = FeaturesTransformer(one_hot_encoder_path, min_max_scaler_path)

    def make_prediction(self, data):
        features = self.features_transformer.transform_features(data=data)
        return self.model.predict_proba(features)

    def get_main_technologies(self):
        main_technologies = self.features_transformer.one_hot_encoder.categories_[0]
        return main_technologies

    def get_seniority_levels(self):
        seniority_levels = self.features_transformer.one_hot_encoder.categories_[1]
        return seniority_levels

    def get_english_levels(self):
        english_levels = self.features_transformer.one_hot_encoder.categories_[2]
        return english_levels



class FeaturesTransformer:
    def __init__(self, one_hot_encoder_path, min_max_scaler_path):
        loaded_one_hot_encoder = pickle.load(open(one_hot_encoder_path, 'rb'))
        loaded_min_max_scaler = pickle.load(open(min_max_scaler_path, 'rb'))
        self.one_hot_encoder = loaded_one_hot_encoder
        self.min_max_scaler = loaded_min_max_scaler

    def transform_features(self, data):
        df = pd.DataFrame(data=data)
        features = df.drop(['subjective_readiness'], axis=1)

        subjective_readiness_scaled = float(self.scale_subjective_readiness(df['subjective_readiness'][0]))

        transformed_features = self.one_hot_encoder.transform(features)
        transformed_features = transformed_features.toarray()
        transformed_features = np.insert(transformed_features, 0, subjective_readiness_scaled, axis=1)

        transformed_labels = np.array(self.one_hot_encoder.get_feature_names_out()).ravel()
        transformed_labels = np.insert(transformed_labels, 0, 'subjective_readiness', axis=0)

        encoded_features = pd.DataFrame(transformed_features, columns=transformed_labels)
        return encoded_features

    def scale_subjective_readiness(self, value):
        return self.min_max_scaler.transform(np.array([value]).reshape(-1, 1))[0][0]

