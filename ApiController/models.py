import pickle
import pandas as pd
import numpy as np


class MLModel:
    def __init__(self, model_path, one_hot_encoder_path, project_complexity_df_path):
        loaded_model = pickle.load(open(model_path, 'rb'))
        self.model = loaded_model
        self.features_transformer = FeaturesTransformer(one_hot_encoder_path, project_complexity_df_path)

    def make_prediction(self, data):
        features = self.features_transformer.transform_features(data=data)
        print(features.columns)
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

    def get_project_names(self):
        project_names = self.features_transformer.project_complexity_df['project_name'].to_list()
        return project_names



class FeaturesTransformer:
    def __init__(self, one_hot_encoder_path, project_complexity_df_path):
        loaded_one_hot_encoder = pickle.load(open(one_hot_encoder_path, 'rb'))
        self.project_complexity_df = pd.read_csv(project_complexity_df_path)
        self.one_hot_encoder = loaded_one_hot_encoder

    def transform_features(self, data):
        df = pd.DataFrame(data=data)
        df = df.drop(['project_name', 'subjective_readiness'], axis=1)

        print(df)


        complexity = float(self.get_complexity(data['project_name']))

        transformed_features = self.one_hot_encoder.transform(df)
        transformed_features = transformed_features.toarray()
        transformed_features = np.insert(transformed_features, 0, complexity)
        transformed_features = np.insert(transformed_features, 1, data['subjective_readiness'])

        transformed_labels = np.array(self.one_hot_encoder.get_feature_names_out()).ravel()
        transformed_labels = np.insert(transformed_labels, 0, 'complexity')
        transformed_labels = np.insert(transformed_labels, 1, 'subjective_readiness')

        encoded_features = pd.DataFrame([transformed_features], columns=transformed_labels)

        return encoded_features

    def get_complexity(self, project_name):
        row = self.project_complexity_df.loc[self.project_complexity_df['project_name'] == project_name]
        return row['complexity']