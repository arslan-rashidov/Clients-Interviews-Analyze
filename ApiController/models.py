import pickle
import pandas as pd
import numpy as np
from config import *

class Candidate:
    def __init__(self, candidate_id, succeed_projects_df, failed_projects_df):
        self.candidate_id = candidate_id
        self.succeed_projects_count, self.failed_projects_count = self.get_characteristics(succeed_projects_df, failed_projects_df)

    # Get number of succeed and failed projects
    def get_characteristics(self, succeed_projects_df, failed_projects_df):
        succeed_projects_count_row = succeed_projects_df[(succeed_projects_df['candidate_id'] == self.candidate_id)]
        succeed_projects_count = succeed_projects_count_row['succeed_projects_count']

        failed_projects_count_row = failed_projects_df[(failed_projects_df['candidate_id'] == self.candidate_id)]
        failed_projects_count = failed_projects_count_row['failed_projects_count']

        if len(succeed_projects_count.to_list()) == 0:
            succeed_projects_count = 0
        else:
            succeed_projects_count = float(succeed_projects_count)

        if len(failed_projects_count.to_list()) == 0:
            failed_projects_count = 0
        else:
            failed_projects_count = float(failed_projects_count)

        return succeed_projects_count, failed_projects_count




class MLModel:
    def __init__(self, model_path, one_hot_encoder_path, project_complexity_df_path):
        loaded_model = pickle.load(open(model_path, 'rb'))
        self.model = loaded_model
        self.features_transformer = FeaturesTransformer(one_hot_encoder_path, project_complexity_df_path)

    def make_prediction(self, data):
        features = self.features_transformer.transform_features(data=data)
        return self.model.predict_proba(features)

    def get_interview_tecnologies(self):
        interview_tecnologies = self.features_transformer.one_hot_encoder.categories_[0]
        return interview_tecnologies

    def get_seniority_levels(self):
        seniority_levels = seniority_level_labels
        return seniority_levels

    def get_english_levels(self):
        english_levels = english_levels_labels
        return english_levels

    def get_project_names(self):
        project_names = list(set(self.features_transformer.project_complexity_df['project_name'].to_list()))
        return project_names



class FeaturesTransformer:
    def __init__(self, one_hot_encoder_path, project_complexity_df_path):
        loaded_one_hot_encoder = pickle.load(open(one_hot_encoder_path, 'rb'))
        self.project_complexity_df = pd.read_csv(project_complexity_df_path)
        self.one_hot_encoder = loaded_one_hot_encoder

    # Trandform API parameters to model inputs
    def transform_features(self, data):
        df = pd.DataFrame(data={'interview_technology': [data['interview_technology']]})
        candidate_english_level = english_levels_labels[data['candidate_english_level']]
        candidate_subjective_readiness = 0
        if data['candidate_subjective_readiness'] == 'undefined':
            candidate_subjective_readiness = 50.
        else:
            candidate_subjective_readiness = float(data['candidate_subjective_readiness'])
        project_name = data['project_name']
        candidate_seniority_level = seniority_level_labels[data['candidate_seniority_level']]
        interview_language = interview_language_labels[data['interview_language']]
        succeed_projects_count = data['succeed_projects_count']
        failed_projects_count = data['failed_projects_count']
        complexity = self.get_complexity(project_name, data['interview_technology'])

        transformed_features = self.one_hot_encoder.transform(df)
        transformed_features = transformed_features.toarray()
        transformed_features = np.insert(transformed_features, 0, candidate_subjective_readiness)
        transformed_features = np.insert(transformed_features, 1, interview_language)
        transformed_features = np.insert(transformed_features, 2, candidate_english_level)
        transformed_features = np.insert(transformed_features, 3, candidate_seniority_level)
        transformed_features = np.insert(transformed_features, 4, complexity)
        transformed_features = np.insert(transformed_features, len(transformed_features), succeed_projects_count)
        transformed_features = np.insert(transformed_features, len(transformed_features), failed_projects_count)


        transformed_labels = np.array(self.one_hot_encoder.get_feature_names_out()).ravel()
        transformed_labels = np.insert(transformed_labels, 0, 'candidate_subjective_readiness')
        transformed_labels = np.insert(transformed_labels, 1, 'interview_language')
        transformed_labels = np.insert(transformed_labels, 2, 'candidate_english_level')
        transformed_labels = np.insert(transformed_labels, 3, 'candidate_seniority_level')
        transformed_labels = np.insert(transformed_labels, 4, 'complexity')
        transformed_labels = np.insert(transformed_labels, len(transformed_labels), 'succeed_projects_count')
        transformed_labels = np.insert(transformed_labels, len(transformed_labels), 'failed_projects_count')

        encoded_features = pd.DataFrame([transformed_features], columns=transformed_labels)


        return encoded_features


    def get_complexity(self, project_name, technology):
        if project_name == 'undefined':
            technology = 'undefined'
        row = self.project_complexity_df[(self.project_complexity_df['project_name'] == project_name) & (self.project_complexity_df['interview_technology'] == technology)]
        if len(row['complexity'].tolist()) == 0:
            return 0.5
        return float(row['complexity'])