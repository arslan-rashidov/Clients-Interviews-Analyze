import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler


class CLientsInterviewsDataPreprocessing:
    def __init__(self):
        self.clients_interview_df_path = "clients_interviews.csv"

    def run(self):
        self.combine_datasets()
        self.prepare_data()

    def combine_datasets(self):
        stop_list = ['CANCEL', 'TERMINATED']

        staffing_candidate_in_request_for_specialist_df = pd.read_csv(
            'data/staffing_candidate_in_request_for_specialist.csv')
        staffing_candidate_in_request_for_specialist_df = staffing_candidate_in_request_for_specialist_df[
            ['status', 'candidate_id']]
        staffing_candidate_in_request_for_specialist_df = staffing_candidate_in_request_for_specialist_df[
            ~staffing_candidate_in_request_for_specialist_df['status'].isin(stop_list)]

        staffing_candidates_df = pd.read_csv('data/staffing_candidates.csv')
        staffing_candidates_df = staffing_candidates_df.rename(columns={'id': 'candidate_id'})
        staffing_candidates_df = staffing_candidates_df[[
            'candidate_id', 'main_technology', 'seniority_level',
            'english_level', 'location_id']]

        clients_interviews_df = staffing_candidate_in_request_for_specialist_df.merge(staffing_candidates_df,
                                                                                      on='candidate_id',
                                                                                      how='left')

        staffing_interview_df = pd.read_csv('data/staffing_interview.csv')
        staffing_interview_df = staffing_interview_df[(~staffing_interview_df['subjective_readiness'].isnull())]
        staffing_interview_df = staffing_interview_df[['candidate_id', 'interview_type', 'subjective_readiness']]
        staffing_interview_df = staffing_interview_df[(staffing_interview_df['interview_type'] != "SOFT")]
        staffing_interview_df = staffing_interview_df[(~staffing_interview_df['candidate_id'].isnull())]

        new_staffing_interview_data = {}
        new_staffing_interview_dict = {'candidate_id': [], 'interview_type': [], 'subjective_readiness': []}

        for index, row in staffing_interview_df.iterrows():
            candidate_id = row['candidate_id']
            subjective_readiness = row['subjective_readiness']
            if candidate_id not in list(new_staffing_interview_data.keys()):
                new_staffing_interview_data[candidate_id] = [0, 0]
            new_staffing_interview_data[candidate_id][0] += subjective_readiness
            new_staffing_interview_data[candidate_id][1] += 1

        for candidate_id in list(new_staffing_interview_data.keys()):
            subjective_readiness_avg = new_staffing_interview_data[candidate_id][0] // \
                                       new_staffing_interview_data[candidate_id][1]
            new_staffing_interview_dict['candidate_id'].append(candidate_id)
            new_staffing_interview_dict['interview_type'].append('TECH')
            new_staffing_interview_dict['subjective_readiness'].append(subjective_readiness_avg)

        new_staffing_interview_df = pd.DataFrame(data=new_staffing_interview_dict)


        # Uncomment to get subjective_readiness
        # clients_interviews_df = clients_interviews_df.merge(new_staffing_interview_df, on='candidate_id', how='left')

        clients_interviews_df.to_csv(self.clients_interview_df_path, index=False)

    def prepare_data(self):
        clients_interviews_df = pd.read_csv(self.clients_interview_df_path)
        clients_interviews_df = clients_interviews_df[
            clients_interviews_df['main_technology'].notna() & clients_interviews_df['seniority_level'].notna()]

        clients_interviews_df['status'] = clients_interviews_df['status'].apply(
            lambda value: 0 if str(value) == 'REJECT' else 1)

        one_hot_encoder = OneHotEncoder()

        transformed_main_technology = one_hot_encoder.fit_transform(clients_interviews_df[['main_technology']])
        clients_interviews_df[one_hot_encoder.categories_[0]] = transformed_main_technology.toarray()

        transformed_seniority_level = one_hot_encoder.fit_transform(clients_interviews_df[['seniority_level']])
        clients_interviews_df[one_hot_encoder.categories_[0]] = transformed_seniority_level.toarray()

        transformed_english_level = one_hot_encoder.fit_transform(clients_interviews_df[['english_level']])
        clients_interviews_df['english_level'] = clients_interviews_df['english_level'].replace(np.nan, "B1")
        clients_interviews_df[one_hot_encoder.categories_[0]] = transformed_english_level.toarray()

        transformed_location_id = one_hot_encoder.fit_transform(clients_interviews_df[['location_id']])
        clients_interviews_df[one_hot_encoder.categories_[0]] = transformed_location_id.toarray()

        clients_interviews_df = clients_interviews_df.drop(['candidate_id', 'main_technology', 'seniority_level', 'english_level', 'location_id'], axis=1)

        clients_interviews_df.to_csv(self.clients_interview_df_path, index=False)


if __name__ == "__main__":
    clients_interviews_data_preprocessing = CLientsInterviewsDataPreprocessing()
    clients_interviews_data_preprocessing.run()
