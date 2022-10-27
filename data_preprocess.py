import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from config import *


class ClientsInterviewsDataPreprocessing:
    def __init__(self, staffing_candidate_in_request_for_specialist_path, staffing_candidates_path, working_dataset_path):
        self.staffing_candidate_in_request_for_specialist_path = staffing_candidate_in_request_for_specialist_path
        self.staffing_candidates_path = staffing_candidates_path
        self.working_dataset_path = working_dataset_path

    # Run Method
    def run(self):
        self.get_and_preapare_data()


    def get_and_preapare_data(self):
        # Read datasets
        staffing_candidate_in_request_df = pd.read_csv(self.staffing_candidate_in_request_for_specialist_path)
        staffing_candidates_df = pd.read_csv(self.staffing_candidates_path)

        # Delete duplicates
        print(staffing_candidate_in_request_df['status'].count())
        staffing_candidate_in_request_df = staffing_candidate_in_request_df.sort_values(["id","request_for_specialist_id","candidate_id","status","result_comment","subjective_readiness","interview_seniority_level","interview_language","technology","project_name"]).drop_duplicates(["id","request_for_specialist_id","candidate_id","status","result_comment","interview_seniority_level","interview_language","technology","project_name"], keep='last', ignore_index=True)
        print(staffing_candidate_in_request_df['status'].count())

        staffing_candidate_in_request_df = staffing_candidate_in_request_df.drop(['interview_seniority_level'], axis=1)
        print(staffing_candidate_in_request_df['status'].count())

        # Rename column 'id' to 'candidate_id'
        staffing_candidates_df = staffing_candidates_df.rename({'id': 'candidate_id'}, axis=1)

        # Get 'english_level' and 'seniority_level' columns
        staffing_candidate_in_request_df = staffing_candidate_in_request_df.merge(
            staffing_candidates_df[['candidate_id', 'english_level', 'seniority_level']], on='candidate_id', how='left')

        # Drop unnecessary columns
        staffing_candidate_in_request_df = staffing_candidate_in_request_df.drop(
            ['id', 'request_for_specialist_id', 'result_comment'], axis=1)

        # Rename columns
        staffing_candidate_in_request_df = staffing_candidate_in_request_df.rename({
            'technology': 'interview_technology',
            'english_level': 'candidate_english_level',
            'seniority_level': 'candidate_seniority_level',
            'subjective_readiness': 'candidate_subjective_readiness'
        }, axis=1)

        # Delete cancelled interviews
        stop_list = ['CANCEL', 'TERMINATED']
        staffing_candidate_in_request_df = staffing_candidate_in_request_df[
            ~staffing_candidate_in_request_df['status'].isin(stop_list)]

        # Encode 'status' column
        staffing_candidate_in_request_df.loc[staffing_candidate_in_request_df['status'] != "REJECT", 'status'] = 1
        staffing_candidate_in_request_df.loc[staffing_candidate_in_request_df['status'] == "REJECT", 'status'] = 0

        # Fill NaNs of 'subjective_readiness' with mean
        staffing_candidate_in_request_df['candidate_subjective_readiness'] = staffing_candidate_in_request_df[
            'candidate_subjective_readiness'].fillna(
            staffing_candidate_in_request_df['candidate_subjective_readiness'].mean())

        # Fill NaNs of 'english_level' in proportions
        staffing_candidate_in_request_df = self.fill_nans_with_proportion(dataframe=staffing_candidate_in_request_df,
                                                                          column_name='candidate_english_level')

        # Encode english levels
        staffing_candidate_in_request_df['candidate_english_level'] = staffing_candidate_in_request_df['candidate_english_level'].replace(english_levels_labels)

        # Encode interview_language
        staffing_candidate_in_request_df['interview_language'] = staffing_candidate_in_request_df['interview_language'].replace(interview_language_labels)

        # Fill NaNs in candidate_seniority_level
        staffing_candidate_in_request_df = self.fill_nans_with_proportion(dataframe=staffing_candidate_in_request_df,
                                                                          column_name='candidate_seniority_level')

        # Encode labels in candidate_seniority_level
        staffing_candidate_in_request_df['candidate_seniority_level'] = staffing_candidate_in_request_df['candidate_seniority_level'].replace(seniority_level_labels)

        # Split technologies
        # print(staffing_candidate_in_request_df['interview_technology'].count())
        # count_tech = 0
        # for i, row in staffing_candidate_in_request_df.iterrows():
        #    technologies = row['interview_technology'].split(',')
        #    if len(technologies) > 1:
        #        print(technologies)
        #        count_tech += len(technologies)
        #        print(staffing_candidate_in_request_df.shape[0])
        #        for technology in technologies:
        #            print(technology)
        #            staffing_candidate_in_request_df.loc[staffing_candidate_in_request_df.shape[0]] = row.tolist()
        #            print(staffing_candidate_in_request_df.shape[0])
        #            staffing_candidate_in_request_df.at[staffing_candidate_in_request_df.shape[0] - 1, 'interview_technology'] = technology
        #        print(staffing_candidate_in_request_df.shape[0])
        # print(count_tech)
        # print(staffing_candidate_in_request_df['interview_technology'].count())

        # Make complexity table
        complexity_df = self.make_project_complexity_by_technologies_table(
            staffing_candidate_in_request_df[['project_name', 'status', 'interview_technology']])[
            ['project_name', 'interview_technology', 'complexity']]

        # Add complexity to the table
        staffing_candidate_in_request_df = staffing_candidate_in_request_df.merge(complexity_df, on=['project_name',
                                                                                                   'interview_technology'],
                                                                                how='left')
        # Delete project_name
        staffing_candidate_in_request_df = staffing_candidate_in_request_df.drop(['project_name'], axis=1)

        # One Hot Encode interview_technology
        columns_to_encode = ['interview_technology']

        one_hot_encoder = OneHotEncoder()

        transformed_features = one_hot_encoder.fit_transform(staffing_candidate_in_request_df[columns_to_encode]).toarray()
        transformed_labels = np.array(one_hot_encoder.get_feature_names_out()).ravel()

        encoded_df = pd.DataFrame(transformed_features, columns=transformed_labels)
        staffing_candidate_in_request_df = pd.concat([staffing_candidate_in_request_df.drop(columns=columns_to_encode).reset_index(drop=True),
                                         encoded_df.reset_index(drop=True)], axis=1)

        # Save One Hot Encoder
        #pickle.dump(one_hot_encoder, open('one_hot_encoder.pkl', 'wb'))

        # Get succeed_projects_count and failed_projects_count

        succeed_projects_count_df, failed_projects_count_df = self.get_succeed_and_failed_projects_for_candidates(staffing_candidate_in_request_df[['status', 'candidate_id']])

        # Merge with succeed_projects_count_df
        staffing_candidate_in_request_df = staffing_candidate_in_request_df.merge(succeed_projects_count_df, on='candidate_id', how='left')

        # Merge with succeed_projects_count_df
        staffing_candidate_in_request_df = staffing_candidate_in_request_df.merge(failed_projects_count_df, on='candidate_id', how='left')

        # Drop candidate_id
        staffing_candidate_in_request_df = staffing_candidate_in_request_df.drop(['candidate_id'], axis=1)

        # Save Interviews
        staffing_candidate_in_request_df.to_csv(working_dataset_path, index=False)

    def fill_nans_with_proportion(self, dataframe, column_name):
        column_value_counts = dataframe[column_name].value_counts()
        column_values_total = dataframe[column_name].count()
        column_values_percentage = {}

        for level in column_value_counts.index.tolist():
            column_values_percentage[level] = column_value_counts[level] / column_values_total

        column_nan_values_total = dataframe[column_name].isna().sum()

        column_nan_distribution = {}

        filled_nans = 0
        for level in list(column_values_percentage.keys()):
            level_filled_nans = column_values_percentage[level] * column_nan_values_total
            column_nan_distribution[level] = round(level_filled_nans)
            filled_nans += level_filled_nans

        while filled_nans < column_nan_values_total + 1:
            most_counted_level = list(column_values_percentage.keys())[0]
            column_nan_distribution[most_counted_level] += 1
            filled_nans += 1

        for level in list(column_nan_distribution.keys()):
            level_fill_nans_total = column_nan_distribution[level]
            level_filled_nans_count = 0
            for i, row in dataframe.iterrows():
                if level_filled_nans_count == level_fill_nans_total:
                    break
                if pd.isnull(row[column_name]):
                    level_filled_nans_count += 1
                    dataframe.at[i, column_name] = level

        return dataframe

    def make_project_complexity_by_technologies_table(self, staffing_candidate_in_request_df):
        projects_failed = staffing_candidate_in_request_df[staffing_candidate_in_request_df['status'] == 0][
            ['project_name', 'interview_technology', 'status']]
        projects_failed = projects_failed[['project_name', 'interview_technology']].value_counts().reset_index(
            name='failed')

        projects_succeed = staffing_candidate_in_request_df[staffing_candidate_in_request_df['status'] == 1][
            ['project_name', 'interview_technology', 'status']]
        projects_succeed = projects_succeed[['project_name', 'interview_technology']].value_counts().reset_index(
            name='succeed')

        projects_total = staffing_candidate_in_request_df[
            ['project_name', 'interview_technology']].value_counts().reset_index(
            name='total')
        projects_total = projects_total.rename({'index': 'project_name'}, axis=1)

        project_complexity_df = projects_total.merge(projects_failed, on=['project_name', 'interview_technology'],
                                                     how='left').merge(
            projects_succeed, on=['project_name', 'interview_technology'], how='left')
        project_complexity_df = project_complexity_df.fillna(0)

        project_complexity_df['complexity'] = project_complexity_df.apply(lambda row: row['failed'] / row['total'],
                                                                          axis=1)
        project_complexity_df.loc[len(project_complexity_df.index)] = ['undefined', 'undefined', 0, 0., 0., 0.5]
        project_complexity_df.to_csv('Data/project_complexity_new_df.csv', index=False)
        return project_complexity_df

    def get_succeed_and_failed_projects_for_candidates(self, candidates_history):
        new_df = candidates_history.groupby(['candidate_id', 'status']).size().sort_values(ascending=False).reset_index(
            name='count')
        print(new_df)
        succeed_projects_count = {}
        failed_projects_count = {}
        for i, row in new_df.iterrows():
            candidate_id = row['candidate_id']
            status = row['status']
            count = row['count']
            if candidate_id not in list(succeed_projects_count.keys()):
                succeed_projects_count[candidate_id] = 0
            if candidate_id not in list(failed_projects_count.keys()):
                failed_projects_count[candidate_id] = 0
            if status == 1:
                succeed_projects_count[candidate_id] += count
            if status == 0:
                failed_projects_count[candidate_id] += count
        succeed_projects_count_df = pd.DataFrame(data={'candidate_id': list(succeed_projects_count.keys()), 'succeed_projects_count': list(succeed_projects_count.values())})
        succeed_projects_count_df.to_csv('Data/succeed_projects_count.csv', index=False)

        failed_projects_count_df = pd.DataFrame(data={'candidate_id': list(failed_projects_count.keys()), 'failed_projects_count': list(failed_projects_count.values())})
        failed_projects_count_df.to_csv('Data/failed_projects_count.csv', index=False)
        return succeed_projects_count_df, failed_projects_count_df


if __name__ == "__main__":
    staffing_candidate_in_request_for_specialist_path = 'Data/staffing_candidate_in_request_for_specialist.csv'
    staffing_candidates_path = 'Data/staffing_candidates.csv'
    working_dataset_path = 'Data/clients_interviews.csv'

    clients_interviews_data_preprocessing = ClientsInterviewsDataPreprocessing(
        staffing_candidate_in_request_for_specialist_path=staffing_candidate_in_request_for_specialist_path,
        staffing_candidates_path=staffing_candidates_path,
        working_dataset_path=working_dataset_path)
    clients_interviews_data_preprocessing.run()
