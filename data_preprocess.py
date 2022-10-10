import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


class ClientsInterviewsDataPreprocessing:
    def __init__(self):
        self.clients_interview_df_path = "clients_interviews.csv"

    # Run Method
    def run(self):
        # self.make_project_complexity_by_technologies_table()
        self.combine_datasets()
        self.prepare_data()

    # Combine Different Tables
    def combine_datasets(self):
        stop_list = ['CANCEL', 'TERMINATED']

        staffing_candidate_in_request_df = pd.read_csv(
            'Data/staffing_candidate_in_request_for_specialist.csv')
        staffing_candidate_in_request_df = staffing_candidate_in_request_df[
            ['status', 'candidate_id', 'request_for_specialist_id']]
        staffing_candidate_in_request_df = staffing_candidate_in_request_df[
            ~staffing_candidate_in_request_df['status'].isin(stop_list)]

        staffing_candidates_df = pd.read_csv('Data/staffing_candidates.csv', dtype={"additional_technologies": "str",
                                                                                    "vacation_from": "str",
                                                                                    "vacation_to": "str",
                                                                                    "last_vacation_date": "str"})
        staffing_candidates_df = staffing_candidates_df.rename(columns={'id': 'candidate_id'})
        staffing_candidates_df = staffing_candidates_df[[
            'candidate_id', 'main_technology', 'seniority_level',
            'english_level']]

        clients_interviews_df = staffing_candidate_in_request_df.merge(staffing_candidates_df,
                                                                       on='candidate_id',
                                                                       how='left')

        # Comment to delete subjective_readiness(don't have info for all candidates)
        # staffing_interview_df = pd.read_csv('data/staffing_interview.csv')
        # staffing_interview_df = staffing_interview_df[(~staffing_interview_df['subjective_readiness'].isnull())]
        # staffing_interview_df = staffing_interview_df[['candidate_id', 'interview_type', 'subjective_readiness']]
        # staffing_interview_df = staffing_interview_df[(staffing_interview_df['interview_type'] != "SOFT")]
        # staffing_interview_df = staffing_interview_df[(~staffing_interview_df['candidate_id'].isnull())]
        #
        # new_staffing_interview_data = {}
        # new_staffing_interview_dict = {'candidate_id': [], 'interview_type': [], 'subjective_readiness': []}
        #
        # for index, row in staffing_interview_df.iterrows():
        #    candidate_id = row['candidate_id']
        #    subjective_readiness = row['subjective_readiness']
        #    if candidate_id not in list(new_staffing_interview_data.keys()):
        #        new_staffing_interview_data[candidate_id] = [0, 0]
        #    new_staffing_interview_data[candidate_id][0] += subjective_readiness
        #    new_staffing_interview_data[candidate_id][1] += 1
        #
        # for candidate_id in list(new_staffing_interview_data.keys()):
        #    subjective_readiness_avg = new_staffing_interview_data[candidate_id][0] // \
        #                               new_staffing_interview_data[candidate_id][1]
        #    new_staffing_interview_dict['candidate_id'].append(candidate_id)
        #    new_staffing_interview_dict['interview_type'].append('TECH')
        #    new_staffing_interview_dict['subjective_readiness'].append(subjective_readiness_avg)
        #
        # new_staffing_interview_df = pd.DataFrame(data=new_staffing_interview_dict)
        #
        # clients_interviews_df = clients_interviews_df.merge(new_staffing_interview_df, on='candidate_id', how='left')
        # clients_interviews_df = clients_interviews_df[~clients_interviews_df['subjective_readiness'].isnull()]
        # clients_interviews_df = clients_interviews_df.drop(["interview_type"], axis=1)
        #

        new_staffing_candidate_in_request_df = pd.read_csv('Data/new_staffing_candidate_in_request_for_specialist.csv')
        clients_interviews_df = clients_interviews_df.merge(
            new_staffing_candidate_in_request_df[['candidate_id', 'request_for_specialist_id', 'project_name']],
            on=['candidate_id', 'request_for_specialist_id'], how='inner')

        clients_interviews_df = clients_interviews_df.drop_duplicates()

        #clients_interviews_df["complexity"] = np.nan

        new_inteviews_df = pd.read_csv('Data/new_interviews.csv')
        clients_interviews_df = clients_interviews_df.merge(
            new_inteviews_df[['candidate_id', 'project_name', 'subjective_readiness']],
            on=['candidate_id', 'project_name'], how='right')

        project_complexity_new_df = pd.read_csv('Data/project_complexity_new_df.csv')
        project_complexity_new_df = project_complexity_new_df.rename({'technology': 'main_technology'}, axis=1)

        clients_interviews_df = clients_interviews_df.merge(project_complexity_new_df[['project_name', 'main_technology', 'complexity']],
                                          on=['project_name', 'main_technology'], how='left')

        clients_interviews_df.to_csv(self.clients_interview_df_path, index=False)

    # Prepare data
    def prepare_data(self):
        clients_interviews_df = pd.read_csv(self.clients_interview_df_path)
        clients_interviews_df = clients_interviews_df[
            clients_interviews_df['main_technology'].notna() & clients_interviews_df['seniority_level'].notna()]

        clients_interviews_df['status'] = clients_interviews_df['status'].apply(
            lambda value: 0 if str(value) == 'REJECT' else 1)

        clients_interviews_df['english_level'] = clients_interviews_df['english_level'].replace(np.nan, "B1")

        project_complexity_new_df = pd.read_csv('Data/project_complexity_new_df.csv')


        for i, row in clients_interviews_df.iterrows():
            if pd.isnull(row['complexity']):
                project_name = row['project_name']
                mean_complexity = project_complexity_new_df[project_complexity_new_df['project_name'] == project_name]['complexity'].mean()
                clients_interviews_df.at[i, 'complexity'] = mean_complexity

        clients_interviews_df['complexity'] = clients_interviews_df['complexity'].replace(np.nan, 0.5)


        columns_to_encode = clients_interviews_df.drop(
            ['status', 'candidate_id', 'request_for_specialist_id', 'project_name', 'complexity',
             'subjective_readiness'], axis=1).columns

        one_hot_encoder = OneHotEncoder()

        transformed_features = one_hot_encoder.fit_transform(clients_interviews_df[columns_to_encode]).toarray()
        transformed_labels = np.array(one_hot_encoder.get_feature_names_out()).ravel()

        encoded_df = pd.DataFrame(transformed_features, columns=transformed_labels)
        clients_interviews_df = pd.concat([clients_interviews_df.drop(columns=columns_to_encode).reset_index(drop=True),
                                           encoded_df.reset_index(drop=True)], axis=1)

        clients_interviews_df['subjective_readiness'] = clients_interviews_df['subjective_readiness'].fillna(50.)

        # min_max_scaler = MinMaxScaler()
        # clients_interviews_df['subjective_readiness'] = min_max_scaler.fit_transform(clients_interviews_df[['subjective_readiness']])

        clients_interviews_df.to_csv(self.clients_interview_df_path, index=False)

        pickle.dump(one_hot_encoder, open('one_hot_encoder.pkl', 'wb'))

    # pickle.dump(min_max_scaler, open('min_max_scaler.pkl', 'wb'))

    def make_project_complexity_table(self):
        staffing_candidate_in_request_df = pd.read_csv('Data/new_staffing_candidate_in_request_for_specialist.csv')
        staffing_candidate_in_request_df.loc[staffing_candidate_in_request_df['status'] != "REJECT", 'status'] = 1
        staffing_candidate_in_request_df.loc[staffing_candidate_in_request_df['status'] == "REJECT", 'status'] = 0

        projects_failed = staffing_candidate_in_request_df[staffing_candidate_in_request_df['status'] == 0][
            ['project_name', 'status']]
        projects_failed = projects_failed.value_counts().reset_index(name='failed').drop('status', axis=1)

        projects_succeed = staffing_candidate_in_request_df[staffing_candidate_in_request_df['status'] == 1][
            ['project_name', 'status']]
        projects_succeed = projects_succeed.value_counts().reset_index(name='succeed').drop('status', axis=1)

        projects_total = staffing_candidate_in_request_df['project_name'].value_counts().reset_index(name='total')
        projects_total = projects_total.rename({'index': 'project_name'}, axis=1)

        project_complexity_df = projects_total.merge(projects_failed, on='project_name', how='left').merge(
            projects_succeed, on='project_name', how='left')
        project_complexity_df = project_complexity_df.fillna(0)

        project_complexity_df['complexity'] = project_complexity_df.apply(lambda row: row['failed'] / row['total'],
                                                                          axis=1)

        project_complexity_df.loc[len(project_complexity_df.index)] = ['undefined', 0, 0., 0., 0.5]

        project_complexity_df.to_csv('Data/project_complexity_df.csv', index=False)

    def get_interview_languages_for_projects(self):
        new_interviews_df = pd.read_csv('Data/new_interviews.csv')
        unique_projects = new_interviews_df['project_name'].unique()
        interview_languages = []
        for project in unique_projects:
            row = new_interviews_df[new_interviews_df['project_name'] == project].iloc[0]
            if row['interview_language'] == 'ENGLISH':
                interview_languages.append(1)
            else:
                interview_languages.append(0)
        interview_languages_for_projects = pd.DataFrame(
            data={'project_name': unique_projects, 'is_interview_language_english': interview_languages})
        interview_languages_for_projects.to_csv("Data/interview_languages_for_projects.csv", index=False)

    def make_project_complexity_by_technologies_table(self):
        stop_list = ['CANCEL', 'TERMINATED']
        staffing_candidate_in_request_df = pd.read_csv(
            'Data/staffing_candidate_in_request_for_specialist_with_technologies.csv')[
            ['project_name', 'status', 'technology']]
        staffing_candidate_in_request_df = staffing_candidate_in_request_df[
            ~staffing_candidate_in_request_df['status'].isin(stop_list)]
        staffing_candidate_in_request_df.loc[staffing_candidate_in_request_df['status'] != "REJECT", 'status'] = 1
        staffing_candidate_in_request_df.loc[staffing_candidate_in_request_df['status'] == "REJECT", 'status'] = 0

        delete_rows = []

        for row in staffing_candidate_in_request_df.iloc():
            row_number = row.name
            technologies = row['technology'].split(',')
            if len(technologies) > 1:
                delete_rows.append(row_number)
                project_name = row['project_name']
                status = row['status']
                for technology in technologies:
                    staffing_candidate_in_request_df.loc[staffing_candidate_in_request_df.shape[0]] = [project_name,status,technology]

        staffing_candidate_in_request_df = staffing_candidate_in_request_df.drop(delete_rows, axis=0)

        # staffing_candidate_in_request_df.to_csv('Data/staffing_candidate_in_request_for_specialist_with_technologies_transformed.csv', index=False)

        projects_failed = staffing_candidate_in_request_df[staffing_candidate_in_request_df['status'] == 0][
            ['project_name', 'technology', 'status']]
        projects_failed.to_csv('projects_failed.csv')
        projects_failed = projects_failed[['project_name', 'technology']].value_counts().reset_index(name='failed')

        projects_succeed = staffing_candidate_in_request_df[staffing_candidate_in_request_df['status'] == 1][
            ['project_name', 'technology', 'status']]
        projects_succeed = projects_succeed[['project_name', 'technology']].value_counts().reset_index(name='succeed')

        projects_total = staffing_candidate_in_request_df[['project_name', 'technology']].value_counts().reset_index(
            name='total')
        projects_total = projects_total.rename({'index': 'project_name'}, axis=1)

        project_complexity_df = projects_total.merge(projects_failed, on=['project_name', 'technology'],
                                                     how='left').merge(
            projects_succeed, on=['project_name', 'technology'], how='left')
        project_complexity_df = project_complexity_df.fillna(0)

        project_complexity_df['complexity'] = project_complexity_df.apply(lambda row: row['failed'] / row['total'],
                                                                          axis=1)

        # project_complexity_df.loc[len(project_complexity_df.index)] = ['undefined', 0, 0., 0., 0.5]

        project_complexity_df.to_csv('Data/project_complexity_new_df.csv', index=False)




if __name__ == "__main__":
    clients_interviews_data_preprocessing = ClientsInterviewsDataPreprocessing()
    clients_interviews_data_preprocessing.run()
