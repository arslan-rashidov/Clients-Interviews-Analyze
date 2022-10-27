import pickle
import random

import pandas as pd

import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import requests
import json
import numpy as np
from ApiController.models import Candidate

from config import *

succeed_projects_df = pd.read_csv('Data/succeed_projects_count.csv')
failed_projects_df = pd.read_csv('Data/failed_projects_count.csv')


class DataExportation:
    def __init__(self, dataset_path, sheet_key, sheet_id, json_credentials_path):
        self.dataset = pd.read_csv(dataset_path)
        self.sheet_key = sheet_key
        self.sheet_id = sheet_id
        self.json_credentials_path = json_credentials_path

    def get_probabilities(self):
        probabilities_df = \
        pd.read_csv('DataProbabilities/staffing_candidate_in_request_for_specialist_with_probabilites.csv')[
            ['id', 'project_name', 'probability']].rename({'id': 'candidate_id'}, axis=1)
        probabilities_df = probabilities_df.sort_values('probability')
        probabilities_df = probabilities_df.drop_duplicates(subset=['candidate_id', 'project_name'], keep='last')
        return probabilities_df

    def run(self):
        dataset = pd.read_csv('sheets_predictions.csv')
        #dataset['prediction'] = dataset['prediction'].replace({1: "Success", 0: "Failed"})
        #print(dataset[['Status', 'prediction']].value_counts())
        #dataset = self.get_from_sheets()
        #dataset = self.tranform_dataset_in_sheets(dataset)
        #dataset.to_csv('sheets_dataset.csv', index=False)
        #validation_dataset = self.transform_dataset_old_model(dataset)
        #validation_dataset.to_csv('validation_dataset.csv', index=False)
        #predictions = self.check_dataframe_predictions(dataset)
        #dataset = dataset.assign(prediction=predictions)
        #dataset.to_csv('sheets_predictions.csv', index=False)

        # dataset.to_csv('validation_dataset_transformed.csv', index=False)
        # print(dataset)
        # random_subset_df = self.get_random_subset(rows_number=100)
        # predictions = self.check_dataframe_predictions(dataset)
        # dataset = dataset.assign(prediction_new=predictions)
        # dataset.to_csv('sheets_predictions.csv')
        # dataset = pd.read_csv('sheets_predictions.csv')
        # dataset = pd.read_csv('validation_dataset.csv')[['prediction_new', 'Candidate Id']]
        # dataset.to_csv('validation_dataset.csv')
        # print(dataset.to_string())
        self.export_to_sheets(dataset)

    def transform_dataset_old_model(self, dataset):
        print(dataset['candidate_id'].value_counts(dropna=False))
        print(dataset['Subjective readiness (%)'].value_counts(dropna=False))
        print(dataset['succeed_projects_count'].value_counts(dropna=False))
        print(dataset['failed_projects_count'].value_counts(dropna=False))
        print(dataset['English'].value_counts(dropna=False))
        print(dataset['Level'].value_counts(dropna=False))


    def check_predictions_locally(self, dataframe):
        loaded_model = pickle.load(open('random_forest_model.sav', 'rb'))
        x_test = dataframe.drop(['status'], axis=1)
        y_test = dataframe['status']
        y_test = y_test.rename({'Success': 1, 'Failed': 0})
        print(y_test)
        print(loaded_model.predict(x_test))
        return loaded_model.predict(x_test)

    def transform_validation_dataset(self, dataframe):
        dataframe = dataframe.drop(
            ['Language of interview', 'Resolved', 'English', 'Candidate from', 'Summary', 'Full name', 'Key'], axis=1)
        dataframe = dataframe.rename(
            {'Project': 'project_name', 'Technology SPT': 'interview_technology', 'Level': 'candidate_seniority_level',
             'Subjective readiness (%)': 'candidate_subjective_readiness', 'Status': 'status',
             'Vacancy level': 'interview_seniority_level'}, axis=1)
        dataframe['interview_seniority_level'] = dataframe['interview_seniority_level'].replace(
            {'Senior': 'S1', 'Middle': 'M2', 'Junior': 'J2', 'Architect': 'S1'})
        dataframe['interview_seniority_level'] = dataframe['interview_seniority_level'].replace(seniority_level_labels)
        dataframe['candidate_seniority_level'] = dataframe['candidate_seniority_level'].replace(seniority_level_labels)
        project_complexity_df = pd.read_csv('Data/project_complexity_new_df.csv')
        dataframe = dataframe.merge(project_complexity_df[['project_name', 'interview_technology', 'complexity']],
                                    on=['project_name', 'interview_technology'], how='left')
        dataframe['complexity'] = dataframe['complexity'].fillna(dataframe['complexity'].mean())
        dataframe.loc[dataframe['status'] != "REJECT", 'status'] = 1
        dataframe.loc[dataframe['status'] == "REJECT", 'status'] = 0
        print(dataframe.columns)
        dataframe = dataframe[['candidate_id', 'project_name', 'status', 'candidate_subjective_readiness', 'interview_seniority_level',
                               'candidate_seniority_level', 'complexity', 'interview_technology',
                               'succeed_projects_count', 'failed_projects_count']]
        loaded_one_hot_encoder = pickle.load(open('one_hot_encoder.pkl', 'rb'))

        dataframe['levels_equal'] = ''
        for i, row in dataframe.iterrows():
            if row['interview_seniority_level'] == row['candidate_seniority_level']:
                levels_equal = 'EQUAL'
            elif row['interview_seniority_level'] > row['candidate_seniority_level']:
                levels_equal = 'HIGHER'
            else:
                levels_equal = 'LOWER'
            dataframe.at[i, 'levels_equal'] = levels_equal

        # Salary_in_usd
        salaries_df = pd.read_csv('Data/staffing_candidates.csv')[
            ['id', 'salary_in_usd']].rename({'id': 'candidate_id'}, axis=1).drop_duplicates()
        print(salaries_df)
        dataframe = dataframe.merge(salaries_df, on='candidate_id',
                                    how='left')

        # Location_id
        location_ids_df = pd.read_csv('Data/staffing_candidates.csv')[
            ['id', 'location_id']].rename({'id': 'candidate_id'}, axis=1).drop_duplicates()
        dataframe = dataframe.merge(location_ids_df,
                                    on='candidate_id',
                                    how='left')

        dataframe['salary_in_usd'] = dataframe[
            'salary_in_usd'].fillna(dataframe['salary_in_usd'].mean())

        dataframe = dataframe.drop(
            ['interview_seniority_level', 'candidate_seniority_level'], axis=1)

        # Get probabilities
        probabilities_df = self.get_probabilities()
        print(dataframe['status'].count())
        dataframe = dataframe.merge(probabilities_df,
                                    on=['candidate_id', 'project_name'],
                                    how='left')
        dataframe['probability'] = dataframe['probability'].fillna(0)
        dataframe['probability'] = dataframe['probability'].replace(
            {100.0: 1, 40.0: 0, 35.0: 0})
        print(dataframe['status'].count())
        dataframe['location_id'] = dataframe['location_id'].replace({66.0: np.nan, 78.0: np.nan})

        new_dataframe = dataframe[['interview_technology', 'levels_equal', 'location_id']]
        transformed_features = loaded_one_hot_encoder.transform(new_dataframe)
        transformed_features = transformed_features.toarray()
        transformed_labels = np.array(loaded_one_hot_encoder.get_feature_names_out()).ravel()
        new_dataframe = pd.DataFrame(transformed_features, columns=transformed_labels)
        new_dataframe.insert(loc=0,
                             column='status',
                             value=dataframe['status'].values)
        new_dataframe.insert(loc=1,
                             column='candidate_subjective_readiness',
                             value=dataframe['candidate_subjective_readiness'].values)
        new_dataframe.insert(loc=2,
                             column='complexity',
                             value=dataframe['complexity'].values)
        new_dataframe.insert(loc=3,
                             column='salary_in_usd',
                             value=dataframe['salary_in_usd'].values)
        new_dataframe.insert(loc=len(new_dataframe.columns),
                             column='succeed_projects_count',
                             value=dataframe['succeed_projects_count'].values)
        new_dataframe.insert(loc=len(new_dataframe.columns),
                             column='failed_projects_count',
                             value=dataframe['failed_projects_count'].values)
        new_dataframe.insert(loc=len(new_dataframe.columns),
                             column='probability',
                             value=dataframe['probability'].values)
        return new_dataframe

    def tranform_dataset_in_sheets(self, dataframe):
        candidate_id_df = pd.read_csv('Data/candidate_id.csv')
        c = 0
        t = 0

        technologies = {
            "interview_tecnologies": [
                "ANDROID",
                "ANDROID,IOS",
                "ANDROID,JAVA,KOTLIN",
                "ANDROID,KOTLIN",
                "ANGULAR",
                "ANGULAR,ANGULAR_JS",
                "ANGULAR,JAVA",
                "ANGULAR_JS",
                "ARCHITECT",
                "AWS",
                "AWS,CLOUD_ADMINISTRATOR,DEVOPS,POSTGRESQL",
                "AWS,POSTGRESQL,PY_TEST,PYTHON",
                "BI_ANALYST",
                "BUSINESS_ANALYST",
                "BUSINESS_ANALYST,SYSTEM_ANALYST",
                "C_PLUSPLUS",
                "C_SHARP",
                "DATA_ANALYST",
                "DATA_ENGINEER",
                "DATA_ENGINEER_DBA",
                "DATA_ENGINEER_ETL",
                "DATA_ENGINEER_ETL,SQL",
                "DATA_SCIENCE_ML",
                "DBA",
                "DELIVERY_MANAGER",
                "DEVOPS",
                "DEVOPS,DEVSECOPS",
                "DRUPAL",
                "EXT_JS",
                "FLUTTER",
                "GO",
                "GO,POSTGRESQL",
                "GO,VUE_JS",
                "GRAPHIC_DESIGNER",
                "GRAPHIC_DESIGNER,WEB_DESIGNER",
                "HTML",
                "IOS",
                "JAVA",
                "JAVA,KOTLIN",
                "JAVA,KOTLIN,SPRING",
                "JAVA,NET,PYTHON,GO,RUBY",
                "JAVA,PYTHON",
                "JAVA,SPRING",
                "KOTLIN",
                "KOTLIN,SPRING,SCALA",
                "NET",
                "NET,C_SHARP",
                "NET,TYPE_SCRIPT",
                "NODE_JS",
                "NODE_JS,TYPE_SCRIPT",
                "NODE_REACT",
                "NODE_REACT,REDUX,TYPE_SCRIPT",
                "ORACLE",
                "PHP",
                "PHP,REACT,TYPE_SCRIPT",
                "PHP,SYMFONY",
                "PHP,VUE_JS",
                "PHP,YII",
                "POSTGRESQL",
                "POSTGRESQL,AWS",
                "POSTGRESQL,PYTHON",
                "POWER_BI,SQL",
                "PRODUCT_OWNER",
                "PROJECT_MANAGEMENT",
                "PROJECT_MANAGER",
                "PYTHON",
                "PYTHON,QA_MOBILE",
                "QA",
                "QA_ANALYST",
                "QA_ANALYST,QA_ENGINEER",
                "QA_ARCHITECT",
                "QA_AUTOMATION",
                "QA_AUTOMATION_C_SHARP",
                "QA_AUTOMATION_ENGINEER_JAVA",
                "QA_AUTOMATION_ENGINEER_JAVA,QA_AUTOMATION_ENGINEER_JS",
                "QA_AUTOMATION_ENGINEER_JS",
                "QA_AUTOMATION_ENGINEER_JS,QA_AUTOMATION_ENGINEER_JAVA",
                "QA_AUTOMATION_ENGINEER_PYTHON",
                "QA_ENGINEER",
                "QA_ENGINEER,QA_MOBILE",
                "QA_LEAD,QA_MANAGER",
                "QA_MOBILE",
                "QA_PERFORMANCE",
                "REACT",
                "REACT,REDUX",
                "REACT,REDUX,C_SHARP,TYPE_SCRIPT,NET",
                "REACT,TYPE_SCRIPT",
                "REACT_NATIVE",
                "ROR",
                "RUBY",
                "SCALA",
                "SCRUM_MASTER",
                "SOFTWARE_ARCHITECT",
                "SOLUTIONS_ARCHITECT",
                "SPRING",
                "SQL",
                "SYSTEM_ANALYST",
                "SYSTEM_ANALYST,BUSINESS_ANALYST",
                "UX_UI",
                "VUE_JS",
                "VUE_JS,TYPE_SCRIPT",
                "WEB_DESIGNER",
                "WPF"
            ]
        }['interview_tecnologies']

        transform_technologies = {
            'QAAUTOMATIONJS': 'QA_AUTOMATION_ENGINEER_JAVA',
            'QAAUTOMATIONCSHARP': 'QA_AUTOMATION_C_SHARP',
            'DATASCIENCE': 'DATA_ANALYST',
            'UXUI': 'UX_UI',
            'REACTNATIVE,RUBY': 'REACT_NATIVE',
            'REACTNATIVE': 'REACT_NATIVE',
            'QA_AUTOMATION_ENGINEER_JAVA,QAAUTOMATIONJS': 'QA_AUTOMATION_ENGINEER_JS,QA_AUTOMATION_ENGINEER_JAVA',
            'PROJECTMANAGEMENT': 'PROJECT_MANAGEMENT',
            'AUTOMATIONJAVA': 'QA_AUTOMATION_ENGINEER_JAVA',
            'QA_ENGINEER,QAMOBILE': 'QA_ENGINEER,QA_MOBILE',
            'NODE_JS,ARCHITECT': 'NODE_JS',
            'NET,ANGULAR': 'NET',
            'QAMOBILE': 'QA_MOBILE',
            'QALEAD': 'QA_LEAD,QA_MANAGER',
            'CPLUSPLUS': 'C_PLUSPLUS',
            'QAAUTOMATIONJS,QAMOBILE': 'QA_MOBILE',
            'ANGULAR,NODE_JS': 'ANGULAR,ANGULAR_JS',
            'NET,REACT,DEVFULLSTACK': 'REACT',
            'BIGDATA': 'DATA_ENGINEER',
            'QAPERFORMANCE': 'QA_PERFORMANCE',
            'REACT,REACTNATIVE': 'REACT_NATIVE',
            'EXTJS': 'EXT_JS',
            'NET,REACT': 'NET',
            'DEVOPS,SECURITY': 'DEVOPS',
            'DOTNET': 'NET',
            'QAMANUAL': 'QA_ENGINEER',
            'SYSTEMANALYST': 'SYSTEM_ANALYST',
            'NODEJS': 'NODE_JS',
            'VUEJS': 'VUE_JS',
            'BUSINESSANALYST': 'BUSINESS_ANALYST',
            'QAMANUAL,QAAUTOMATIONJAVA': 'QA_AUTOMATION_ENGINEER_JAVA'
        }
        zeros = 0
        total = 0

        for i, init_row in dataframe.iterrows():
            eng_levels = ['A1', 'B1', 'C1']
            if init_row['English'] == '':
                dataframe.at[i, 'English'] = random.choice(eng_levels)

            name = init_row['Full name']
            row = candidate_id_df[(candidate_id_df['name_rus'] == name) | (
                    candidate_id_df['name_eng'] == name)]
            candidate_id = row['id']
            if len(candidate_id.to_list()) != 0:
                dataframe.at[i, 'Candidate Id'] = candidate_id.iloc[0]

            technology = init_row['Technology SPT'].upper()
            technology = technology.replace(" ", "")
            if technology not in technologies:
                print(technology)
                technology = transform_technologies[technology]
            dataframe.at[i, 'Technology SPT'] = technology

            if float(init_row['Subjective readiness (%)']) > 100:
                dataframe.at[i, 'Subjective readiness (%)'] = float(
                    str(init_row['Subjective readiness (%)'])[0] + str(init_row['Subjective readiness (%)'])[0])

        dataframe['Subjective readiness (%)'] = dataframe['Subjective readiness (%)'].fillna(
            dataframe['Subjective readiness (%)'].mean())

        dataframe = dataframe.rename({'Candidate Id': 'candidate_id'}, axis=1)

        dataframe = dataframe.merge(succeed_projects_df, on='candidate_id', how='left')
        dataframe = dataframe.merge(failed_projects_df, on='candidate_id', how='left')

        dataframe['succeed_projects_count'] = dataframe['succeed_projects_count'].fillna(0)
        dataframe['failed_projects_count'] = dataframe['failed_projects_count'].fillna(0)
        print(dataframe.to_string())

        dataframe = dataframe[dataframe['candidate_id'].notna()]
        dataframe['Level'] = dataframe['Level'].fillna('M1')

        dataframe['English'] = dataframe['English'].fillna('B1')

        return dataframe

    def check_dataframe_predictions(self, dataframe):
        predictions = []
        status_count = 0
        status_total = 0
        for i, row in dataframe.iterrows():
            print(row)

            status = row['Status']

            candidate_id = row['candidate_id']
            if len(str(candidate_id)) == 0:
                predictions.append(None)
                continue
            if len(str(row['Subjective readiness (%)'])) == 0:
                predictions.append(None)
                continue
            if len(str(row['Level'])) == 0:
                predictions.append(None)
                continue

            candidate = Candidate(candidate_id, succeed_projects_df, failed_projects_df)

            params = {
                'candidate_subjective_readiness': str(row['Subjective readiness (%)']),
                'interview_language': str(row['Language of interview']).upper(),
                'interview_technology': str(row['Technology SPT']),
                'candidate_english_level': str(row['English']),
                'candidate_seniority_level': str(row['Level']).upper(),
                'project_name': str(row['Project']),
                'candidate_id': candidate_id
            }
            # status_prediction = 0
            result = requests.get('http://127.0.0.1:8000/predict/', params=params)
            response = json.loads(result.text)
            print(response)
            pred = ""
            if response['chance_of_success'] > response['chance_of_failure']:
                status_prediction = f"Success - {response['chance_of_success']}"
                pred = 'Success'
            else:
                status_prediction = f"Failed - {response['chance_of_failure']}"
                pred = 'Failed'
            if pred == status:
                status_count += 1
            status_total += 1

            # if str(status_prediction) == str(status):
            #    total_true_preictions += 1
            # else:
            #    if str(row['succeed_projects_count']) == "0":
            #        if response['chance_of_success'] != 0:
            #            print(
            #            f"{row['succeed_projects_count']} - {row['failed_projects_count']} - {response['chance_of_success']} - {False}")

            predictions.append(status_prediction)
        print(f"{status_count}/{status_total}")
        return predictions

    def get_random_subset(self, rows_number):
        random_subset_df = self.dataset.sample(n=100)
        return random_subset_df

    def export_to_sheets(self, dataframe):
        scopes = ['https://www.googleapis.com/auth/spreadsheets',
                  'https://www.googleapis.com/auth/drive']

        credentials = Credentials.from_service_account_file(self.json_credentials_path, scopes=scopes)

        gc = gspread.authorize(credentials)

        gauth = GoogleAuth()
        drive = GoogleDrive(gauth)

        # open a google sheet
        gs = gc.open_by_key(self.sheet_key)
        # select a work sheet from its name
        worksheet1 = gs.worksheet('Лист1')

        worksheet1.clear()
        set_with_dataframe(worksheet=worksheet1, dataframe=dataframe, include_index=False,
                           include_column_header=True, resize=True)

    def get_from_sheets(self):
        scopes = ['https://www.googleapis.com/auth/spreadsheets',
                  'https://www.googleapis.com/auth/drive']

        credentials = Credentials.from_service_account_file(self.json_credentials_path, scopes=scopes)

        gc = gspread.authorize(credentials)

        gauth = GoogleAuth()
        drive = GoogleDrive(gauth)

        # open a google sheet
        gs = gc.open_by_key(self.sheet_key)
        # select a work sheet from its name
        worksheet1 = gs.worksheet('Лист1')

        dataframe = pd.DataFrame(worksheet1.get_all_records())
        return dataframe


if __name__ == '__main__':
    data_exportation = DataExportation(dataset_path='Data/clients_interviews.csv',
                                       sheet_key='1Gha_QCoiie4vBAs0mBeasvjHKdNQBvSTudAcAVEg5Xo', sheet_id='0',
                                       json_credentials_path='atomic-venture-341006-b463fe7af64b.json')
    data_exportation.run()
