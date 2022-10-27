from typing import Union

from fastapi import FastAPI, HTTPException, status

from .models import MLModel, Candidate

import pandas as pd

app = FastAPI()
ml_model = MLModel(model_path='random_forest_model.sav', one_hot_encoder_path='one_hot_encoder.pkl',
                   project_complexity_df_path='Data/project_complexity_new_df.csv')
succeed_projects_df = pd.read_csv('Data/succeed_projects_count.csv')
failed_projects_df = pd.read_csv('Data/failed_projects_count.csv')
staffing_candidates_df = pd.read_csv('Data/staffing_candidates.csv').rename({'id':'candidate_id'}, axis=1)


@app.get("/predict")
async def make_predict(candidate_id: float, interview_technology: str, candidate_seniority_level: str, candidate_english_level: str, interview_language: str, candidate_subjective_readiness: Union[str, None] = 'undefined', project_name: Union[str, None] = "undefined"):
    try:
        candidate = Candidate(candidate_id=candidate_id, succeed_projects_df=succeed_projects_df, failed_projects_df=failed_projects_df)

        data = {'succeed_projects_count': candidate.succeed_projects_count, 'failed_projects_count': candidate.failed_projects_count, 'interview_technology': interview_technology, 'candidate_english_level': candidate_english_level, 'candidate_subjective_readiness': candidate_subjective_readiness,
                'project_name': project_name, 'candidate_seniority_level': candidate_seniority_level, 'interview_language': interview_language}

        prediction = ml_model.make_prediction(data=data)
        response = {
            "chance_of_success": (prediction[0][1] * 100).round(),
            "chance_of_failure": (prediction[0][0] * 100).round()
        }
        return response
    except Exception as e:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/interview_tecnologies")
async def get_interview_tecnologies():
    try:
        interview_tecnologies = ml_model.get_interview_tecnologies().tolist()
        response = {
            'interview_tecnologies': interview_tecnologies
        }
        return response
    except Exception as e:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/seniority_levels")
async def get_seniority_levels():
    try:
        seniority_levels = list(ml_model.get_seniority_levels().keys())
        response = {
            'seniority_levels': seniority_levels
        }
        return response
    except Exception as e:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/english_levels")
async def get_english_levels():
    try:
        english_levels = list(ml_model.get_english_levels().keys())
        response = {
            'english_levels': english_levels
        }
        return response
    except Exception as e:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/project_names")
async def get_project_names():
    try:
        project_names = ml_model.get_project_names()
        response = {
            'project_names': project_names
        }
        return response
    except Exception as e:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
