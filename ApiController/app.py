from typing import Union

from fastapi import FastAPI, HTTPException, status

from .models import MLModel

app = FastAPI()
ml_model = MLModel(model_path='random_forest_model.sav', one_hot_encoder_path='one_hot_encoder.pkl', project_complexity_df_path='Data/project_complexity_df.csv')


@app.get("/predict")
async def make_predict(main_technology: str, seniority_level: str, english_level: str, project_name: Union[str, None] = "undefined"):
    try:
        data = {'main_technology': [main_technology], 'seniority_level': [seniority_level],
                'english_level': [english_level], 'project_name': project_name}
        prediction = ml_model.make_prediction(data=data)
        response = {
            "chance_of_success": (prediction[0][1] * 100).round(),
            "chance_of_failure": (prediction[0][0] * 100).round()
        }
        return response
    except Exception as e:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@app.get("/main_technologies")
async def get_main_tecnologies():
    try:
        main_tecnologies = ml_model.get_main_technologies().tolist()
        response = {
            'main_technologies': main_tecnologies
        }
        return response
    except Exception as e:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@app.get("/seniority_levels")
async def get_seniority_levels():
    try:
        seniority_levels = ml_model.get_seniority_levels().tolist()
        response = {
            'seniority_levels': seniority_levels
        }
        return response
    except Exception as e:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@app.get("/english_levels")
async def get_english_levels():
    try:
        english_levels = ml_model.get_english_levels().tolist()
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
