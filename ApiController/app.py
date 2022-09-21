from fastapi import FastAPI, HTTPException, status

from .models import MLModel

app = FastAPI()
ml_model = MLModel(model_path='random_forest_model.sav', one_hot_encoder_path='one_hot_encoder.pkl')


@app.get("/predict")
async def make_predict(main_technology: str, seniority_level: str, english_level: str, location_id: int):
    try:
        data = {'main_technology': [main_technology], 'seniority_level': [seniority_level],
                'english_level': [english_level], 'location_id': [location_id]}
        prediction = ml_model.make_prediction(data=data)
        response = {
            "chance_of_failure": prediction[0][0],
            "chance_of_success": prediction[0][1]
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

@app.get("/location_ids")
async def get_location_ids():
    try:
        location_ids = ml_model.get_location_ids().tolist()
        response = {
            'location_ids': location_ids
        }
        return response
    except Exception as e:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))