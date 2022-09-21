#Clients Interviews Analyze

## API

1. cd to the folder
2. python3 -m pip install -r requirements.txt
3. python3 -m ApiController
4. http://127.0.0.1:8000/docs


### Used Libraries

1. Uvicorn
2. FastApi


## Model Building

### Used libraries

1. Pandas
2. Numpy
3. Sklearn

### Data Preprocessing (data_preprocess.py)

Our current dataset consists of:
1. Main Technology of a candidate
2. Seniority level of a candidate
3. English level of a candidate
4. Location ID of a candidate

Features were prepared with OneHotEncoder.

### Analyze (interviews_analyze.py)

Train - 80%
Test - 20%

Best parameters were chosen with GridSearchCV.

### Algorithms

Random Forest
accuracy - 73.6%

