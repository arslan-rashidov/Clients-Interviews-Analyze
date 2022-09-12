# Clients Interviews Analyze

# Used libraries

1. Pandas
2. Numpy
3. Sklearn

# Data Preprocessing (data_preprocess.py)

Our current dataset consists of:
1. Main Technology of a candidate
2. Seniority level of a candidate
3. English level of a candidate
4. Location ID of a candidate

Features were prepared with OneHotEncoder.

# Analyze (interviews_analyze.py)

Train - 80%
Test - 20%

Best parameters were chosen with GridSearchCV.

# Algorithms

Random Forest
accuracy - 73.6%

Gradient Boosting
accuracy - 75.1% (best for now)

