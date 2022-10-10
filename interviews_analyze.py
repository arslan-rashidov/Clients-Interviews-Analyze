import pickle

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from matplotlib import pyplot as plt


class ClientsInterviewsAnalyzer:
    def __init__(self):
        self.clients_interview_df_path = "clients_interviews.csv"

    # Run Method
    def run(self):
        self.main()

    # Train Model
    def main(self):
        clients_interview_df = pd.read_csv(self.clients_interview_df_path)

        y = np.asarray(clients_interview_df['status'])
        x = clients_interview_df.drop(['status', 'request_for_specialist_id', 'candidate_id', 'project_name'], axis=1)

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            train_size=0.8,
                                                            random_state=42)

        # max - 85.2%
        #rfc = RandomForestClassifier(n_estimators=74, max_depth=10, random_state=42,
        #                             max_features=16, min_samples_leaf=4,
        #                             n_jobs=-1, oob_score=True, verbose=True, criterion='gini')

        rfc = RandomForestClassifier(n_estimators=300, max_depth=30, random_state=42,
                                     max_features=17, min_samples_leaf=2,
                                     n_jobs=-1, oob_score=True, verbose=True, criterion='entropy')

        rfc.fit(x_train, y_train)

        print(x_train.iloc[0])
        print(rfc.predict_proba(np.array(x_train.iloc[0]).reshape(1, -1)))

        print(f"Random Forest Accuracy - {rfc.score(x_test, y_test)}")

        #plt.barh(x.columns, rfc.feature_importances_)
        #plt.show()

        self.save_model(rfc, "random_forest_model")

    # Save Model
    def save_model(self, model, model_name):
        filename = f'{model_name}.sav'
        pickle.dump(model, open(filename, 'wb'))


if __name__ == "__main__":
    clients_interviews_analyze = ClientsInterviewsAnalyzer()
    clients_interviews_analyze.run()
