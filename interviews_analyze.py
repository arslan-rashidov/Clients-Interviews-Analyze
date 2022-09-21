import pickle

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split


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
        x = clients_interview_df.drop(['status'], axis=1)

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            train_size=0.8,
                                                            random_state=42)

        # max - 73.6%
        rfc = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=2,
                                     max_features=100, min_samples_leaf=6,
                                     n_jobs=-1, oob_score=True, verbose=True)
        rfc.fit(x_train, y_train)
        print(f"Random Forest Accuracy - {rfc.score(x_test, y_test)}")

        self.save_model(rfc, "random_forest_model")

    # Save Model
    def save_model(self, model, model_name):
        filename = f'{model_name}.sav'
        pickle.dump(model, open(filename, 'wb'))


if __name__ == "__main__":
    clients_interviews_analyze = ClientsInterviewsAnalyzer()
    clients_interviews_analyze.run()
