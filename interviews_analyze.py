import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split


class ClientsInterviewsAnalyze:
    def __init__(self):
        self.clients_interview_df_path = "clients_interviews.csv"

    def run(self):
        self.main()

    def main(self):
        clients_interview_df = pd.read_csv(self.clients_interview_df_path)

        y = np.asarray(clients_interview_df['status'])
        x = clients_interview_df.drop(['status'], axis=1)

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            train_size=0.8,
                                                            random_state=42)
        # max - 73.6%
        rfc = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42,
                                     max_features=100, min_samples_leaf=6,
                                     n_jobs=-1, oob_score=True, verbose=True)
        rfc.fit(x_train, y_train)
        print(f"Random Forest Accuracy - {rfc.score(x_test, y_test)}")


        # max - 75.1%
        clf = GradientBoostingClassifier(n_estimators=70, learning_rate=1.0, max_depth=1, random_state=0,
                                         min_samples_split=10, min_samples_leaf=3, max_features=100, verbose=True)
        clf.fit(x_train, y_train)
        print(f"Gradient Boosting Accuracy - {clf.score(x_test, y_test)}")


if __name__ == "__main__":
    clients_interviews_analyze = ClientsInterviewsAnalyze()
    clients_interviews_analyze.run()
