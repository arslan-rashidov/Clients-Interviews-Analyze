import pickle

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image


class ClientsInterviewsAnalyzer:
    def __init__(self, working_dataset_path):
        self.clients_interview_df_path = working_dataset_path

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
        print(x_train.columns)

        # max - 97.47%
        rfc = RandomForestClassifier(n_estimators=135, max_depth=18, random_state=42,
                                     max_features=39, min_samples_leaf=2, min_samples_split=2,
                                     n_jobs=-1, oob_score=True, verbose=True, criterion='entropy')

        rfc.fit(x_train, y_train)

        print(f"Random Forest Accuracy - {rfc.score(x_test, y_test)}")

        feature_importance = self.get_feature_importance(x.columns, rfc.feature_importances_)
        self.show_feature_importance(feature_importance)

        self.save_model(rfc, "random_forest_model")

        #self.save_tree_png(rfc, x_train.columns, 5)

    def save_model(self, model, model_name):
        filename = f'{model_name}.sav'
        pickle.dump(model, open(filename, 'wb'))

    def show_feature_importance(self, feature_importance):
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        for i in range(len(importances)):
            importances[i] = round(importances[i] * 100, 2)

        plt.figure()

        p1 = plt.bar(features, importances)

        for rect1 in p1:
            height = rect1.get_height()
            plt.annotate("{}%".format(height), (rect1.get_x() + rect1.get_width() / 2, height + .05), ha="center",
                         va="bottom", fontsize=10)
        plt.xticks(rotation=30, ha='right')

        plt.show()

    def get_feature_importance(self, columns, feature_importance_list):
        feature_importance = {}
        for i in range(len(columns)):
            feature_importance[columns[i]] = feature_importance_list[i]
        interview_technology_importance = 0
        for key in list(feature_importance.keys()):
            value = feature_importance[key]
            if 'interview_technology' in key:
                interview_technology_importance += value
                del (feature_importance[key])
        feature_importance['interview_technology'] = interview_technology_importance
        feature_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1])}
        return feature_importance

    def save_tree_png(self, model, feature_names, tree_n):
        estimator = model.estimators_[tree_n]

        # Export as dot file
        export_graphviz(estimator, out_file='tree.dot',
                        feature_names=feature_names,
                        class_names=['failure', 'success'],
                        rounded=True, proportion=False,
                        precision=2, filled=True)

        # Convert to png using system command (requires Graphviz)
        call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

        # Display in jupyter notebook
        Image(filename='tree.png')


if __name__ == "__main__":
    working_dataset_path = 'Data/clients_interviews.csv'

    clients_interviews_analyze = ClientsInterviewsAnalyzer(working_dataset_path=working_dataset_path)
    clients_interviews_analyze.run()
