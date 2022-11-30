import json

import joblib
import pandas as pd
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./data/train.csv")
metrics = ['precision', 'recall', 'f1', 'accuracy']


def save_best_estimator(clf, name):
    best_model_stats = {}

    for metric in metrics:
        best_model_stats[metric] = {}
        rank_key = f"rank_test_{metric}"
        important_index = clf.cv_results_[rank_key].tolist().index(1)

        for key, val in clf.cv_results_.items():
            real_val = val[important_index]
            if 'numpy' in str(type(real_val)):
                if 'float' in str(type(real_val)):
                    best_model_stats[metric][key] = float(real_val)
                else:
                    best_model_stats[metric][key] = int(real_val)
            else:
                best_model_stats[metric][key] = real_val
    joblib.dump(clf.best_estimator_, f'{name}.pkl', compress=1)
    with open(f"result/{name}_stats.json", 'w') as jf:
        json.dump(best_model_stats, jf, indent=4)


x = df.drop(columns="Lead")

y = df['Lead'].replace({"Male": 1, "Female": -1})

parameters = {'model__max_depth': range(3, 20), 'model__min_samples_split': range(2, 15)}
model = Pipeline([('scaler', StandardScaler()), ('model', tree.DecisionTreeClassifier())])

clf = GridSearchCV(model, parameters, n_jobs=4, cv=5, refit="accuracy",
                   scoring=metrics)
clf.fit(X=x, y=y)
tree_model = clf.best_estimator_
save_best_estimator(clf, "decisionTree")
print(clf.best_score_, clf.best_params_)

# 0.8065356744704572 {'model__max_depth': 7, 'model__min_samples_split': 3}


parameters = {'model__n_estimators': range(10, 150, 10), 'model__bootstrap': [True, False],
              'model__min_samples_split': range(2, 15)}

model = Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier())])

clf = GridSearchCV(model, parameters, n_jobs=4, cv=5, refit="accuracy", scoring=metrics)
clf.fit(X=x, y=y)
randomforest_model = clf.best_estimator_
save_best_estimator(clf, "randomForest")
print(clf.best_score_, clf.best_params_)

# 0.8536696395392047 {'model__bootstrap': False, 'model__min_samples_split': 4, 'model__n_estimators': 130}


parameters = {'model__n_estimators': range(5, 150, 5), 'model__bootstrap': [True, False],
              'model__bootstrap_features': [True, False]}
model = Pipeline([('scaler', StandardScaler()), ('model', BaggingClassifier())])

clf = GridSearchCV(model, parameters, n_jobs=4, cv=5, refit="accuracy", scoring=metrics)
clf.fit(X=x, y=y)
save_best_estimator(clf, "bagging")
print(clf.best_score_, clf.best_params_)

# 0.8613851727982164 {'model__bootstrap': True, 'model__bootstrap_features': False, 'model__n_estimators': 105}

clf.cv_results_.keys()
