# Import modules
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

def make_folds(features, labels, random_state=554):
    cv_folds = list(KFold(n_splits=5,
                          random_state=random_state,
                          shuffle=True).split(features))
    folds_dict = {}
    for i in range(5):
        folds_dict[f"X_train{i}"] = features.iloc[cv_folds[i][0]]
        folds_dict[f"X_valid{i}"] = features.iloc[cv_folds[i][1]]
        folds_dict[f"Y_train{i}"] = labels.iloc[cv_folds[i][0]]
        folds_dict[f"Y_valid{i}"] = labels.iloc[cv_folds[i][1]]
    return(folds_dict)


def train_models(X_train, y_train):
    seed = 31415
    np.random.seed(seed)

    # Create empty dict and list to fill
    best_models = {}
    cv_results = []

    # Use standard scaler for all models
    scaler = StandardScaler()

    # Elastic net regression
    model = ElasticNet(max_iter=20000)
    pipe = Pipeline(steps=[("scaler", scaler),
                           ("model", model)])
    params = {'model__alpha': [0.001, 0.01, 0.1, 0.5],
              'model__l1_ratio': [0, 0.01, 0.1, 0.5, 1]}
    search = GridSearchCV(pipe, params, scoring='r2', n_jobs=-2)
    result = search.fit(X_train, y_train)
    cv_results.append([[model, x, y] for x, y in zip(result.cv_results_['params'],
                                                      result.cv_results_['mean_test_score'])])
    best_models["enet"] = result
    enet = result.best_estimator_
    print(f"Elastic net trained with CV r2: {(result.best_score_).round(4)}")

    # Gradient boosting regression
    model = GradientBoostingRegressor(n_estimators=200,
                                      loss="huber",
                                      random_state=seed)
    pipe = Pipeline(steps=[("scaler", scaler),
                           ("model", model)])
    params = {'model__max_depth': [3, 4]}
    search = GridSearchCV(pipe, params, scoring='r2', n_jobs=-2)
    result = search.fit(X_train, y_train)
    cv_results.append([[model, x, y] for x, y in zip(result.cv_results_['params'],
                                                      result.cv_results_['mean_test_score'])])
    best_models["gb"] = result
    gb = result.best_estimator_
    print(f"Gradient boosting regressor trained with CV r2: {(result.best_score_).round(4)}")

    # SVM regression
    model = SVR()
    pipe = Pipeline(steps=[("scaler", scaler),
                           ("model", model)])
    params = {'model__C': [1, 5, 10, 20, 30],
              'model__epsilon': [0.1, 0.2]}
    search = GridSearchCV(pipe, params, scoring='r2', n_jobs=-2)
    result = search.fit(X_train, y_train)
    cv_results.append([[model, x, y] for x, y in zip(result.cv_results_['params'],
                                                      result.cv_results_['mean_test_score'])])
    best_models["svr"] = result
    svr = result.best_estimator_
    print(f"SVR trained with CV r2: {(result.best_score_).round(4)}")

    # K nearest neighbor regressor
    model = KNeighborsRegressor()
    pipe = Pipeline(steps=[("scaler", scaler),
                           ("model", model)])
    params = {'model__n_neighbors': [3, 5, 7, 10, 15, 20],
              'model__weights': ["uniform", "distance"]}
    search = GridSearchCV(pipe, params, scoring='r2', n_jobs=-2)
    result = search.fit(X_train, y_train)
    cv_results.append([[model, x, y] for x, y in zip(result.cv_results_['params'],
                                                      result.cv_results_['mean_test_score'])])
    best_models["neigh"] = result
    neigh = result.best_estimator_
    print(f"Knearest neighbor trained with CV r2: {(result.best_score_).round(4)}")

    # Neural network
    model = MLPRegressor(max_iter=1000, random_state=seed)
    pipe = Pipeline(steps=[("scaler", scaler), ("model", model)])

    params = {'model__hidden_layer_sizes': [(100, 100,),
                                            (200, 200,),
                                            (200, 200, 200,)],
              'model__alpha': [0.0001, 0.001, 0.01, 1],
              'model__early_stopping': [True, False]}

    search = GridSearchCV(pipe, params, scoring='r2', n_jobs=-2)
    result = search.fit(X_train, y_train)
    cv_results.append([[model, x, y] for x, y in zip(result.cv_results_['params'],
                                                      result.cv_results_['mean_test_score'])])
    best_models["mlp"] = result
    mlp = result.best_estimator_
    print(f"MLP trained with CV r2: {(result.best_score_).round(4)}")

    # Stacked model
    estimators = [("enet", enet),
                  ("gb", gb),
                  ("svr", svr),
                  ("neigh", neigh),
                  ("mlp", mlp)]

    stack_model = StackingRegressor(estimators=estimators, n_jobs=-2)

    result = stack_model.fit(X_train, y_train)
    best_models["stack"] = result
    print(f"Stacked model trained.")

    # Convert performance list to dataframe
    cv_results = [item for sublist in cv_results for item in sublist]
    cv_results = pd.DataFrame(cv_results, columns=["Model", "Parameters", "Score"])

    # Return results for all models
    return best_models, cv_results
