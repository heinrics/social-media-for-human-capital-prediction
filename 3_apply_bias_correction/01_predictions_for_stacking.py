
#-------------------------------------------------------------------------------
################################################################################
# Generate train and test set predictions for each fold for stacking
################################################################################
#-------------------------------------------------------------------------------



################################################################################
# Import modules
################################################################################

# Import modules
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', 20)

################################################################################
# Define functions
################################################################################

# Generate crossvalidation folds
def make_folds(features, labels, n=5):
    cv_folds = list(KFold(n_splits=5, random_state=554, shuffle=True).split(features))
    folds_dict = {}
    for i in range(n):
        folds_dict[f"X_train{i}"] = features.iloc[cv_folds[i][0]]
        folds_dict[f"X_valid{i}"] = features.iloc[cv_folds[i][1]]
        folds_dict[f"Y_train{i}"] = labels.iloc[cv_folds[i][0]]
        folds_dict[f"Y_valid{i}"] = labels.iloc[cv_folds[i][1]]
    return folds_dict


# Make model predictions on train and test set
def make_preds(X_train, y_train, X_test, y_test):

    test_preds = {}

    # Elastic net regression
    scaler = StandardScaler()
    model = ElasticNet(max_iter=20000)
    pipe = Pipeline(steps=[("scaler", scaler),
                           ("model", model)])
    params = {'model__alpha': [0.001, 0.01, 0.1, 0.5],
              'model__l1_ratio': [0, 0.01, 0.1, 0.5, 1]}
    search = GridSearchCV(pipe, params, scoring='r2', n_jobs=-2)
    enet = search.fit(X_train, y_train).best_estimator_
    test_preds["enet"] = enet.predict(X_test)

    # Gradient boosting regression
    model = GradientBoostingRegressor(n_estimators=200,
                                      loss="huber",
                                      random_state=525)
    pipe = Pipeline(steps=[("scaler", scaler),
                           ("model", model)])
    params = {'model__max_depth': [3, 4]}
    search = GridSearchCV(pipe, params, scoring='r2', n_jobs=-2)
    gb = search.fit(X_train, y_train).best_estimator_
    test_preds["gb"] = gb.predict(X_test)

    # Support vector regression
    model = SVR()
    pipe = Pipeline(steps=[("scaler", scaler),
                           ("model", model)])
    params = {'model__C': [1, 5, 10, 20, 30],
              'model__epsilon': [0.1, 0.2]}
    search = GridSearchCV(pipe, params, scoring='r2', n_jobs=-2)
    svm = search.fit(X_train, y_train).best_estimator_
    test_preds["svm"] = svm.predict(X_test)


    # Nearest neighbor
    model = KNeighborsRegressor()
    pipe = Pipeline(steps=[("scaler", scaler),
                           ("model", model)])
    params = {'model__n_neighbors': [3, 5, 7, 10, 15, 20],
              'model__weights': ["uniform", "distance"]}
    search = GridSearchCV(pipe, params, scoring='r2', n_jobs=-2)
    knn = search.fit(X_train, y_train).best_estimator_
    test_preds["knn"] = knn.predict(X_test)


    # MLP
    model = MLPRegressor(max_iter=1000, random_state=545)
    pipe = Pipeline(steps=[("scaler", scaler), ("model", model)])

    params = {'model__hidden_layer_sizes': [(100, 100,),
                                            (200, 200,),
                                            (200, 200, 200,)],
              'model__alpha': [0.0001, 0.001, 0.01, 1],
              'model__early_stopping': [True, False]}

    search = GridSearchCV(pipe, params, scoring='r2', n_jobs=-2)
    mlp = search.fit(X_train, y_train).best_estimator_
    test_preds["mlp"] = mlp.predict(X_test)


    # Combine test set predictions
    test_preds = pd.concat([y_test.reset_index(), pd.DataFrame(test_preds)], axis=1)


    # Create (out-of sample) predictions for training set
    # ------------------------------------------------------------------------

    # Create additional subfolds (within fold) for stacking
    folds_stacking = make_folds(X_train, y_train)

    # Make out of sample predictions for each subfold
    stack_preds = {}
    for stack_fold in range(5):

        # Define X
        X_train_stack = folds_stacking[f"X_train{stack_fold}"]
        X_valid_stack = folds_stacking[f"X_valid{stack_fold}"]

        # Define y
        y_train_stack = folds_stacking[f"Y_train{stack_fold}"]
        y_valid_stack = folds_stacking[f"Y_valid{stack_fold}"]

        # Retrain models and make predictions on validations set
        model_preds = {}
        model_preds["enet"] = enet.fit(X_train_stack, y_train_stack).predict(X_valid_stack)
        model_preds["gb"] = gb.fit(X_train_stack, y_train_stack).predict(X_valid_stack)
        model_preds["svm"] = svm.fit(X_train_stack, y_train_stack).predict(X_valid_stack)
        model_preds["knn"] = knn.fit(X_train_stack, y_train_stack).predict(X_valid_stack)
        model_preds["mlp"] = mlp.fit(X_train_stack, y_train_stack).predict(X_valid_stack)

        # Combine into dataframe
        df = pd.concat([y_valid_stack.reset_index(), pd.DataFrame(model_preds)], axis=1)

        stack_preds[f"fold{stack_fold}"] = df

        print(f"Subfold {stack_fold} trained.")

    # Return training and test predictions
    # ------------------------------------------------------------------------

    # Combine and training set predictions
    train_preds = pd.concat(stack_preds.values(), axis=0)

    return train_preds, test_preds


################################################################################
# Make predictions for MX
################################################################################

# Set paths for MX
from paths import paths_mx as paths
input_path = paths["train_data"]
output_path = paths["results"] + r"\stacking_preds"

# Import data
features = pd.read_csv(input_path + r"\weeks\features_week09.csv",
                       dtype={"GEOID": str},
                       index_col="GEOID",
                       sep=";")
labels = pd.read_csv(input_path + r"\labels.csv",
                     dtype={"GEOID": str},
                     index_col="GEOID",
                     sep=";")
os.chdir(output_path)

# Make folds
folds_dict = make_folds(features, labels)

# Iterate through outcomes
for outcome, outcome_name in zip(labels.columns,
                                 ["yschooling", "pbasic", "secondary", "primary"]):
    print(f"--------------  {outcome_name}  ------------------")

    # Iterate through folds
    for fold in range(5):

        # Define train and test set
        X_train = folds_dict[f"X_train{fold}"]
        X_test = folds_dict[f"X_valid{fold}"]
        y_train = folds_dict[f"Y_train{fold}"][outcome]
        y_test = folds_dict[f"Y_valid{fold}"][outcome]

        # Make and export predictions for fold
        train_preds, test_preds = make_preds(X_train, y_train, X_test, y_test)
        train_preds.to_csv(f"{outcome_name}_train{fold}.csv")
        test_preds.to_csv(f"{outcome_name}_test{fold}.csv")
        print(f"Fold {fold} trained.")


################################################################################
# Make predictions for the US
################################################################################

# Set paths for MX
from paths import paths_us as paths

input_path = paths["train_data"]
output_path = paths["results"] + r"\stacking_preds"

# Import data
features = pd.read_csv(input_path + r"\weeks\features_week09.csv",
                       dtype={"GEOID": str},
                       index_col="GEOID",
                       sep=";")
labels = pd.read_csv(input_path + r"\labels.csv",
                     dtype={"GEOID": str},
                     index_col="GEOID",
                     sep=";")
os.chdir(output_path)

# Make folds
folds_dict = make_folds(features, labels)

# Iterate through outcomes
for outcome, outcome_name in zip(labels.columns,
                                 ["yschooling", "bachelor", "some_college", "high_school"]):
    print(f"--------------  {outcome_name}  ------------------")

    # Iterate through folds
    for fold in range(5):
        # Define train and test set
        X_train = folds_dict[f"X_train{fold}"]
        X_test = folds_dict[f"X_valid{fold}"]
        y_train = folds_dict[f"Y_train{fold}"][outcome]
        y_test = folds_dict[f"Y_valid{fold}"][outcome]

        # Make and export predictions for fold
        train_preds, test_preds = make_preds(X_train, y_train, X_test, y_test)
        train_preds.to_csv(f"{outcome_name}_train{fold}.csv")
        test_preds.to_csv(f"{outcome_name}_test{fold}.csv")
        print(f"Fold {fold} trained.")
