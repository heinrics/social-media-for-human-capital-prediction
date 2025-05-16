
import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
module_path = os.getcwd() + r"\_modules"
sys.path.insert(1, module_path)
from train_models import train_models
from visualize_results import performance_table
import shap

def get_model_scores(folds_dict, outcome, nfolds=5, shapl=False):

    all_tables = {}
    model_score_tables = []
    model_scores = []
    model_weights = []
    model_predictions = []
    gboost_importances = []
    gboost_importances2 = []
    enet_importances = []

    if shapl==True:
        shapley_values = []

    for fold in range(nfolds):

        print(f"-----------Training fold {fold} --------------")

        # Define X
        X_train = folds_dict[f"X_train{fold}"]
        X_valid = folds_dict[f"X_valid{fold}"]

        # Define y
        y_train = folds_dict[f"Y_train{fold}"][outcome]
        y_valid = folds_dict[f"Y_valid{fold}"][outcome]

        # Train models
        model_dict, full_cv_results = train_models(X_train, y_train)  # Takes long!

        # Model score table
        score_table = performance_table(model_dict, X_train, y_train, X_valid, y_valid)
        score_table["fold"] = fold
        model_score_tables.append(score_table)

        # R2 per model
        r2_scores = {}
        for model in model_dict:
            r2_scores[model] = r2_score(y_valid, model_dict[model].predict(X_valid))
        model_scores.append(r2_scores)

        # Stacking weight per model
        weights = {x: y for x, y in zip([x[0] for x in model_dict["stack"].estimators],
                                        model_dict["stack"].final_estimator_.coef_)}
        model_weights.append(weights)

        # Predictions
        y_hat = pd.Series(model_dict["stack"].predict(X_valid), index=X_valid.index)
        model_predictions.append(y_hat)

        # Shapley values

        # rather than use the whole training set to estimate expected values, we summarize with
        # a set of weighted kmeans, each weighted by the number of points they represent.
        if shapl == True:

            explainer = shap.explainers.Permutation(model_dict["stack"].predict,
                                                    X_train,
                                                    seed=1)

            shap_values_valid = explainer(X_valid)
            shap_df = pd.DataFrame(shap_values_valid.values,
                                   columns=X_valid.columns,
                                   index=X_valid.index)
            shap_df['base_values'] = shap_values_valid.base_values

            shapley_values.append(shap_df)


        # Table with gboost feature importances: permutation
        permut = permutation_importance(
            model_dict["gb"], X_train, y_train, n_repeats=10, random_state=42, n_jobs=2, scoring="r2")
        gb_imp = pd.DataFrame({"mean": permut.importances_mean,
                               "std": permut.importances_std}, index=X_train.columns)
        gb_imp = gb_imp.melt(ignore_index=False)
        gb_imp.columns = ["stat", f"fold{fold}"]
        gboost_importances.append(gb_imp)

        # Table with gboost feature importances: impurity decrease
        imps = model_dict["gb"].best_estimator_._final_estimator.feature_importances_
        stds = np.std(
            [tree[0].feature_importances_ for tree in model_dict["gb"].best_estimator_._final_estimator.estimators_],
            axis=0)
        gb_imp2 = pd.DataFrame({"mean": imps,
                                "std": stds}, index=X_train.columns)
        gb_imp2 = gb_imp2.melt(ignore_index=False)
        gb_imp2.columns = ["stat", f"fold{fold}"]
        gboost_importances2.append(gb_imp2)

        # Table with enet feature importances
        enet_imp = model_dict["enet"].best_estimator_._final_estimator.coef_
        enet_imp = pd.Series(enet_imp, index=X_train.columns)
        enet_importances.append(enet_imp)

    # Get model score table
    model_df = model_score_tables[0]
    for fold in range(nfolds-1):
        #model_df = model_df.append(model_score_tables[fold+1])
        model_df = pd.concat([model_df, model_score_tables[fold + 1]])
    all_tables["scores_table"] = model_df

    # Get model scores
    scores_df = pd.DataFrame(model_scores).transpose()
    scores_df.columns = [f"fold{col}" for col in scores_df.columns]
    all_tables["r2"] = scores_df

    # Get weights
    weights_df = pd.DataFrame(model_weights).transpose()
    weights_df.columns = [f"fold{col}" for col in weights_df.columns]
    all_tables["weights"] = weights_df

    # Get predictions
    pred_df = pd.DataFrame(model_predictions[0])
    pred_df["fold"] = 0
    for fold in range(nfolds-1):
        fold_df = pd.DataFrame(model_predictions[fold+1])
        fold_df["fold"] = fold+1
        pred_df = pred_df.append(fold_df)
    pred_df.columns = ["y_hat", "fold"]
    all_tables["predictions"] = pred_df

    # Get shapley values
    if shapl == True:
        shapley_values = pd.concat(shapley_values, axis=0)
        all_tables["shapley"] = shapley_values

    # Get gboost feature importances: permutations
    gboost_df = gboost_importances[0].reset_index()
    for fold in range(nfolds-1):
        gboost_df = gboost_df.merge(gboost_importances[fold+1].reset_index(),
                                    on=["stat", "index"])
    gboost_df = gboost_df.set_index("index")
    gboost_df.index.name = None
    all_tables["gboost"] = gboost_df

    # Get gboost feature importances: impurity decrease
    gboost_df2 = gboost_importances2[0].reset_index()
    for fold in range(nfolds-1):
        gboost_df2 = gboost_df2.merge(gboost_importances2[fold+1].reset_index(),
                                      on=["stat", "index"])
    gboost_df2 = gboost_df2.set_index("index")
    gboost_df2.index.name = None
    all_tables["gboost2"] = gboost_df2

    # Get enet feature importances
    enet_df = pd.DataFrame(enet_importances).transpose()
    enet_df.columns = [f"fold{col}" for col in enet_df]
    all_tables["enet"] = enet_df

    return all_tables