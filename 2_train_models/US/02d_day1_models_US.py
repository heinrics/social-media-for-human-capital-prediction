
#-------------------------------------------------------------------------------
################################################################################
# Train main models for day 1 in the US:
# -----------------------------------------------------------
# - Performs 5-fold crossvalidation
# - Trains models for each fold for the following outcomes:
#       - Years of schooling
#       - Share with bachelor
#       - Share with some college
#       - Share with high school
# - Exports the follwoing csv files for each outcome:
#       - Validation R2 for each fold
#       - Stacking weights for each fold
#       - Predictions on validation set
#       - Feature importances based on gradient boosting for each fold
#       - Feature importances based on elastic net for each fold
#
# - Custom modules used:
#       - train_models: trains enet, gboost, svr, knearest, mlp and stacking regressor
#           including hyperparam tuning
#       - get_model_scores: creates tables with r2, stacking weights, predictions and
#           feature importances for each fold

################################################################################
#-------------------------------------------------------------------------------



################################################################################
# Import modules
################################################################################

import pandas as pd
import os
import sys
from sklearn.model_selection import KFold
pd.set_option('display.max_columns', 20)

################################################################################
# Specify paths to folders
################################################################################

from paths import tpaths_us as paths
input_path = paths["train_data"]
output_path = paths["results"] + r"\day1"
module_path = os.getcwd() + paths["modules"]


################################################################################
# Import custom functions
################################################################################

sys.path.insert(1, module_path)
from create_model_tables import get_model_scores # requires train_models and performance_table

################################################################################
# Import data
################################################################################

# Import data
features = pd.read_csv(input_path + r"\days\features_day01.csv", dtype={"GEOID":str},
                       index_col="GEOID", sep=";")
labels = pd.read_csv(input_path + r"\labels.csv", dtype={"GEOID":str},
                     index_col="GEOID", sep=";")

# Import feature names (for plots)
flabels = pd.read_excel(module_path + r"\feature_labels.xlsx")
label_dict = dict(zip(flabels.col_name, flabels.label))
fnames = [label_dict[x] for x in features.columns]


################################################################################
# Generate crossvalidation folds
################################################################################

cv_folds = list(KFold(n_splits=5, random_state=54, shuffle=True).split(features))
folds_dict = {}
for i in range(5):
    folds_dict[f"X_train{i}"] = features.iloc[cv_folds[i][0]]
    folds_dict[f"X_valid{i}"] = features.iloc[cv_folds[i][1]]
    folds_dict[f"Y_train{i}"] = labels.iloc[cv_folds[i][0]]
    folds_dict[f"Y_valid{i}"] = labels.iloc[cv_folds[i][1]]
folds_dict["X_valid0"]



################################################################################
# Train models and export results
################################################################################

# Main outcome: years of schooling
#------------------------------------------------------------------------------

# Train models
outcome = "edu_years_schooling"
outcome_name = "yschooling"
yschooling = get_model_scores(folds_dict, outcome)

# Print model scores
print(yschooling["scores_table"])
tabl = yschooling["scores_table"].reset_index()
print(f"Crossvalidation R2 (yschooling): "
      f"{tabl.loc[tabl['model'] == 'stack', 'validation_set_performance'].mean()}")

# Export results
for df_name in ['r2', 'weights', 'gboost', 'gboost2', 'enet']:
    yschooling[df_name].to_csv(output_path + fr"\{outcome_name}_{df_name}.csv",
                               encoding="UTF8", sep=";")
pred_df = pd.DataFrame(yschooling["predictions"]).merge(labels[outcome],
                                                        left_index=True,
                                                        right_index=True)
pred_df.columns = ["y_hat", "fold", "y"]
pred_df = pred_df[["y", "y_hat", "fold"]]
pred_df.to_csv(output_path + fr"\{outcome_name}_predictions.csv", encoding="UTF8", sep=";")


# Secondary outcome 1: % with bachelor degree or more
#------------------------------------------------------------------------------

# Train models
outcome = "edu_bachelor"
outcome_name = "bachelor"
bachelor = get_model_scores(folds_dict, outcome)

# Print model scores
bachelor["scores_table"]

# Export results
for df_name in ['r2', 'weights', 'gboost', 'gboost2', 'enet']:
    bachelor[df_name].to_csv(output_path + fr"\{outcome_name}_{df_name}.csv", encoding="UTF8", sep=";")
pred_df = pd.DataFrame(bachelor["predictions"]).merge(labels[outcome], left_index=True, right_index=True)
pred_df.columns = ["y_hat", "fold", "y"]
pred_df = pred_df[["y", "y_hat", "fold"]]
pred_df.to_csv(output_path + fr"\{outcome_name}_predictions.csv", encoding="UTF8", sep=";")


# Secondary outcome 2: % with some college or more
#------------------------------------------------------------------------------

# Train models
outcome = "edu_some_college"
outcome_name = "some_college"
college = get_model_scores(folds_dict, outcome)

# Print model scores
print(college["scores_table"].loc["stack"])

# Export results
for df_name in ['r2', 'weights', 'gboost', 'gboost2', 'enet']:
    college[df_name].to_csv(output_path + fr"\{outcome_name}_{df_name}.csv", encoding="UTF8", sep=";")
pred_df = pd.DataFrame(college["predictions"]).merge(labels[outcome], left_index=True, right_index=True)
pred_df.columns = ["y_hat", "fold", "y"]
pred_df = pred_df[["y", "y_hat", "fold"]]
pred_df.to_csv(output_path + fr"\{outcome_name}_predictions.csv", encoding="UTF8", sep=";")

# Secondary outcome 3: % with high school or more
#------------------------------------------------------------------------------

# Train models
outcome = "edu_high"
outcome_name = "high_school"
high = get_model_scores(folds_dict, outcome)

# Print model scores
print(high["scores_table"].loc["stack"])

# Export results
for df_name in ['r2', 'weights', 'gboost', 'gboost2', 'enet']:
    high[df_name].to_csv(output_path + fr"\{outcome_name}_{df_name}.csv", encoding="UTF8", sep=";")
pred_df = pd.DataFrame(high["predictions"]).merge(labels[outcome], left_index=True, right_index=True)
pred_df.columns = ["y_hat", "fold", "y"]
pred_df = pred_df[["y", "y_hat", "fold"]]
pred_df.to_csv(output_path + fr"\{outcome_name}_predictions.csv", encoding="UTF8", sep=";")



################################################################################
# Train models for subgroups of variables
################################################################################

# TODO: Which variable should we take as main outcome?

# Define column groups to inspect
col_dict = {}
col_dict["pop_cols"] = ["population", "log_pop_density"]
col_dict["count_cols"] = ["nr_tweets", "log_tweet_density", "nr_users", "log_user_density"]
col_dict["tweet_cols"] = list(features.loc[:,"share_weekdays":"log_emoji_per_tweet"].columns)
col_dict["error_cols"] = list(features.loc[:,"log_error_per_char":"log_char_TYPOS"].columns)
col_dict["topic_cols"] = list(features.loc[:,"log_arts_&_culture":"log_youth_&_student_life"].columns)
col_dict["sentiment_cols"] = list(features.loc[:,"negative":"log_offensive"].columns)
col_dict["network_cols"] = list(features.loc[:,"log_ref_in_deg":"log_ref_pagerank"].columns)
col_groups = list(col_dict.values())
for col_group in col_groups:
    cluster_cols = ["cluster_"+col for col in col_group]
    for col in cluster_cols:
        col_group.append(col)

# Train models for all variable groups
subset_results = {}
for group_name, group_vars in col_dict.items():
    group_folds_dict = folds_dict.copy()
    for fold in range(5):
        group_folds_dict[f"X_train{fold}"] = group_folds_dict[f"X_train{fold}"][group_vars]
        group_folds_dict[f"X_valid{fold}"] = group_folds_dict[f"X_valid{fold}"][group_vars]
    result = get_model_scores(group_folds_dict, "edu_years_schooling")
    subset_results[group_name] = result


# Export r2 for each group and fold
stack_r2 = {}
for group_name in col_dict:
    stack_r2[group_name] = list(subset_results[group_name]["r2"].iloc[-1, :])
vargroups_r2 = pd.DataFrame(stack_r2)
vargroups_r2.index = [f"fold_{x}" for x in vargroups_r2.index]
vargroups_r2 = vargroups_r2.transpose()
vargroups_r2.to_csv(output_path + fr"\yschooling_vargroups_r2.csv", encoding="UTF8", sep=";")


# Train models for all variable groups
subset_results = {}
for group_name, group_vars in col_dict.items():
    group_folds_dict = folds_dict.copy()
    for fold in range(5):
        group_folds_dict[f"X_train{fold}"] = group_folds_dict[f"X_train{fold}"][group_vars]
        group_folds_dict[f"X_valid{fold}"] = group_folds_dict[f"X_valid{fold}"][group_vars]
    result = get_model_scores(group_folds_dict, "edu_bachelor")
    subset_results[group_name] = result


# Export r2 for each group and fold
stack_r2 = {}
for group_name in col_dict:
    stack_r2[group_name] = list(subset_results[group_name]["r2"].iloc[-1,:])
vargroups_r2 = pd.DataFrame(stack_r2)
vargroups_r2.index = [f"fold_{x}" for x in vargroups_r2.index]
vargroups_r2 = vargroups_r2.transpose()
vargroups_r2.to_csv(output_path + fr"\bachelor_vargroups_r2.csv", encoding="UTF8", sep=";")