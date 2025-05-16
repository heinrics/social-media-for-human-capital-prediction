
#-------------------------------------------------------------------------------
################################################################################
# Train main models for Mexico:
# -----------------------------------------------------------
# - Performs 5-fold crossvalidation
# - Trains models for each fold for the following outcomes:
#       - years of schooling
#       - share with postbasic education
#       - share with secondary education
#       - Share with primary education
# - Exports the following csv files for each outcome:
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
# Import modules and set parameters
################################################################################

import pandas as pd
import os
import sys
from sklearn.metrics import r2_score
pd.set_option('display.max_columns', 20)


# Parameters
shapl = True  # Set to False if shapley values should not be computed (takes very long)

if shapl:
    all_dfs = ['r2', 'weights', 'gboost', 'gboost2', 'enet', 'shapley']
else:
    all_dfs = ['r2', 'weights', 'gboost', 'gboost2', 'enet']


################################################################################
# Specify paths to folders
################################################################################

from paths import paths_mx as paths
input_path = paths["train_data"]
exhibits_path = paths["exhibits"]
output_path = paths["results"]
module_path = os.getcwd() + paths["modules"]


################################################################################
# Import custom functions
################################################################################

sys.path.insert(1, module_path)
from create_model_tables import get_model_scores  # requires train_models and performance_table
from train_models import make_folds

################################################################################
# Import data
################################################################################

# Import data
# features = pd.read_csv(input_path + r"\features.csv", dtype={"GEOID":str},
#                        index_col="GEOID", sep=";")
features = pd.read_csv(input_path + r"\weeks\features_week09.csv", dtype={"GEOID": str},
                       index_col="GEOID", sep=";")
labels = pd.read_csv(input_path + r"\labels.csv", dtype={"GEOID": str},
                     index_col="GEOID", sep=";")

# Import feature names (for plots)
flabels = pd.read_excel(module_path + r"\feature_labels.xlsx")
label_dict = dict(zip(flabels.col_name, flabels.label))
fnames = [label_dict[x] for x in features.columns]


################################################################################
# Generate crossvalidation folds
################################################################################

random_state = 554
folds_dict = make_folds(features, labels, random_state=random_state)

################################################################################
# Train models and export results
################################################################################

# Main outcome: years of schooling
#------------------------------------------------------------------------------

# Train models
outcome = "edu_years_schooling"
outcome_name = "yschooling"
yschooling = get_model_scores(folds_dict, outcome, shapl=shapl)

# Print model scores
yschooling["scores_table"]
yschooling["scores_table"].to_csv(output_path + r"\hyperparams\yschooling.csv")

# Export results
for df_name in all_dfs:
    yschooling[df_name].to_csv(output_path + fr"\{outcome_name}_{df_name}.csv",
                               encoding="UTF8", sep=";")
pred_df = pd.DataFrame(yschooling["predictions"]).merge(labels[outcome],
                                                        left_index=True,
                                                        right_index=True)
pred_df.columns = ["y_hat", "fold", "y"]
pred_df = pred_df[["y", "y_hat", "fold"]]
pred_df.to_csv(output_path + fr"\{outcome_name}_predictions.csv", encoding="UTF8", sep=";")


# Secondary outcome 1: % with post-basic degree
#------------------------------------------------------------------------------

# Train models
outcome = "edu_postbasic"
outcome_name = "pbasic"
pbasic = get_model_scores(folds_dict, outcome, shapl=shapl)

# Print model scores
pbasic["scores_table"]
pbasic["scores_table"].to_csv(output_path + r"\hyperparams\pbasic.csv")

# Export results
for df_name in all_dfs:
    pbasic[df_name].to_csv(output_path + fr"\{outcome_name}_{df_name}.csv", encoding="UTF8", sep=";")
pred_df = pd.DataFrame(pbasic["predictions"]).merge(labels[outcome], left_index=True, right_index=True)
pred_df.columns = ["y_hat", "fold", "y"]
pred_df = pred_df[["y", "y_hat", "fold"]]
pred_df.to_csv(output_path + fr"\{outcome_name}_predictions.csv", encoding="UTF8", sep=";")


# Secondary outcome 2: % with secondary degree
#------------------------------------------------------------------------------

# Train models
outcome = "edu_secondary"
outcome_name = "secondary"
secondary = get_model_scores(folds_dict, outcome, shapl=shapl)

# Print model scores
secondary["scores_table"]
secondary["scores_table"].to_csv(output_path + r"\hyperparams\secondary.csv")

# Export results
for df_name in all_dfs:
    secondary[df_name].to_csv(output_path + fr"\{outcome_name}_{df_name}.csv",
                              encoding="UTF8",
                              sep=";")

pred_df = pd.DataFrame(secondary["predictions"]).merge(labels[outcome],
                                                       left_index=True,
                                                       right_index=True)
pred_df.columns = ["y_hat", "fold", "y"]
pred_df = pred_df[["y", "y_hat", "fold"]]
pred_df.to_csv(output_path + fr"\{outcome_name}_predictions.csv",
               encoding="UTF8",
               sep=";")


# Secondary outcome 3: % with primary degree
#------------------------------------------------------------------------------

# Train models
outcome = "edu_primary"
outcome_name = "primary"
primary = get_model_scores(folds_dict, outcome, shapl=shapl)

# Print model scores
primary["scores_table"]
primary["scores_table"].to_csv(output_path + r"\hyperparams\primary.csv")

# Export results
for df_name in all_dfs:
    primary[df_name].to_csv(output_path + fr"\{outcome_name}_{df_name}.csv",
                            encoding="UTF8",
                            sep=";")
pred_df = pd.DataFrame(primary["predictions"]).merge(labels[outcome],
                                                     left_index=True,
                                                     right_index=True)
pred_df.columns = ["y_hat", "fold", "y"]
pred_df = pred_df[["y", "y_hat", "fold"]]
pred_df.to_csv(output_path + fr"\{outcome_name}_predictions.csv",
               encoding="UTF8",
               sep=";")


################################################################################
# Train models for subgroups of variables
################################################################################

# Define column groups to inspect
col_dict = {}
col_dict["pop_cols"] = ["population", "log_pop_density"]
col_dict["count_cols"] = ["nr_tweets", "log_tweet_density", "nr_users", "log_user_density"]
col_dict["tweet_cols"]= list(features.loc[:,"share_weekdays":"log_emoji_per_tweet"].columns)
col_dict["error_cols"] = list(features.loc[:,"log_error_per_char":"log_char_TYPOS"].columns)
col_dict["topic_cols"] = list(features.loc[:,"log_arts_&_culture":"log_youth_&_student_life"].columns)
col_dict["sentiment_cols"] = list(features.loc[:,"negative":"log_offensive"].columns)
col_dict["network_cols"] = list(features.loc[:,"log_ref_in_deg":"log_ref_pagerank"].columns)
col_groups = list(col_dict.values())
for col_group in col_groups:
    cluster_cols = ["cluster_" + col for col in col_group]
    for col in cluster_cols:
        col_group.append(col)

col_dict["mun_cols"] = [col for col in features.columns if not col.startswith('cluster')]
col_dict["cluster_cols"] = [col for col in features.columns if col.startswith('cluster')]


# Train models for all variable groups
subset_results = {}
for group_name, group_vars in col_dict.items():
    group_folds_dict = folds_dict.copy()
    for fold in range(5):
        group_folds_dict[f"X_train{fold}"] = group_folds_dict[f"X_train{fold}"][group_vars]
        group_folds_dict[f"X_valid{fold}"] = group_folds_dict[f"X_valid{fold}"][group_vars]
    result = get_model_scores(group_folds_dict, "edu_years_schooling")
    subset_results[group_name] = result

# Get r2 for each group and fold
stack_r2 = {}
for group_name in col_dict:
    stack_r2[group_name] = list(subset_results[group_name]["r2"].iloc[-1, :])
vargroups_r2 = pd.DataFrame(stack_r2)
vargroups_r2.index = [f"fold_{x}" for x in vargroups_r2.index]
vargroups_r2 = vargroups_r2.transpose()

# Add overall r2 (computed with predictions)
r2_full = {}
for group_name in col_dict:
    pred_df = subset_results[f"{group_name}"]["predictions"]
    pred_df = pd.DataFrame(pred_df).merge(labels["edu_years_schooling"],
                                          left_index=True,
                                          right_index=True)
    pred_df.columns = ["y_hat", "fold", "y"]
    pred_df = pred_df[["y", "y_hat", "fold"]]
    full_score = r2_score(pred_df["y"], pred_df["y_hat"])
    r2_full[f"{group_name}"] = full_score

r2_add = pd.DataFrame(pd.Series(r2_full), columns=["full"])

full_r2_df = vargroups_r2.merge(r2_add,
                                left_index=True,
                                right_index=True)

# Export final dataset
full_r2_df.to_csv(output_path + fr"\yschooling_vargroups_r2.csv",
                  encoding="UTF8", sep=";")