
################################################################################
# Import modules
################################################################################

import pandas as pd
import os
import sys
from sklearn.metrics import r2_score
pd.set_option('display.max_columns', 20)

################################################################################
# Specify paths to folders
################################################################################

from paths import paths_mx as paths
input_path = paths["train_data"]
exhibits_path = paths["exhibits"]
output_path = paths["results"] + r"\validity"
module_path = os.getcwd() + paths["modules"]
labels_path = paths["econ_data"]


################################################################################
# Import custom functions
################################################################################

sys.path.insert(1, module_path)
from create_model_tables import get_model_scores  # requires train_models and performance_table
from train_models import make_folds

################################################################################
# Import data and set parameters
################################################################################

# Import data
features = pd.read_csv(input_path + r"\weeks\features_week09.csv", dtype={"GEOID":str},
                       index_col="GEOID", sep=";")

labels = pd.read_csv(labels_path + r"\wealth_idx.csv", dtype={"GEOID": str},
                     index_col="GEOID", sep=";")

features = features.merge(labels, how="inner",
                          left_index=True,
                          right_index=True).iloc[:, :-1]
labels = features.merge(labels, how="inner",
                        left_index=True,
                        right_index=True).iloc[:, -1]
labels = pd.DataFrame(labels)

outcome = "wealth_idx"

################################################################################
# Train models
################################################################################


# Predict wealth with Twitter features
#------------------------------------------------------------------------------

# Make folds
random_state = 554
folds_dict = make_folds(features, labels, random_state=random_state)

# Train models
wealth = get_model_scores(folds_dict, outcome)

# Export results
for df_name in ['r2', 'weights', 'gboost', 'gboost2', 'enet']:
    wealth[df_name].to_csv(output_path + fr"\wealth0_{df_name}.csv",
                           encoding="UTF8", sep=";")
pred_df = pd.DataFrame(wealth["predictions"]).merge(labels[outcome],
                                                    left_index=True,
                                                    right_index=True)
pred_df.columns = ["y_hat", "fold", "y"]
pred_df = pred_df[["y", "y_hat", "fold"]]
pred_df.to_csv(output_path + fr"\wealth0_predictions.csv", encoding="UTF8", sep=";")

# Inspect results
wealth['r2'].mean(axis=1)
r2_score(pred_df["y"], pred_df["y_hat"])


# Predict wealth just with education (years of schooling)
#------------------------------------------------------------------------------
edu = pd.read_csv(input_path + r"\labels.csv",
                  dtype={"GEOID": str},
                  index_col="GEOID", sep=";")
edu = edu.merge(labels, how="inner",
                left_index=True,
                right_index=True).iloc[:, :-1]
labels = edu.merge(labels, how="inner",
                   left_index=True,
                   right_index=True).iloc[:,-1]
edu = pd.DataFrame(edu["edu_years_schooling"])
labels = pd.DataFrame(labels)

folds_dict = make_folds(edu, labels, random_state=random_state)

# Train models
wealth_edu = get_model_scores(folds_dict, outcome)

# Export results
for df_name in ['r2', 'weights', 'gboost', 'gboost2', 'enet']:
    wealth_edu[df_name].to_csv(output_path + fr"\wealth1_{df_name}_only_edu.csv",
                               encoding="UTF8", sep=";")
pred_df = pd.DataFrame(wealth_edu["predictions"]).merge(labels[outcome],
                                                        left_index=True,
                                                        right_index=True)
pred_df.columns = ["y_hat", "fold", "y"]
pred_df = pred_df[["y", "y_hat", "fold"]]
pred_df.to_csv(output_path + fr"\wealth1_predictions_only_edu.csv", encoding="UTF8", sep=";")

# Inspect results
print(wealth_edu['r2'].mean(axis=1))
print(r2_score(pred_df["y"], pred_df["y_hat"]))


# Predict wealth just with Twitter features and education
#------------------------------------------------------------------------------

edu = pd.merge(features, edu,
               left_index=True,
               right_index=True)

folds_dict = make_folds(edu, labels, random_state=random_state)

# Train models
wealth_both = get_model_scores(folds_dict, outcome)

# Export results
for df_name in ['r2', 'weights', 'gboost', 'gboost2', 'enet']:
    wealth_both[df_name].to_csv(output_path + fr"\wealth2_{df_name}_only_edu.csv",
                                encoding="UTF8", sep=";")
pred_df = pd.DataFrame(wealth_both["predictions"]).merge(labels[outcome],
                                                         left_index=True,
                                                         right_index=True)
pred_df.columns = ["y_hat", "fold", "y"]
pred_df = pred_df[["y", "y_hat", "fold"]]
pred_df.to_csv(output_path + fr"\wealth2_predictions_only_edu.csv", encoding="UTF8", sep=";")

# Inspect results
print(wealth_both['r2'].mean(axis=1))
print(r2_score(pred_df["y"], pred_df["y_hat"]))
