#-------------------------------------------------------------------------------
################################################################################
# Train models for permutation test:
# -----------------------------------------------------------
# - Trains model for each permutation
# - Exports file with R2 by permutation
################################################################################
#-------------------------------------------------------------------------------



################################################################################
# Import modules
################################################################################

import pandas as pd
import os
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import time

start = time.time()

pd.set_option('display.max_columns', 20)

################################################################################
# Specify paths to folders
################################################################################

from paths import paths_mx as paths
input_path = paths["train_data"]
output_path = paths["results"]
module_path = os.getcwd() + paths["modules"]

################################################################################
# Import custom functions
################################################################################

sys.path.insert(1, module_path)
from create_model_tables import get_model_scores # requires train_models and performance_table


################################################################################
# Import data
################################################################################
features = pd.read_csv(input_path + r"\weeks\features_week09.csv",
                       dtype={"GEOID": str},
                       index_col="GEOID",
                       sep=";")\
    .reset_index(drop=True)

labels = pd.read_csv(input_path + r"\labels.csv",
                     dtype={"GEOID": str},
                     index_col="GEOID",
                     sep=";").reset_index(drop=True)


################################################################################
# Function to generate crossvalidation folds
################################################################################

def make_folds(features, labels):
    cv_folds = list(KFold(n_splits=5, random_state=554, shuffle=True).split(features))
    folds_dict = {}
    for i in range(5):
        folds_dict[f"X_train{i}"] = features.iloc[cv_folds[i][0]]
        folds_dict[f"X_valid{i}"] = features.iloc[cv_folds[i][1]]
        folds_dict[f"Y_train{i}"] = labels.iloc[cv_folds[i][0]]
        folds_dict[f"Y_valid{i}"] = labels.iloc[cv_folds[i][1]]
    return(folds_dict)


################################################################################
# Estimate models for each permutation
################################################################################


# Set parameters
outcome = "edu_years_schooling"
n_permut = 100


# Estimate models

results = {}
for permut in range(n_permut):

    print("Permutation:", permut)
    print("-----------------------------------------")

    # Shuffle labels
    labels = labels.sample(frac=1,
                           random_state=45)\
        .reset_index(drop=True)

    # Create folds
    folds_dict = make_folds(features, labels)

    # Estimate models
    permut_scores = get_model_scores(folds_dict, outcome)
    results[permut] = permut_scores


# Get r2 scores for each fold and permutation
r2 = {}
for permut in range(n_permut):
    r2[permut] = list(results[permut]["r2"].loc["stack"])

r2_df = pd.DataFrame(r2).transpose()
r2_df.columns = results[permut]["r2"].loc["stack"].index
print(r2_df.mean(axis=1))


# Add overall r2 (computed with predictions)
r2_full = {}

for permut in range(n_permut):
    pred_df = results[permut]["predictions"]
    pred_df = pd.DataFrame(pred_df).merge(labels[outcome],
                                          left_index=True,
                                          right_index=True)
    pred_df.columns = ["y_hat", "fold", "y"]
    pred_df = pred_df[["y", "y_hat", "fold"]]
    full_score = r2_score(pred_df["y"], pred_df["y_hat"])
    r2_full[permut] = full_score

r2_add = pd.DataFrame(pd.Series(r2_full), columns=["full"])

r2_df = r2_df.merge(r2_add,
                    left_index=True,
                    right_index=True)


# Export R2s for permutations
r2_df.to_csv(os.path.join(output_path, "yschooling_by_permut.csv"), sep=";")

