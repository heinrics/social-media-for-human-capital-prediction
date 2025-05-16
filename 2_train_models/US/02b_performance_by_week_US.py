
#-------------------------------------------------------------------------------
################################################################################
# Train weekly models for the US:
# -----------------------------------------------------------
# - Trains model for each period
# - Exports files with R2 by week and by day
################################################################################
#-------------------------------------------------------------------------------



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

from paths import paths_us as paths
input_path = paths["train_data"]
exhibits_path = paths["exhibits"]
output_path = paths["results"]
module_path = os.getcwd() + paths["modules"]


################################################################################
# Import custom functions
################################################################################

sys.path.insert(1, module_path)
from create_model_tables import get_model_scores # requires train_models and performance_table
from train_models import make_folds


################################################################################
# Train models and export results
################################################################################

################################################################################
# Train models and export results for each week and day
################################################################################

# Set parameters and import labels
results = {}
outcome = "edu_years_schooling"
nr_weeks = 9
nr_days = 7
random_state = 54

labels = pd.read_csv(input_path + r"\labels.csv", dtype={"GEOID": str},
                     index_col="GEOID", sep=";")

dfs = {}
for period, period_length in zip(["week", "day"],
                                 [nr_weeks, nr_days]):

    # Get estimates for population only (week/day 0)
    features = pd.read_csv(input_path + fr"\{period}s\features_{period}01.csv", dtype={"GEOID": str},
                           index_col="GEOID", sep=";")
    features = pd.concat([features.loc[:, ["population", "log_pop_density"]],
                         features.loc[:, ['cluster_population', "cluster_log_pop_density"]]], axis=1)
    assert (all(features.index == labels.index))  # Check if same order as labels
    folds_dict = make_folds(features, labels, random_state=random_state)
    scores0 = get_model_scores(folds_dict, outcome)
    results[f"{period}0"] = scores0


    # Fit models for each week/day
    for i in range(1, period_length + 1):
        print(f"================= {period} {i} =================\n")
        # Import data
        i = str(i).zfill(2)
        features = pd.read_csv(input_path + fr"\{period}s\features_{period}{i}.csv", dtype={"GEOID": str},
                               index_col="GEOID", sep=";")
        assert (all(features.index == labels.index))  # Check if same order as labels
        folds_dict = make_folds(features, labels, random_state=random_state)
        period_scores = get_model_scores(folds_dict, outcome)
        results[f"{period}{int(i)}"] = period_scores


    # Get r2 scores for each fold and week/day
    r2 = {}
    for i in range(period_length + 1):
        r2[f"{period}{i}"] = list(results[f"{period}{i}"]["r2"].loc["stack"])

    r2_df = pd.DataFrame(r2).transpose()
    r2_df.columns = results[f"{period}{i}"]["r2"].loc["stack"].index
    print(r2_df.mean(axis=1))


    # Add overall r2 (computed with predictions)
    r2_full = {}

    for i in range(period_length + 1):
        pred_df = results[f"{period}{i}"]["predictions"]
        pred_df = pd.DataFrame(pred_df).merge(labels[outcome],
                                              left_index=True,
                                              right_index=True)
        pred_df.columns = ["y_hat", "fold", "y"]
        pred_df = pred_df[["y", "y_hat", "fold"]]
        full_score = r2_score(pred_df["y"], pred_df["y_hat"])
        r2_full[f"{period}{i}"] = full_score

    r2_add = pd.DataFrame(pd.Series(r2_full), columns=["full"])

    r2_df = r2_df.merge(r2_add,
                        left_index=True,
                        right_index=True)

    dfs[period] = r2_df

# Export R2 by week
dfs["week"].to_csv(os.path.join(output_path, "yschooling_by_week.csv"), sep=";")
dfs["day"].to_csv(os.path.join(output_path, "yschooling_by_day.csv"), sep=";")