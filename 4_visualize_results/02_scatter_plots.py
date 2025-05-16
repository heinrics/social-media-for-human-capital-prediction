
#-------------------------------------------------------------------------------
################################################################################
# Scatter and bubble plots
# -----------------------------------------------------------
# - Creates and exports a scatter and bubble plot per outcome and country
################################################################################
#-------------------------------------------------------------------------------


################################################################################
# Import modules
################################################################################

import pandas as pd
import os
import sys


pd.set_option('display.max_columns', 20)

################################################################################
# Specify paths to folders
################################################################################

from paths import paths_mx as paths
mx_input_path = paths["results"] + "/"
mx_train_path = paths["train_data"] + "/"
exhibits_path = paths["exhibits"] + r"/figures/analysis/"
module_path = os.getcwd() + paths["modules"]

from paths import paths_us as paths
us_input_path = paths["results"] + "/"
us_train_path = paths["train_data"] + "/"

sys.path.insert(1, module_path)
from visualize_results import scatter
from visualize_results import bubble

# Set parameters
################################################################################

# Colors
mx_color = "#00008B"
us_color = "#8B0000"

# Outcomes
mx_outcomes = ["yschooling",
               "pbasic",
               "secondary",
               "primary"]
us_outcomes = ["yschooling",
               "bachelor",
               "some_college",
               "high_school"]

# Units and labels for outcomes
mx_units = [2, 10, 10, 10]
us_units = [1, 10, 10, 5]

mx_labels = ["Years of schooling",
             "% with post-basic degree",
             "% with secondary degree",
             "% with primary degree"]

us_labels = ["Years of schooling",
             "% with bachelor degree",
             "% with some college",
             "% with high school"]

# Population data
mx_pop = pd.read_csv(mx_train_path + r"features.csv",
                     sep=";",
                     index_col="GEOID",
                     usecols=["GEOID",
                              "population"])

us_pop = pd.read_csv(us_train_path + r"features.csv",
                     sep=";",
                     index_col="GEOID",
                     usecols=["GEOID",
                              "population"])



# #######################################################################################
# Scatter plots
# #######################################################################################


# Mexico
for outcome, unit, label in zip(mx_outcomes, mx_units, mx_labels):
    preds = pd.read_csv(mx_input_path + fr"{outcome}_predictions.csv",
                        sep=";",
                        index_col="GEOID")
    if unit == 10:
        preds = preds * 100
    preds = preds.merge(mx_pop, right_index=True, left_index=True)

    scatter(preds["y"],
            preds["y_hat"],
            label,
            f"Predicted {label.lower()}",
            unit=unit,
            color=mx_color,
            save=exhibits_path + fr"mx_{outcome}_scatter.pdf")

    bubble(preds["y"],
           preds["y_hat"],
           preds["population"]/5000,
           label, f"Predicted {label.lower()}",
           unit=unit,
           color=mx_color,
           save=exhibits_path + fr"mx_{outcome}_bubble.pdf")


# USA
min_outliers = [1, 1, 1, 5]
for outcome, unit, label, nr_outliers, in zip(us_outcomes, us_units, us_labels, min_outliers):
    preds = pd.read_csv(us_input_path + fr"{outcome}_predictions.csv",
                        sep=";",
                        index_col="GEOID")
    preds = preds.merge(us_pop, right_index=True, left_index=True)

    scatter(preds["y"],
            preds["y_hat"],
            label,
            f"Predicted {label.lower()}",
            unit=unit,
            color=us_color,
            outlier_min=nr_outliers,
            save=exhibits_path + fr"us_{outcome}_scatter.pdf")

    bubble(preds["y"],
           preds["y_hat"],
           preds["population"]/5000,
           label, f"Predicted {label.lower()}",
           unit=unit,
           color=us_color,
           outlier_min=nr_outliers,
           save=exhibits_path + fr"us_{outcome}_bubble.pdf")
