
#-------------------------------------------------------------------------------
################################################################################
# Prepare features per week for the US
# -----------------------------------------------------------
# - Imports indicator data for each time period
# - Creates cluster level indicators
# - Takes log
# - Exports features for each time period
################################################################################
#-------------------------------------------------------------------------------



################################################################################
# Import modules
################################################################################

import pandas as pd
import os
import geopandas
import sys
pd.set_option('display.max_columns', 20)

################################################################################
# Specify paths to folders
################################################################################


from paths import paths_us as paths
input_path = paths["processed_data"]
exhibits_path = paths["exhibits"]
geo_path = paths["geo_data"]
train_path = paths["train_data"]
output_path = paths["results"]
module_path = os.getcwd() + paths["modules"]


################################################################################
# Import functions from custom modules
################################################################################

sys.path.insert(1, module_path)
from features_full import prepare_features


################################################################################
# Import geodata
################################################################################

geo = geopandas.read_file(geo_path + r"\cb_2018_us_county_500k.shp")
geo = geo.rename(columns={"ADM2_PCODE": "GEOID"})
geo = geo[["GEOID", "geometry"]]
len(geo)

################################################################################
# Create feature and label data for each week
################################################################################


for week in range(1, 10):
    print(f"------ WEEK {week} ------")
    # Import data
    i = str(week).zfill(2)
    df = pd.read_csv(input_path + fr"\weeks\week_{i}\county-level-complete-US-{i}.csv",
                     sep=";", dtype={'county_id_user': 'string'})
    df = df.rename(columns={"county_id_user": "GEOID"})

    # Create and save feature dataset for each week
    feature_data = prepare_features(df, geo, merge_var="GEOID")
    feature_data.to_csv(os.path.join(train_path, "weeks", f"features_week{i}.csv"),
                        sep=";", encoding="UTF8")


################################################################################
# Create feature and label data for each day
################################################################################

for day in range(1, 8):
    print(f"------ DAY {day} ------")
    # Import data
    i = str(day).zfill(2)
    df = pd.read_csv(input_path + fr"\days\day_{i}\county-level-complete-US-{i}.csv",
                     sep=";", dtype={"county_id_user": str})
    df = df.rename(columns={"county_id_user": "GEOID"})

    # Create and save feature dataset for each day
    feature_data = prepare_features(df, geo, merge_var="GEOID")
    feature_data.to_csv(os.path.join(train_path, "days", f"features_day{i}.csv"),
                        sep=";", encoding="UTF8")
