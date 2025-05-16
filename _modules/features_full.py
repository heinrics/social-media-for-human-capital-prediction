
#-------------------------------------------------------------------------------
################################################################################
# Prepare features for prediction models at MX municipality levels:
# -----------------------------------------------------------
# - Merges municipality-level features with education and geodata
# - Computes weighted cluster averages for each municipality cluster
# - Inspects distributions of variables and takes log of right-skewed variables
# - Imputes values for municipality with 0 tweets based on cluster averages
# - Handles outliers
# - Exports features and labels
################################################################################
#-------------------------------------------------------------------------------


################################################################################
# Import modules
################################################################################

import sys
import pandas as pd
import numpy as np
import os
import geopandas
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

# Custom modules
module_path = os.getcwd() + r"\_modules"
sys.path.insert(1, module_path)
from prepare_features import move_col
from prepare_features import log_with0
from prepare_features import is_outlier
from prepare_features import impute_na


################################################################################
# Import and merge data
################################################################################


def prepare_features(indicator_df, geo_df, merge_var="GEOID"):
    """
    :param indicator_df: county level df with all indicators + ID
    :param geo_df: county level geo dataframe containing only geometry + ID
    :return: feature_data
    """
    df = indicator_df.merge(geo_df, on=merge_var)
    df = geopandas.GeoDataFrame(df)
    df = df.to_crs("EPSG:5071")

    len_df = len(df.columns)
    df = df.dropna(axis=1, how="all") # Drop columns that are all NA (only one case)
    print(len_df-len(df.columns), "columns with all NA dropped.")

    ################################################################################
    # Add densities and define groups of columns
    ################################################################################


    # Compute population, tweet and user density
    df["pop_density"] = df["population"]/(df.area / (1000 * 1000))
    df["tweet_density"] = df["nr_tweets"]/df["population"] * 1000
    df["user_density"] = (df["nr_users"])/df["population"] * 1000

    # Define groups of columns
    id_cols = ['GEOID']
    geo_cols = ['geometry']
    pop_cols = ['population', 'pop_density']
    count_cols = ['nr_tweets', 'tweet_density', 'nr_users', 'user_density']
    tweet_cols = list(df.loc[:,'share_weekdays':'emoji_per_tweet'].columns)
    topic_cols = list(df.loc[:,'arts_&_culture':'youth_&_student_life'].columns)
    sentiment_cols = list(df.loc[:,'negative':'offensive'].columns)
    network_cols = list(df.loc[:,'ref_in_deg':'ref_pagerank'].columns)

    error_cols = [x for x in df.columns if x not in
                  (id_cols+geo_cols+pop_cols+count_cols+topic_cols+tweet_cols+sentiment_cols+network_cols)]

    X_cols = pop_cols+count_cols+tweet_cols+topic_cols+error_cols+sentiment_cols+network_cols

    # Order dataframe
    df = df[list(id_cols+X_cols+geo_cols)]
    df.columns

    ################################################################################
    # Create cluster level dataframe
    ################################################################################

    # Make neighbor identification dataframe
    # ------------------------------------------------------------------------------
    neighbors = df[["GEOID", "geometry"]].sjoin(df[["GEOID", "geometry"]].rename(columns={"GEOID": "N_GEOID"}),
                                                predicate='intersects') # Spatial join with itself

    neighbors = neighbors[["GEOID", "N_GEOID"]].sort_values(by="GEOID")

    neighbors["GEOID"].value_counts().value_counts().sort_index()

    # Get cluster-level values
    # ------------------------------------------------------------------------------
    df_clusters = neighbors.merge(df.rename(columns={"GEOID": "N_GEOID_merge"}),
                                  left_on="N_GEOID", right_on="N_GEOID_merge")
    df_clusters = df_clusters.drop(columns=['N_GEOID', 'N_GEOID_merge', 'geometry'])
    df_clusters.columns

    # Function to aggregate values at cluster level
    def cluster_values(df_clusters):
        # Get sum for population, tweet count and user count per cluster
        sums = df_clusters[['GEOID', 'population', 'nr_tweets', 'nr_users']].groupby(['GEOID']).sum()

        # Get population weighted averages per cluster
        df_clusters = df_clusters[df_clusters["nr_users"] > 0].copy() # Keep only merged counties with tweets
        df_clusters = df_clusters.drop(columns=['nr_tweets', 'nr_users']) # drop sum columns

        df_clusters.reset_index(inplace=True, drop=True)
        wm = lambda x: np.average(x, weights=df_clusters.loc[x.index, "population"])
        w_avgs = df_clusters.groupby(["GEOID"]).agg(wm).reset_index()
        w_avgs.drop(columns="population", inplace=True)

        # Merge and inspect
        df_clusters = sums.merge(w_avgs, how="left", on=["GEOID"])
        no_cluster = sum(df_clusters["median_friends"].isna())
        print(f"Municipalities with no cluster level values: {no_cluster}")
        return df_clusters

    # Get cluster level averages
    df_clusters = cluster_values(df_clusters)


    # Get values for second order clusters for cases where no tweets in cluster
    # ------------------------------------------------------------------------------

    # Select municipalities without cluster values
    missing_clusters = set(df_clusters.loc[df_clusters["median_friends"].isna(), "GEOID"])
    neighbors2 = neighbors[neighbors["GEOID"].isin(missing_clusters)]

    # Get neighbors of neighbors
    neighbors2 = neighbors2.merge(neighbors.rename(columns={"GEOID": "GEOID_merge",
                                                            "N_GEOID": "N2_GEOID"}),
                                      left_on="N_GEOID",
                                      right_on="GEOID_merge") # Get neighbors of neighbors
    neighbors2 = neighbors2.drop(columns=["GEOID_merge", "N_GEOID"])
    neighbors2 = neighbors2.drop_duplicates() # Drop duplicates

    # Merge with data for neighbors of neighbors
    df_clusters2 = neighbors2.merge(df.rename(columns={"GEOID": "N_GEOID_merge"}),
                                  left_on="N2_GEOID", right_on="N_GEOID_merge")
    df_clusters2 = df_clusters2.drop(columns=['N2_GEOID', 'N_GEOID_merge', 'geometry'])

    # Get cluster level averages for second order clusters
    df_clusters2 = cluster_values(df_clusters2)
    df_clusters2 = df_clusters2.sort_values(by="GEOID")

    # Insert values for second order clusters if no tweets in cluster
    len(df_clusters)
    df_clusters["zero_in_cluster"] = df_clusters["median_friends"].isna().astype(int)
    df_clusters = df_clusters[df_clusters["zero_in_cluster"] == 0] # Delete empty rows
    df_clusters2["zero_in_cluster"] = 1
    df_clusters = pd.concat([df_clusters, df_clusters2])
    len(df_clusters)


    # Get values for third order clusters for cases where no tweets in second order cluster
    # --------------------------------------------------------------------------------------

    # Select municipalities without second order cluster values
    missing_clusters2 = set(df_clusters2.loc[df_clusters2["median_friends"].isna(), "GEOID"])
    if len(missing_clusters2) > 0:
        neighbors3 = neighbors2[neighbors2["GEOID"].isin(missing_clusters2)]

        # Get neighbors of neighbors of neighbors
        neighbors3 = neighbors3.merge(neighbors.rename(columns={"GEOID": "GEOID_merge",
                                                                "N_GEOID": "N3_GEOID"}),
                                          left_on="N2_GEOID",
                                          right_on="GEOID_merge") # Get neighbors of neighbors of neighbors
        neighbors3 = neighbors3.drop(columns=["GEOID_merge", "N2_GEOID"])
        neighbors3 = neighbors3.drop_duplicates() # Drop duplicates

        # Merge with data for neighbors of neighbors of neighbors
        df_clusters3 = neighbors3.merge(df.rename(columns={"GEOID": "N_GEOID_merge"}),
                                      left_on="N3_GEOID", right_on="N_GEOID_merge")
        df_clusters3 = df_clusters3.drop(columns=['N3_GEOID', 'N_GEOID_merge', 'geometry'])

        # Get cluster level averages for third order clusters
        df_clusters3 = cluster_values(df_clusters3)

        # Insert values for third order clusters if no tweets in second order cluster
        df_clusters["zero_in_cluster2"] = df_clusters["median_friends"].isna().astype(int)
        df_clusters = df_clusters[df_clusters["zero_in_cluster2"] == 0] # Delete empty rows
        df_clusters3["zero_in_cluster2"] = 1
        df_clusters3["zero_in_cluster"] = 1
        df_clusters = pd.concat([df_clusters, df_clusters3])


    # Get values for 4. order clusters for cases where no tweets in third order cluster
    # --------------------------------------------------------------------------------------

        # Select municipalities without third order cluster values
        missing_clusters3 = set(df_clusters3.loc[df_clusters3["median_friends"].isna(), "GEOID"])
        if len(missing_clusters3) > 0:
            neighbors4 = neighbors3[neighbors3["GEOID"].isin(missing_clusters3)]

            # Get neighbors of neighbors of neighbors of neighbors
            neighbors4 = neighbors4.merge(neighbors.rename(columns={"GEOID": "GEOID_merge",
                                                                    "N_GEOID": "N4_GEOID"}),
                                          left_on="N3_GEOID",
                                          right_on="GEOID_merge")  # Get neighbors of neighbors of neighbors
            neighbors4 = neighbors4.drop(columns=["GEOID_merge", "N3_GEOID"])
            neighbors4 = neighbors4.drop_duplicates()  # Drop duplicates

            # Merge with data for neighbors of neighbors of neighbors
            df_clusters4 = neighbors4.merge(df.rename(columns={"GEOID": "N_GEOID_merge"}),
                                            left_on="N4_GEOID", right_on="N_GEOID_merge")
            df_clusters4 = df_clusters4.drop(columns=['N4_GEOID', 'N_GEOID_merge', 'geometry'])

            # Get cluster level averages for 4. order clusters
            df_clusters4 = cluster_values(df_clusters4)

            # Insert values for 4. order clusters if no tweets in third order cluster
            df_clusters["zero_in_cluster3"] = df_clusters["median_friends"].isna().astype(int)
            df_clusters = df_clusters[df_clusters["zero_in_cluster3"] == 0]  # Delete empty rows
            df_clusters4["zero_in_cluster3"] = 1
            df_clusters4["zero_in_cluster2"] = 1
            df_clusters4["zero_in_cluster"] = 1
            df_clusters = pd.concat([df_clusters, df_clusters4])
            del df_clusters4

        del df_clusters3
    del df_clusters2

    # Finalize
    # ------------------------------------------------------------------------------
    move_col(df_clusters, 'pop_density', 'nr_tweets')
    move_col(df_clusters, 'nr_tweets', 'tweet_density')
    move_col(df_clusters, 'nr_users', 'user_density')


    ################################################################################
    # Prepare and inspect features for county and cluster dataframes
    ################################################################################


    # Transform features for df and cluster df
    # ------------------------------------------------------------------------------

    # Define set of variables that will not be logged:
    nolog = ["population", "pop_density", "nr_tweets", "nr_users",
             "share_weekdays", "share_workhours", "share_iphone",
             "account_age_absolute_year",
             "positive", "negative", "ref_clos_cent",
             "zero_in_cluster", "zero_in_cluster2",
             "zero_in_cluster2"] # "is_quote_status_mean", "is_reply_mean",

    # Take log for county and cluster-level data
    df_counties = df.copy()
    del df

    for df in [df_counties, df_clusters]:
        # Take log for population density
        df["pop_density"] = np.log(df["pop_density"])
        df.rename(columns={"pop_density": "log_pop_density"}, inplace=True)

        # Add small number and then take log for other right-skewed variables
        for col in [x for x in X_cols if x not in nolog]:
            df[col] = log_with0(df[col])
            df.rename(columns={col: f"log_{col}"}, inplace=True)
    del df

    # Update columns groups
    log_cols = ["log_pop_density"] + ["log_" + x for x in X_cols if x not in nolog]
    X_cols = [x for x in df_counties.columns if x not in id_cols + geo_cols]
    pop_cols = [f"log_{x}" if f"log_{x}" in log_cols else x for x in pop_cols]
    count_cols = [f"log_{x}" if f"log_{x}" in log_cols else x for x in count_cols]
    tweet_cols = [f"log_{x}" if f"log_{x}" in log_cols else x for x in tweet_cols]
    topic_cols = [f"log_{x}" if f"log_{x}" in log_cols else x for x in topic_cols]
    error_cols = [f"log_{x}" if f"log_{x}" in log_cols else x for x in error_cols]
    sentiment_cols = [f"log_{x}" if f"log_{x}" in log_cols else x for x in sentiment_cols]
    network_cols = [f"log_{x}" if f"log_{x}" in log_cols else x for x in network_cols]


    ################################################################################
    # Impute values for missings in county level data (if 0 tweets)
    ################################################################################

    # Merge county with cluster data
    df_clusters.columns = list(df_clusters.columns[:1]) + list(["cluster_" + x for x in df_clusters.columns[1:]])
    df = df_counties.merge(df_clusters, on=["GEOID"])
    #print(len(df) == len(df_counties) == len(df_clusters))


    # Set parameters
    cluster_cols = list(df_clusters.columns[1:])
    Xvars = count_cols + pop_cols + cluster_cols
    scaler = StandardScaler()
    enet = ElasticNet(alpha=0.01, max_iter=1000, l1_ratio=0.9)
    enet_model = Pipeline(steps=[("scaler", scaler),
                          ("enet", enet)])

    # Impute values
    df["zero_tweets"] = df["log_median_friends"].isna().astype(int)
    for col in tweet_cols + topic_cols + error_cols + sentiment_cols + network_cols:
        impute_na(df=df, yvar=col, Xvars=Xvars,
                  model=enet_model, miss_indicator="zero_tweets", verbose=False)
    df.drop(columns="zero_tweets", axis=1, inplace=True)


    ################################################################################
    # Impute values for outliers if in counties with less than 5 tweets
    ################################################################################
    #https://towardsdatascience.com/univariate-outlier-detection-in-python-40b621295bc5


    # Define columns for outlier imputation
    impute_cols = tweet_cols + topic_cols + error_cols + sentiment_cols + network_cols

    # Replace outliers with missing if based on less than 5 users
    for colname in impute_cols:
        condition = (is_outlier(df[colname], verbose=False)) & (df["nr_users"] < 5)
        df.loc[condition, colname] = np.nan
        #print(f"{condition.sum()} outliers removed.")

    # Replace all missings with imputed values
    for col in impute_cols:
        df["miss_indicator"] = df[col].isna()
        if df["miss_indicator"].sum() > 0:
            impute_na(df=df, yvar=col, Xvars=Xvars, model=enet_model,
                      miss_indicator="miss_indicator", verbose=False)
            #print(f"{col}: {df['miss_indicator'].sum()} outliers replaced with imputed values.")
    df.drop(columns="miss_indicator", inplace=True)



    ################################################################################
    # Finalize feature data
    ################################################################################


    # Drop geographic variables
    df.drop(columns=['geometry'], inplace=True)
    df.set_index(df["GEOID"], inplace=True)
    df.drop(columns="GEOID", inplace=True)


    # Shuffle data
    df = df.sample(frac=1, random_state=56)

    return df

# Test MX
if __name__ == "__main__":
    from paths import tpaths_mx as paths

    input_path = paths["processed_data"]
    geo_path = paths["geo_data"]
    module_path = os.getcwd() + paths["modules"]

    # Import data
    indicator_df = pd.read_csv(input_path + r"\03-indicators\weeks\week_09\mun-level-complete-MX-09.csv",
                     sep=";")
    indicator_df = indicator_df.rename(columns={"mun_id_user": "GEOID"})

    # Import geodata
    geo_df = geopandas.read_file(geo_path + "/mex_admbnda_adm2_govmex_20210618.shp")
    geo_df = geo_df.rename(columns={"ADM2_PCODE": "GEOID"})
    geo_df = geo_df[["GEOID", "geometry"]]

    merge_var = "GEOID"

    # Apply function
    feature_data = prepare_features(indicator_df,
                                    geo_df,
                                    merge_var="GEOID")
