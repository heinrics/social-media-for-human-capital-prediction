
#-------------------------------------------------------------------------------
################################################################################
# Heterogeneity plots
# -----------------------------------------------------------
# - R2 by nr of user quantiles
# - R2 by nr of tweets quantiles
# - R2 by population quantiles
# - R2 by nr of weeks included
################################################################################
#-------------------------------------------------------------------------------



################################################################################
# Import modules
################################################################################

import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from math import sqrt
import matplotlib

pd.set_option('display.max_columns', 20)

################################################################################
# Specify paths to folders
################################################################################

from paths import paths_mx as paths
mx_input_path = paths["results"] + "/"
mx_train_path = paths["train_data"] + "/"
exhibits_path = paths["exhibits"] + r"/figures/sample-subset/"
module_path = os.getcwd() + paths["modules"] + "/"

from paths import paths_us as paths
us_input_path = paths["results"]
us_train_path = paths["train_data"]

################################################################################
# Import functions and labels
################################################################################

# Import functions
sys.path.insert(1, module_path)
from visualize_results import plot_hetero

# Import feature names (for plots)
flabels = pd.read_excel(module_path + r"feature_labels.xlsx")
label_dict = dict(zip(flabels.col_name, flabels.label))

################################################################################
# Prepare data and set parameters
################################################################################

# Colors
mx_color = "#00008B"
us_color = "#8B0000"

mx_color_alpha = (0, 0, 0.545, 0.3)
us_color_alpha = (0.545, 0, 0, 0.3)

# Font size
matplotlib.rcParams.update({'axes.labelsize': 18})

# Feature data
mx_features = pd.read_csv(mx_train_path + r"features.csv",
                          sep=";",
                          index_col="GEOID")
mx_nrf = len(mx_features.columns) # Nr of features
mx_features = mx_features[["nr_users", "nr_tweets", "population"]] # Keep counts and population

us_features = pd.read_csv(us_train_path + r"features.csv",
                          sep=";",
                          index_col="GEOID",)
us_nrf = len(us_features.columns)  # Nr of features
us_features = us_features[["nr_users", "nr_tweets", "population"]]  # Keep counts and population

# Prediction data
mx_preds = pd.read_csv(mx_input_path + r"yschooling_predictions.csv",
                       sep=";",
                       index_col="GEOID")

mx_preds = mx_preds.merge(mx_features,
                          right_index=True,
                          left_index=True)

us_preds = pd.read_csv(us_input_path + r"yschooling_predictions.csv",
                       sep=";",
                       index_col="GEOID")

us_preds = us_preds.merge(us_features,
                          right_index=True,
                          left_index=True)




# #######################################################################################
# Performance heterogeneity by user, tweet and population threshold
# #######################################################################################


# Plot by user count
for df, ylabel, color, pref, n_features in zip([mx_preds, us_preds],
                                               ["Share of included units",
                                                "Share of included units"],
                                               [mx_color, us_color],
                                               ["mx", "us"],
                                               [mx_nrf, us_nrf]):
    plot_hetero(df["y"],
                df["y_hat"],
                df["fold"],
                df["nr_users"],
                xlabel="User count cutoff",
                ylabel=ylabel,
                color=color,
                uncertainties="se",
                n_features=n_features,
                save=exhibits_path + fr"{pref}_by_user_count.pdf")

# Plot by tweet count
mx_cutoffs = range(0, 52, 2)
us_cutoffs = range(0, 155, 5)
for df, cutoffs, ylabel, color, pref, n_features in zip([mx_preds, us_preds],
                                                        [mx_cutoffs, us_cutoffs],
                                                        ["Share of included units",
                                                         "Share of included units"],
                                                        [mx_color, us_color],
                                                        ["mx", "us"],
                                                        [mx_nrf, us_nrf]):
    plot_hetero(df["y"],
                df["y_hat"],
                df["fold"],
                df["nr_tweets"],
                xlabel="Tweet count cutoff",
                ylabel=ylabel,
                color=color,
                cutoffs=cutoffs,
                uncertainties="se",
                n_features=n_features,
                save=exhibits_path + fr"{pref}_by_tweet_count.pdf")


# Plot by population
mx_cutoffs = range(0, 31000, 1000)
us_cutoffs = range(0, 46500, 1500)
for df, cutoffs, ylabel, color, pref, n_features in zip([mx_preds, us_preds],
                                                        [mx_cutoffs, us_cutoffs],
                                                        ["Share of included units",
                                                         "Share of included units"],
                                                        [mx_color, us_color],
                                                        ["mx", "us"],
                                                        [mx_nrf, us_nrf]):
    plot_hetero(df["y"],
                df["y_hat"],
                df["fold"],
                df["population"],
                xlabel="Population cutoff",
                ylabel=ylabel,
                color=color,
                cutoffs=cutoffs,
                uncertainties="se",
                n_features=n_features,
                save=exhibits_path + fr"{pref}_by_population.pdf")


# Inspect R2 for low vs. high education (only works with split at median)
for preds in [mx_preds, us_preds]:
    levels = ["low", "high"]
    preds["quantile"] = pd.qcut(preds["y"], len(levels), labels=levels)
    for level in levels:
        y = preds.loc[preds["quantile"] == level, "y"]
        y_hat = preds.loc[preds["quantile"] == level, "y_hat"]
        print(1-((y - y_hat)**2).sum()/((y - preds["y"].mean())**2).sum())

    preds.drop(columns="quantile", inplace=True)


# #######################################################################################
# Performance based on data collection period
# #######################################################################################

# Set parameters
nr_weeks = 9
nr_days = 7

# Font size
matplotlib.rcParams.update({'axes.labelsize': 15})

# Prepare data on performance per week
# #######################################################################################
mx_by_week = pd.read_csv(mx_input_path + r"yschooling_by_week.csv",
                         sep=";")
mx_by_week.drop(columns=mx_by_week.columns[0],
                inplace=True)

us_by_week = pd.read_csv(us_input_path + r"yschooling_by_week.csv",
                         sep=";")
us_by_week.drop(columns=us_by_week.columns[0],
                inplace=True)


# Prepare data on performance per day
# #######################################################################################
mx_by_day = pd.read_csv(mx_input_path + r"yschooling_by_day.csv", sep=";")
mx_by_day.drop(columns=mx_by_day.columns[0],
               inplace=True)

us_by_day = pd.read_csv(us_input_path + r"yschooling_by_day.csv", sep=";")
us_by_day.drop(columns=us_by_day.columns[0],
               inplace=True)



# Prepare data on share of units without tweets for each week and day
# #######################################################################################

dfs = {}

for period, period_length in zip(["week", "day"],
                                 [nr_weeks, nr_days]):
    no_tweets = {}
    for train_path, pref in zip([mx_train_path, us_train_path],
                                ["mx_shares", "us_shares"]):

        shares = [1]
        for i in range(1, period_length+1):

            # Import data
            i = str(i).zfill(2)
            # features = pd.read_csv(train_path + fr"\{period}s\features_{period}{i}.csv",
            features = pd.read_csv(train_path + fr"{period}s/features_{period}{i}.csv",
                                   dtype={"GEOID": str},
                                   index_col="GEOID",
                                   sep=";")

            # Get share of communities without tweets
            shares.append((features["nr_tweets"] == 0).mean())

        no_tweets[pref] = shares

    no_tweets = pd.DataFrame(no_tweets)
    dfs[period] = no_tweets


no_tweets_week = dfs["week"]
no_tweets_day = dfs["day"]


# Define function and set parameter for uncertainties
# #######################################################################################
uncertainties = "se" # Or: "fold_sd"
def se_r2(R2, n, k):
    ses = sqrt((4 * R2 * ((1 - R2) ** 2) * ((n - k - 1) ** 2)) / ((n ** 2 - 1) * (n + 3)))
    return(ses)


# Plot by week
# #######################################################################################

# Create figure
fig, ax = plt.subplots()


# Plot R2 and uncertainties for MX and US
for df, color, n, k in zip([mx_by_week, us_by_week],
                           [mx_color, us_color],
                           [len(mx_preds), len(us_preds)],
                           [mx_nrf, us_nrf]):
    # R2
    ax.plot(df["full"],
            color=color,
            marker="o",
            zorder=10,
            clip_on=False)

    # Standard deviation of fold R2s
    if uncertainties == "fold_sd":
        ax.fill_between(range(0, nr_weeks+1),
                        df["full"] + df["sd"],
                        df["full"] - df["sd"],
                        color=color,
                        alpha=.1)

    # Standard error based on nr of observations and features
    if uncertainties == "se":
        r2se = df.apply(lambda x: se_r2(x["full"], n, k), axis=1)
        ax.fill_between(range(0, nr_weeks+1),
                        (df["full"] + r2se),
                        (df["full"] - r2se),
                        color=color,
                        alpha=.1)


# Customize grid and axes
ax.grid(which='major',
        linestyle='--',
        color="#e2e4e0")
ax.set_ylim(0.3, 0.75)
ax.set_xlim(0, nr_weeks)
ax.set_ylabel("Crossvalidation $r^2$")
ax.set_xlabel("Weeks of data collection")
ax.spines['top'].set_visible(False)


# Add additional ax to plot % of units without tweets
ax2 = ax.twinx()
mx_shares = ax2.plot(range(0, nr_weeks+1),
                     no_tweets_week["mx_shares"]*100,
                     color=mx_color,
                     alpha=0.4,
                     zorder=10,
                     linestyle="--")

us_shares = ax2.plot(range(0, nr_weeks+1),
                     no_tweets_week["us_shares"]*100,
                     color=us_color,
                     alpha=0.4,
                     zorder=10,
                     linestyle="--")
ax2.set_ylim(0, 100)
ax2.set_ylabel("Share of units without tweets")
ax2.spines['top'].set_visible(False)

fig.tight_layout()
fig.savefig(exhibits_path + fr"yschooling_by_week.pdf", bbox_inches='tight')
fig.show()



# Plot by day
# #######################################################################################

# Create figure
fig, ax = plt.subplots()


# Plot R2 and uncertainties for MX and US
for df, color, n, k in zip([mx_by_day, us_by_day],
                           [mx_color, us_color],
                           [len(mx_preds), len(us_preds)],
                           [mx_nrf, us_nrf]):
    # R2
    ax.plot(df["full"],
            color=color,
            marker="o",
            zorder=10,
            clip_on=False)

    # Standard deviation of fold R2s
    if uncertainties == "fold_sd":
        ax.fill_between(range(0, nr_days+1),
                        df["full"] + df["sd"],
                        df["full"] - df["sd"],
                        color=color,
                        alpha=.1)

    # Standard error based on nr of observations and features
    if uncertainties == "se":
        r2se = df.apply(lambda x: se_r2(x["full"], n, k), axis=1)
        ax.fill_between(range(0, nr_days+1),
                        (df["full"] + r2se),
                        (df["full"] - r2se),
                        color=color,
                        alpha=.1)


# Customize grid and axes
ax.grid(which='major',
        linestyle='--',
        color="#e2e4e0")
ax.set_ylim(0.3, 0.75)
ax.set_xlim(0, nr_days)
ax.set_ylabel("Crossvalidation $r^2$")
ax.set_xlabel("Days of data collection")
ax.spines['top'].set_visible(False)


# Add additional ax to plot % of units without tweets
ax2 = ax.twinx()
mx_shares = ax2.plot(range(0, nr_days+1),
                     no_tweets_day["mx_shares"]*100,
                     color=mx_color,
                     alpha=0.4,
                     zorder=10,
                     linestyle="--")

us_shares = ax2.plot(range(0, nr_days+1),
                     no_tweets_day["us_shares"]*100,
                     color=us_color,
                     alpha=0.4,
                     zorder=10,
                     linestyle="--")
ax2.set_ylim(0, 100)
ax2.set_ylabel("Share of units without tweets")
ax2.spines['top'].set_visible(False)

fig.tight_layout()
fig.savefig(exhibits_path + fr"yschooling_by_day.pdf")
fig.show()
