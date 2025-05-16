
#-------------------------------------------------------------------------------
################################################################################
# Boxplots with R2 results for MX and US
# -----------------------------------------------------------
# - Boxplot with r2 for each outcome
# - Boxplot with r2 for each model (main outcome)
# - Boxplot with r2 for each variable group
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
from sklearn.metrics import r2_score
from matplotlib.patches import Rectangle


pd.set_option('display.max_columns', 20)

################################################################################
# Specify paths to folders
################################################################################

from paths import paths_mx as paths
mx_input_path = paths["results"] + "/"
exhibits_path = paths["exhibits"] + r"/figures/analysis/"
from paths import paths_us as paths
us_input_path = paths["results"] + "/"


# Set parameters
################################################################################

# Colors
mx_color = "#00008B"
us_color = "#8B0000"

mx_color_alpha = (0, 0, 0.545, 0.3)
us_color_alpha = (0.545, 0, 0, 0.3)


# Outcomes
mx_outcomes = ["yschooling",
               "pbasic",
               "secondary",
               "primary"]
us_outcomes = ["yschooling",
               "bachelor",
               "some_college",
               "high_school"]


# #######################################################################################
# Boxplots
# #######################################################################################


# Boxplot with R2 scores for each outcome
# ---------------------------------------------------------------------------------------

## V1: horizontal
#################

# Import and prepare data
all_results = {}
for outcome in mx_outcomes:
    stack = list(pd.read_csv(mx_input_path + rf"{outcome}_r2.csv", sep=";", index_col=0)\
         .transpose()["stack"])
    all_results[f"mx_{outcome}"] = stack
for outcome in us_outcomes:
    stack = list(pd.read_csv(us_input_path + rf"{outcome}_r2.csv", sep=";", index_col=0)\
         .transpose()["stack"])
    all_results[f"us_{outcome}"] = stack
all_outcomes = pd.DataFrame(all_results)
all_outcomes.index = [f"fold{x}" for x in range(5)]
all_outcomes = all_outcomes[all_outcomes.columns[::-1]]


# Set parameters for figure
boxprops = dict(linewidth=1)
medianprops = dict(color='black')
meanlineprops = dict(linestyle='--', linewidth=1, color='black')
tick_labels = ["MX: Years of Schooling",
               "MX: % Post-basic",
               "MX: % Secondary",
               "MX: % Primary",
               "US: Years of Schooling",
               "US: % Bachelor",
               "US: % Some College",
               "US: % High School"]
linecolors = [us_color]*4 + [mx_color]*4
facecolors = [us_color_alpha]*4 + [mx_color_alpha] * 4



# Generate plot
fig, ax = plt.subplots(figsize=(8, 8))

bplot = ax.boxplot(all_outcomes,
                   vert=False,
                   whis=1000,
                   meanline=True,
                   showmeans=True,
                   patch_artist=True,
                   boxprops=boxprops,
                   medianprops=medianprops,
                   meanprops=meanlineprops,
                   positions=[1, 2, 3, 4, 5.5, 6.5, 7.5, 8.5],
                   widths=0.8)

# Customize grid and axes
ax.grid(which='major',
        axis='x',
        linestyle='--',
        color="#e2e4e0")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticklabels(tick_labels[::-1])
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_xlim(np.floor(all_outcomes.min().min()*20)/20,
            np.ceil(all_outcomes.max().max()*20)/20)
ax.set_ylim(0.25, 9.25)
ax.axhline(4.75,
           color="black",
           linestyle='--',
           linewidth=0.75)

# Change colors
for patch, lcolor, fcolor, in zip(bplot["boxes"], linecolors, facecolors):
    patch.set_color(lcolor)
    patch.set_facecolor(fcolor)

fig.tight_layout()
fig.savefig(exhibits_path + fr"main_results_h.pdf")
fig.show()



## V2: vertical
#################

# Reverse column order
all_outcomes = all_outcomes[all_outcomes.columns[::-1]]

# Generate plot
fig, ax = plt.subplots(figsize=(8, 8))

bplot = ax.boxplot(all_outcomes,
                   vert=True,
                   whis=1000,
                   meanline=True,
                   showmeans=True,
                   patch_artist=True,
                   boxprops=boxprops,
                   medianprops=medianprops,
                   meanprops=meanlineprops,
                   positions=[1, 2, 3, 4, 5.5, 6.5, 7.5, 8.5],
                   widths=0.8)

# Customize grid and axes
ax.grid(which='major',
        axis='y',
        linestyle='--',
        color="#e2e4e0")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticklabels(tick_labels, rotation=90)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_ylim(np.floor(all_outcomes.min().min()*20)/20,
            np.ceil(all_outcomes.max().max()*20)/20)
ax.set_xlim(0.25, 9.25)
ax.axvline(4.75,
           color="black",
           linestyle='--',
           linewidth=0.75)

# Change colors
for patch, lcolor, fcolor, in zip(bplot["boxes"], linecolors[::-1], facecolors[::-1]):
    patch.set_color(lcolor)
    patch.set_facecolor(fcolor)

fig.tight_layout()
fig.savefig(exhibits_path + fr"main_results.pdf")
fig.show()


# Model results: Boxplot with R2 scores for each model, main outcome (years of schooling)
# ---------------------------------------------------------------------------------------

outcome = "yschooling"


## V1: horizontal
#################

# Import and prepare data
r2s_mx = pd.read_csv(mx_input_path + rf"{outcome}_r2.csv",
                     sep=";",
                     index_col=0).iloc[::-1].transpose()

r2s_us = pd.read_csv(us_input_path + rf"{outcome}_r2.csv",
                     sep=";",
                     index_col=0).iloc[::-1].transpose()

r2s_mx.columns = ["mx_" + x for x in r2s_mx.columns]
r2s_us.columns = ["us_" + x for x in r2s_us.columns]
r2s = pd.concat([r2s_mx, r2s_us], axis=1)
r2s = r2s[list(i for tupl in zip(r2s_us, r2s_mx) for i in tupl)]

# Set parameters for figure
models = ["Elastic Net", "Gradient Boosting", "Support Vector",
          "Nearest Neighbor", "Neural Net", "Stacking"]

boxprops = dict(linewidth=1)
medianprops = dict(color='black')
meanlineprops = dict(linestyle='--',
                     linewidth=1,
                     color='black')
linecolors = [us_color,
              mx_color]*6

facecolors = [us_color_alpha,
              mx_color_alpha]*6

# Generate plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.add_patch(Rectangle((0, 0), 1, 3,
                       facecolor="black",
                       alpha=0.05))

bplot = ax.boxplot(r2s,
                   vert=False,
                   whis=1000,
                   meanline=True,
                   showmeans=True,
                   patch_artist=True,
                   boxprops=boxprops,
                   medianprops=medianprops,
                   meanprops=meanlineprops,
                   positions=[1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17],
                   widths=0.95)

# Customize grid and axes
ax.grid(which='major', axis='x', linestyle='--', color="#e2e4e0")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks(np.arange(1.5, 17, step=3))
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(models[::-1])
ax.set_xlim(np.floor(r2s.min().min()*20)/20,
            np.ceil(r2s.max().max()*20)/20)
ax.set_ylim(0, 18)

for i in np.arange(3, 16, step=3):
    plt.axhline(i,
                color="black",
                linestyle='--',
                linewidth=0.75)

# Color by country
for patch, lcolor, fcolor, in zip(bplot["boxes"], linecolors, facecolors):
    patch.set_color(lcolor)
    patch.set_facecolor(fcolor)

fig.tight_layout()
fig.savefig(exhibits_path + fr"{outcome}_models_h.pdf",
            bbox_inches='tight')
fig.show()


## V2: vertical
#################

# Change column order
r2s = r2s[r2s.columns[::-1]]

# Change order of colors
linecolors = [mx_color,
              us_color]*6

facecolors = [mx_color_alpha,
              us_color_alpha]*6


# Generate plot
fig, ax = plt.subplots(figsize=(10, 7))
ax.add_patch(Rectangle((15, 0), 3, 1,
                       facecolor="black",
                       alpha=0.05))

bplot = ax.boxplot(r2s,
                   vert=True,
                   whis=1000,
                   meanline=True,
                   showmeans=True,
                   patch_artist=True,
                   boxprops=boxprops,
                   medianprops=medianprops,
                   meanprops=meanlineprops,
                   positions=[1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17],
                   widths=0.95)

# Customize grid and axes
ax.grid(which='major', axis='y', linestyle='--', color="#e2e4e0")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(np.arange(1.5, 17, step=3))
ax.set_xticklabels(models, rotation=90)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_ylim(np.floor((r2s.min().min()-0.01)*20)/20,
            np.ceil(r2s.max().max()*20)/20)
ax.set_xlim(0, 18)

for i in np.arange(3, 16, step=3):
    plt.axvline(i,
                color="black",
                linestyle='--',
                linewidth=0.75)

# Color by country
for patch, lcolor, fcolor, in zip(bplot["boxes"], linecolors, facecolors):
    patch.set_color(lcolor)
    patch.set_facecolor(fcolor)

fig.tight_layout()
fig.savefig(exhibits_path + fr"{outcome}_models.pdf",
            bbox_inches='tight')
fig.show()


# Boxplot with R2 scores for each outcome
# ---------------------------------------------------------------------------------------

outcome = "yschooling"

#  Boxplot with outcomes for each variable group
# ################################################

# Import and prepare data
mx_groups = pd.read_csv(mx_input_path + rf"{outcome}_vargroups_r2.csv",
                        sep=";",
                        index_col=0).transpose()

us_groups = pd.read_csv(us_input_path + rf"{outcome}_vargroups_r2.csv",
                        sep=";",
                        index_col=0).transpose()


mx_groups.columns = ["mx_" + x for x in mx_groups.columns]
us_groups.columns = ["us_" + x for x in us_groups.columns]

var_groups = pd.concat([mx_groups, us_groups], axis=1)
var_groups = var_groups[list(i for tupl in zip(mx_groups, us_groups) for i in tupl)]


# Set parameters for figure
boxprops = dict(linewidth=1)
medianprops = dict(color='black')
meanlineprops = dict(linestyle='--',
                     linewidth=1,
                     color='black')

linecolors = ["#00008B", "#8B0000"]*9
facecolors = [(0, 0, 0.545, 0.3), (0.545, 0, 0, 0.3)]*9


# Create figure
fig, ax = plt.subplots(figsize=(13, 7))
bplot = ax.boxplot(var_groups.dropna(),
                   vert=True,
                   whis=1000,
                   meanline=True,
                   showmeans=True,
                   patch_artist=True,
                   boxprops=boxprops,
                   medianprops=medianprops,
                   meanprops=meanlineprops,
                   positions=[1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26],
                   widths=0.75)

ax.grid(which='major', axis='y', linestyle='--', color="#e2e4e0")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(np.arange(1.5, 28, step=3))
ax.set_xticklabels(["Cluster values",
                    "Unit values",
                    "Networks",
                    "Sentiments",
                    "Topics",
                    "Errors",
                    "Usage statistics",
                    "Twitter penetration",
                    "Population"][::-1], rotation=90)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_xlim(np.floor(var_groups.min().min()*20)/20,
            np.ceil(var_groups.max().max()*20)/20)
ax.set_xlim(0, 27)
ax.set_ylim(0.18, 0.66)

for i in np.arange(3, 21, step=3):
    ax.axvline(i, color="black", linestyle='--', linewidth=0.75)

for patch, lcolor, fcolor, in zip(bplot["boxes"], linecolors, facecolors):
    patch.set_color(lcolor)
    patch.set_facecolor(fcolor)

ax.axvline(24, color="black", linestyle='--', linewidth=0.75)
ax.axvline(21.05, color="black", linewidth=1)
ax.axvline(20.95, color="black", linewidth=1)


fig.tight_layout()
fig.savefig(exhibits_path + fr"{outcome}_vargroups.pdf")
fig.show()
