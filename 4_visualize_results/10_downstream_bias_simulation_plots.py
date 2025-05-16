import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
import matplotlib

# Set directory and seed
# os.chdir(r"H:\analysis_mapping_mistakes\Exhibits\analysis")

################################################################################
# Specify paths to folders
################################################################################

# tpaths_mx spaths_mx
from paths import paths_mx as paths
exhibits_path = paths["exhibits"] + r"/figures/simulations/"

seed = 1625
np.random.seed(1625)


# Set styling parameters
dotalpha = 0.15
dotsize = 5
arrow_width = 0.005
arrow_alpha = 1
arrow_color = "black"
sample_size = 1000
truth_color = (0, 0, 0.545, 0.3)
pred_color = (0.545, 0, 0, 0.3)
matplotlib.rcParams.update({'font.size': 20})
minmax = np.array([-3, 3])

# Function to create figure
def create_figure(edu, edu_hat, econ, var_increase=True, save=False):
    fig, axs = plt.subplots(2, 1, figsize=(5.5, 10))

    # Plot edu as y
    axs[0].scatter(econ, edu, color=truth_color, s=dotsize, alpha=dotalpha)
    m, b = np.polyfit(econ, edu, 1)
    axs[0].plot(minmax, m * minmax + b, color=truth_color,
                linewidth=2)

    axs[0].scatter(econ, edu_hat, color=pred_color, s=dotsize, alpha=dotalpha)
    m, b = np.polyfit(econ, edu_hat, 1)
    axs[0].plot(minmax, m * minmax + b, color=pred_color, linestyle="dashed",
                linewidth=2)

    # Plot edu as x
    axs[1].scatter(edu, econ, color=truth_color, s=dotsize, alpha=dotalpha)
    m, b = np.polyfit(edu, econ, 1)
    axs[1].plot(minmax, m * minmax + b, color=truth_color)
    axs[1].scatter(edu_hat, econ, color=pred_color, s=dotsize, alpha=dotalpha)
    m, b = np.polyfit(edu_hat, econ, 1)
    axs[1].plot(minmax, m * minmax + b, color=pred_color,
                linestyle='--', linewidth=2)

    for i in [0, 1]:
        axs[i].set_xlim(-3, 3)
        axs[i].set_ylim(-3, 3)
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
        axs[i].set_xticks(range(-3, 3 + 1))
        axs[i].set_yticks(range(-3, 3 + 1))

    # axs[1].set_ylabel("$edu$ / " + r"\textcolor{red}{$\hat{edu}$}")
    axs[0].set_ylabel("$edu$ (or $\widehat{edu}$)")
    axs[0].set_xlabel("$econ$")

    axs[1].set_ylabel("$econ$")
    axs[1].set_xlabel("$edu$ (or $\widehat{edu}$)")

    # Identify and plot typical point movements
    diff = edu_hat - edu
    diff2 = edu - econ
    df = pd.DataFrame(zip(edu, edu_hat, econ, diff, diff2),
                      columns=["edu", "edu_hat", "econ", "diff", "diff2"])

    if var_increase:
        below = df[(df["diff"] < -0.5) & (df["diff"] > -2) &
                   (df["diff2"] < -0.2) &
                   (df["edu_hat"] > -2.9)].sort_values("econ")
        interval = round(len(below) / 5)
        below = list(below.index[[interval, interval * 4]])

        above = df[(df["diff"] > 0.5) & (df["diff"] < 2) &
                   (df["diff2"] > 0.2) &
                   (df["edu_hat"] < 2.9)].sort_values("econ")
        interval = round(len(above) / 5)
        above = list(above.index[[interval, interval * 4]])

    if var_increase==False:
        diff2 = edu_hat - econ
        df["diff2"] = diff2
        below = df[(df["diff"] > 0.5) & (df["diff"] < 2) &
                   (df["diff2"] < 0)].sort_values("econ")
        interval = round(len(below) / 8)
        below = list(below.index[[interval*2, interval * 6]])

        above = df[(df["diff"] < -0.5) & (df["diff"] > -2) &
                   (df["diff2"] > 0)].sort_values("econ")
        interval = round(len(above) / 7)
        above = list(above.index[[interval, interval * 6]])


    for i in below + above:
        old_point = [econ[i], edu[i]]
        new_point = [econ[i], edu_hat[i]]
        dist_points = edu_hat[i] - edu[i]
        axs[0].scatter(old_point[0], old_point[1],
                       color=truth_color,
                       alpha=0.7)
        axs[0].scatter(new_point[0], new_point[1],
                       color=pred_color,
                       alpha=0.7)
        axs[0].arrow(old_point[0], old_point[1], 0, dist_points,
                     width=arrow_width,
                     length_includes_head=True,
                     color="black",
                     head_width=arrow_width * 30,
                     head_length=arrow_width * 30,
                     alpha=arrow_alpha,
                     zorder=10
                     )

        axs[1].scatter(old_point[1], old_point[0],
                       color=truth_color,
                       alpha=0.7)
        axs[1].scatter(new_point[1], new_point[0],
                       color=pred_color,
                       alpha=0.7)
        axs[1].arrow(old_point[1], old_point[0], dist_points, 0,
                     width=arrow_width,
                     length_includes_head=True,
                     color=arrow_color,
                     head_width=arrow_width * 30,
                     head_length=arrow_width * 30,
                     alpha=arrow_alpha,
                     zorder=10)

    fig.tight_layout()
    if save != False:
        fig.savefig(save)
    fig.show()


# Random measurement error
# ------------------------------------------------------------------

econ = np.random.normal(size=sample_size)
edu = econ + 0.5*np.random.normal(size=sample_size)
edu_hat = edu + 0.8*np.random.normal(size=sample_size)

# Check correlations
edu_error = edu-edu_hat
print(np.corrcoef(edu, econ))
print(np.cov([edu_hat, econ]))
print(np.corrcoef(edu_hat, econ))
print(np.cov([edu_hat, econ]))
print(np.corrcoef(edu, edu_error))
print(np.corrcoef(econ, edu_error))

cov_edu = np.cov([edu, econ])
cov_edu_hat = np.cov([edu_hat, econ])
print(cov_edu[0, 1]/cov_edu[0, 0])
print(cov_edu_hat[0, 1]/cov_edu_hat[0, 0])
print(cov_edu[0, 1]/cov_edu[1, 1])
print(cov_edu_hat[0, 1]/cov_edu_hat[1, 1])


# Attenuation correction
var_error = np.var(edu_hat-edu)

print(cov_edu[0, 1]/cov_edu[0, 0])
print(cov_edu_hat[0, 1]/(cov_edu_hat[0, 0]-var_error)) # With variance of error

from sklearn.metrics import r2_score
r2 = r2_score(edu, edu_hat)
print(cov_edu_hat[0, 1]/(cov_edu_hat[0, 0]-(np.var(edu)*(1-r2)))) # "With R2 and variance of ground truth

# Create figure
create_figure(edu, edu_hat, econ, save=exhibits_path + "sim1_attenuation.pdf")

# Berkson-type error
# ------------------------------------------------------

# Create data
edu_hat = 0.5*edu #+ 0.1*np.random.normal(size=500)

# Check correlations
edu_error = edu-edu_hat
print(np.corrcoef(edu, econ))
print(np.corrcoef(edu_hat, econ))
print(np.cov([edu, econ]))
print(np.cov([edu_hat, econ]))
print(np.corrcoef(edu, edu_error))
print(np.corrcoef(econ, edu_error))

# Create figure
create_figure(edu, edu_hat, econ, save=exhibits_path + "sim2_berkson.pdf")


# # Error correlated with econ
# -----------------------------------------------------

# Create data
econ = np.random.normal(size=sample_size)
edu = econ + 0.8*np.random.normal(size=sample_size)

m, b = np.polyfit(econ, edu, 1)
edu_hat = m*econ+b + 0.25*np.random.normal(size=sample_size)

# Check correlations
edu_error = edu-edu_hat
print(np.corrcoef(edu, econ))
print(np.corrcoef(edu_hat, econ))
print(np.cov([edu, econ]))
print(np.cov([edu_hat, econ]))
print(np.corrcoef(edu, edu_error))
print(np.corrcoef(econ, edu_error))

print(np.cov(edu, edu_error))
print(np.cov(econ, edu_error))


# Create figure
create_figure(edu, edu_hat, econ, var_increase=False, save=exhibits_path + "sim3_correcon.pdf")
