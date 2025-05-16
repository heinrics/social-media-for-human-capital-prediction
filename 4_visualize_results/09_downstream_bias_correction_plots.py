################################################################################
# Load packages
################################################################################
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

pd.options.display.width = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


################################################################################
# Specify paths
################################################################################


# Specify paths
from paths import paths_mx as paths
stack_path_mx = paths["results"] + '/bias_correction/'
out_path = paths["exhibits"] + '/figures/bias/'

from paths import paths_mx as paths
stack_path_us = paths["results"] + '/bias_correction/'


################################################################################
# Load datasets
################################################################################

mx_no_cor = pd.read_csv(stack_path_mx + 'bias-correction-yschooling-False-0.csv', sep=';')
mx_semi_cor = pd.read_csv(stack_path_mx + 'bias-correction-yschooling-True-1.csv', sep=';')
mx_full_cor = pd.read_csv(stack_path_mx + 'bias-correction-yschooling-True-3.csv', sep=';')

us_no_cor = pd.read_csv(stack_path_us + 'bias-correction-yschooling-False-0.csv', sep=';')
us_semi_cor = pd.read_csv(stack_path_us + 'bias-correction-yschooling-True-1.csv', sep=';')
us_full_cor = pd.read_csv(stack_path_us + 'bias-correction-yschooling-True-3.csv', sep=';')

xlabel = 'Years of schooling'
ylabel = 'Predicted years of schooling'

mx_color = "#00008B"
us_color = "#8B0000"

plt.rcParams.update({'axes.labelsize': 18})

round(np.corrcoef(mx_no_cor['y_true'], mx_no_cor['y_pred'] - mx_no_cor['y_true'])[1,0], 3)
# -0.597
round(np.corrcoef(us_no_cor['y_true'], us_no_cor['y_pred'] - us_no_cor['y_true'])[1,0], 3)
# -0.628

# Scatter plot
def scatter(y, y_hat, xlabel, ylabel, color="#8B0000", unit=10, outlier_min=0, outlier_max=0, title=False, save=False):

    # Make scatter plot
    fig, ax = plt.subplots(figsize=(5,5))
    plt.scatter(y, y_hat, alpha=0.12, s=15, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    # Add regression line
    b1, b0 = np.polyfit(y, y_hat, 1) # Estimate slope and intercept
    max_x = np.floor(y.nlargest(outlier_max+1).iloc[-1] / unit + 1) * unit - unit / 100
    min_x = np.floor(y.nsmallest(outlier_min+1).iloc[-1] / unit) * unit
    x = np.array([0, max_x])
    plt.plot(x, x, color='black', linestyle='--') # Plot line
    plt.plot(x, b1*x + b0, color=color) # Plot line
    ax.set_axisbelow(True)
    plt.grid(which='major', linestyle='--', color ="#e2e4e0")

    # Customize
    r2 = round(r2_score(y, y_hat), 3)
    b1 = round(b1, 3)
    plt.xlim([min_x, max_x])
    plt.ylim([min_x, max_x])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.text(min_x+unit/3, np.floor(max_x/unit)*unit-unit/2, f'$R^2$={r2:.3f}\n$\\beta_1$={b1:.3f}',
             fontsize=16)

    plt.tight_layout()

    if title != False:
        plt.title(title)
    if save != False:
        plt.savefig(save)

    plt.show()


# Mexico
scatter(mx_no_cor['y_true'],
        mx_no_cor['y_pred'],
        xlabel,
        ylabel,
        color=mx_color,
        unit=1,
        outlier_min=0,
        outlier_max=13,
        save=out_path + 'mx-no-correction.png')

scatter(mx_semi_cor['y_true'],
        mx_semi_cor['y_pred'],
        xlabel,
        ylabel,
        color=mx_color,
        unit=1,
        outlier_min=0,
        outlier_max=13,
        save=out_path + 'mx-semi-correction.png')

scatter(mx_full_cor['y_true'],
        mx_full_cor['y_pred'],
        xlabel,
        ylabel,
        color=mx_color,
        unit=1,
        outlier_min=0,
        outlier_max=13,
        save=out_path + 'mx-full-correction.png')


# USA
scatter(us_no_cor['y_true'],
        us_no_cor['y_pred'],
        xlabel,
        ylabel,
        color=us_color,
        unit=0.5,
        outlier_min=1,
        outlier_max=1,
        save=out_path + 'us-no-correction.png')

scatter(us_semi_cor['y_true'],
        us_semi_cor['y_pred'],
        xlabel,
        ylabel,
        color=us_color,
        unit=0.5,
        outlier_min=1,
        outlier_max=1,
        save=out_path + 'us-semi-correction.png')

scatter(us_full_cor['y_true'],
        us_full_cor['y_pred'],
        xlabel,
        ylabel,
        color=us_color,
        unit=0.5,
        outlier_min=1,
        outlier_max=1,
        save=out_path + 'us-full-correction.png')
