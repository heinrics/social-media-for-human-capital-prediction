# Import modules
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.inspection import permutation_importance


# Table with model performance results
# ------------------------------------------------------------------------------
def performance_table(model_dict, X_train, y_train, X_valid, y_valid):

    r2_scores = []
    for model in [x for x in model_dict if x != "stack"]:
        mscores = [model,
                   r2_score(y_train, model_dict[model].predict(X_train)),
                   r2_score(y_valid, model_dict[model].predict(X_valid)),
                   model_dict[model].best_params_]
        r2_scores.append(mscores)
    weights = {x: y for x, y in zip([x[0] for x in model_dict["stack"].estimators],
                                    model_dict["stack"].final_estimator_.coef_.round(3))}
    mscores = ["stack",
               r2_score(y_train, model_dict["stack"].predict(X_train)),
               r2_score(y_valid, model_dict["stack"].predict(X_valid)),
               weights]
    r2_scores.append(mscores)

    # Convert to table
    scores_df = pd.DataFrame(r2_scores,
                             columns=['model', 'training_set_performance', 'validation_set_performance', "hyperparams"])
    scores_df = scores_df.set_index('model')
    return(scores_df)


# Feature importance based on elastic net
# ------------------------------------------------------------------------------
def enet_importances(imps, sds, fnames, figsize=(5.5, 5), nfeatures=20, color="#8B0000",save=False):

    abs_imp = [abs(x) for x in imps]
    abs_imp, imp, sd, names = zip(*sorted(zip(abs_imp, imps, sds, fnames)))
    imp, sd, names = zip(*sorted(zip(imp[-nfeatures:],
                                     sd[-nfeatures:],
                                     names[-nfeatures:])))

    fig, ax = plt.subplots(figsize=figsize)
    plt.rcParams['axes.axisbelow'] = True
    plt.axvline(x=0, color='black')
    plt.grid(which='major', axis='x')
    plt.barh(range(len(names)), imp, align='center', color=color, alpha=0.5,
             edgecolor="black", xerr=sd, capsize=2)
    plt.yticks(range(len(names)), names)
    plt.grid(which='major', axis='x', linestyle='--', color="#e2e4e0")
    plt.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if save != False:
        plt.savefig(save)
    plt.show()


# Feature importance plot for gradient boosting
# ------------------------------------------------------------------------------
def gboost_importances(imps, sds, fnames, xlabel="Mean variance decrease",
                       figsize=(5.5, 5), nfeatures=20,
                       se=False, color="#8B0000", save=False):

    importances = pd.DataFrame({"importance": imps,
                                "std": sds})
    importances.index = fnames
    importances = importances.sort_values(by="importance")[-nfeatures:]

    fig, ax = plt.subplots(figsize=figsize)
    if se == False:
        importances["importance"].plot.barh(ax=ax, width=0.7, color=color,
                                            alpha=0.5, edgecolor="black")
    elif se == True:
        importances["importance"].plot.barh(ax=ax, width=0.8, color=color,
                                            alpha=0.5, edgecolor="black",
                                            xerr=importances["std"], capsize=2)
    else:
        raise Exception('Please provide a valid input for se (True or False).')
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    ax.set_axisbelow(True)
    plt.grid(which='major', axis='x', linestyle='--', color="#e2e4e0")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if save != False:
        plt.savefig(save)
    plt.show()


# Scatter plot
def scatter(y, y_hat, xlabel, ylabel, color="#8B0000", unit=10, outlier_min=0, outlier_max=0, save=False):

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
    plt.plot(x, b1*x + b0, color="black") # Plot line
    ax.set_axisbelow(True)
    plt.grid(which='major', linestyle='--', color ="#e2e4e0")

    # Customize
    r2 = round(r2_score(y, y_hat), 3)
    plt.xlim([min_x, max_x])
    plt.ylim([min_x, max_x])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.text(min_x+unit/3, np.floor(max_x/unit)*unit-unit/2, f'$R^2$={r2:.3f}')
    if save != False:
        plt.savefig(save)
    plt.show()

# Bubble plot
def bubble(y, y_hat, weights, xlabel, ylabel, color="#8B0000",
           unit=10, lfit=True, eline=False, outlier_min=0, outlier_max=0, save=False):

    # Make scatter plot
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.scatter(y, y_hat, s=weights, alpha=0.12, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    # Add regression line or 45-degree line
    max_x = np.floor(y.nlargest(outlier_max+1).iloc[-1] / unit + 1) * unit - unit / 100
    min_x = np.floor(y.nsmallest(outlier_min+1).iloc[-1] / unit) * unit
    x = np.array([0, max_x])

    if lfit:
        b1, b0 = np.polyfit(y, y_hat, 1) # Estimate slope and intercept
        plt.plot(x, b1*x + b0, color=color, linewidth=1) # Plot line
    if eline:
        plt.plot(x, x, color=color, linewidth=1, alpha=0.2)  # Plot 45 degree line
    ax.set_axisbelow(True)
    plt.grid(which='major', linestyle='--', color ="#e2e4e0")

    # Customize
    r2 = r2_score(y, y_hat)
    w_r2 = r2_score(y, y_hat, sample_weight=weights)
    plt.xlim([min_x, max_x])
    plt.ylim([min_x, max_x])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.text(min_x+unit/3, np.floor(max_x/unit)*unit-unit*0.85, f'$R^2$={r2:.3f}\n'
                                                                f'Weighted $R^2$={w_r2:.3f}')

    if save != False:
        plt.savefig(save)
    plt.show()


def plot_hetero(y, y_hat, fold_nrs, x, xlabel, ylabel="Share of included municipalities",
                cutoffs=range(29), uncertainties=None, n_features=100, color="#8B0000",
                save=False):

    # Set variables
    folds = ["fold0", "fold1", "fold2", "fold3", "fold4"]
    nr_units = len(y)
    max_cutoff = max(cutoffs)

    # Create dataframe with r2, share of included muns and fold by cutoff
    by_cutoff = []
    for cut in cutoffs:
        included_percent = (x >= cut).mean()*100
        y_cut = y[x >= cut]
        y_hat_cut = y_hat[x >= cut]
        fold_r2s = []
        for fold in range(5):
            y_fold = y[(x >= cut) & (fold_nrs == fold)]
            y_hat_fold = y_hat[(x >= cut) & (fold_nrs == fold)]
            fold_r2s.append(r2_score(y_fold, y_hat_fold))
        by_cutoff.append([cut, r2_score(y_cut, y_hat_cut), included_percent] + fold_r2s)

    df = pd.DataFrame(by_cutoff, columns=["cutoff", "r2", "share_muns",
                                          "fold0", "fold1", "fold2", "fold3", "fold4",
                                          ])

    # Plot R2
    fig, ax = plt.subplots()

    ax.plot(df["cutoff"],
            df["r2"],
            color=color,
            marker="o",
            zorder=10,
            clip_on=False)

    # Add uncertainties: standard deviation of fold R2s
    if uncertainties == "fold_sd":
        r2std = df[folds].std(axis=1) #/ 4
        ax.fill_between(df["cutoff"],
                        (df["r2"] + r2std),
                        (df["r2"] - r2std),
                        color=color,
                        alpha=.1)

    # Add uncertainties: quantiles
    if uncertainties == "fold_qs":
        qmin = df[folds].min(axis=1)
        qmax = df[folds].max(axis=1)
        q20 = df[folds].quantile(0.2,
                                 axis=1)
        q80 = df[folds].quantile(0.8,
                                 axis=1)
        ax.fill_between(df["cutoff"],
                        q80,
                        q20,
                        color=color,
                        alpha=.1)
        ax.fill_between(df["cutoff"],
                        qmax,
                        qmin,
                        color=color,
                        alpha=.1)

    # Standard error of r2 (depending on nr of units and features)
    # https://agleontyev.netlify.app/post/2019-09-05-calculating-r-squared-confidence-intervals/
    if uncertainties == "se":
        def se_r2(R2, n, k):
            SE = sqrt((4 * R2 * ((1 - R2) ** 2) * ((n - k - 1) ** 2)) / ((n ** 2 - 1) * (n + 3)))
            return (SE)
        r2se = df.apply(lambda x: se_r2(x["r2"], x["share_muns"]*nr_units/100, n_features), axis=1)
        ax.fill_between(df["cutoff"],
                        (df["r2"] + r2se),
                        (df["r2"] - r2se),
                        color=color,
                        alpha=.1)

    # Customize axes and grid
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Crossvalidation $r^2$")
    ax.set_axisbelow(True)
    ax.grid(which='major',
            linestyle='--',
            color="#e2e4e0")
    ax.set_xlim(0, max_cutoff)
    ax.set_xticks(np.arange(0, max_cutoff,
                            step=round(max_cutoff/10)))
    ax.spines['top'].set_visible(False)

    # Add additional ax with share of included units
    ax2 = ax.twinx()
    ax2.plot(df["cutoff"],
             df["share_muns"],
             color=color,
             alpha=0.4,
             linestyle="--",
             zorder=10,
             clip_on=False)
    ax2.set_ylabel(ylabel)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylim(0, 105)
    ax2.set_yticks(np.arange(0, 101, step=10))

    fig.tight_layout()
    if save != False:
        fig.savefig(save)

    fig.show()
