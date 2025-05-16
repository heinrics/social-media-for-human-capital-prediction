
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Function to move columns
def move_col(df, col_name, move_where):
    col_to_move = df[col_name]
    df.pop(col_name)
    col_loc = df.columns.get_loc(move_where)
    df.insert(col_loc, col_name, col_to_move, allow_duplicates=False)

# Function to add small number and then take log
def log_with0(x):
    q25 = x[x > 0].quantile(0.25)
    q75 = x[x > 0].quantile(0.75)
    c = (q25**2)/q75
    return np.log(x+c)
    # https://aosmith.rbind.io/2018/09/19/the-log-0-problem/#common-choices-of-c

# Function to display many histograms in one plot
def make_hists(df, vnr, hnr, height=10, width=7):
    data = df.transpose().to_numpy()
    f, a = plt.subplots(vnr, hnr)
    a = a.ravel()
    cols = df.columns
    for idx, ax in enumerate(a):
        ax.hist(data[idx], bins=50, color="black")
        ax.set_title(cols[idx])
    plt.tight_layout()
    f.set_figheight(height)
    f.set_figwidth(width)
    plt.subplots_adjust(left=0.1,
                        bottom=0.05,
                        right=0.95,
                        top=0.95,
                        wspace=0.4,
                        hspace=0.4)
    plt.show()

# Function to find outliers
def is_outlier(col, c=3, verbose=True):
    lq = col.quantile(0.25)
    uq = col.quantile(0.75)
    iqr = uq - lq
    lower_bound = lq - c * iqr
    upper_bound = uq + c * iqr
    if lower_bound != upper_bound:
        outlier_mask = (col <= lower_bound) | (col >= upper_bound)
        if verbose == True:
            print(f"{col.name}: {outlier_mask.sum()} outliers detected.")
    else:
        outlier_mask = col != col # Select no values
        if verbose == True:
            print(f"{col.name}: Invalid, IQ is 0.")
    return outlier_mask

# Function that imputes predicted values for NAs
def impute_na(df, yvar, Xvars, model, miss_indicator, verbose=True):
    X_nomiss = df.loc[df[miss_indicator] == False, Xvars]
    y_nomiss = df.loc[df[miss_indicator] == False, yvar]
    X_miss = df.loc[df[miss_indicator] == True, Xvars]
    model = model
    result = model.fit(X_nomiss, y_nomiss)
    cv_mean = cross_val_score(model, X_nomiss, y_nomiss, cv=5).mean()
    y_pred = result.predict(X_miss)
    df.loc[df[miss_indicator] == True, yvar] = y_pred
    if verbose == True:
        print(f"Crossvalidation score for {yvar}: {cv_mean}")
    return [yvar, cv_mean]
