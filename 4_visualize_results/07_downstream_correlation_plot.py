################################################################################
# Load packages
################################################################################
import pandas as pd
from scipy import stats

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

pd.options.display.width = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


################################################################################
# Load data
################################################################################

# Mexico #######################################################################

# Specify paths
from paths import paths_mx as paths
input_path = paths["results"] + "/"
econ_path = paths["econ_data"] + "/"
exhibits_path = paths["exhibits"] + "/tables/"

pbasic_df = pd.read_csv(input_path + 'pbasic_predictions.csv',
                        sep=';')
primary_df = pd.read_csv(input_path + 'primary_predictions.csv',
                         sep=';')
secondary_df = pd.read_csv(input_path + 'secondary_predictions.csv',
                           sep=';')
yschooling_df = pd.read_csv(input_path + 'yschooling_predictions.csv',
                            sep=';')

wealth_idx = pd.read_csv(econ_path + 'wealth_idx.csv',
                         sep=';',
                         encoding='utf-8')

# United States ################################################################

# Specify paths
from paths import paths_us as paths
input_path = paths["results"] + "/"
econ_path = paths["econ_data"] + "/"
exhibits_path = paths["exhibits"]

us_edu_bach = pd.read_csv(input_path + 'bachelor_predictions.csv',
                          sep=';',
                          dtype={'GEOID': str})
us_edu_some_col = pd.read_csv(input_path + 'some_college_predictions.csv',
                              sep=';',
                              dtype={'GEOID': str})
us_edu_high_school = pd.read_csv(input_path + 'high_school_predictions.csv',
                                 sep=';',
                                 dtype={'GEOID': str})
us_edu_y_schooling = pd.read_csv(input_path + 'yschooling_predictions.csv',
                                 sep=';',
                                 dtype={'GEOID': str})

us_income = pd.read_csv(econ_path + 'income.csv',
                        sep=';',
                        encoding='utf-8',
                        dtype={'county_id_user': str})

################################################################################
# Merge data
################################################################################

# Mexico
mx_pbasic_df = pbasic_df.merge(wealth_idx, on='GEOID').drop(['GEOID'], axis=1)
mx_primary_df = primary_df.merge(wealth_idx, on='GEOID').drop(['GEOID'], axis=1)
mx_secondary_df = secondary_df.merge(wealth_idx, on='GEOID').drop(['GEOID'], axis=1)
mx_yschooling_df = yschooling_df.merge(wealth_idx, on='GEOID').drop(['GEOID', 'fold'], axis=1)

# United States
us_bach_df = us_edu_bach.merge(us_income, how='left', left_on='GEOID', right_on='county_id_user')
us_col_df = us_edu_some_col.merge(us_income, how='left', left_on='GEOID', right_on='county_id_user')
us_high_df = us_edu_high_school.merge(us_income, how='left', left_on='GEOID', right_on='county_id_user')
us_yschooling_df = us_edu_y_schooling.merge(us_income, how='left', left_on='GEOID', right_on='county_id_user')


################################################################################
# Compute correlations between education and wealth variables ##################
################################################################################

# Mexico #######################################################################

mx_predictions = [[mx_yschooling_df, 'Years of School.'],
                  [mx_pbasic_df, 'Post Basic Edu.'],
                  [mx_secondary_df, 'Sec. Edu.'],
                  [mx_primary_df, 'Prim. Edu.']]

mx_corr = []

for edu in mx_predictions:

    y_r = stats.pearsonr(edu[0]['y'],
                         edu[0]['wealth_idx'])
    y_ci = y_r.confidence_interval()

    mx_corr.append([y_r[0],
                    y_ci[0],
                    y_ci[1],
                    edu[1],
                    'y',
                    'MX',
                    'red'])

    y_hat_r = stats.pearsonr(edu[0]['y_hat'],
                             edu[0]['wealth_idx'])
    y_hat_ci = y_hat_r.confidence_interval()

    mx_corr.append([y_hat_r[0],
                    y_hat_ci[0],
                    y_hat_ci[1],
                    edu[1],
                    'ŷ',
                    'MX',
                    'red'])

mx_corr_df = pd.DataFrame(mx_corr,
                          columns=['corr',
                                   'conf_min',
                                   'conf_max',
                                   'edu',
                                   'type',
                                   'ctry',
                                   'color'])



# United States ################################################################

us_predictions = [[us_yschooling_df, 'Years of School.'],
                  [us_bach_df, 'Bach. Degree'],
                  [us_col_df, 'Some College'],
                  [us_high_df, 'High School']]
us_corr = []


for edu in us_predictions:

    edu[0].dropna(inplace=True)

    y_r = stats.pearsonr(edu[0]['y'],
                         edu[0]['income'])
    y_ci = y_r.confidence_interval()

    us_corr.append([y_r[0],
                    y_ci[0],
                    y_ci[1],
                    edu[1],
                    'y',
                    'US',
                    'blue'])

    y_hat_r = stats.pearsonr(edu[0]['y_hat'],
                             edu[0]['income'])
    y_hat_ci = y_hat_r.confidence_interval()

    us_corr.append([y_hat_r[0],
                    y_hat_ci[0],
                    y_hat_ci[1],
                    edu[1],
                    'ŷ',
                    'US',
                    'blue'])

us_corr_df = pd.DataFrame(us_corr,
                          columns=['corr',
                                   'conf_min',
                                   'conf_max',
                                   'edu',
                                   'type',
                                   'ctry',
                                   'color'])


corr_df = pd.concat([mx_corr_df, us_corr_df]).reset_index()


################################################################################
# Plot data
################################################################################
# Distances: 0.5 within, 0.75 between
mx_pos = [0, 0.5,
          1.25, 1.75,
          2.5, 3,
          3.75, 4.25]

us_pos = [5.25, 5.75,
          6.5, 7.0,
          7.75, 8.25,
          9.0, 9.5]

fig, ax = plt.subplots()
plt.grid(which='major', axis='x', linestyle='--', color="#e2e4e0")

for item in mx_corr_df.iterrows():

    color = '#00008B'
    alpha = 1
    if item[1]['type'] == 'ŷ':
        color = '#8080c5'
        alpha = 0.45

    ax.errorbar(mx_pos[item[0]],
                item[1]['corr'],
                fmt='o',
                yerr=item[1]['corr'] - item[1]['conf_min'],
                capsize=3,
                color=color,
                elinewidth=1)

plt.axvline(mx_pos[-1] + 0.5, color="black", linestyle='--', linewidth=0.5)

for item in us_corr_df.iterrows():

    alpha = 1
    color ='#8B0000'
    if item[1]['type'] == 'ŷ':
        color = '#c58080'
        alpha = 0.45

    ax.errorbar(us_pos[item[0]],
                item[1]['corr'],
                fmt='o',
                yerr=item[1]['corr'] - item[1]['conf_min'],
                capsize=3,
                color=color,
                elinewidth=1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.xaxis.set_major_locator(ticker.FixedLocator(mx_pos + us_pos))
ax.xaxis.set_major_formatter(ticker.FixedFormatter(corr_df['edu'] +
                                                   ' - ' +
                                                   corr_df['type']))

plt.xticks(rotation=90)

ax.set_ylim(0.31,
            np.ceil(corr_df['conf_max'].max()*20)/20)

plt.tight_layout()

# Save plot to file
plt.savefig(exhibits_path + '/figures/validity/wealth-income-education-correlations.pdf',
            dpi=350,
            bbox_inches='tight')

plt.show()
