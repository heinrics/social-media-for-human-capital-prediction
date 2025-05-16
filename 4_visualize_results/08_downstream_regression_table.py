################################################################################
# Load packages
################################################################################
import pandas as pd
import stata_setup
from sklearn.preprocessing import StandardScaler

pd.options.display.width = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Stata ########################################################################

# Windows example
stata_setup.config("C:/Program Files/Stata18/", "mp")  # Path may need to be adapted !!

# Mac example
# stata_setup.config("/Applications/Stata/", "se")

from pystata import config
from pystata import stata
from sfi import Scalar, Matrix


################################################################################
# Load data
################################################################################

col_names = ['GEOID', 'y', 'y_hat']

# Mexico #######################################################################

# Specify paths
from paths import paths_mx as paths
input_path = paths["results"] + "/"
econ_path = paths["econ_data"] + "/"
exhibits_path = paths["exhibits"] + "/tables/"

# Predictions
pbasic_df = pd.read_csv(input_path + 'pbasic_predictions.csv',
                        sep=';')

secondary_df = pd.read_csv(input_path + 'secondary_predictions.csv',
                           sep=';')

primary_df = pd.read_csv(input_path + 'primary_predictions.csv',
                         sep=';')

yschooling_df = pd.read_csv(input_path + 'yschooling_predictions.csv',
                            sep=';')

# Corrected predictions
pbasic_c_df = pd.read_csv(input_path + 'bias_correction/bias-correction-pbasic-True-15.csv',
                          sep=';')
pbasic_c_df.columns = col_names

secondary_c_df = pd.read_csv(input_path + 'bias_correction/bias-correction-secondary-True-15.csv',
                             sep=';')
secondary_c_df.columns = col_names

primary_c_df = pd.read_csv(input_path + 'bias_correction/bias-correction-primary-True-15.csv',
                           sep=';')
primary_c_df.columns = col_names

yschooling_c_df = pd.read_csv(input_path + 'bias_correction/bias-correction-yschooling-True-3.csv',
                              sep=';')
yschooling_c_df.columns = col_names

# Economic indicator
wealth_idx = pd.read_csv(econ_path + 'wealth_idx.csv',
                         sep=';',
                         encoding='utf-8')


# United States ################################################################

# Specify paths
from paths import paths_us as paths
input_path = paths["results"] + "/"
econ_path = paths["econ_data"] + "/"

# Predictions
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

# Corrected predictions
us_edu_bach_c = pd.read_csv(input_path + 'bias_correction/bias-correction-bachelor-True-15.csv',
                            sep=';',
                            dtype={'GEOID': str})
us_edu_bach_c.columns = col_names
us_edu_bach_c['GEOID'] = us_edu_bach_c['GEOID'].str.pad(5, side='left', fillchar='0')

us_edu_some_col_c = pd.read_csv(input_path + 'bias_correction/bias-correction-some_college-True-15.csv',
                                sep=';',
                                dtype={'GEOID': str})
us_edu_some_col_c.columns = col_names
us_edu_some_col_c['GEOID'] = us_edu_some_col_c['GEOID'].str.pad(5, side='left', fillchar='0')

us_edu_high_school_c = pd.read_csv(input_path + 'bias_correction/bias-correction-high_school-True-15.csv',
                                   sep=';',
                                   dtype={'GEOID': str})
us_edu_high_school_c.columns = col_names
us_edu_high_school_c['GEOID'] = us_edu_high_school_c['GEOID'].str.pad(5, side='left', fillchar='0')

us_edu_y_schooling_c = pd.read_csv(input_path + 'bias_correction/bias-correction-yschooling-True-3.csv',
                                   sep=';',
                                   dtype={'GEOID': str})
us_edu_y_schooling_c.columns = col_names
us_edu_y_schooling_c['GEOID'] = us_edu_y_schooling_c['GEOID'].str.pad(5, side='left', fillchar='0')

# Economic indicator
us_income = pd.read_csv(econ_path + 'income.csv',
                        sep=';',
                        encoding='utf-8',
                        dtype={'county_id_user': str})

us_income['income'] = us_income['income'] / 1000


################################################################################
# Merge data
################################################################################

# Mexico ########################################################################

# Predictions
mx_pbasic_df = pbasic_df.merge(wealth_idx,
                               on='GEOID').drop(['GEOID'], axis=1)
mx_primary_df = primary_df.merge(wealth_idx,
                                 on='GEOID').drop(['GEOID'], axis=1)
mx_secondary_df = secondary_df.merge(wealth_idx,
                                     on='GEOID').drop(['GEOID'], axis=1)
mx_yschooling_df = yschooling_df.merge(wealth_idx,
                                       on='GEOID').drop(['GEOID', 'fold'], axis=1)

# Corrected predictions
mx_pbasic_c_df = pbasic_c_df.merge(wealth_idx,
                                   on='GEOID').drop(['GEOID'], axis=1)
mx_primary_c_df = primary_c_df.merge(wealth_idx,
                                     on='GEOID').drop(['GEOID'], axis=1)
mx_secondary_c_df = secondary_c_df.merge(wealth_idx,
                                         on='GEOID').drop(['GEOID'], axis=1)
mx_yschooling_c_df = yschooling_c_df.merge(wealth_idx,
                                           on='GEOID').drop(['GEOID'], axis=1)

# United States ################################################################

# Predictions
us_bach_df = us_edu_bach.merge(us_income,
                               how='left',
                               left_on='GEOID',
                               right_on='county_id_user')
us_col_df = us_edu_some_col.merge(us_income,
                                  how='left',
                                  left_on='GEOID',
                                  right_on='county_id_user')
us_high_df = us_edu_high_school.merge(us_income,
                                      how='left',
                                      left_on='GEOID',
                                      right_on='county_id_user')
us_yschooling_df = us_edu_y_schooling.merge(us_income,
                                            how='left',
                                            left_on='GEOID',
                                            right_on='county_id_user')

# Corrected predictions
us_bach_c_df = us_edu_bach_c.merge(us_income,
                                   how='left',
                                   left_on='GEOID',
                                   right_on='county_id_user')
us_col_c_df = us_edu_some_col_c.merge(us_income,
                                      how='left',
                                      left_on='GEOID',
                                      right_on='county_id_user')
us_high_c_df = us_edu_high_school_c.merge(us_income,
                                          how='left',
                                          left_on='GEOID',
                                          right_on='county_id_user')
us_yschooling_c_df = us_edu_y_schooling_c.merge(us_income,
                                                how='left',
                                                left_on='GEOID',
                                                right_on='county_id_user')

################################################################################
# Standardize variables
################################################################################

scaler = StandardScaler()

# Mexico #######################################################################

# Post basic prediction
scaler.fit(mx_pbasic_df[['y']].values)
mx_pbasic_df['y'] = scaler.transform(mx_pbasic_df[['y']]).flatten()
mx_pbasic_df['y_hat'] = scaler.transform(mx_pbasic_df[['y_hat']]).flatten()

mx_pbasic_df['wealth_idx'] = scaler.fit_transform(mx_pbasic_df[['wealth_idx']]).flatten()

# Post basic corrected
scaler.fit(mx_pbasic_c_df[['y']].values)
mx_pbasic_c_df['y'] = scaler.transform(mx_pbasic_c_df[['y']]).flatten()
mx_pbasic_c_df['y_hat'] = scaler.transform(mx_pbasic_c_df[['y_hat']]).flatten()

mx_pbasic_c_df['wealth_idx'] = scaler.fit_transform(mx_pbasic_c_df[['wealth_idx']]).flatten()


# Primary predictions
scaler.fit(mx_primary_df[['y']].values)
mx_primary_df['y'] = scaler.transform(mx_primary_df[['y']]).flatten()
mx_primary_df['y_hat'] = scaler.transform(mx_primary_df[['y_hat']]).flatten()

mx_primary_df['wealth_idx'] = scaler.fit_transform(mx_primary_df[['wealth_idx']]).flatten()

# Primary corrected
scaler.fit(mx_primary_c_df[['y']].values)
mx_primary_c_df['y'] = scaler.transform(mx_primary_c_df[['y']]).flatten()
mx_primary_c_df['y_hat'] = scaler.transform(mx_primary_c_df[['y_hat']]).flatten()

mx_primary_c_df['wealth_idx'] = scaler.fit_transform(mx_primary_c_df[['wealth_idx']]).flatten()


# Secondary predictions
scaler.fit(mx_secondary_df[['y']].values)
mx_secondary_df['y'] = scaler.transform(mx_secondary_df[['y']]).flatten()
mx_secondary_df['y_hat'] = scaler.transform(mx_secondary_df[['y_hat']]).flatten()

mx_secondary_df['wealth_idx'] = scaler.fit_transform(mx_secondary_df[['wealth_idx']]).flatten()

# Secondary corrections
scaler.fit(mx_secondary_c_df[['y']].values)
mx_secondary_c_df['y'] = scaler.transform(mx_secondary_c_df[['y']]).flatten()
mx_secondary_c_df['y_hat'] = scaler.transform(mx_secondary_c_df[['y_hat']]).flatten()

mx_secondary_c_df['wealth_idx'] = scaler.fit_transform(mx_secondary_c_df[['wealth_idx']]).flatten()


# Years of schooling predictions
scaler.fit(mx_yschooling_df[['y']].values)
mx_yschooling_df['y'] = scaler.transform(mx_yschooling_df[['y']]).flatten()
mx_yschooling_df['y_hat'] = scaler.transform(mx_yschooling_df[['y_hat']]).flatten()

mx_yschooling_df['wealth_idx'] = scaler.fit_transform(mx_yschooling_df[['wealth_idx']]).flatten()


# Years of schooling corrections
scaler.fit(mx_yschooling_c_df[['y']].values)
mx_yschooling_c_df['y'] = scaler.transform(mx_yschooling_c_df[['y']]).flatten()
mx_yschooling_c_df['y_hat'] = scaler.transform(mx_yschooling_c_df[['y_hat']]).flatten()

mx_yschooling_c_df['wealth_idx'] = scaler.fit_transform(mx_yschooling_c_df[['wealth_idx']]).flatten()


# United States ################################################################

# Bachelor predictions
scaler.fit(us_bach_df[['y']].values)
us_bach_df['y'] = scaler.transform(us_bach_df[['y']]).flatten()
us_bach_df['y_hat'] = scaler.transform(us_bach_df[['y_hat']]).flatten()

us_bach_df['income'] = scaler.fit_transform(us_bach_df[['income']]).flatten()

# Bachelor corrections
scaler.fit(us_bach_c_df[['y']].values)
us_bach_c_df['y'] = scaler.transform(us_bach_c_df[['y']]).flatten()
us_bach_c_df['y_hat'] = scaler.transform(us_bach_c_df[['y_hat']]).flatten()

us_bach_c_df['income'] = scaler.fit_transform(us_bach_c_df[['income']]).flatten()


# Some collage predictions
scaler.fit(us_col_df[['y']].values)
us_col_df['y'] = scaler.transform(us_col_df[['y']]).flatten()
us_col_df['y_hat'] = scaler.transform(us_col_df[['y_hat']]).flatten()

us_col_df['income'] = scaler.fit_transform(us_col_df[['income']]).flatten()

# Some collage corrections
scaler.fit(us_col_c_df[['y']].values)
us_col_c_df['y'] = scaler.transform(us_col_c_df[['y']]).flatten()
us_col_c_df['y_hat'] = scaler.transform(us_col_c_df[['y_hat']]).flatten()

us_col_c_df['income'] = scaler.fit_transform(us_col_c_df[['income']]).flatten()


# High school predictions
scaler.fit(us_high_df[['y']].values)
us_high_df['y'] = scaler.transform(us_high_df[['y']]).flatten()
us_high_df['y_hat'] = scaler.transform(us_high_df[['y_hat']]).flatten()

us_high_df['income'] = scaler.fit_transform(us_high_df[['income']]).flatten()


# High school predictions
scaler.fit(us_high_c_df[['y']].values)
us_high_c_df['y'] = scaler.transform(us_high_c_df[['y']]).flatten()
us_high_c_df['y_hat'] = scaler.transform(us_high_c_df[['y_hat']]).flatten()

us_high_c_df['income'] = scaler.fit_transform(us_high_c_df[['income']]).flatten()


# Years of schooling predictions
scaler.fit(us_yschooling_df[['y']].values)
us_yschooling_df['y'] = scaler.transform(us_yschooling_df[['y']]).flatten()
us_yschooling_df['y_hat'] = scaler.transform(us_yschooling_df[['y_hat']]).flatten()

us_yschooling_df['income'] = scaler.fit_transform(us_yschooling_df[['income']]).flatten()

# Years of schooling corrections
scaler.fit(us_yschooling_c_df[['y']].values)
us_yschooling_c_df['y'] = scaler.transform(us_yschooling_c_df[['y']]).flatten()
us_yschooling_c_df['y_hat'] = scaler.transform(us_yschooling_c_df[['y_hat']]).flatten()

us_yschooling_c_df['income'] = scaler.fit_transform(us_yschooling_c_df[['income']]).flatten()



################################################################################
# Regression Analysis
################################################################################

rows = ['b',
        'se',
        't',
        'pvalue',
        'll',
        'ul',
        'df',
        'crit',
        'eform']

cols = ['b1', 'b0']


def sig_level(p_val):

    if p_val < 0.01:
        return '*' * 3
    elif p_val < 0.05:
        return '*' * 2
    elif p_val < 0.1:
        return '*' * 1
    else:
        return ''


def stata_reg(command_str, cols=['b1']):

    stata.run(command_str)
    df = pd.DataFrame(Matrix.get('r(table)'), columns=cols)
    df['ind'] = rows

    coef = df[df.ind.isin(['b', 'se', 'pvalue'])]['b1'].reset_index(drop=True)
    se = format(round(coef[1], 3), '.3f')
    coef = format(round(coef[0], 3), '.3f') # + sig_level(coef[2])
    coef = '\\makecell{' + coef + '\\\\(' + se + ')}'

    return coef



################################################################################
# MEXICO #######################################################################
################################################################################


################################################################################
# Years of Schooling ###########################################################
################################################################################

yschool_mx = []

# edu ~ wealth #################################################################
################################################################################

# Predicted ####################################################################
stata.pdataframe_to_data(mx_yschooling_df, True)
# stata.run('''est sto clear''')

yschool_mx.append(stata_reg(''' reg y wealth_idx
                                estimates store model1 ''',
                            cols=cols))

yschool_mx.append(stata_reg(''' reg y_hat wealth_idx
                                estimates store model2 ''',
                            cols=cols))

yschool_mx.append(stata_reg(''' suest model1 model2
                                nlcom (_b[model2_mean:wealth_idx] - _b[model1_mean:wealth_idx]) '''))

# Corrected ####################################################################
stata.pdataframe_to_data(mx_yschooling_c_df, True)
stata.run('''est sto clear''')

stata_reg(''' reg y wealth_idx
              estimates store model1 ''',
          cols=cols)

yschool_mx.insert(2, stata_reg(''' reg y_hat wealth_idx
                                   estimates store model2 ''',
                               cols=cols))

yschool_mx.append(stata_reg(''' suest model1 model2
                                nlcom (_b[model2_mean:wealth_idx] - _b[model1_mean:wealth_idx]) '''))


# wealth ~ edu #################################################################
################################################################################

# Predicted ####################################################################
stata.pdataframe_to_data(mx_yschooling_df, True)
stata.run(''' est sto  clear ''')

yschool_mx.append(stata_reg(''' reg wealth_idx y
                                estimates store model1 ''',
                            cols=cols))

yschool_mx.append(stata_reg(''' reg wealth_idx y_hat
                                estimates store model2 ''',
                            cols=cols))

yschool_mx.append(stata_reg(''' suest model1 model2
                                nlcom (_b[model2_mean:y_hat] - _b[model1_mean:y]) '''))

# Corrected ####################################################################
stata.pdataframe_to_data(mx_yschooling_c_df, True)
stata.run('''est sto clear''')

stata_reg(''' reg wealth_idx y
              estimates store model1 ''',
          cols=cols)

yschool_mx.insert(7, stata_reg(''' reg wealth_idx y_hat
                                   estimates store model2 ''',
                               cols=cols))

yschool_mx.append(stata_reg(''' suest model1 model2
                                nlcom (_b[model2_mean:y_hat] - _b[model1_mean:y]) '''))



################################################################################
# Post basic ###################################################################
################################################################################

pbasic_mx = []

# edu ~ wealth #################################################################
################################################################################

# Predicted ####################################################################
stata.pdataframe_to_data(mx_pbasic_df, True)
stata.run(''' est sto  clear ''')

pbasic_mx.append(stata_reg(''' reg y wealth_idx
                               estimates store model1 ''',
                           cols=cols))

pbasic_mx.append(stata_reg(''' reg y_hat wealth_idx
                                estimates store model2 ''',
                           cols=cols))

pbasic_mx.append(stata_reg(''' suest model1 model2
                               nlcom (_b[model2_mean:wealth_idx] - _b[model1_mean:wealth_idx]) '''))


stata.pdataframe_to_data(mx_pbasic_c_df, True)
stata.run(''' est sto  clear ''')

stata_reg(''' reg y wealth_idx
              estimates store model1 ''',
          cols=cols)

pbasic_mx.insert(2, stata_reg(''' reg y_hat wealth_idx
                                  estimates store model2 ''',
                              cols=cols))

pbasic_mx.append(stata_reg(''' suest model1 model2
                               nlcom (_b[model2_mean:wealth_idx] - _b[model1_mean:wealth_idx]) '''))


# wealth ~ edu #################################################################
################################################################################

# Predicted ####################################################################
stata.pdataframe_to_data(mx_pbasic_df, True)
stata.run(''' est sto  clear ''')

pbasic_mx.append(stata_reg(''' reg wealth_idx y
                               estimates store model1 ''',
                           cols=cols))

pbasic_mx.append(stata_reg(''' reg wealth_idx y_hat
                               estimates store model2 ''',
                           cols=cols))

pbasic_mx.append(stata_reg(''' suest model1 model2
                               nlcom (_b[model2_mean:y_hat] - _b[model1_mean:y]) '''))


stata.pdataframe_to_data(mx_pbasic_c_df, True)
stata.run(''' est sto  clear ''')


stata_reg(''' reg wealth_idx y
              estimates store model1 ''',
          cols=cols)

pbasic_mx.insert(7, stata_reg(''' reg wealth_idx y_hat
                                  estimates store model2 ''',
                              cols=cols))

pbasic_mx.append(stata_reg(''' suest model1 model2
                               nlcom (_b[model2_mean:y_hat] - _b[model1_mean:y]) '''))


################################################################################
# Secondary ####################################################################
################################################################################

second_mx = []

# edu ~ wealth #################################################################
################################################################################

# Predicted ####################################################################
stata.pdataframe_to_data(mx_secondary_df, True)
stata.run(''' est sto  clear ''')

second_mx.append(stata_reg(''' reg y wealth_idx
                               estimates store model1 ''',
                           cols=cols))

second_mx.insert(2, stata_reg(''' reg y_hat wealth_idx
                                  estimates store model2 ''',
                              cols=cols))

second_mx.append(stata_reg(''' suest model1 model2
                               nlcom (_b[model2_mean:wealth_idx] - _b[model1_mean:wealth_idx]) '''))


stata.pdataframe_to_data(mx_secondary_c_df, True)
stata.run(''' est sto  clear ''')

stata_reg(''' reg y wealth_idx
              estimates store model1 ''',
          cols=cols)

second_mx.insert(2, stata_reg(''' reg y_hat wealth_idx
                                  estimates store model2 ''',
                              cols=cols))


second_mx.append(stata_reg(''' suest model1 model2
                               nlcom (_b[model2_mean:wealth_idx] - _b[model1_mean:wealth_idx]) '''))


# wealth ~ edu #################################################################
################################################################################

# Predicted ####################################################################
stata.pdataframe_to_data(mx_secondary_df, True)
stata.run(''' est sto  clear ''')

second_mx.append(stata_reg(''' reg wealth_idx y
                               estimates store model1 ''',
                           cols=cols))

second_mx.append(stata_reg(''' reg wealth_idx y_hat
                               estimates store model2 ''',
                           cols=cols))

second_mx.append(stata_reg(''' suest model1 model2
                               nlcom (_b[model2_mean:y_hat] - _b[model1_mean:y]) '''))

stata.pdataframe_to_data(mx_secondary_c_df, True)
stata.run(''' est sto  clear ''')

stata_reg(''' reg wealth_idx y
              estimates store model1 ''',
          cols=cols)

second_mx.insert(7, stata_reg(''' reg wealth_idx y_hat
                                  estimates store model2 ''',
                              cols=cols))

second_mx.append(stata_reg(''' suest model1 model2
                               nlcom (_b[model2_mean:y_hat] - _b[model1_mean:y]) '''))


################################################################################
# Primary ######################################################################
################################################################################

prim_mx = []

# edu ~ wealth #################################################################
################################################################################

# Predicted ####################################################################
stata.pdataframe_to_data(mx_primary_df, True)
stata.run(''' est sto  clear ''')

prim_mx.append(stata_reg(''' reg y wealth_idx
                               estimates store model1 ''',
                         cols=cols))

prim_mx.append(stata_reg(''' reg y_hat wealth_idx
                                estimates store model2 ''',
                         cols=cols))

prim_mx.append(stata_reg(''' suest model1 model2
                               nlcom (_b[model2_mean:wealth_idx] - _b[model1_mean:wealth_idx]) '''))


stata.pdataframe_to_data(mx_primary_c_df, True)
stata.run(''' est sto  clear ''')

stata_reg(''' reg y wealth_idx
              estimates store model1 ''',
          cols=cols)

prim_mx.insert(2, stata_reg(''' reg y_hat wealth_idx
                                estimates store model2 ''',
                            cols=cols))

prim_mx.append(stata_reg(''' suest model1 model2
                             nlcom (_b[model2_mean:wealth_idx] - _b[model1_mean:wealth_idx]) '''))



# wealth ~ edu #################################################################
################################################################################

# Predicted ####################################################################
stata.pdataframe_to_data(mx_primary_df, True)
stata.run(''' est sto  clear ''')

prim_mx.append(stata_reg(''' reg wealth_idx y
                               estimates store model1 ''',
                         cols=cols))

prim_mx.append(stata_reg(''' reg wealth_idx y_hat
                               estimates store model2 ''',
                         cols=cols))

prim_mx.append(stata_reg(''' suest model1 model2
                               nlcom (_b[model2_mean:y_hat] - _b[model1_mean:y]) '''))


stata.pdataframe_to_data(mx_primary_c_df, True)
stata.run(''' est sto  clear ''')

stata_reg(''' reg wealth_idx y
              estimates store model1 ''',
          cols=cols)

prim_mx.insert(7, stata_reg(''' reg wealth_idx y_hat
                                estimates store model2 ''',
                            cols=cols))

prim_mx.append(stata_reg(''' suest model1 model2
                             nlcom (_b[model2_mean:y_hat] - _b[model1_mean:y]) '''))



################################################################################
# UNITED STATES ################################################################
################################################################################


################################################################################
# Years of Schooling ###########################################################
################################################################################

yschool_us = []

# edu ~ wealth #################################################################
################################################################################

# Predicted ####################################################################
stata.pdataframe_to_data(us_yschooling_df, True)
stata.run(''' est sto  clear ''')

yschool_us.append(stata_reg(''' reg y income
                                estimates store model1 ''',
                            cols=cols))

yschool_us.append(stata_reg(''' reg y_hat income
                                estimates store model2 ''',
                            cols=cols))

yschool_us.append(stata_reg(''' suest model1 model2
                                nlcom (_b[model2_mean:income] - _b[model1_mean:income]) '''))


stata.pdataframe_to_data(us_yschooling_c_df, True)
stata.run(''' est sto  clear ''')

stata_reg(''' reg y income
              estimates store model1 ''',
          cols=cols)

yschool_us.insert(2, stata_reg(''' reg y_hat income
                                   estimates store model2 ''',
                               cols=cols))

yschool_us.append(stata_reg(''' suest model1 model2
                                nlcom (_b[model2_mean:income] - _b[model1_mean:income]) '''))


# wealth ~ edu #################################################################
################################################################################

# Predicted ####################################################################
stata.pdataframe_to_data(us_yschooling_df, True)
stata.run(''' est sto  clear ''')

yschool_us.append(stata_reg(''' reg income y
                                estimates store model1 ''',
                            cols=cols))

yschool_us.append(stata_reg(''' reg income y_hat
                                estimates store model2 ''',
                            cols=cols))

yschool_us.append(stata_reg(''' suest model1 model2
                                nlcom (_b[model2_mean:y_hat] - _b[model1_mean:y]) '''))


stata.pdataframe_to_data(us_yschooling_c_df, True)
stata.run(''' est sto  clear ''')

stata_reg(''' reg income y
              estimates store model1 ''',
          cols=cols)

yschool_us.insert(7, stata_reg(''' reg income y_hat
                                   estimates store model2 ''',
                               cols=cols))

yschool_us.append(stata_reg(''' suest model1 model2
                                nlcom (_b[model2_mean:y_hat] - _b[model1_mean:y]) '''))



################################################################################
# Bachelor #####################################################################
################################################################################

bach_us = []

# edu ~ wealth #################################################################
################################################################################

# Predicted ####################################################################
stata.pdataframe_to_data(us_bach_df, True)
stata.run(''' est sto  clear ''')

bach_us.append(stata_reg(''' reg y income
                             estimates store model1 ''',
                         cols=cols))

bach_us.append(stata_reg(''' reg y_hat income
                             estimates store model2 ''',
                         cols=cols))


bach_us.append(stata_reg(''' suest model1 model2
                             nlcom (_b[model2_mean:income] - _b[model1_mean:income]) '''))


stata.pdataframe_to_data(us_bach_c_df, True)
stata.run(''' est sto  clear ''')

stata_reg(''' reg y income
              estimates store model1 ''',
          cols=cols)

bach_us.insert(2, stata_reg(''' reg y_hat income
                                estimates store model2 ''',
                            cols=cols))

bach_us.append(stata_reg(''' suest model1 model2
                             nlcom (_b[model2_mean:income] - _b[model1_mean:income]) '''))



# wealth ~ edu #################################################################
################################################################################

# Predicted ####################################################################
stata.pdataframe_to_data(us_bach_df, True)
stata.run(''' est sto  clear ''')

bach_us.append(stata_reg(''' reg income y
                               estimates store model1 ''',
                         cols=cols))

bach_us.append(stata_reg(''' reg income y_hat
                             estimates store model2 ''',
                         cols=cols))


bach_us.append(stata_reg(''' suest model1 model2
                             nlcom (_b[model2_mean:y_hat] - _b[model1_mean:y]) '''))


stata.pdataframe_to_data(us_bach_c_df, True)
stata.run(''' est sto  clear ''')

stata_reg(''' reg income y
              estimates store model1 ''',
          cols=cols)

bach_us.insert(7, stata_reg(''' reg income y_hat
                                estimates store model2 ''',
                            cols=cols))

bach_us.append(stata_reg(''' suest model1 model2
                             nlcom (_b[model2_mean:y_hat] - _b[model1_mean:y]) '''))


################################################################################
# College ######################################################################
################################################################################

some_col_us = []

# edu ~ wealth #################################################################
################################################################################

# Predicted ####################################################################
stata.pdataframe_to_data(us_col_df, True)
stata.run(''' est sto  clear ''')

some_col_us.append(stata_reg(''' reg y income
                               estimates store model1 ''',
                             cols=cols))

some_col_us.append(stata_reg(''' reg y_hat income
                                estimates store model2 ''',
                             cols=cols))

some_col_us.append(stata_reg(''' suest model1 model2
                                 nlcom (_b[model2_mean:income] - _b[model1_mean:income]) '''))

stata.pdataframe_to_data(us_col_c_df, True)
stata.run(''' est sto  clear ''')

stata_reg(''' reg y income
              estimates store model1 ''',
          cols=cols)

some_col_us.insert(2, stata_reg(''' reg y_hat income
                                    estimates store model2 ''',
                                cols=cols))

some_col_us.append(stata_reg(''' suest model1 model2
                                 nlcom (_b[model2_mean:income] - _b[model1_mean:income]) '''))


# wealth ~ edu #################################################################
################################################################################

# Predicted ####################################################################
stata.pdataframe_to_data(us_col_df, True)
stata.run(''' est sto  clear ''')

some_col_us.append(stata_reg(''' reg income y
                                 estimates store model1 ''',
                             cols=cols))

some_col_us.append(stata_reg(''' reg income y_hat
                                 estimates store model2 ''',
                             cols=cols))

some_col_us.append(stata_reg(''' suest model1 model2
                                 nlcom (_b[model2_mean:y_hat] - _b[model1_mean:y]) '''))


stata.pdataframe_to_data(us_col_c_df, True)
stata.run(''' est sto  clear ''')

stata_reg(''' reg income y
              estimates store model1 ''',
          cols=cols)

some_col_us.insert(7, stata_reg(''' reg income y_hat
                                    estimates store model2 ''',
                                cols=cols))

some_col_us.append(stata_reg(''' suest model1 model2
                                 nlcom (_b[model2_mean:y_hat] - _b[model1_mean:y]) '''))



################################################################################
# High School ##################################################################
################################################################################

high_us = []

# edu ~ wealth #################################################################
################################################################################

# Predicted ####################################################################
stata.pdataframe_to_data(us_high_df, True)
stata.run(''' est sto  clear ''')

high_us.append(stata_reg(''' reg y income
                             estimates store model1 ''',
                         cols=cols))

high_us.append(stata_reg(''' reg y_hat income
                             estimates store model2 ''',
                         cols=cols))

high_us.append(stata_reg(''' suest model1 model2
                             nlcom (_b[model2_mean:income] - _b[model1_mean:income]) '''))

stata.pdataframe_to_data(us_high_c_df, True)
stata.run(''' est sto  clear ''')

stata_reg(''' reg y income
              estimates store model1 ''',
          cols=cols)

high_us.insert(2, stata_reg(''' reg y_hat income
                                estimates store model2 ''',
                            cols=cols))

high_us.append(stata_reg(''' suest model1 model2
                             nlcom (_b[model2_mean:income] - _b[model1_mean:income]) '''))

# wealth ~ edu #################################################################
################################################################################

# Predicted ####################################################################
stata.pdataframe_to_data(us_high_df, True)
stata.run(''' est sto  clear ''')

high_us.append(stata_reg(''' reg income y
                             estimates store model1 ''',
                         cols=cols))

high_us.append(stata_reg(''' reg income y_hat
                             estimates store model2 ''',
                         cols=cols))

high_us.append(stata_reg(''' suest model1 model2
                             nlcom (_b[model2_mean:y_hat] - _b[model1_mean:y]) '''))


stata.pdataframe_to_data(us_high_c_df, True)
stata.run(''' est sto  clear ''')

stata_reg(''' reg income y
              estimates store model1 ''',
          cols=cols)

high_us.insert(7, stata_reg(''' reg income y_hat
                                estimates store model2 ''',
                            cols=cols))

high_us.append(stata_reg(''' suest model1 model2
                             nlcom (_b[model2_mean:y_hat] - _b[model1_mean:y]) '''))




################################################################################
# Joint result table ###########################################################
################################################################################
row_names = ['$\\beta_t\!:\, edu \sim  econ $',
             '$\\beta_p\!:\, \widehat{edu} \sim econ $',
             '$\\beta_c\!:\, \widehat{edu_c} \sim econ $',
             '$\\beta_t - \\beta_p$',
             '$\\beta_t - \\beta_c$',
             '$\\beta_t\!:\, econ \sim edu$',
             '$\\beta_p\!:\, econ \sim \widehat{edu}$',
             '$\\beta_c\!:\, econ \sim \widehat{edu_c}$',
             '$\\beta_t - \\beta_p$',
             '$\\beta_t - \\beta_c$']

n_obs = ['N'] + ['\makecell{2,457}'] * 4 + ['\makecell{3,140}'] * 4
n_obs = pd.DataFrame.from_records([n_obs])

reg_table = pd.DataFrame.from_records([row_names,
                                       yschool_mx,
                                       pbasic_mx,
                                       second_mx,
                                       prim_mx,
                                       yschool_us,
                                       bach_us,
                                       some_col_us,
                                       high_us]).T

reg_table = pd.concat([reg_table, n_obs], axis=0)

multi_cols = [('', ''),
              (r'\makecell{Mexico}', r'\makecell{Years of\\Schooling}'),
              (r'\makecell{Mexico}', r'\makecell{Post-\\Basic}'),
              (r'\makecell{Mexico}', r'\makecell{Secondary}'),
              (r'\makecell{Mexico}', r'\makecell{Primary}'),
              (r'\makecell{United States}', r'\makecell{Years of\\Schooling}'),
              (r'\makecell{United States}', r'\makecell{Bachelor}'),
              (r'\makecell{United States}', r'\makecell{College}'),
              (r'\makecell{United States}', r'\makecell{High\\School}')]

# Create multi-level index
reg_table.columns = pd.MultiIndex.from_tuples(multi_cols)

n_rows, n_cols = reg_table.shape

tex_str = reg_table.to_latex(column_format="@{}l" + "r" * (n_cols - 1)  + '@{}',
                             escape=False,
                             multicolumn=True,
                             multicolumn_format='c',
                             index=False) \
    .splitlines()

tex_str.insert(3, '\cmidrule(l){2-5}\cmidrule(l){6-9}')
tex_str.insert(9, '\cmidrule(l){2-9}')
tex_str.insert(12, '\midrule')
tex_str.insert(16, '\cmidrule(l){2-9}')
tex_str.insert(19, '\midrule')

tex_str = '\n'.join(tex_str)

# Export
with open(exhibits_path + 'edu-econ-reg-table.tex', "w") as text_file:
    print(tex_str, file=text_file)
