
# This was implemented with Python 3.7.6 environment end respective pandas version

################################################################################
# Import modules
################################################################################
import pandas as pd

pd.options.display.width = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


################################################################################
# Load and merge data
################################################################################

# MX ###########################################################################

# Specify data location
from paths import paths_mx as paths
input_path = paths["processed_data"]
edu_path = paths["edu_data"]
output_path = paths["exhibits"] + r"/tables"


# Education
mx_edu = pd.read_excel(edu_path + "/education-data.xlsx",
                       dtype={'GEOID': 'string'})

# Indicators
mx_ind = pd.read_csv(input_path + '/weeks/week_09/mun-level-complete-MX-09.csv',
                     sep=';',
                     encoding='utf-8')

mx_df = mx_ind.merge(mx_edu,
                     how='left',
                     left_on='mun_id_user',
                     right_on='GEOID')

# US ###########################################################################

# Specify data location
from paths import paths_us as paths
input_path = paths["processed_data"]
edu_path = paths["edu_data"]

# Education
us_edu = pd.read_csv(edu_path + '/education-data.csv',
                     sep=';',
                     encoding='utf-8')

# Indicators
us_ind = pd.read_csv(input_path + '/weeks/week_09/county-level-complete-US-09.csv',
                     sep=';',
                     encoding='utf-8')

us_df = us_ind.merge(us_edu,
                     how='left',
                     left_on='county_id_user',
                     right_on='GEOID')

# Variable labels ##############################################################

# Terminal
# var_labels = pd.read_excel('../../05-prediction/_modules/feature_labels.xlsx')
# Console
var_labels = pd.read_excel('_modules/feature_labels.xlsx',
                           usecols = ['col_name', 'label'])

# Remove log_ to match original variables
var_labels.col_name = var_labels.col_name.str.replace('log_', '')

################################################################################
# Preprocess data
################################################################################

# Calculate densities ##########################################################

# MX
mx_df['user_density'] = (mx_df['nr_users']) / mx_df['population'] * 1000
mx_df['tweet_density'] = (mx_df['nr_tweets']) / mx_df['population'] * 1000

# US
us_df['user_density'] = (us_df['nr_users']) / us_df['population'] * 1000
us_df['tweet_density'] = (us_df['nr_tweets']) / us_df['population'] * 1000


# Select counties (>= 5 users) #################################################

# MX
mx_df = mx_df[mx_df['nr_users'] >= 1]
mx_df['unit_count_total'] = mx_df['mun_id_user'].nunique()

# US
us_df = us_df[us_df['nr_users'] >= 1]
us_df['unit_count_total'] = us_df['county_id_user'].nunique()


# Select variables #############################################################

select_feat = ['user_density',
               'tweet_density',
               'avg_chars_tweet',
               'account_age_absolute_year',
               'statuses_count_per_year',
               'favourites_per_statuses',
               'error_per_char',
               'char_GRAMMAR',
               'char_TYPOS',
               'science_&_technology',
               'relationships',
               'positive',
               'offensive',
               'ref_clos_cent']

mx_df = mx_df[['mun_id_user',
               'edu_years_schooling',
               'unit_count_total'] + select_feat]

mx_df[['science_&_technology', 'relationships']] = mx_df[['science_&_technology', 'relationships']] * 100


us_df = us_df[['county_id_user',
               'edu_years_schooling',
               'unit_count_total'] + select_feat]

us_df[['science_&_technology', 'relationships']] = us_df[['science_&_technology', 'relationships']] * 100



################################################################################
# Table MX
################################################################################

p_mx_25_quant = (mx_df.edu_years_schooling <= mx_df.edu_years_schooling.quantile(.25))
mx_df['unit_count_25'] = mx_df.loc[p_mx_25_quant, 'mun_id_user'].nunique()

p_mx_75_quant = (mx_df.edu_years_schooling >= mx_df.edu_years_schooling.quantile(.75))
mx_df['unit_count_75'] = mx_df.loc[p_mx_75_quant, 'mun_id_user'].nunique()


p_mx_25_df = mx_df.loc[p_mx_25_quant, select_feat + ['unit_count_25']] \
    .mean() \
    .rename({'unit_count_25': 'unit_count'})

p_mx_75_df = mx_df.loc[p_mx_75_quant, select_feat + ['unit_count_75']] \
    .mean() \
    .rename({'unit_count_75': 'unit_count'})

p_mx_all_df = mx_df[select_feat + ['unit_count_total']] \
    .mean() \
    .rename({'unit_count_total': 'unit_count'})


################################################################################
# Table percent_edu_high
################################################################################

p_us_25_quant = (us_df.edu_years_schooling <= us_df.edu_years_schooling.quantile(.25))
us_df['unit_count_25'] = us_df.loc[p_us_25_quant, 'county_id_user'].nunique()

p_us_75_quant = (us_df.edu_years_schooling >= us_df.edu_years_schooling.quantile(.75))
us_df['unit_count_75'] = us_df.loc[p_us_75_quant, 'county_id_user'].nunique()

p_us_25_df = us_df.loc[p_us_25_quant, select_feat + ['unit_count_25']] \
    .mean() \
    .rename({'unit_count_25': 'unit_count'})

p_us_75_df = us_df.loc[p_us_75_quant, select_feat + ['unit_count_75']] \
    .mean() \
    .rename({'unit_count_75': 'unit_count'})

p_us_all_df = us_df[select_feat + ['unit_count_total']] \
    .mean() \
    .rename({'unit_count_total': 'unit_count'})


################################################################################
# Final table
################################################################################

desc_df = pd.concat([p_mx_25_df,
                     p_mx_75_df,
                     p_mx_all_df,
                     p_us_25_df,
                     p_us_75_df,
                     p_us_all_df],
                    axis=1) \
    .round(decimals=2) \
    .reset_index()

# Use labels form var table
desc_df = desc_df.merge(var_labels,
                        how='left',
                        left_on='index',
                        right_on='col_name')

desc_df['index'] = desc_df['label']
desc_df.drop(columns=['col_name', 'label'], inplace=True)

desc_df['index'].fillna('Number of Areas', inplace=True)

cols = [('', ''),
        (r'\makecell{Mexico}', r'\makecell{Bottom 25\%}'),
        (r'\makecell{Mexico}', r'\makecell{Top 25\%}'),
        (r'\makecell{Mexico}', r'\makecell{All}'),
        (r'\makecell{United States}', r'\makecell{Bottom 25\%}'),
        (r'\makecell{United States}', r'\makecell{Top 25}\%'),
        (r'\makecell{United States}', r'\makecell{All}')]

# Create multi-level index
desc_df.columns = pd.MultiIndex.from_tuples(cols)

# Thousand separator
desc_df.iloc[:, 1:] = desc_df.iloc[:, 1:].applymap('{:,.2f}'.format)

n_rows, n_cols = desc_df.shape

# Remove decimals for N
desc_df.iloc[-1, 1:] = desc_df.iloc[-1, 1:].str.replace(".00", "")

# Return latex table as list of strings, split by lines
tex_str = desc_df.to_latex(column_format="@{}l" + "r" * (n_cols - 1) + '@{}',
                           multicolumn_format='c',
                           multirow=True,
                           escape=False,
                           index=False) \
    .splitlines()

# Add vertical lines after first headers
tex_str.insert(3, '\cmidrule(lr){2-4}\cmidrule(l){5-7}')
tex_str.insert(8, '\midrule')
tex_str.insert(13, '\midrule')
tex_str.insert(17, '\midrule')
tex_str.insert(20, '\midrule')
tex_str.insert(23, '\midrule')
tex_str.insert(-3, '\midrule')
tex_str.insert(-4, '\midrule')

# Rejoin string
tex_str = '\n'.join(tex_str)

with open(output_path + '/indicator-descriptive-stats.tex', "w") as text_file:
    print(tex_str, file=text_file)
    # text_file.write(tex_str)