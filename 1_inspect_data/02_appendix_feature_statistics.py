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
econ_path = paths["econ_data"]
output_path = paths["exhibits"] + r"/tables/appendix/feature_statistics/"


# Education
mx_edu = pd.read_excel(edu_path + "/education-data.xlsx",
                       dtype={'GEOID': 'string'})

# Wealth index
wealth_idx_mx = pd.read_csv(econ_path + '/wealth_idx.csv',
                            sep=';',
                            encoding='utf-8')

# Indicators
mx_ind = pd.read_csv(input_path + '/weeks/week_09/mun-level-complete-MX-09.csv',
                     sep=';',
                     encoding='utf-8')

mx_df = mx_ind.merge(mx_edu,
                     how='left',
                     left_on='mun_id_user',
                     right_on='GEOID')

# Survey
survey_mx = mx_ind[['mun_id_user', 'population']] \
    .merge(wealth_idx_mx,
           left_on='mun_id_user',
           right_on='GEOID',
           how='left') \
    .merge(mx_edu,
           left_on='mun_id_user',
           right_on='GEOID',
           how='left').drop(columns=['GEOID_x',
                                     'GEOID_y',
                                     'NAME']) \
    [['edu_years_schooling',
      'edu_postbasic',
      'edu_secondary',
      'edu_primary',
      'population',
      'wealth_idx']]


# US ###########################################################################


# Specify data location
from paths import paths_us as paths
input_path = paths["processed_data"]
edu_path = paths["edu_data"]
econ_path = paths["econ_data"]

# Education
us_edu = pd.read_csv(edu_path + '/education-data.csv',
                     sep=';',
                     encoding='utf-8')

# Income
us_income = pd.read_csv(econ_path + '/income.csv',
                        sep=';',
                        encoding='utf-8')

# Indicators
us_ind = pd.read_csv(input_path + '/weeks/week_09/county-level-complete-US-09.csv',
                     sep=';',
                     encoding='utf-8')

us_ind.rename(columns={'char_CONFUSED_WORDS': 'char_CONFUSIONS'}, inplace=True)

us_df = us_ind.merge(us_edu,
                     how='left',
                     left_on='county_id_user',
                     right_on='GEOID')

# Survey
survey_us = us_ind[['county_id_user', 'population']]\
    .merge(us_income,
           on='county_id_user',
           how='left') \
    .merge(us_edu,
           left_on='county_id_user',
           right_on='GEOID',
           how='left').drop(columns=['GEOID', 'NAME'])\
    [['edu_years_schooling',
      'edu_bachelor',
      'edu_some_college',
      'edu_high',
      'population',
      'income']]


# Variable labels ##############################################################

# Terminal
var_labels = pd.read_excel('_modules/feature_labels.xlsx')

# Remove log_ to match original variables
var_labels.col_name = var_labels.col_name.str.replace('log_', '')


# Define variable groups for tables ############################################

tweet_stats_mx_us_1 = ['nr_tweets', 'nr_users',
                       'share_weekdays', 'share_workhours',
                       'median_followers', 'median_friends',
                       'median_statuses',
                       'avg_mobility',
                       'share_iphone', 'share_insta',
                       'favourites_per_statuses',
                       'statuses_count_per_year',
                       'account_age_absolute_year']

tweet_stats_mx_us_2 = ['account_age_absolute_year',
                       'listed_count',
                       'followers_per_friends',
                       'is_quote_status_mean',
                       'is_reply_mean',
                       'share_verified',
                       'avg_chars_tweet',
                       'hashtag_per_tweet', 'mention_per_tweet',
                       'url_per_tweet', 'emoji_per_tweet']

topics_1 = ['arts_&_culture',
            'business_&_entrepreneurs',
            'celebrity_&_pop_culture',
            'diaries_&_daily_life',
            'family',
            'fashion_&_style',
            'film_tv_&_video',
            'fitness_&_health',
            'food_&_dining',
            'gaming']

topics_2 = ['learning_&_educational',
            'music',
            'news_&_social_concern',
            'other_hobbies',
            'relationships',
            'science_&_technology',
            'sports',
            'travel_&_adventure',
            'youth_&_student_life']


errors_mx = ['error_per_char',
             'char_AGREEMENT_NOUNS',
             'char_AGREEMENT_VERBS',
             'char_CAMBIOS_NORMA',
             'char_CASING',
             'char_CONFUSIONS',
             'char_CONTEXT',
             'char_DIACRITICS',
             'char_GRAMMAR',
             'char_INCORRECT_EXPRESSIONS',
             'char_LANGUAGE_VARIANTS',
             'char_MISC',
             'char_MISSPELLING',
             'char_PREPOSITIONS',
             'char_PROPER_NOUNS',
             'char_PUNCTUATION',
             'char_REDUNDANCIES',
             'char_REPETITIONS',
             'char_REPETITIONS_STYLE',
             'char_SEMANTICS',
             'char_STYLE',
             'char_TYPOGRAPHY',
             'char_TYPOS']

errors_us = ['error_per_char',
             'char_LANGUAGE_VARIANTS',
             'char_CASING',
             'char_COLLOCATIONS',
             'char_COMPOUNDING',
             'char_CONFUSIONS',
             'char_GRAMMAR',
             'char_MISC',
             'char_NONSTANDARD_PHRASES',
             'char_PUNCTUATION',
             'char_REDUNDANCY',
             'char_REPETITIONS_STYLE',
             'char_SEMANTICS',
             'char_STYLE',
             'char_TYPOGRAPHY',
             'char_TYPOS']

errors_shared = list(set(errors_mx) & set(errors_us))
errors_mx = list(set(errors_mx) - set(errors_shared))
errors_us = list(set(errors_us) - set(errors_shared))

sent_mx_us = ['negative',
              'positive',
              'hate',
              'offensive']

network_mx_us = ['ref_in_deg',
                 'ref_out_deg',
                 'ref_clos_cent',
                 'ref_pagerank']


################################################################################
# Generate Variable Statistics Tables
################################################################################

# Base table generation ########################################################
def generate_table(table_mx, table_us):

    # US
    table_us = table_us.describe() \
        .transpose() \
        .reset_index() \
        .rename(columns={'index': 'Variables'}) \
        .drop(columns=['25%', '75%'])

    table_us['Country'] = 'US'

    # Merge variable names
    table_us = table_us.merge(var_labels,
                              how='left',
                              left_on='Variables',
                              right_on='col_name')

    table_us['label'] = table_us['label'] .str.replace('&', '\&')

    # MX
    table_mx = table_mx.describe() \
        .transpose() \
        .reset_index() \
        .rename(columns={'index': 'Variables'}) \
        .drop(columns=['25%', '75%'])

    table_mx['Country'] = 'MX'

    # Merge variable names
    table_mx = table_mx.merge(var_labels,
                              how='left',
                              left_on='Variables',
                              right_on='col_name')

    table_mx['label'] = table_mx['label'] .str.replace('&', '\&')

    joint_table = pd.concat([table_mx, table_us], ignore_index=True) \
        .sort_index() \
        [['label',
          'Country',
          'mean',
          'std',
          'min',
          '50%',
          'max']]

    # Thousand separator
    joint_table.loc[:, 'mean':'max'] = joint_table.loc[:,
                                       'mean':'max'].applymap('{:,.2f}'.format)

    joint_table.columns = ['Variable',
                           'Country',
                           'Mean',
                           'SD',
                           'Min.',
                           'Median',
                           'Max.']

    return joint_table

def generate_latex(df):

    df.index = pd.MultiIndex.from_frame(df[['Variable',
                                            'Country']])

    df.drop(columns=['Variable', 'Country'], inplace=True)

    tweets_stats_table = df.round(decimals=2)

    n_rows, n_cols = tweets_stats_table.shape

    tex_str = tweets_stats_table.to_latex(column_format="@{}lc" + "r" * n_cols + '@{}',
                                          escape=False,
                                          multicolumn=True,
                                          multirow=True,
                                          multicolumn_format='c') \
        .splitlines()

    tex_str = [row if row != '\\cline{1-7}' else '\midrule' for row in tex_str]
    tex_str = [row if not row.startswith('\\multirow{2}{*}') else row.replace('\\\\', '\\\\ \cmidrule(l){3-7}') for row in tex_str]

    tex_str[2] = 'Variable & Country & Mean & SD & Min & Median & Max \\\\'
    tex_str.pop(3)

    tex_str = '\n'.join(tex_str)

    return tex_str


################################################################################
# Survey Latex Table ###########################################################
################################################################################

survey_table = generate_table(survey_mx, survey_us)

survey_table.loc[survey_table['Variable'] == 'Years of Schooling', 'Variable'] = '\multirow{2}{*}{Years of Schooling}'
survey_table.loc[survey_table['Variable'] == 'Population', 'Variable'] = '\multirow{2}{*}{Population}'


survey_table.index = pd.MultiIndex.from_frame(survey_table[['Variable',
                                                            'Country']])

survey_table.drop(columns=['Variable', 'Country'],
                  inplace=True)

survey_table = survey_table.round(decimals=2)

n_rows, n_cols = survey_table.shape

tex_str = survey_table.to_latex(column_format="@{}lc" + "r" * n_cols + '@{}',
                                escape=False,
                                multicolumn=True,
                                multicolumn_format='c') \
    .splitlines()

tex_str[2] = 'Variable & Country & Mean & SD & Min & Median & Max \\\\'
tex_str.pop(3)

tex_str.insert(5, '\cmidrule(l){3-7}')
tex_str.insert(7, '\midrule')
tex_str.insert(9, '\midrule')
tex_str.insert(11, '\midrule')
tex_str.insert(13, '\midrule')
tex_str.insert(15, '\midrule')
tex_str.insert(17, '\midrule')
tex_str.insert(19, '\midrule')
tex_str.insert(21, '\cmidrule(l){3-7}')
tex_str.insert(23, '\midrule')
tex_str.insert(25, '\midrule')

tex_str = '\n'.join(tex_str)

# print(tex_str)
with open(output_path + 'app-survey-feature-stats.tex', "w") as text_file:
    print(tex_str, file=text_file)


################################################################################
# Tweet Statistics Latex Table #################################################
################################################################################

tweets_stats_table_1 = generate_table(mx_ind[tweet_stats_mx_us_1],
                                      us_ind[tweet_stats_mx_us_1])

tweets_stats_table_2 = generate_table(mx_ind[tweet_stats_mx_us_2],
                                      us_ind[tweet_stats_mx_us_2])

tweet_stats_table_1 = generate_latex(tweets_stats_table_1)
tweet_stats_table_2 = generate_latex(tweets_stats_table_2)

with open(output_path + 'app-tweet-feature-stats-1.tex', "w") as text_file:
    print(tweet_stats_table_1, file=text_file)

with open(output_path + 'app-tweet-feature-stats-2.tex', "w") as text_file:
    print(tweet_stats_table_2, file=text_file)


################################################################################
# Topics Latex Table ###########################################################
################################################################################

topics_table_1 = generate_table(mx_ind[topics_1] * 100,
                                us_ind[topics_1] * 100)

topics_table_2 = generate_table(mx_ind[topics_2] * 100,
                                us_ind[topics_2] * 100)


topics_table_1 = generate_latex(topics_table_1)
topics_table_2 = generate_latex(topics_table_2)

with open(output_path + 'app-topic-stats-1.tex', "w") as text_file:
    print(topics_table_1, file=text_file)

with open(output_path + 'app-topic-stats-2.tex', "w") as text_file:
    print(topics_table_2, file=text_file)


################################################################################
# Errors Latex Table ###########################################################
################################################################################

# Shared error types ###########################################################
errors_us_mx = generate_table(mx_ind[errors_shared],
                              us_ind[errors_shared])

errors_us_mx_table = generate_latex(errors_us_mx)

with open(output_path + 'app-error-mx-us-stats.tex', "w") as text_file:
    print(errors_us_mx_table, file=text_file)


# Excluded error types #########################################################
errors_disjoint = generate_table(mx_ind[errors_mx],
                                 us_ind[errors_us])

errors_disjoint.sort_values(by='Country', inplace=True)
errors_disjoint_table = errors_disjoint.round(decimals=2)


n_rows, n_cols = errors_disjoint_table.shape

errors_disjoint_table = errors_disjoint_table.to_latex(column_format="@{}lc" + "r" * n_cols + '@{}',
                                                       escape=False,
                                                       multicolumn=True,
                                                       index=False,
                                                       multicolumn_format='c') \
    .splitlines()

tex_str = [row if row.endswith('Max. \\\\') or row.startswith('   Error nonstandard') else row.replace('\\\\', '\\\\ \\midrule') for row in errors_disjoint_table]

errors_disjoint_table = '\n'.join(tex_str)

with open(output_path + 'app-error-disjoint-stats.tex', "w") as text_file:
    print(errors_disjoint_table, file=text_file)


################################################################################
# Sentiment Latex Table #######################################################
################################################################################

sentiment_table = generate_table(mx_ind[sent_mx_us],
                                 us_ind[sent_mx_us])

sentiment_table = generate_latex(sentiment_table)

with open(output_path + 'app-sentiment-stats.tex', "w") as text_file:
    print(sentiment_table, file=text_file)


################################################################################
# Networks Latex Table #########################################################
################################################################################

network_table = generate_table(mx_ind[network_mx_us],
                               us_ind[network_mx_us])

network_table = generate_latex(network_table)

with open(output_path + 'app-network-stats.tex', "w") as text_file:
    print(network_table, file=text_file)
