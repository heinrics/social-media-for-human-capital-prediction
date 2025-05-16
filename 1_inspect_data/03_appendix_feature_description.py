################################################################################
# Import modules
################################################################################
import pandas as pd

pd.options.display.width = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

pd.set_option('display.max_colwidth', None)


################################################################################
# Load and merge data
################################################################################

# MX ###########################################################################

# Specify data location
from paths import paths_mx as paths
input_path = paths["processed_data"]
edu_path = paths["edu_data"]
econ_path = paths["econ_data"]
output_path = paths["exhibits"] + r"/tables/appendix/feature_descriptions/"

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
survey_mx = mx_ind[['mun_id_user', 'population']]\
                  .merge(wealth_idx_mx,
                         left_on='mun_id_user',
                         right_on='GEOID',
                         how='left')\
                  .merge(mx_edu,
                         left_on='mun_id_user',
                         right_on='GEOID',
                         how='left').drop(columns=['GEOID_x',
                                                   'GEOID_y',
                                                   'NAME'])\
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
                         how='left')\
                  .merge(us_edu,
                         left_on='county_id_user',
                         right_on='GEOID',
                         how='left').drop(columns=['GEOID', 'NAME']) \
                  [['edu_years_schooling',
                    'edu_bachelor',
                    'edu_some_college',
                    'edu_high',
                    'population',
                    'income']]


# Variable labels ##############################################################

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
# Generate Variable eDescription Tables
################################################################################

def latex_generator(df, out_path):
    df = df[['label', 'description']]

    df.columns = ['Label', 'Description']

    tex_list = df.to_latex(
        column_format=r'p{0.25\textwidth}p{0.75\textwidth}',
        index=False,
        escape=True) \
        .splitlines()

    # Adding lines to every row, preventing duplicate lines at beginning and end
    tex_list = [row.replace('\\\\', '\\\\ \midrule') if ind > 2 and ind < len(tex_list) - 3 else row for ind, row in enumerate(tex_list)]

    tex_str = '\n'.join(tex_list)

    with open(out_path, "w") as text_file:
        print(tex_str, file=text_file)


################################################################################
# Survey Latex Table ###########################################################
################################################################################

survey_table = var_labels[var_labels['col_name'].isin(list(survey_mx.columns) +
                                                      list(survey_us.columns))]

latex_generator(survey_table.copy(), output_path + 'survey-descriptions.tex')


################################################################################
# Tweet Statistics Latex Table #################################################
################################################################################
tweet_table_1 = var_labels[var_labels['col_name'].isin(tweet_stats_mx_us_1)]
tweet_table_2 = var_labels[var_labels['col_name'].isin(tweet_stats_mx_us_2)]

latex_generator(tweet_table_1.copy(), output_path + 'tweet-descriptions-1.tex')
latex_generator(tweet_table_2.copy(), output_path + 'tweet-descriptions-2.tex')


################################################################################
# Topics Latex Table ###########################################################
################################################################################
topic_table_1 = var_labels[var_labels['col_name'].isin(topics_1)]
topic_table_2 = var_labels[var_labels['col_name'].isin(topics_2)]

latex_generator(topic_table_1.copy(), output_path + 'topic-descriptions-1.tex')
latex_generator(topic_table_2.copy(), output_path + 'topic-descriptions-2.tex')


################################################################################
# Errors Latex Table ###########################################################
################################################################################
errors_table_joint = var_labels[var_labels['col_name'].isin(errors_shared)]
errors_table_disjoint = var_labels[var_labels['col_name'].isin(errors_mx + errors_us)]

latex_generator(errors_table_joint.copy(), output_path + 'errors-joint-descriptions-1.tex')
latex_generator(errors_table_disjoint.copy(), output_path + 'errors-disjoint-descriptions-2.tex')


################################################################################
# Sentiment Latex Table #######################################################
################################################################################
sentiment_table = var_labels[var_labels['col_name'].isin(sent_mx_us)]

latex_generator(sentiment_table.copy(), output_path + 'sentiment-descriptions.tex')


################################################################################
# Network Latex Table ##########################################################
################################################################################
network_table = var_labels[var_labels['col_name'].isin(network_mx_us)]

latex_generator(network_table.copy(), output_path + 'network-descriptions.tex')
