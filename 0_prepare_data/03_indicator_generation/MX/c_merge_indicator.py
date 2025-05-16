import pandas as pd
import numpy as np
import geopandas as gpd
from glob import glob
import os

pd.options.display.width = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


################################################################################
# JOIN ALL DATASETS ############################################################
################################################################################
def merge_indicators(output_folder_name, time_str=''):
    base_dir = "PATH TO FOLDER"
    data_dir = base_dir + '01_data/MX/03-indicators/' + output_folder_name

    # Municipalities ###########################################################
    municipality_df = gpd.read_file(base_dir + "01_data/MX/01-raw-data/geodata/municipalities_mx/mex_admbnda_adm2_govmex_20210618.shp")
    municipality_df.rename({"ADM2_ES": "mun_name_user",
                            "ADM2_PCODE": "mun_id_user"},
                           axis=1,
                           inplace=True)

    municipality_df = municipality_df[['mun_id_user', 'mun_name_user']]

    # Population ###############################################################

    selected_var = ['ENTIDAD',
                    'MUN',
                    'NOM_MUN',
                    'NOM_LOC',
                    'POBTOT']

    mx_population = pd.read_csv(base_dir + '01_data/MX/01-raw-data/income/iter_00_cpv2020/conjunto_de_datos/conjunto_de_datos_iter_00CSV20.csv',
                                sep=',',
                                low_memory=False,
                                usecols=selected_var)

    mx_population = mx_population[mx_population['NOM_LOC'] == 'Total del Municipio']

    mx_population['mun_id_user'] = ("MX" +
                                    mx_population.ENTIDAD.astype("str").str.zfill(2) +
                                    mx_population.MUN.astype("str").str.zfill(3))

    mx_population.rename(columns={'POBTOT': 'population'}, inplace=True)

    mx_population = mx_population[['mun_id_user', 'population']]

    # Time and platform ########################################################
    time_platform_df = pd.read_csv(data_dir + 'time-platform/mun-level-time-platform-indicators-MX.csv',
                                   sep=';',
                                   encoding='utf-8')

    # Mistakes #################################################################
    mistakes_df = pd.read_csv(data_dir + 'text-mistakes-data/mun-level-mistakes-text-indicators-MX.csv',
                              sep=';',
                              encoding='utf-8')

    mistakes_cat_df = pd.read_csv(data_dir + 'text-mistakes-data/mun-level-mistakes-category-per-char-MX.csv',
                                  sep=';',
                                  encoding='utf-8')

    # Merge error categories
    mistakes_cat_df['char_STYLE'] = mistakes_cat_df['char_ESTILO'] + \
                                    mistakes_cat_df['char_STYLE']
    mistakes_cat_df.drop(columns=['char_ESTILO'], inplace=True)

    # For first day dataset, variable is missing
    if not 'char_CONFUSED_WORDS' in mistakes_cat_df.columns:
        mistakes_cat_df['char_CONFUSED_WORDS'] = np.nan

    mistakes_cat_df['char_CONFUSIONS'] = mistakes_cat_df['char_CONFUSIONS'] + \
                                         mistakes_cat_df['char_CONFUSIONS2'] + \
                                         mistakes_cat_df['char_CONFUSED_WORDS']

    mistakes_cat_df.drop(columns=['char_CONFUSED_WORDS', 'char_CONFUSIONS2'], inplace=True)


    # Topics ###################################################################
    topics_df = pd.read_csv(data_dir + 'topics/mun-level-topic-indicators-MX.csv',
                            sep=';',
                            encoding='utf-8')

    # Sentiment ################################################################
    sentiment_df = pd.read_csv(data_dir + 'sentiment/mun-level-sentiment-indicators-MX.csv',
                               sep=';',
                               encoding='utf-8')

    # Social ###################################################################
    social_df = pd.read_csv(data_dir + 'social/mun-level-social-indicators-MX.csv',
                            sep=';',
                            encoding='utf-8')

    # Network ##################################################################
    network_df = pd.read_csv(data_dir + 'network/mun-level-network-indicators-MX.csv',
                             sep=';',
                             encoding='utf-8')

    print(municipality_df.shape)
    # (2457, 2)

    # Merge all datasets #######################################################
    full_df = municipality_df.merge(mx_population,
                                    on='mun_id_user',
                                    how='left')\
                             .merge(time_platform_df,
                                    on='mun_id_user',
                                    how='left')\
                             .merge(mistakes_df,
                                    on='mun_id_user',
                                    how='left')\
                             .merge(mistakes_cat_df,
                                    on='mun_id_user',
                                    how='left')\
                             .merge(topics_df,
                                    on='mun_id_user',
                                    how='left')\
                             .merge(sentiment_df,
                                    on='mun_id_user',
                                    how='left')\
                             .merge(social_df,
                                    on='mun_id_user',
                                    how='left')\
                             .merge(network_df,
                                    on='mun_id_user',
                                    how='left')

    print(full_df.shape)
    # (2457, 78)

    # Fill missing values for tweet and user count
    full_df[['tweet_count', 'user_count']] = full_df[['tweet_count', 'user_count']].fillna(0)

    # Rename and select variables
    full_df = full_df.rename(columns={'tweet_count': 'nr_tweets',
                                      'user_count': 'nr_users',
                                      'text_len_clean_avg': 'avg_chars_tweet',
                                      'verified_share': 'share_verified'})\
                     [['mun_id_user',
                       'population',
                       'nr_tweets',
                       'nr_users',
                       'share_weekdays',
                       'share_workhours',
                       'median_followers',
                       'median_friends',
                       'median_statuses',
                       'avg_mobility',
                       'share_iphone',
                       'share_insta',
                       'favourites_per_statuses',
                       'statuses_count_per_year',
                       'account_age_absolute_year',
                       'listed_count',
                       'followers_per_friends',
                       'is_quote_status_mean',
                       'is_reply_mean',
                       'share_verified',
                       'avg_chars_tweet',
                       'hashtag_per_tweet',
                       'mention_per_tweet',
                       'url_per_tweet',
                       'emoji_per_tweet',
                       'arts_&_culture',
                       'business_&_entrepreneurs',
                       'celebrity_&_pop_culture',
                       'diaries_&_daily_life',
                       'family',
                       'fashion_&_style',
                       'film_tv_&_video',
                       'fitness_&_health',
                       'food_&_dining',
                       'gaming',
                       'learning_&_educational',
                       'music',
                       'news_&_social_concern',
                       'other_hobbies',
                       'relationships',
                       'science_&_technology',
                       'sports',
                       'travel_&_adventure',
                       'youth_&_student_life',
                       'error_per_char',
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
                       'char_TYPOS',
                       'negative',
                       'positive',
                       'hate',
                       'offensive',
                       'ref_in_deg',
                       'ref_out_deg',
                       'ref_clos_cent',
                       'ref_pagerank']]

    full_df.to_csv(data_dir + 'mun-level-complete-MX' + time_str + '.csv',
                   sep=';',
                   encoding='utf-8',
                   index=False)


# Export full dataset ##########################################################
# merge_indicators('total/')


# Export weeks #################################################################

# Get all folders in weeks
week_dir = "PATH TO FOLDER"
for week in sorted([os.path.basename(x) for x in glob(week_dir + 'week_*', recursive=False)]):
    print(week)
    merge_indicators('weeks/' + week + '/', week.replace('week_', '-'))

# Get all merged files
# week_lists = sorted(list(glob(f"{week_dir}/*/*.csv",
#                     recursive=False)))


# Export days ##################################################################
day_dir = "PATH TO FOLDER"
for week in sorted([os.path.basename(x) for x in glob(day_dir + '*', recursive=False)]):
    print(week)
    merge_indicators('days/' + week + '/', week.replace('day_', '-'))

# Get all merged files
# day_lists = sorted(list(glob("f{day_dir}/*/*.csv",
#                    recursive=False)))