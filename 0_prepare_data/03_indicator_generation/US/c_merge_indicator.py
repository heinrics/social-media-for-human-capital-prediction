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

    data_dir = base_dir + '01_data/US/03-indicators/' + output_folder_name

    # Counties #################################################################
    counties_df = gpd.read_file(base_dir + "/01_data/US/01-raw-data/geodata/counties_us/cb_2018_us_county_500k.shp") \
                     [["GEOID", "NAME"]]\
                     .rename(columns={'GEOID': 'county_id_user',
                                      'NAME': 'county_name_user'})

    # Drop counties outside US mainland
    counties_df['state_id_user'] = counties_df['county_id_user'].str[:2].astype(int)

    (counties_df['state_id_user'] < 60).value_counts()

    # Only mainland US counties are used
    counties_df = counties_df[counties_df['state_id_user'] < 60].drop('state_id_user', axis=1)

    # Remove single county with no education data
    counties_df = counties_df[counties_df['county_id_user'] != '02261']

    # Population ###############################################################

    us_population = pd.read_csv(base_dir + '/01_data/US/01-raw-data/population/PopulationEstimates.csv',
                                sep=',',
                                encoding='latin-1',
                                dtype={'FIPStxt': str},
                                usecols=['FIPStxt',
                                         'Attribute',
                                         'Value'])

    us_population.rename(columns={'FIPStxt': 'county_id_user',
                                  'Value': 'population'},
                         inplace=True)

    # Pad county IDs to original format
    us_population['county_id_user'] = us_population['county_id_user'].str.zfill(5)
    # Select 2021 observations and subsequently needed variables
    us_population = us_population[us_population['Attribute'] == 'Population 2021']\
                                 [['county_id_user',
                                   'population']]

    # Time and platform ########################################################
    time_platform_df = pd.read_csv(data_dir + 'time-platform/county-level-time-platform-indicators-US.csv',
                                   sep=';',
                                   encoding='utf-8',
                                   dtype={'county_id_user': str})

    # Mistakes #################################################################
    mistakes_df = pd.read_csv(data_dir + 'text-mistakes-data/county-level-mistakes-text-indicators-US.csv',
                              sep=';',
                              encoding='utf-8',
                                   dtype={'county_id_user': str})

    mistakes_cat_df = pd.read_csv(data_dir + 'text-mistakes-data/county-level-mistakes-category-per-char-US.csv',
                                  sep=';',
                                  encoding='utf-8',
                                   dtype={'county_id_user': str})

    mistakes_cat_df['char_LANGUAGE_VARIANTS'] = mistakes_cat_df['char_AMERICAN_ENGLISH_STYLE'] + \
                                    mistakes_cat_df['char_BRITISH_ENGLISH']
    mistakes_cat_df.drop(columns=['char_AMERICAN_ENGLISH_STYLE', 'char_BRITISH_ENGLISH'], inplace=True)


    # Topics ###################################################################
    topics_df = pd.read_csv(data_dir + 'topics/county-level-topic-indicators-US.csv',
                            sep=';',
                            encoding='utf-8',
                                   dtype={'county_id_user': str})

    # Sentiment ################################################################
    sentiment_df = pd.read_csv(data_dir + 'sentiment/county-level-sentiment-indicators-US.csv',
                               sep=';',
                               encoding='utf-8',
                                   dtype={'county_id_user': str})

    # Social ###################################################################
    social_df = pd.read_csv(data_dir + 'social/county-level-social-indicators-US.csv',
                            sep=';',
                            encoding='utf-8',
                                   dtype={'county_id_user': str})

    # Network ##################################################################
    network_df = pd.read_csv(data_dir + 'network/county-level-network-indicators-US.csv',
                             sep=';',
                             encoding='utf-8',
                                   dtype={'county_id_user': str})


    # Merge all datasets #######################################################
    print(counties_df.shape)
    # (3141, 2)

    full_df = counties_df.merge(us_population,
                                    on='county_id_user',
                                    how='left')\
                         .merge(time_platform_df,
                                    on='county_id_user',
                                    how='left')\
                         .merge(mistakes_df,
                                on='county_id_user',
                                how='left')\
                         .merge(mistakes_cat_df,
                                on='county_id_user',
                                how='left')\
                         .merge(topics_df,
                                on='county_id_user',
                                how='left')\
                         .merge(sentiment_df,
                                on='county_id_user',
                                how='left')\
                         .merge(social_df,
                                on='county_id_user',
                                how='left')\
                         .merge(network_df,
                                on='county_id_user',
                                how='left')

    print(full_df.shape)
    # (3141, 71)

    # Fill missing values for tweet and user count
    full_df[['tweet_count', 'user_count']] = full_df[['tweet_count', 'user_count']].fillna(0)

    # Rename and select variables
    full_df = full_df.rename(columns={'tweet_count': 'nr_tweets',
                                      'user_count': 'nr_users',
                                      'text_len_clean_avg': 'avg_chars_tweet',
                                      'verified_share': 'share_verified'})\
                     [['county_id_user',
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
                       'char_LANGUAGE_VARIANTS',
                       'char_CASING',
                       'char_COLLOCATIONS',
                       'char_COMPOUNDING',
                       'char_CONFUSED_WORDS',
                       'char_GRAMMAR',
                       'char_MISC',
                       'char_NONSTANDARD_PHRASES',
                       'char_PUNCTUATION',
                       'char_REDUNDANCY',
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

    print(full_df.shape)
    # (3141, 68)

    full_df.to_csv(data_dir + 'county-level-complete-US' + time_str + '.csv',
                   sep=';',
                   encoding='utf-8',
                   index=False)


# Export full dataset ##########################################################
# merge_indicators('total/')


# Export weeks #################################################################

# Get all folders in weeks
week_dir = "PATH TO FOLDER"
for week in [os.path.basename(x) for x in glob(week_dir + '*', recursive=False)]:
    print(week)
    merge_indicators('weeks/' + week + '/', week.replace('week_', '-'))

# week_lists = sorted(list(glob(f"{week_dir}/*/*.csv",
#                     recursive=False)))


# Export days ##################################################################

# Get all folders in days
day_dir = "PATH TO FOLDER"

for day in [os.path.basename(x) for x in glob(day_dir + '*', recursive=False)]:
    print(day)
    merge_indicators('days/' + day + '/', day.replace('day_', '-'))

# day_lists = sorted(list(glob(f"{day_dir}/*/*.csv",
#                    recursive=False)))
