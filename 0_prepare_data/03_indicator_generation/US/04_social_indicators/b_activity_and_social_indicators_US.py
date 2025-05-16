################################################################################
# Load packages
################################################################################

import glob
import json
import time

import dask
import dask.bag as db

import pandas as pd
import numpy as np

pd.options.display.width = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


################################################################################
# Specify data location
################################################################################

base_path = "PATH TO FOLDER"
input_dir = base_path + '/01_data/US/02-processed-data/social/'
output_files_indicator = base_path + '/01_data/US/03-indicators/'

################################################################################
# Load data
################################################################################

def load_data(filter_date_start, filter_date_end, out_dir):

    social_df = pd.read_csv(input_dir + 'tweet-level-indicators-US.csv',
                            sep=';',
                            encoding='utf-8',
                            dtype={'county_id_user': 'str'})

    # print(social_df.shape)
    # (6086256, 13)
    # print(social_df['user_id'].nunique())
    # 206363

    social_df['created_at_tm'] = social_df['created_at'].apply(lambda x: time.strptime(x, '%a %b %d %H:%M:%S +0000 %Y'))

    # Filter tweets by date
    if filter_date_start:
        social_df = social_df[social_df['created_at_tm'] >= filter_date_start]

    if filter_date_end:
        social_df = social_df[social_df['created_at_tm'] <= filter_date_end]

    # Load and update correct user county data
    social_df.rename(columns={'county_id_user': 'county_id_user_full',
                              'county_name_user': 'county_name_user_full'},
                     inplace=True)

    cols = ['user_id',
            'county_id_user',
            'county_name_user']

    user_id_df = pd.read_csv('/'.join(out_dir.split('/')[0:-2]).replace('03-indicators/',
                                                                        '03-indicators/selected_con/') +
                             "/tweet_data_selection.csv",
                             sep=";",
                             low_memory=False,
                             lineterminator='\n',
                             encoding="utf-8",
                             usecols=cols,
                             dtype={'county_id_user': 'str'})

    social_df = social_df.merge(user_id_df.drop_duplicates('user_id'),
                                on='user_id',
                                how='left')

    return social_df



def user_social(df, out_path):

    # Aggregate tweet related metrics
    df[['is_quote_status_mean',
        'is_reply_mean']] = df[['user_id',
                                'is_quote_status',
                                'is_reply']]\
                              .groupby('user_id')\
                              .transform('mean')

    # Get user information from most recent tweet
    user_social_df = df.sort_values(by=['user_id',
                                        'created_at_tm'],
                                    ascending=False)\
                       .groupby('user_id')\
                       .nth(0)\
                       .reset_index()

    # print(user_social_df.shape[0])



    # Relative to time #############################################################
    ################################################################################


    # Get account creation time structure to compute account age
    user_social_df['account_created_at_tm'] = user_social_df['account_created_at']\
                                                            .apply(lambda x: time.strptime(x, '%a %b %d %H:%M:%S +0000 %Y'))

    # Year age, relative to last tweet. Used for per year measure below
    user_social_df['account_age_last_tweet_year'] = user_social_df.apply((lambda x: (time.mktime(x['created_at_tm']) -
                                                                                     time.mktime(x['account_created_at_tm'])) / 60 / 60 / 24 / 365),
                                                                         axis=1)

    # print(user_social_df['account_age_last_tweet_year'].describe())


    # Absolute account age, relative to time last tweet (overall) was sampled
    most_recent_tweet_creation_time = user_social_df.sort_values(by=['created_at_tm'],
                                                                 ascending=False) \
                                                    ['created_at_tm'][0]

    user_social_df['account_age_absolute_year'] = user_social_df.apply((lambda x: (time.mktime(most_recent_tweet_creation_time) -
                                                                                   time.mktime(x['account_created_at_tm'])) / 60 / 60 / 24 / 365),
                                                                       axis=1)
    # print(user_social_df['account_age_absolute_year'].describe())


    # There is one account with creation time set to unix epoch
    invalid_creation_time_user_id = user_social_df[user_social_df['account_created_at_tm']\
                                                                 .apply(lambda x: x.tm_year < 2006)]\
                                                                 ['user_id']

    cols = ['account_age_last_tweet_year',
            'account_age_absolute_year']
    selector = user_social_df['user_id'].isin(invalid_creation_time_user_id)
    user_social_df.loc[selector, cols] = np.nan

    # Check how many rows & columns have been set to 0
    # print(user_social_df.isnull().sum())


    # Relative to 'tweets' #########################################################
    ################################################################################

    # favourites_count per statuses_count
    user_social_df['favourites_per_statuses'] = user_social_df['favourites_count'] / user_social_df['statuses_count']

    # statuses_count per account_age_last_tweet_year
    user_social_df['statuses_count_per_year'] = user_social_df['statuses_count'] / user_social_df['account_age_last_tweet_year']

    # followers_count / (friends_count + 1)
    user_social_df['followers_per_friends'] = user_social_df['followers_count'] / (user_social_df['friends_count'] + 1)



    # Export user level ############################################################
    ################################################################################
    user_social_df = user_social_df[['user_id',
                                     'tweet_id',
                                     'created_at',
                                     'county_id_user',
                                     'county_id_tweet',
                                     'account_created_at',
                                     'statuses_count',
                                     'favourites_count',
                                     'followers_count',
                                     'friends_count',
                                     'followers_per_friends',
                                     'favourites_per_statuses',
                                     'listed_count',
                                     'statuses_count_per_year',
                                     'account_age_last_tweet_year',
                                     'account_age_absolute_year',
                                     'is_quote_status_mean',
                                     'is_reply_mean']]

    # Aggregate on county level ################################################
    county_social_df = user_social_df[['county_id_user',
                                       'favourites_per_statuses',
                                       'statuses_count_per_year',
                                       'account_age_absolute_year',
                                       'listed_count',
                                       'followers_per_friends',
                                       'is_quote_status_mean',
                                       'is_reply_mean']]\
                                     .groupby('county_id_user')\
                                     .agg({'favourites_per_statuses': 'median',
                                           'statuses_count_per_year': 'median',
                                           'account_age_absolute_year': 'mean',
                                           'listed_count': 'mean',
                                           'followers_per_friends': 'median',
                                           'is_quote_status_mean': 'mean',
                                           'is_reply_mean': 'mean'})\
                                     .reset_index()

    # Export user level indicators
    county_social_df.to_csv(out_path,
                            index=False,
                            sep=';',
                            encoding='utf-8')


def compute_indicators(filter_date_start, filter_date_end, out_dir):

    df = load_data(filter_date_start, filter_date_end, out_dir)
    user_social(df, out_dir + 'county-level-social-indicators-US.csv')
