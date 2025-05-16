################################################################################
# Load packages
################################################################################

import glob
import json
import random

import dask
import dask.bag as db

import pandas as pd


################################################################################
# Specify data location
################################################################################
base_path = "PATH TO FOLDER"
input_dir = base_path + '/02-preprocessing/01-spell-checking/english/'
input_files = input_dir + 'americas_tweets-*.txt'

output_dir = base_path + '/02-preprocessing/02-add-county-data/english-with-county/'
output_files = base_path + '/02-preprocessing/02-add-county-data/english-with-county/americas_tweets-*.txt'

input_file_list = glob.glob(input_files)
output_file_list = [path.replace(input_dir, output_dir) for path in input_file_list]

################################################################################
# Load selected tweets #########################################################
################################################################################


if __name__ == '__main__':

    print('Start loading ids dataframe')

    # Load pandas dataframe in every pipline
    selected_tw_df = pd.read_csv(base_path +
                                 '/01_data/US/2-processed_data/ids.csv',
                                 sep=';',
                                 encoding='utf-8')

    selected_tw_df.drop_duplicates('id', inplace=True)

    selected_tw_df.sort_values(by='id', inplace=True)

    # Derive id list
    tweet_id_list = set(selected_tw_df['id'])

    selected_tw_df.set_index('id', inplace=True)
    selected_tw_df.rename_axis(index=None, inplace=True)

    selected_tw_df_list = [selected_tw_df.copy() for i in range(32)]

    # Filter for existing tweets
    def filter_tweets(tweet):
        try:

            if 'id' in tweet:
                return tweet['id'] in tweet_id_list
            else:
                return False

        except Exception as e:
            print('filter_tweets')
            print(e)
            # print(file_path)

            return False


    # Append county data to tweet
    def append_county_info(tweet, id_df_list):
        try:

            tweet['county_id_user'] = None
            tweet['county_id_tweet'] = None

            if tweet['id']:

                id_df = random.choice(id_df_list)

                county_data = id_df.iloc[id_df.index.get_loc(tweet['id'])]
                tweet['county_id_user'] = int(county_data['county_id_user'])
                tweet['county_id_tweet'] = int(county_data['county_id_tweet'])

                del county_data
                del id_df

            return tweet

        except Exception as e:
            print('append_county_info')
            print(e)
            # print(file_path)

            return tweet


    print('Start processing outer bag')

    print('Finished processing')
