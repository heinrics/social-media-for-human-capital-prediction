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

base_path ="PATH TO FOLDER"

# Two sources of tweets:
source = 'americas' # 'la_america, la_tweets, la_tweets_server_update'

if source == 'americas':
    input_dir = base_path + '/02-preprocessing/01-spell-checking/spanish/'
    input_files = input_dir + 'americas_tweets-*.txt.gz'

    output_dir = base_path + '/02-preprocessing/02-add-county-data/spanish-with-municipality/'

if source == 'la_tweets':
    input_dir = base_path + '/02-preprocessing/01-spell-checking/spanish-2/'
    input_files = input_dir + 'la_tweets-*.txt'

    output_dir = base_path + '/02-preprocessing/02-add-county-data/spanish-with-municipality-2/'

if source == 'la_tweets_server_update':
    input_dir = base_path + '/02-preprocessing/01-spell-checking/spanish-3/'
    input_files = input_dir + 'la_tweets-*.txt'

    output_dir = base_path + '/02-preprocessing/02-add-county-data/spanish-with-municipality-3/'

input_file_list = glob.glob(input_files)
output_file_list = [path.replace(input_dir, output_dir) + '.gz' for path in input_file_list]


################################################################################
# Load selected tweets #########################################################
################################################################################


if __name__ == '__main__':

    print('Start loading ids dataframe')

    # Load pandas dataframe in every pipline
    selected_tw_df = pd.read_csv(base_path +
                                 '/01_data/MX/02-processed-data/ids.csv',
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


    # Append municipality data to tweet
    def append_municipality_info(tweet, id_df_list):
        try:

            tweet['mun_id_user'] = None
            tweet['mun_id_tweet'] = None

            if tweet['id']:

                id_df = random.choice(id_df_list)

                mun_data = id_df.iloc[id_df.index.get_loc(tweet['id'])]
                tweet['mun_id_user']  = mun_data['mun_id_user']
                tweet['mun_id_tweet'] = mun_data['mun_id_tweet']

                del mun_data
                del id_df

            return tweet

        except Exception as e:
            print('append_mun_info')
            print(e)
            # print(file_path)

            return tweet


    print('Start processing outer bag')


    with dask.config.set(scheduler='threads'):

        db.read_text(input_file_list) \
          .map(json.loads) \
          .filter(filter_tweets) \
          .map(append_municipality_info, id_df_list=selected_tw_df_list)\
          .map(json.dumps) \
          .to_textfiles(output_file_list,
                        encoding='utf-8')

    print('Finished processing')
