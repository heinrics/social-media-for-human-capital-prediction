################################################################################
# Load packages
################################################################################
import json
import dask.bag as db
import pandas as pd

pd.options.display.width = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

################################################################################
# Specify data location
################################################################################
base_path = "PATH TO FOLDER"

input_files_1 = base_path + '/02-preprocessing/02-add-county-data/spanish-with-municipality/americas_tweets-*.txt.gz'
output_files = base_path + '/01_data/MX/02-processed-data/social/'

################################################################################
# Load data
################################################################################
print('Start loading ids dataframe')

# Load tweet data ############### input_files_3]
en_tw_bag = db.read_text(input_files_1) \
              .map(json.loads)

# There must be 6086256 tweets in the final sample
# (this count contains duplicates)
# print(en_tw_bag.count().compute())
# 6094721

# For documentation of data see:
# Tweet object:
# https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/object-model/tweet
# User object:
# https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/object-model/user


# Derive user statistics dataframe #############################################
user_stats_bag = en_tw_bag.map(lambda x: {'tweet_id': x['id'],
                                          'created_at': x['created_at'],
                                          'user_id': x['user']['id_str'],
                                          'tweet_created_at': x['created_at'],
                                          'mun_id_user': x['mun_id_user'],
                                          'mun_id_tweet': x['mun_id_tweet'],
                                          'account_created_at': x['user']['created_at'],
                                          'followers_count': x['user']['followers_count'],
                                          'friends_count': x['user']['friends_count'],
                                          'listed_count': x['user']['listed_count'],
                                          'favourites_count': x['user']['favourites_count'],
                                          'statuses_count': x['user']['statuses_count'],
                                          'is_quote_status': x['is_quote_status'],
                                          'is_reply': x['in_reply_to_status_id'] != None})

print('Start processing data')
social_df = pd.DataFrame(user_stats_bag)\
              .drop_duplicates(subset='tweet_id')

print(social_df.shape)
# (2686779, 14)
# print(social_df['user_id'].nunique())
# 123309


# Export tweet indicators
social_df.to_csv(output_files + 'tweet-level-indicators-MX.csv',
                 index=False,
                 sep=';',
                 encoding='utf-8')
