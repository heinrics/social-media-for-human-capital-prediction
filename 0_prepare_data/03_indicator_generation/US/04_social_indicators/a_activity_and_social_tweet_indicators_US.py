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


base_path = '/mnt/4e9e0a0f-64a6-4182-b861-0904a6a7d78d/Mapping_Mistakes'
input_files = base_path + '/02-preprocessing/02-add-county-data/english-with-county/americas_tweets-*.txt'
output_files = base_path + '/01_data/US/02-processed-data/social/'


################################################################################
# Load data
################################################################################
print('Start loading ids dataframe')

# Load tweet data ##############################################################
en_tw_bag = db.read_text(input_files) \
              .map(json.loads)

# For documentation of data see:
# Tweet object:
# https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/object-model/tweet
# User object:
# https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/object-model/user


# Derive user statistics dataframe #############################################
user_stats_bag = en_tw_bag.map(lambda x: {'tweet_id': x['id'],
                                          'created_at': x['created_at'],
                                          'user_id': x['user']['id_str'],
                                          'county_id_user': x['county_id_user'],
                                          'county_id_tweet': x['county_id_tweet'],
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

# print(social_df.shape)
# (22610134, 12)
# print(social_df['user_id'].nunique())
# 943164

# pad county ids
social_df['county_id_user'] = social_df['county_id_user']\
                                       .astype(str)\
                                       .str\
                                       .pad(5,
                                            side='left',
                                            fillchar='0')

social_df['county_id_tweet'] = social_df['county_id_tweet']\
                                        .astype(str)\
                                        .str\
                                        .pad(5,
                                             side='left',
                                             fillchar='0')

# Export tweet indicators
# social_df.to_csv(output_files + 'tweet-level-indicators-US.csv',
#                  index=False,
#                  sep=';',
#                  encoding='utf-8')
