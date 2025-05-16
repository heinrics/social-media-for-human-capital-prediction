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
base_path_sebastian_linux_ssd = '/mnt/4e9e0a0f-64a6-4182-b861-0904a6a7d78d/Mapping_Mistakes'
base_path_sebastian_linux_dropbox = '/mnt/7adaf322-ecbb-4b5d-bc6f-4c54f7f808eb/Dropbox/mapping_mistakes'

input_files_1 = base_path_sebastian_linux_ssd + '/02-preprocessing/02-add-county-data/spanish-with-municipality/americas_tweets-*.txt.gz'
# input_files_2 = base_path_sebastian_linux_ssd + '/02-preprocessing/02-add-county-data/spanish-with-municipality-2/la_tweets-*.txt.gz'
# input_files_3 = base_path_sebastian_linux_ssd + '/02-preprocessing/02-add-county-data/spanish-with-municipality-3/la_tweets-*.txt.gz'

output_files_user = base_path_sebastian_linux_dropbox + '/01_data/MX/02-processed-data/network/'


################################################################################
# Load data
################################################################################

# Load tweet data ##############################################################
# [input_files_1, input_files_2, input_files_3]
en_tw_bag = db.read_text(input_files_1) \
              .map(json.loads)


# For documentation of data see:
# Tweet object:
# https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/object-model/tweet
# User object:
# https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/object-model/user


# en_tw_bag.filter(lambda x: x.get('quoted_status', False)).take(10)


# Derive tweet 'links' dataframe ###############################################

def get_quote(tweet):

    if tweet.get('quoted_status', False):

        selected = {}
        quote = tweet['quoted_status']

        selected['q_id'] = quote['id']
        selected['q_source'] = quote['source']
        selected['q_geo'] = quote['geo']
        selected['q_coordinates'] = quote['coordinates']
        selected['q_place'] = quote['place']
        selected['q_user_id'] = quote['user']['id_str']
        selected['q_user_created_at'] = quote['user']['created_at']
        selected['q_user_location'] = quote['user']['location']
        selected['q_user_verified'] = quote['user']['verified']
        selected['q_user_followers_count'] = quote['user']['followers_count']
        selected['q_user_friends_count'] = quote['user']['friends_count']
        selected['q_user_listed_count'] = quote['user']['listed_count']
        selected['q_user_favourites_count'] = quote['user']['favourites_count']
        selected['q_user_statuses_count'] = quote['user']['statuses_count']

        return selected

    else:
        return None


link_bag = en_tw_bag.map(lambda x: {'tweet_id': x['id'],
                                    'user_id': x['user']['id_str'],
                                    'tweet_created_at': x['created_at'],
                                    'mun_id_user': x['mun_id_user'],
                                    'mun_id_tweet': x['mun_id_tweet'],
                                    'user_mention_ids': [mention['id_str'] for mention in x['entities']['user_mentions']],
                                    'user_mentions_no': len(x['entities']['user_mentions']),
                                    'quoted_status': get_quote(x)})

print('Start processing data')
link_df = pd.DataFrame(link_bag)\
            .drop_duplicates(subset='tweet_id')

print(link_df.shape)
# (2686779, 8)



# MENTION Network ##############################################################
################################################################################

# Check how many MENTIONED accounts are in original sample #####################

# List of all unique user ids occurring in mentions
all_mentioned_ids = link_df['user_mention_ids'].explode().drop_duplicates().dropna()

print(all_mentioned_ids.shape[0])
# 341816
print(all_mentioned_ids[all_mentioned_ids.isin(link_df['user_id'])].shape[0])
# 23150
print(23150 / 341816 * 100)
# Relative to all mentions, 6.77% have location data available from the original sample.

print(link_df['user_id'].nunique())
# 123309
print(link_df.drop_duplicates('user_id')['user_id'][link_df['user_id'].isin(all_mentioned_ids)].shape[0])
# 23150
print(23150 / 123309 * 100)
# Of the original sample 18.77% have been mentioned throughout the samplin period.


# Construct MENTION edge list for in-sample nodes (accounts) ###################

ment_edge_df = link_df[['user_id',
                        # 'mun_id_user',
                        'user_mention_ids',
                        'tweet_created_at']]\
                       .explode('user_mention_ids')

print(ment_edge_df.shape)
# (3442829, 3)

ment_edge_df = ment_edge_df[ment_edge_df['user_mention_ids']\
                                        .isin(link_df['user_id'])]

print(ment_edge_df.shape)
# (232833, 3)

ment_edge_df.columns = ['source_user_id',
                        # 'source_mun_id',
                        'target_user_id',
                        'source_created_at']


# Store edge list to file for subsequent analysis
# ment_edge_df.to_csv(output_files_user + 'mentions-edge-list-MX.csv',
#                     index=False,
#                     sep=';',
#                     encoding='utf-8')



# QUOTED Network ###############################################################
################################################################################

# Queck how many QUOTED accounts are in original sample ########################
print(link_df[~link_df['quoted_status'].isna()].shape[0])
# 333425

quotes_df = pd.DataFrame(link_df[~link_df['quoted_status'].isna()]['quoted_status'].to_list())
print(quotes_df['q_user_id'].nunique())
# 81465

# List of all unique user ids occurring in quotes
all_quotes_ids = quotes_df['q_user_id'].explode().drop_duplicates().dropna()

print(all_quotes_ids.shape[0])
# 81465
print(all_quotes_ids[all_quotes_ids.isin(link_df['user_id'])].shape[0])
# 9046
print(9046 / 81465 * 100)
# Relative to all mentions, 11.10% have location data available from the original sample.

print(link_df['user_id'].nunique())
# 123309
print(link_df.drop_duplicates('user_id')['user_id'][link_df['user_id'].isin(all_quotes_ids)].shape[0])
# 9046
print(9046 / 123309 * 100)
# Of the original sample 7.336% have been quoted throughout the sampling period.


# Has geo:
print(quotes_df[~quotes_df['q_coordinates'].isna()].drop_duplicates('q_user_id').shape[0])
# 35


# Construct QUOTED edge list for in-sample nodes (accounts) ####################

quotes_edge_df = link_df[~link_df['quoted_status'].isna()]\
                        [['user_id',
                          # 'mun_id_user',
                          'quoted_status',
                          'tweet_created_at']]

quotes_edge_df.columns = ['source_user_id',
                          # 'source_mun_id',
                          'target_user_id',
                          'source_created_at']

quotes_edge_df['target_user_id'] = quotes_edge_df['target_user_id'].apply(lambda x: x['q_user_id'])

print(quotes_edge_df.shape)
# (333425, 3)

# Store edge list to file for subsequent analysis
# quotes_edge_df.to_csv(output_files_user + 'quotes-edge-list-MX.csv',
#                       index=False,
#                       sep=';',
#                       encoding='utf-8')
