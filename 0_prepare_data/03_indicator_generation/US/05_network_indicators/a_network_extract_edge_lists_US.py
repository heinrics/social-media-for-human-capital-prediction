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

input_files = base_path_sebastian_linux_ssd + '/02-preprocessing/02-add-county-data/english-with-county/americas_tweets-*.txt'

output_files_user = base_path_sebastian_linux_dropbox + '/01_data/US/02-processed-data/social/'


################################################################################
# Load data
################################################################################

# Load tweet data ##############################################################
en_tw_bag = db.read_text(input_files) \
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
                                    'county_id_user': x['county_id_user'],
                                    'county_id_tweet': x['county_id_tweet'],
                                    'user_mention_ids': [mention['id_str'] for mention in x['entities']['user_mentions']],
                                    'user_mentions_no': len(x['entities']['user_mentions']),
                                    'quoted_status': get_quote(x)})

print('Start processing data')
link_df = pd.DataFrame(link_bag)\
               .drop_duplicates(subset='tweet_id')

print(link_df.shape)
# (22610134, 15)

# pad county ids
link_df['county_id_user'] = link_df['county_id_user']\
                                   .astype(str)\
                                   .str\
                                   .pad(5,
                                        side='left',
                                        fillchar='0')

link_df['county_id_tweet'] = link_df['county_id_tweet']\
                                    .astype(str)\
                                    .str\
                                    .pad(5,
                                         side='left',
                                         fillchar='0')



# MENTION Network ##############################################################
################################################################################

# Check how many MENTIONED accounts are in original sample #####################

# List of all unique user ids occurring in mentions
all_mentioned_ids = link_df['user_mention_ids'].explode().drop_duplicates().dropna()

print(all_mentioned_ids.shape[0])
# 2768184
print(all_mentioned_ids[all_mentioned_ids.isin(link_df['user_id'])].shape[0])
# 232896
print(232896 / 2768184 * 100)
# Relative to all mentions, 8.41% have location data available from the original sample.

print(link_df['user_id'].nunique())
# 943164
print(link_df.drop_duplicates('user_id')['user_id'][link_df['user_id'].isin(all_mentioned_ids)].shape[0])
# 232896
print(232896 / 943164 * 100)
# Of the original sample 24.69% have been mentioned throughout the samplin period.


# Construct MENTION edge list for in-sample nodes (accounts) ###################

ment_edge_df = link_df[['user_id',
                        # 'county_id_tweet',
                        'user_mention_ids',
                        'tweet_created_at']]\
                       .explode('user_mention_ids')

print(ment_edge_df.shape)
# (28527621, 4)

ment_edge_df = ment_edge_df[ment_edge_df['user_mention_ids']\
                                        .isin(link_df['user_id'])]

print(ment_edge_df.shape)
# (2040994, 4)

ment_edge_df.columns = ['source_user_id',
                        # 'source_county_id',
                        'target_user_id',
                        'source_created_at']

# ment_edge_df = ment_edge_df.merge(link_df[['user_id',
#                                            'county_id_tweet']]\
#                                          .drop_duplicates()\
#                                          .rename(columns={'user_id': 'target_user_id',
#                                                           'county_id_tweet': 'target_county_id'}),
#                                   on='target_user_id',
#                                   how='left')

print(ment_edge_df.shape)
# (2040994, 4)


# [['source_county_id',
#               'target_county_id',
#               'source_created_at']]\


# Store edge list to file for subsequent analysis
# ment_edge_df.to_csv(output_files_user + 'mentions-edge-list.csv',
#                     index=False,
#                     sep=';',
#                     encoding='utf-8')



# QUOTED Network ###############################################################
################################################################################

# Queck how many QUOTED accounts are in original sample ########################
print(link_df[~link_df['quoted_status'].isna()].shape[0])
# 3722045

quotes_df = pd.DataFrame(link_df[~link_df['quoted_status'].isna()]['quoted_status'].to_list())
print(quotes_df['q_user_id'].nunique())
# 747486

# List of all unique user ids occurring in quotes
all_quotes_ids = quotes_df['q_user_id'].explode().drop_duplicates().dropna()

print(all_quotes_ids.shape[0])
# 747486
print(all_quotes_ids[all_quotes_ids.isin(link_df['user_id'])].shape[0])
# 120713
print(120713 / 747486 * 100)
# Relative to all mentions, 16.15% have location data available from the original sample.

print(link_df['user_id'].nunique())
# 943164
print(link_df.drop_duplicates('user_id')['user_id'][link_df['user_id'].isin(all_quotes_ids)].shape[0])
# 120713
print(120713 / 943164 * 100)
# Of the original sample 12.8% have been quoted throughout the sampling period.


# Has geo:
print(quotes_df[~quotes_df['q_coordinates'].isna()].drop_duplicates('q_user_id').shape[0])
# 596


# Construct QUOTED edge list for in-sample nodes (accounts) ####################

quotes_edge_df = link_df[~link_df['quoted_status'].isna()]\
                        [['user_id',
                          # 'county_id_tweet',
                          'quoted_status',
                          'tweet_created_at']]

quotes_edge_df.columns = ['source_user_id',
                          # 'source_county_id',
                          'target_user_id',
                          'source_created_at']

quotes_edge_df['target_user_id'] = quotes_edge_df['target_user_id'].apply(lambda x: x['q_user_id'])

print(quotes_edge_df.shape)
# (3722045, 3)

# quotes_edge_df = quotes_edge_df.merge(link_df[['user_id',
#                                                'county_id_tweet']]\
#                                              .drop_duplicates()\
#                                              .rename(columns={'user_id': 'target_user_id',
#                                                               'county_id_tweet': 'target_county_id'}),
#                                       on='target_user_id',
#                                       how='left')
#
# print(quotes_edge_df.shape)
# #  (3722045, 5)

# Drop edges with no county information in target
# quotes_edge_df = quotes_edge_df[~quotes_edge_df['target_county_id'].isna()]

# print(quotes_edge_df.shape)
# (607622, 5)


# [['source_county_id',
#   'target_county_id',
#   'source_created_at']]


# Store edge list to file for subsequent analysis
# quotes_edge_df.to_csv(output_files_user + 'quotes-edge-list.csv',
#                       index=False,
#                       sep=';',
#                       encoding='utf-8')
