################################################################################
# Load packages
################################################################################

import dask.bag as db
import json
import re

import emoji
import pandas as pd
import numpy as np

import matplotlib


print(matplotlib.get_backend())
matplotlib.use('module://backend_interagg')
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.options.display.width = 0
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


################################################################################
# Specify data location
################################################################################

base_path = "PATH TO FOLDER"
input_files_1 = base_path + '/02-preprocessing/02-add-county-data/spanish-with-municipality/americas_tweets-*.txt.gz'
output_files = base_path + '01_data/MX/02-processed-data/text-mistakes-data/'


################################################################################
# Import data
################################################################################


# Tweet datasets ###############################################################

en_tw_bag = db.read_text(input_files_1)\
              .map(json.loads)

################################################################################
# Preprocess mistakes data
################################################################################

def preprocess_tweet_text(tweet):

    # Consolidate tweet text location
    if "extended_tweet" in tweet:
        tweet['tweet_text'] = tweet["extended_tweet"]['full_text']

    elif "text" in tweet:
        tweet['tweet_text'] = tweet['text']

    else:
        tweet['tweet_text'] = np.nan

    tweet['text_len_full'] = len(tweet['tweet_text'])


    # Clean tweet from hashtags, mentions and urls
    tweet_text = tweet['tweet_text']

    # Remove hashtags
    for hashtags in tweet['entities']['hashtags']:
        ind = hashtags['indices']
        tweet_text = tweet_text[0:ind[0]] + ' '*(ind[1] - ind[0]) + tweet_text[ind[1]:]

    # Remove mentions
    for user_mentions in tweet['entities']['user_mentions']:
        ind = user_mentions['indices']
        tweet_text = tweet_text[0:ind[0]] + ' '* (ind[1] - ind[0]) + tweet_text[ind[1]:]

    # Remove urls
    for urls in tweet['entities']['urls']:
        ind = urls['indices']
        tweet_text = tweet_text[0:ind[0]] + ' '*(ind[1] - ind[0]) + tweet_text[ind[1]:]

    # Remove emojis
    tweet_text = emoji.replace_emoji(tweet_text, replace='')

    # remove duplicate whitespaces
    tweet['text_cleaned'] = re.sub(' +', ' ', tweet_text).strip()

    # Get length of cleaned text
    tweet['text_len_clean'] = len(tweet['text_cleaned'])

    return tweet


en_tw_bag = en_tw_bag.map(preprocess_tweet_text)


# Extract data needed for further analysis
mistake_bag = en_tw_bag.map(lambda x: {'tweet_id': x['id'],
                                       'created_at': x['created_at'],
                                       'user_id': x['user']['id_str'],
                                       'verified': x['user']['verified'],
                                       'err_count': len(x['lang_tool']),
                                       'hashtags_count': len(x['entities']['hashtags']),
                                       'user_mentions_count': len(x['entities']['user_mentions']),
                                       'urls_count': len(x['entities']['hashtags']),
                                       'emoji_count': len(x['emoji']),
                                       'text_len_full': x['text_len_full'],
                                       'text_len_clean': x['text_len_clean'],
                                       'rule': [mistake['ruleId'] for mistake in x['lang_tool']],
                                       'category': [mistake['category'] for mistake in x['lang_tool']],
                                       'ruleIssueType': [mistake['ruleIssueType'] for mistake in x['lang_tool']],
                                       'mun_id_user': x['mun_id_user'],
                                       'mun_id_tweet': x['mun_id_tweet']})

mistake_df = pd.DataFrame(mistake_bag)\
               .drop_duplicates(subset='tweet_id')

# There must be 6086256 tweets
# print(mistake_df.shape)
# (6086256, 15)

# Describe tweet level indicators
mistake_df[['err_count',
            'hashtags_count',
            'user_mentions_count',
            'urls_count',
            'emoji_count',
            'text_len_full',
            'text_len_clean']].describe()

#         err_count  hashtags_count  user_mentions_count  urls_count  emoji_count  text_len_full  text_len_clean
# count 2686779.000     2686779.000          2686779.000 2686779.000  2686779.000    2686779.000     2686779.000
# mean        1.362           0.174                0.785       0.174        0.728         91.732          72.747
# std         1.644           0.739                1.151       0.739        1.971         72.861          62.350
# min         0.000           0.000                0.000       0.000        0.000          4.000           1.000
# 25%         0.000           0.000                0.000       0.000        0.000         40.000          29.000
# 50%         1.000           0.000                1.000       0.000        0.000         67.000          53.000
# 75%         2.000           0.000                1.000       0.000        1.000        118.000          97.000
# max        76.000          15.000               13.000      15.000      136.000        999.000         873.000


# Export tweet indicators
mistake_df.to_csv(output_files + 'tweet-level-indicators-MX.csv',
                  index=False,
                  sep=';',
                  encoding='utf-8')
