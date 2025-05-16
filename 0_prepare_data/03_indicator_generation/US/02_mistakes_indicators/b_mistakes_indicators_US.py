################################################################################
# Load packages
################################################################################

import pandas as pd
from ast import literal_eval
import time

# Pandas settings
pd.set_option('display.float_format', lambda x: '%.3f' % x)

pd.options.display.width = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


################################################################################
# Specify data location
################################################################################

main_path = "PATH TO FOLDER"
file_dir = main_path + '01_data/US/02-processed-data/text-mistakes-data/'
input_dir_period = main_path + '01_data/MX/03-indicators/'

################################################################################
# Import data
################################################################################

# Tweet datasets ###############################################################

def load_data(filter_date_start, filter_date_end, out_dir):

    # takes a while to load, due to evaluation of arrays
    mistake_df = pd.read_csv(file_dir + 'tweet-level-indicators.csv',
                             sep=';',
                             encoding='utf-8',
                             dtype={'county_id_user': str,
                                    'county_id_tweet': str},
                             converters={'rule': literal_eval,
                                         'category': literal_eval,
                                         'ruleIssueType': literal_eval})

    # print(mistake_df.shape)
    # (22610134, 15)

    mistake_df['created_at_tm'] = mistake_df['created_at'].apply(lambda x: time.strptime(x, '%a %b %d %H:%M:%S +0000 %Y'))

    # Filter tweets by date
    if filter_date_start:
        mistake_df = mistake_df[mistake_df['created_at_tm'] >= filter_date_start]

    if filter_date_end:
        mistake_df = mistake_df[mistake_df['created_at_tm'] <= filter_date_end]

    # Load and update correct user county data
    mistake_df.rename(columns={'county_id_user': 'county_id_user_full',
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

    mistake_df = mistake_df.merge(user_id_df.drop_duplicates('user_id'),
                                  on='user_id',
                                  how='left')

    return mistake_df

################################################################################
# High level indicators
################################################################################
def user_mistakes(df, out_path):
    # Aggregate tweet to user level ############################################

    user_err_count_df = df.groupby('user_id')\
                          .agg({'tweet_id': 'nunique',
                                'verified': 'max',
                                'err_count': ['sum', 'mean'],
                                'hashtags_count':  ['sum', 'mean'],
                                'user_mentions_count':  ['sum', 'mean'],
                                'urls_count':  ['sum', 'mean'],
                                'emoji_count':  ['sum', 'mean'],
                                'text_len_full':  ['sum', 'mean'],
                                'text_len_clean':  ['sum', 'mean'],
                                'county_id_user': 'max'})\
                          .reset_index()\
                          .rename(columns={'tweet_id': 'tweet_count'})

    # Rename columns
    user_err_count_df.columns = ['_'.join(col).strip() for col in user_err_count_df.columns.values]
    user_err_count_df.rename(columns={'user_id_': 'user_id'}, inplace=True)
    user_err_count_df.rename(columns={'county_id_user_max': 'county_id_user'}, inplace=True)

    # Compute error rates relative to length (per 1000 chars)
    user_err_count_df['err_per_char'] = user_err_count_df['err_count_sum'] / (user_err_count_df['text_len_clean_sum'] / 1000)

    # print(user_err_count_df.shape)
    # (943164, 19)

    user_err_count_df = user_err_count_df.groupby('county_id_user') \
        .agg({'tweet_count_nunique': 'sum',
              'user_id': 'nunique',
              'verified_max': 'mean',
              'err_count_mean': 'mean',
              'err_per_char': 'mean',
              'text_len_full_mean': 'mean',
              'text_len_clean_mean': 'mean',
              'hashtags_count_mean': 'mean',
              'user_mentions_count_mean': 'mean',
              'urls_count_mean': 'mean',
              'emoji_count_mean': 'mean'}) \
        .reset_index()

    # print(user_err_count_df['user_id'].sum())
    # 943164
    # print(user_err_count_df['tweet_count_nunique'].sum())
    # 22610134

    user_err_count_df.columns = ['county_id_user',
                                 'tweet_count',
                                 'user_count',
                                 'verified_share',
                                 'error_per_tweet',
                                 'error_per_char',
                                 'text_len_full_avg',
                                 'text_len_clean_avg',
                                 'hashtag_per_tweet',
                                 'mention_per_tweet',
                                 'url_per_tweet',
                                 'emoji_per_tweet']

    user_err_count_df.to_csv(out_path,
                             index=False,
                             sep=';',
                             encoding='utf-8')


################################################################################
# Detailed mistakes indicators
################################################################################

def user_stats(df):

    user_stats = df.groupby('user_id')\
                   .agg({'tweet_id': 'nunique',
                         'text_len_clean': 'sum',
                         'verified': 'max',
                         'county_id_user': 'max'})\
                   .reset_index()

    # print(user_stats.tweet_id.sum())
    # 22610134

    user_stats.columns = ['user_id',
                          'tweet_count',
                          'text_len_clean',
                          'verified',
                          'county_id_user']

    return user_stats


# Generate CATEGORY variables ##################################################
################################################################################

def user_mistakes_cat(df, user_stats, out_path):

    cat_mistakes_df = df[['user_id',
                          'category',
                          'tweet_id']]\
                        .explode('category')

    cat_mistakes_df['category'].fillna('no_err', inplace=True)

    cat_mistakes_df = cat_mistakes_df.groupby(['user_id',
                                               'category'])\
                                     .size()\
                                     .reset_index(name='err_cat_count')

    # Merge number of errors in category to user level stats
    cat_mistakes_df = cat_mistakes_df.merge(user_stats,
                                            on='user_id')

    # Calculate numer of errors total characters written of user (in 1000)
    cat_mistakes_df['err_cat_per_char'] = cat_mistakes_df['err_cat_count'] / \
                                          (cat_mistakes_df['text_len_clean'] / 1000)

    # print(cat_mistakes_df[['err_cat_per_tweet',
    #                        'err_cat_per_char']]\
    #                      .describe())

    #        err_cat_per_tweet  err_cat_per_char
    # count        3098934.000       3098934.000
    # mean               0.426             7.865
    # std                0.486            14.133
    # min                0.000             0.000
    # 25%                0.077             1.121
    # 50%                0.280             3.676
    # 75%                0.653             9.412
    # max               47.000           555.556


    # Per character error indicators ###############################################

    cat_mistakes_per_char_df = cat_mistakes_df.pivot(index=['user_id'],
                                                      columns='category',
                                                      values='err_cat_per_char')\
                                               .reset_index()

    cat_mistakes_per_char_df.fillna(0, inplace=True)
    cat_mistakes_per_char_df.drop('no_err', axis=1, inplace=True)

    # Rename columns
    cat_mistakes_per_char_df.columns = ['char_' + str(col) for col in cat_mistakes_per_char_df.columns]
    cat_mistakes_per_char_df.rename(columns={'char_user_id': 'user_id'}, inplace=True)

    # Add user counties
    cat_mistakes_per_char_df = cat_mistakes_per_char_df.merge(user_stats[['user_id',
                                                                          'county_id_user']],
                                                              on='user_id')


    # Aggregate to county level ################################################

    cat_mistakes_per_char_df = cat_mistakes_per_char_df.drop('user_id', axis=1) \
                                                       .groupby('county_id_user') \
                                                       .mean() \
                                                       .reset_index()

    # Export to csv
    cat_mistakes_per_char_df.to_csv(out_path,
                                    index=False,
                                    sep=';',
                                    encoding='utf-8')


def compute_indicators(filter_date_start, filter_date_end, out_dir):

    df = load_data(filter_date_start, filter_date_end, out_dir)

    # Compute and store high level indicators
    user_mistakes(df, out_dir + 'county-level-mistakes-text-indicators-US.csv')

    # User statistics as input in subsequent indicator generation
    user_statistics = user_stats(df)

    # Store data
    user_mistakes_cat(df, user_statistics, out_dir + 'county-level-mistakes-category-per-char-US.csv')
