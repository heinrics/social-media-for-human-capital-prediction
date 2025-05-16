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
tapir = "sebastian" # Or: "tina"

if tapir == "sebastian":
    path_sebastian_linux_dropbox = '/mnt/7adaf322-ecbb-4b5d-bc6f-4c54f7f808eb/Dropbox/mapping_mistakes/'

    input_path = path_sebastian_linux_dropbox
    input_dir = input_path + '01_data/MX/02-processed-data/text-mistakes-data/'
    input_dir_period = path_sebastian_linux_dropbox + '01_data/MX/03-indicators/'


################################################################################
# Import and filter data
################################################################################

# Tweet datasets ###############################################################

def load_data(time_split, filter_date_end, week, out_dir):

    # takes a while to load, due to evaluation of arrays
    mistake_df = pd.read_csv(input_dir + 'tweet-level-indicators-MX.csv',
                             sep=';',
                             encoding='utf-8',
                             converters={'rule': literal_eval,
                                         'category': literal_eval,
                                         'ruleIssueType': literal_eval})

    # print(mistake_df.shape)
    # (6086256, 15)

    mistake_df['created_at_tm'] = mistake_df['created_at'].apply(lambda x: time.strptime(x, '%a %b %d %H:%M:%S +0000 %Y'))

    # Filter tweets by date
    if week <= 9:
        if filter_date_end:
            mistake_df = mistake_df[(mistake_df['created_at_tm'] > time_split) &
                                    (mistake_df['created_at_tm'] < filter_date_end)]
        else:
            mistake_df = mistake_df[(mistake_df['created_at_tm'] > time_split)]

    if (week > 9) and filter_date_end and time_split:
        print('Not used data!')
        exit()
        # mistake_df = mistake_df[(mistake_df['created_at_tm'] < filter_date_end) |
        #                         (mistake_df['created_at_tm'] > time_split)]

    # Load and update correct user county data
    mistake_df.rename(columns={'mun_id_user': 'mun_id_user_full',
                               'mun_name_user': 'mun_name_user_full'},
                      inplace=True)

    cols = ['user_id',
            'mun_id_user',
            'mun_name_user']

    user_id_df = pd.read_csv('/'.join(out_dir.split('/')[0:-2]).replace('03-indicators/',
                                                                        '03-indicators/selected_mun/') +
                             "/tweet_data_selection.csv",
                             sep=";",
                             low_memory=False,
                             lineterminator='\n',
                             encoding="utf-8",
                             usecols=cols)

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
                                'mun_id_user': 'max'})\
                          .reset_index()\
                          .rename(columns={'tweet_id': 'tweet_count'})

    # Rename columns
    user_err_count_df.columns = ['_'.join(col).strip() for col in user_err_count_df.columns.values]
    user_err_count_df.rename(columns={'user_id_': 'user_id'}, inplace=True)
    user_err_count_df.rename(columns={'mun_id_user_max': 'mun_id_user'}, inplace=True)

    # Compute error rates relative to length (per 1000 chars)
    user_err_count_df['err_per_char'] = user_err_count_df['err_count_sum'] / (user_err_count_df['text_len_clean_sum'] / 1000)

    # Aggregate to municipality level ##########################################
    user_err_count_df = user_err_count_df.groupby('mun_id_user') \
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
    # 206363
    # print(user_err_count_df['tweet_count_nunique'].sum())
    # 6086256

    user_err_count_df.columns = ['mun_id_user',
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

    # Export to csv
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
                         'mun_id_user': 'max'})\
                   .reset_index()

    # print(user_stats.tweet_id.sum())
    # 6086256

    user_stats.columns = ['user_id',
                          'tweet_count',
                          'text_len_clean',
                          'verified',
                          'mun_id_user']

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
    # count         850039.000        850039.000
    # mean               0.445             7.584
    # std                0.646            13.795
    # min                0.000             0.001
    # 25%                0.062             0.857
    # 50%                0.250             3.344
    # 75%                0.579             8.929
    # max               33.000           555.556


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
                                                                          'mun_id_user']],
                                                              on='user_id')


    # Aggregate to municipality level ##########################################

    cat_mistakes_per_char_df = cat_mistakes_per_char_df.drop('user_id', axis=1) \
                                                       .groupby('mun_id_user') \
                                                       .mean() \
                                                       .reset_index()

    # Export to csv
    cat_mistakes_per_char_df.to_csv(out_path,
                                    index=False,
                                    sep=';',
                                    encoding='utf-8')


def compute_indicators(filter_date_start, filter_date_end, week, out_dir):

    df = load_data(filter_date_start, filter_date_end, week, out_dir)

    # Compute and store high level indicators
    user_mistakes(df, out_dir + 'mun-level-mistakes-text-indicators-MX.csv')

    # User statistics as input in subsequent indicator generation
    user_statistics = user_stats(df)

    # Store data
    user_mistakes_cat(df, user_statistics, out_dir + 'mun-level-mistakes-category-per-char-MX.csv')
