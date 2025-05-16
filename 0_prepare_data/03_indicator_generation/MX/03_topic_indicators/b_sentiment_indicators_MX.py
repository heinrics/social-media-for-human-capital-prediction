################################################################################
# Load packages
################################################################################
import pandas as pd
import time

pd.options.display.width = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


################################################################################
# Specify data location
################################################################################

input_path = "PATH TO FOLDER"
output_path = "PATH TO FOLDER"


################################################################################
# Import data
################################################################################
def load_data(time_split, filter_date_end, week, out_dir):

    sent_df = pd.read_csv(input_path + "/text-sentiment-data-mx.csv",
                          dtype={'mud_id_user': str,
                                 'mud_id_tweet': str},
                          low_memory=False,
                          lineterminator='\n',
                          encoding="utf-8",
                          sep=";")

    sent_df['created_at_tm'] = sent_df['created_at'].apply(lambda x: time.strptime(x, '%Y-%m-%d %H:%M:%S+00:00'))

    # Filter tweets by date
    if week <= 9:
        if filter_date_end:
            sent_df = sent_df[(sent_df['created_at_tm'] > time_split) &
                              (sent_df['created_at_tm'] < filter_date_end)]
        else:
            sent_df = sent_df[(sent_df['created_at_tm'] > time_split)]

    if (week > 9) and filter_date_end and time_split:
        print('Not used data!')
        exit()
        # sent_df = sent_df[(sent_df['created_at_tm'] < filter_date_end) |
        #                     (sent_df['created_at_tm'] > time_split)]


    # Load and update correct user county data
    sent_df.rename(columns={'mun_id_user': 'mun_id_user_full',
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

    sent_df = sent_df.merge(user_id_df.drop_duplicates('user_id'),
                            on='user_id',
                            how='left')

    return sent_df


################################################################################
# Aggregation
################################################################################

def user_sentiment(df, out_path):

    sent_list = ['negative',
                 'positive',
                 'hate',
                 'offensive']

    # Aggregate on user level ####################################################
    user_sent_df = df.groupby(['user_id', 'mun_id_user']) \
                     [sent_list] \
                     .mean() \
                     .reset_index()

    # print(user_sent_df.shape)
    # (206363, 5)

    # Aggregate to municipality level ##########################################
    county_sent_df = user_sent_df.groupby('mun_id_user') \
                                  [sent_list] \
                                  .mean() \
                                  .reset_index()

    county_sent_df.to_csv(out_path,
                          index=False,
                          sep=';',
                          encoding='utf-8')

def compute_indicators(filter_date_start, filter_date_end, week, out_dir):

    df = load_data(filter_date_start, filter_date_end, week, out_dir)

    user_sentiment(df, out_dir + 'mun-level-sentiment-indicators-MX.csv')
