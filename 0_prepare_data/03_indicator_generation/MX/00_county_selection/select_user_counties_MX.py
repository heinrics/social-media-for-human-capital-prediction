import pandas as pd
import time
import statistics as st
import numpy as np
import os

pd.options.display.width = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# Set directories ##############################################################
################################################################################

main_path = "PATH TO FOLDER"
input_path = main_path + '01_data/MX/02-processed-data'


# Load data ####################################################################

# Time periods
selected_days = pd.read_csv(main_path + '01_data/MX/02-processed-data/selected_dates-days.csv',
                            sep=';')

selected_weeks = pd.read_csv(main_path + '01_data/MX/02-processed-data/selected_dates.csv',
                             sep=';')

# Tweet data
tweet_df = pd.read_csv(input_path + "/tweet_data.csv",
                       sep=";",
                       # usecols=cols,
                       dtype={'mun_id_tweet': 'str',
                              'mun_id_user': 'str'})

tweet_df.rename(columns={'mun_id_user': 'mun_id_user_full',
                         'mun_name_user': 'mun_name_user_full'},
                inplace=True)

tweet_df['created_at_tm'] = tweet_df['created_at']\
                                    .apply(lambda x: time.strptime(x, '%Y-%m-%d %H:%M:%S+00:00'))


def filter_time_window(df, time_split, filter_date_end, week):

    # Filter tweets by date
    if week <= 9:
        if filter_date_end:
            df = df[(df['created_at_tm'] > time_split) &
                    (df['created_at_tm'] < filter_date_end)]
        else:
            df = df[(df['created_at_tm'] > time_split)]

    if (week > 9) and filter_date_end and time_split:
        print('wrong time period')
        exit()
        # df = df[(df['created_at_tm'] < filter_date_end) |
        #         (df['created_at_tm'] > time_split)]

    df.drop_duplicates(subset='id', keep='first', inplace=True)

    return df


def select_user_muns(df, out_path):

    # Get main muns for each user
    df["main_mun_all"] = df.groupby("user_id")["mun_id_tweet"].transform(lambda x: ",".join(st.multimode(x)))
    df["nr_main_counties"] = df.groupby("user_id")["mun_id_tweet"].transform(lambda x: len(st.multimode(x)))

    # Get main muns/counties for each user excluding tweets
    # during workhours or weekends (when people might not be at home)
    df["mun_home"] = df["mun_id_tweet"] * 1
    df["mun_home"].loc[(df["weekday"] == False) |
                       (df["workhours"] == True)] = np.nan
    df["main_mun_home"] = df.groupby("user_id")["mun_home"]\
                            .transform(lambda x: ",".join(st.multimode(x[x.notna()])))

    df["nr_home_counties"] = df.groupby("user_id")["mun_home"].transform(lambda x: len(st.multimode(x[x.notna()])))

    df.reset_index(drop=True, inplace=True)

    df["main_muns"] = df.apply(lambda x: x["main_mun_all"] if x["nr_main_counties"] == 1
                                           else (x["main_mun_home"] if x["nr_home_counties"] == 1
                                                 else np.nan),
                                           axis=1)

    df = df[~df["main_muns"].isna()]

    df.drop(['main_mun_all',
             'nr_main_counties',
             'mun_home',
             'main_mun_home',
             'nr_home_counties'],
            inplace=True,
            axis=1)

    df.rename({"main_muns": "mun_id_user"},
              axis=1, inplace=True)

    df = df.merge(df[["mun_id_tweet", "mun_name_tweet"]]
                    .sort_values('mun_name_tweet')
                    .rename({"mun_name_tweet": "mun_name_user",
                             "mun_id_tweet": "mun_id_user"}, axis=1)
                    .drop_duplicates(subset='mun_id_user'),
                  on="mun_id_user",
                  how="left")

    df.drop("created_at_tm", inplace=True, axis=1)
    # df.rename({"NAME": "mun_name_user"}, axis=1, inplace=True)

    os.makedirs(main_path + \
                '01_data/MX/03-indicators/selected_mun/' + \
                out_path, exist_ok=True)

    df.to_csv(main_path +
              '01_data/MX/03-indicators/selected_mun/' +
              out_path +
              'tweet_data_selection.csv',
              index=False,
              sep=';',
              encoding='utf-8')


# Run function for cumulative time slices ######################################
for row in selected_weeks.iterrows():

    if row[1]['week'] <= 9:

        print(row[1]['week'])

        min_date = 0
        max_date = 0

        if row[1]['min_date'] == row[1]['min_date']:
            min_date = time.strptime(row[1]['min_date'], '%Y-%m-%d')

        if row[1]['max_date'] == row[1]['max_date']:
            max_date = time.strptime(row[1]['max_date'], '%Y-%m-%d')

        output_folder = 'weeks/week_' + str(row[1]['week']).zfill(2) + '/'

        selected_df = filter_time_window(tweet_df.copy(), min_date, max_date, row[1]['week'])
        select_user_muns(selected_df, output_folder)


# Days datasets ################################################################
for row in selected_days.iterrows():

    print(row[1]['day'])

    min_date = 0
    max_date = 0

    if row[1]['min_date'] == row[1]['min_date']:
        min_date = time.strptime(row[1]['min_date'], '%Y-%m-%d')

    if row[1]['max_date'] == row[1]['max_date']:
        max_date = time.strptime(row[1]['max_date'], '%Y-%m-%d')

    output_folder = 'days/day_' + str(row[1]['day']).zfill(2) + '/'

    selected_df = filter_time_window(tweet_df.copy(), min_date, max_date, row[1]['day'])
    select_user_muns(selected_df, output_folder)
