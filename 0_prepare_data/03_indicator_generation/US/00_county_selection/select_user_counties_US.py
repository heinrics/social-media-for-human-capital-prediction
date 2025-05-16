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
input_path = main_path + '01_data/US/02-processed-data'

# Load data ####################################################################

# Time periods
selected_days = pd.read_csv(main_path  + '01_data/US/02-processed-data/selected_dates-days.csv',
                             sep=';')

selected_weeks = pd.read_csv(main_path + '01_data/US/02-processed-data/selected_dates.csv',
                             sep=';')

# Tweet data
tweet_df = pd.read_csv(input_path + "/tweet_data.csv",
                           sep=";",
                           # usecols=cols,
                           dtype={'county_id_tweet': 'str',
                                  'county_id_user': 'str'})

tweet_df.rename(columns={'county_id_user': 'county_id_user_full',
                         'county_name_user': 'county_name_user_full'},
                inplace=True)

tweet_df['created_at_tm'] = tweet_df['created_at']\
                                    .apply(lambda x: time.strptime(x, '%Y-%m-%d %H:%M:%S+00:00'))


def filter_time_window(df, filter_date_start, filter_date_end):

    # Filter tweets by date
    if filter_date_start:
        df = df[df['created_at_tm'] > filter_date_start]

    if filter_date_end:
        df = df[df['created_at_tm'] < filter_date_end]

    df.drop_duplicates(subset='id', keep='first', inplace=True)

    return df


def select_user_county(df, out_path):

    # Get main county/counties for each user
    df["main_county_all"] = df.groupby("user_id")["county_id_tweet"].transform(lambda x: ",".join(st.multimode(x)))
    df["nr_main_counties"] = df.groupby("user_id")["county_id_tweet"].transform(lambda x: len(st.multimode(x)))

    # Get main county/counties for each user excluding tweets
    # during workhours or weekends (when people might not be at home)
    df["county_home"] = df["county_id_tweet"] * 1
    df["county_home"].loc[(df["weekday"] == False) |
                          (df["workhours"] == True)] = np.nan
    df["main_county_home"] = df.groupby("user_id")["county_home"]\
                               .transform(lambda x: ",".join(st.multimode(x[x.notna()])))

    df["nr_home_counties"] = df.groupby("user_id")["county_home"].transform(lambda x: len(st.multimode(x[x.notna()])))

    df.reset_index(drop=True, inplace=True)

    df["main_county"] = df.apply(lambda x: x["main_county_all"] if x["nr_main_counties"] == 1
                                           else (x["main_county_home"] if x["nr_home_counties"] == 1
                                                 else np.nan),
                                           axis=1)

    df = df[~df["main_county"].isna()]

    df.drop(['main_county_all',
             'nr_main_counties',
             'county_home',
             'main_county_home',
             'nr_home_counties'],
            inplace=True,
            axis=1)

    df.rename({"main_county": "county_id_user"},
              axis=1, inplace=True)

    df = df.merge(df[["county_id_tweet", "county_name_tweet"]]
                    .sort_values('county_name_tweet')
                    .rename({"county_name_tweet": "county_name_user",
                             "county_id_tweet": "county_id_user"}, axis=1)
                    .drop_duplicates(),
                  on="county_id_user",
                  how="left")

    df.drop("created_at_tm", inplace=True, axis=1)
    # df.rename({"NAME": "county_name_user"}, axis=1, inplace=True)


    os.makedirs(main_path + \
                '01_data/US/03-indicators/selected_con/' + \
                out_path, exist_ok=True)

    df.to_csv(main_path +
              '01_data/US/03-indicators/selected_con/' +
              out_path +
              'tweet_data_selection.csv',
              index=False,
              sep=';',
              encoding='utf-8')


# Run function for cumulative time slices ######################################
for row in selected_weeks.iterrows():

    print(row[1]['week'])

    min_date = None
    max_date = None

    if row[1]['min_date'] == row[1]['min_date']:
        min_date = time.strptime(row[1]['min_date'], '%Y-%m-%d')

    if row[1]['max_date'] == row[1]['max_date']:
        max_date = time.strptime(row[1]['max_date'], '%Y-%m-%d')

    output_folder = 'weeks/week_' + str(row[1]['week']).zfill(2) + '/'

    df = filter_time_window(tweet_df.copy(), min_date, max_date)
    select_user_county(df, output_folder)


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

    df = filter_time_window(tweet_df.copy(), min_date, max_date)
    select_user_county(df, output_folder)