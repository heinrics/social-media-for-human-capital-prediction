import pandas as pd
import time

pd.options.display.width = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# Set directories ##############################################################
################################################################################

main_path = "PATH TO FOLDER"
input_path = main_path + '01_data/MX/03-indicators/'

def load_data(time_split, filter_date_end, week, out_dir):

    cols = ['id',
            'created_at',
            'user_id',
            'mun_id_user',
            "weekday",
            "workhours",
            "user_followers",
            "user_friends",
            "user_statuses",
            "user_nr_muns",
            "iphone",
            "insta"]

    time_plat_df = pd.read_csv('/'.join(out_dir.split('/')[0:-2]).replace('03-indicators/',
                                                                          '03-indicators/selected_mun/') +
                               "/tweet_data_selection.csv",
                               sep=";",
                               usecols=cols,
                               low_memory=False,
                               lineterminator='\n',
                               encoding="utf-8")

    return time_plat_df


def user_time_platform(df):

    user_df = df.groupby(["user_id",
                          "mun_id_user"])[["weekday",
                                           "workhours",
                                           "user_followers",
                                           "user_friends",
                                           "user_statuses",
                                           "user_nr_muns",
                                           "iphone",
                                           "insta"]].mean().reset_index()

    return user_df


def mun_time_platform(df, out_path):

    mun_user = df.groupby("mun_id_user")\
                 .agg({"weekday": "mean",
                       "workhours": "mean",
                       "user_followers": "median",
                       "user_friends": "median",
                       "user_statuses": "median",
                       "user_nr_muns": "mean",
                       "iphone": "mean",
                       "insta": "mean"}).reset_index()

    mun_user.rename({'weekday': 'share_weekdays',
                     'workhours': 'share_workhours',
                     'user_followers': 'median_followers',
                     'user_friends': 'median_friends',
                     'user_statuses': 'median_statuses',
                     'user_nr_muns': 'avg_mobility',
                     'iphone': 'share_iphone',
                     'insta': 'share_insta'},
                    inplace=True,
                    axis=1)

    mun_user.to_csv(out_path,
                    index=False,
                    sep=';',
                    encoding='utf-8')


def compute_indicators(filter_date_start, filter_date_end, week, out_dir):

    df = load_data(filter_date_start, filter_date_end, week, out_dir)

    df = user_time_platform(df)

    mun_time_platform(df, out_dir + 'mun-level-time-platform-indicators-MX.csv')
