import pandas as pd
import time

pd.options.display.width = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Set directories ##############################################################
################################################################################

main_path = "PATH TO FOLDER"
input_path = main_path + '01_data/MX/02-processed-data'


# Load data ####################################################################
################################################################################

cols = ['mun_id_user',
        'created_at',
        'user_id',
        'tweet']

tweet_df = pd.read_csv(input_path + "/tweet_data.csv",
                       sep=";",
                       usecols=cols)

tweet_df['created_at_tm'] = tweet_df['created_at'].apply(lambda x: time.strptime(x, '%Y-%m-%d %H:%M:%S+00:00'))

min_date = tweet_df['created_at_tm'].min()
max_date = tweet_df['created_at_tm'].max()

tweet_df['created_at_year_month'] = tweet_df['created_at'].str[0:7]
print(tweet_df['created_at_year_month'].value_counts().sort_index())
# 2020-12     342651
# 2021-01    1406626
# 2021-02    1227788
# 2021-03     420503
# 2021-06          1
# 2021-07    1282158
# 2021-08    1378971
# 2021-09      27558

tweet_df['created_at_year_month_day'] = tweet_df['created_at'].str[0:10]
sample_days = tweet_df['created_at_year_month_day'].value_counts().sort_index()
print(len(sample_days))
# 141


# Week Sample 1 ################################################################

sampled_days_1 = tweet_df[tweet_df['created_at_year_month'].str.startswith(('2021-06',
                                                                            '2021-07',
                                                                            '2021-08',
                                                                            '2021-09'))]\
                         ['created_at_year_month_day'].value_counts().sort_index()
print(len(sampled_days_1))
# 64

# Intervals need be the same as US
# ignore single tweet day
select_dates_1 = sampled_days_1.iloc[1::7].index.to_list()
select_dates_1.append(None)
select_dates_1 = select_dates_1[1:]

select_dates_1 = pd.DataFrame(select_dates_1,
                              columns=['max_date'])
# To exclude first sampling period
select_dates_1['min_date'] = '2021-04-01'


# Week Sample 2 ################################################################
sampled_days_2 = tweet_df[tweet_df['created_at_year_month'].str.startswith(('2020-12',
                                                                            '2021-01',
                                                                            '2021-02',
                                                                            '2021-03'))]\
                         ['created_at_year_month_day'].value_counts().sort_index()
print(len(sampled_days_2))
# 77

select_dates_2 = sampled_days_2.iloc[::7].index.to_list()
select_dates_2.append(None)
select_dates_2 = select_dates_2[1:]


select_dates_2 = pd.DataFrame(select_dates_2,
                              columns=['max_date'])
# To exclude first sampling period
select_dates_2['min_date'] = None
select_dates_2.loc[:(select_dates_2.shape[0]-2), 'min_date'] = '2021-04-01'

# Joint week sample ############################################################
selected_dates = pd.concat([select_dates_1, select_dates_2], axis=0)\
                   .reset_index(drop=True)

selected_dates['week'] = selected_dates.index + 1

selected_dates.to_csv(main_path + '01_data/MX/02-processed-data/selected_dates.csv',
                      sep=';',
                      index=False)


# Day sample ###################################################################

week_sample = pd.DataFrame(sampled_days_1[3:10].index,
                           columns=['max_date'])
week_sample['min_date'] = sampled_days_1[1:2].index[0]

week_sample['day'] = week_sample.index + 1

week_sample.to_csv(main_path + '01_data/MX/02-processed-data/selected_dates-days.csv',
                   sep=';',
                   index=False)
