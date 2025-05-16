import pandas as pd
import time

pd.options.display.width = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Set directories ##############################################################
################################################################################

base_path = "PATH TO FOLDER"
input_path = base_path + '01_data/US/02-processed-data'


# Load data ####################################################################
################################################################################

cols = ['county_id_user',
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
# 2021-07    10915865
# 2021-08    11474565
# 2021-09      219704

tweet_df['created_at_year_month_day'] = tweet_df['created_at'].str[0:10]
sampled_days = tweet_df['created_at_year_month_day'].value_counts().sort_index()
print(len(sampled_days))
# 63

# Week Sample ##################################################################

select_dates = sampled_days.iloc[::7].index.to_list()
select_dates.append(None)
select_dates = select_dates[1:]

selected_dates = pd.DataFrame(select_dates,
                              columns=['max_date'])
selected_dates['min_date'] = None
selected_dates['week'] = selected_dates.index + 1

#      max_date min_date  week
# 0  2021-07-08     None     1
# 1  2021-07-15     None     2
# 2  2021-07-22     None     3
# 3  2021-07-29     None     4
# 4  2021-08-05     None     5
# 5  2021-08-12     None     6
# 6  2021-08-19     None     7
# 7  2021-08-26     None     8
# 8        None     None     9

# selected_dates.to_csv(path_sebastian_linux_dropbox + '01_data/US/02-processed-data/selected_dates.csv',
#                       sep=';',
#                       index=False)


# Day sample ###################################################################

week_sample = pd.DataFrame(sampled_days[2:9].index,
                           columns=['max_date'])
week_sample['min_date'] = sampled_days[0:1].index[0]

week_sample['day'] = week_sample.index + 1

# week_sample.to_csv(path_sebastian_linux_dropbox + '01_data/US/02-processed-data/selected_dates-days.csv',
#                    sep=';',
#                    index=False)
