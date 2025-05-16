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
def load_data(filter_date_start, filter_date_end, out_dir):

    topic_df = pd.read_csv(input_path + "/topic-data.csv",
                           dtype={'county_id_user': str,
                                  'county_id_tweet': str},
                           low_memory=False,
                           lineterminator='\n',
                           encoding="utf-8",
                           sep=";")

    # pad county ids
    topic_df['county_id_user'] = topic_df['county_id_user']\
                                         .astype(str)\
                                         .str\
                                         .pad(5,
                                              side='left',
                                              fillchar='0')

    topic_df['county_id_tweet'] = topic_df['county_id_tweet']\
                                           .astype(str)\
                                           .str\
                                           .pad(5,
                                                side='left',
                                                fillchar='0')

    topic_df['created_at_tm'] = topic_df['created_at'].apply(lambda x: time.strptime(x, '%Y-%m-%d %H:%M:%S+00:00'))

    # Filter tweets by date
    if filter_date_start:
        topic_df = topic_df[topic_df['created_at_tm'] >= filter_date_start]

    if filter_date_end:
        topic_df = topic_df[topic_df['created_at_tm'] <= filter_date_end]

    # Load and update correct user county data
    topic_df.rename(columns={'county_id_user': 'county_id_user_full',
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

    topic_df = topic_df.merge(user_id_df.drop_duplicates('user_id'),
                              on='user_id',
                              how='left')

    return topic_df


################################################################################
# Generate Topic Indicators
################################################################################

# Aggregate on user level ######################################################

def user_topics(df, out_path):

    topic_list = ['arts_&_culture',
                  'business_&_entrepreneurs',
                  'celebrity_&_pop_culture',
                  'diaries_&_daily_life',
                  'family',
                  'fashion_&_style',
                  'film_tv_&_video',
                  'fitness_&_health',
                  'food_&_dining',
                  'gaming',
                  'learning_&_educational',
                  'music',
                  'news_&_social_concern',
                  'other_hobbies',
                  'relationships',
                  'science_&_technology',
                  'sports',
                  'travel_&_adventure',
                  'youth_&_student_life']


    # Aggregate on user level ##################################################
    user_topic_df = df.groupby(['user_id', 'county_id_user']) \
                      [topic_list]\
                      .mean()\
                      .reset_index()

    # Standardize topics
    user_topic_df['score_sum'] = user_topic_df[topic_list].sum(axis=1)

    for topic in topic_list:
        user_topic_df[topic] = user_topic_df[topic] / user_topic_df['score_sum']

    # print(user_topic_df[topic_list].sum(axis=1))


    # Aggregate on county level ####################################################
    county_topic_df = user_topic_df.groupby('county_id_user')\
                                   [topic_list]\
                                   .mean()\
                                   .reset_index()

    county_topic_df.to_csv(out_path,
                           index=False,
                           sep=';',
                           encoding='utf-8')

def compute_indicators(filter_date_start, filter_date_end, out_dir):

    df = load_data(filter_date_start, filter_date_end, out_dir)

    user_topics(df, out_dir + 'county-level-topic-indicators-US.csv')
