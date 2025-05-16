
# ######################################################################################################################
# Topic predictions
# ######################################################################################################################

# - Predicts topics for each tweet
# - Exports csv with topic summary statistics per county
# - exports csv with topic and ID per tweet

import tweetnlp # https://github.com/cardiffnlp/tweetnlp
import pandas as pd
import numpy as np
import time
# import swifter
# import time
import dask.dataframe as dd


pd.options.display.width = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Set directories
# ----------------------------------------------------------------------------------------------------------------------

input_path_raw_tweets = "PATH TO FOLDER"
input_path_existing_topics = "PATH TO FOLDER"
output_path = "PATH TO FOLDER"


# Import tweet data
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Import data
    cols = ['id',
            'created_at',
            'user_id',
            'tweet',
            'county_id_tweet',
            'county_name_tweet',
            'county_id_user',
            'county_name_user']


    df = pd.read_csv(input_path_raw_tweets + '/tweet_data.csv',
                     dtype={'county_id_user': str,
                            'county_id_tweet': str},
                     sep=';',
                     encoding='utf-8',
                     usecols=cols)
    print(df.shape[0])
    # 22610134


    # Check which tweets are already in old output file

    existing_topic_df = pd.read_csv(input_path_existing_topics + "/topic-data-old.csv",
                                    dtype={'county_id_user': str,
                                           'county_id_tweet': str},
                                    low_memory=False,
                                    lineterminator='\n',
                                    encoding="utf-8",
                                    sep=";")
    print(existing_topic_df.shape[0])
    # 23139418

    # pad county ids
    existing_topic_df['county_id_user'] = existing_topic_df['county_id_user'] \
                                                           .str \
                                                           .pad(5,
                                                                side='left',
                                                                fillchar='0')

    existing_topic_df['county_id_tweet'] = existing_topic_df['county_id_tweet'] \
                                                            .str \
                                                            .pad(5,
                                                                 side='left',
                                                                 fillchar='0')


    print(df[df['id'].isin(existing_topic_df['id'])]['id'].nunique())
    # 22609453
    # Tweets that newly entering the sample -> need to be newly classified
    print(df[~(df['id'].isin(existing_topic_df['id']))]['id'].nunique())
    # 681
    # Tweets that do no longer belong to sample -> need to be removed
    print(existing_topic_df[~existing_topic_df['id'].isin(df['id'])]['id'].nunique())
    # 529965

    # (old sample - new sample) = (exits + entries)
    print((23139418 - 22610134) == (529965 - 681))
    # True
    print((22609453 + 681) == 22610134)
    # True

    # Select tweets that need to be newly classified
    new_tweets_df = df[~df['id'].isin(existing_topic_df['id'])].reset_index(drop=True)
    print(new_tweets_df.shape)
    # (681, 8)

    ddf = dd.from_pandas(new_tweets_df, chunksize=1500)
    print("data loaded")


if __name__ == '__main__':

    # Load and test topic model
    model = tweetnlp.load_model('topic_classification')
    # print(model.topic(ddf["tweet"][0], return_probability=True))

    print('Begin topic computation')
    # Apply topic model
    start = time.time()

    def compute_topics(part_df):
        print('Start partition', time.strftime("%H:%M:%S", time.localtime()))

        topics = model.topic(part_df['tweet'].to_list(), return_probability=True)
        topics = [x['probability'] for x in topics]
        return part_df.assign(topic=topics)

    meta = [('county_id_tweet', np.int64),
            ('county_name_tweet', object),
            ('county_id_user', np.int64),
            ('county_name_user', object),
            ('id', np.int64),
            ('created_at', object),
            ('tweet', object),
            ('user_id', np.int64),
            ('topic', object)]

    ddf = ddf.map_partitions(lambda part_df: compute_topics(part_df), meta=meta)

    topic_df = ddf.compute(scheduler='threads', num_workers=3)

    end = time.time()
    print('End topic computation')
    print(end - start)


if __name__ == '__main__':

    # Expand columns
    new_topic_df = pd.concat([topic_df.drop('topic', axis=1),
                              pd.DataFrame(topic_df["topic"].tolist())], axis=1)
    print(new_topic_df.shape[0])
    # 681

    # Select tweets with existing classification
    selected_existing_topic_df = existing_topic_df[existing_topic_df['id'].isin(df['id'])]
    print(selected_existing_topic_df.shape[0])
    # 22609453

    print(new_topic_df.shape[0] + selected_existing_topic_df.shape[0])
    # 22610134
    print(df.shape[0])
    # 22610134

    print(df.shape[0] == (new_topic_df.shape[0] +
                          selected_existing_topic_df.shape[0]))
    # True

    # Merge new and existing topic classifications
    updated_topic_df = pd.concat([new_topic_df,
                                  selected_existing_topic_df],
                                 axis=0)

    # Update county information
    cols_to_replace = ['county_id_tweet',
                       'county_name_tweet',
                       'county_id_user',
                       'county_name_user']

    updated_topic_df.drop(labels=cols_to_replace,
                          axis=1,
                          inplace=True)

    updated_topic_df = updated_topic_df.merge(df[['id'] + cols_to_replace],
                                              on='id',
                                              how='left')

    print(updated_topic_df.shape[0])
    # 22610134
    print(updated_topic_df.shape[0] == df.shape[0])
    # True

    # Save topic dataset
    updated_topic_df.to_csv(output_path + "/topic-data.csv",
                            encoding="utf-8",
                            index=False,
                            sep=";")
    print("exported data")
