
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
input_path_existing_sentiment = "PATH TO FOLDER"
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

    existing_sent_df = pd.read_csv(input_path_existing_sentiment + "/text-sentiment-data-old.csv",
                                    dtype={'county_id_user': str,
                                           'county_id_tweet': str},
                                    low_memory=False,
                                    lineterminator='\n',
                                    encoding="utf-8",
                                    sep=";")
    print(existing_sent_df.shape[0])
    # 23139418

    # pad county ids
    existing_sent_df['county_id_user'] = existing_sent_df['county_id_user'] \
                                                         .str \
                                                         .pad(5,
                                                              side='left',
                                                              fillchar='0')

    existing_sent_df['county_id_tweet'] = existing_sent_df['county_id_tweet'] \
                                                          .str \
                                                          .pad(5,
                                                               side='left',
                                                               fillchar='0')


    print(df[df['id'].isin(existing_sent_df['id'])]['id'].nunique())
    # 22609453
    # Tweets that newly entering the sample -> need to be newly classified
    print(df[~(df['id'].isin(existing_sent_df['id']))]['id'].nunique())
    # 681
    # Tweets that do no longer belong to sample -> need to be removed
    print(existing_sent_df[~existing_sent_df['id'].isin(df['id'])]['id'].nunique())
    # 529965

    # (old sample - new sample) = (exits + entries)
    print((23139418 - 22610134) == (529965 - 681))
    # True
    print((22609453 + 681) == 22610134)
    # True

    # Select tweets that need to be newly classified
    new_tweets_df = df[~df['id'].isin(existing_sent_df['id'])].reset_index(drop=True)
    print(new_tweets_df.shape)
    # (681, 8)

    ddf = dd.from_pandas(new_tweets_df, chunksize=1500)
    print("data loaded")


if __name__ == '__main__':

    # Load and test topic model
    sent_model = tweetnlp.load_model('sentiment')
    hate_model = tweetnlp.load_model('hate')
    offl_model = tweetnlp.load_model('offensive')

    print('Begin sentiment computation')

    # Apply topic model
    start = time.time()


    # Sentiment
    def compute_sentiment(part_df):
        print('Start sentiment partition',
              time.strftime("%H:%M:%S", time.localtime()))

        sentiment = sent_model.sentiment(part_df['tweet'].to_list(),
                                         return_probability=True)
        sentiment = [x['probability'] for x in sentiment]


        hate_speech = hate_model.hate(part_df['tweet'].to_list(),
                                      return_probability=True)
        hate_speech = [x['probability'] for x in hate_speech]


        offen_lang = offl_model.offensive(part_df['tweet'].to_list(),
                                          return_probability=True)
        offen_lang = [x['probability'] for x in offen_lang]


        return part_df.assign(sentiment=sentiment,
                              hate_speech=hate_speech,
                              offen_lang=offen_lang)


    meta = [('county_id_tweet', np.int64),
            ('county_name_tweet', object),
            ('county_id_user', np.int64),
            ('county_name_user', object),
            ('id', np.int64),
            ('created_at', object),
            ('tweet', object),
            ('user_id', np.int64),
            ('sentiment', object),
            ('hate_speech', object),
            ('offen_lang', object)]


    ddf = ddf.map_partitions(lambda part_df: compute_sentiment(part_df),
                             meta=meta)

    sent_df = ddf.compute(scheduler='threads', num_workers=3)

    end = time.time()
    print('End NLP computation')
    print(end - start)


if __name__ == '__main__':

    # Expand columns
    new_sent_df = pd.concat([sent_df.drop(['sentiment', 'hate_speech', 'offen_lang'], axis=1),
                             pd.DataFrame(sent_df["sentiment"].tolist()),
                             pd.DataFrame(sent_df["hate_speech"].tolist()),
                             pd.DataFrame(sent_df["offen_lang"].tolist())], axis=1)
    print(new_sent_df.shape[0])
    # 681

    # Select tweets with existing classification
    selected_existing_sent_df = existing_sent_df[existing_sent_df['id'].isin(df['id'])]
    print(selected_existing_sent_df.shape[0])
    # 22609453

    print(new_sent_df.shape[0] + selected_existing_sent_df.shape[0])
    # 22610134
    print(df.shape[0])
    # 22610134

    print(df.shape[0] == (new_sent_df.shape[0] +
                          selected_existing_sent_df.shape[0]))
    # True

    # Merge new and existing topic classifications
    updated_sent_df = pd.concat([new_sent_df,
                                 selected_existing_sent_df],
                                 axis=0)

    # Update county information
    cols_to_replace = ['county_id_tweet',
                       'county_name_tweet',
                       'county_id_user',
                       'county_name_user']

    updated_sent_df.drop(labels=cols_to_replace,
                         axis=1,
                         inplace=True)

    updated_sent_df = updated_sent_df.merge(df[['id'] + cols_to_replace],
                                            on='id',
                                            how='left')

    print(updated_sent_df.shape[0])
    # 22610134
    print(updated_sent_df.shape[0] == df.shape[0])
    # True

    # Save topic dataset
    updated_sent_df.to_csv(output_path + "/text-sentiment-data.csv",
                           encoding="utf-8",
                           index=False,
                           sep=";")
    print("exported data")
