
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

input_path = "PATH TO FOLDER"
output_path = "PATH TO FOLDER"


# Import tweet data
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Import data
    #ddf = dd.read_csv(input_path + r"\tweet_data.csv", sep=";", on_bad_lines='warn')
    cols = ['id',
            'created_at',
            'user_id',
            'tweet',
            'county_id_tweet',
            'county_name_tweet',
            'county_id_user',
            'county_name_user']

    df = pd.read_csv(input_path + "/tweet_data.csv",
                     sep=";",
                     usecols=cols)

    ddf = dd.from_pandas(df, chunksize=1500)
    print("data loaded")


if __name__ == '__main__':

    # Load and test topic model
    sent_model = tweetnlp.load_model('sentiment')
    hate_model = tweetnlp.load_model('hate')
    offl_model = tweetnlp.load_model('offensive')

    print('Begin sentiment computation')

    # Apply NLP predictions
    start = time.time()

    # Sentiment
    def compute_sentiment(part_df):
        print('Start sentiment partition', time.strftime("%H:%M:%S", time.localtime()))

        sentiment = sent_model.sentiment(part_df['tweet'].to_list(), return_probability=True)
        sentiment = [x['probability'] for x in sentiment]

        hate_speech = hate_model.hate(part_df['tweet'].to_list(), return_probability=True)
        hate_speech = [x['probability'] for x in hate_speech]

        offen_lang = offl_model.offensive(part_df['tweet'].to_list(), return_probability=True)
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

    ddf = ddf.map_partitions(lambda part_df: compute_sentiment(part_df), meta=meta)

    sentiment_df = ddf.compute(scheduler='threads', num_workers=3)

    end = time.time()
    print('End NLP computation')
    print(end - start)
    # 212616.96738886833

if __name__ == '__main__':

    # Expand columns
    sentiment_df = pd.concat([sentiment_df.drop(columns=['sentiment', 'hate_speech', 'offen_lang'], axis=1),
                              pd.DataFrame(sentiment_df["sentiment"].tolist()),
                              pd.DataFrame(sentiment_df["hate_speech"].tolist()),
                              pd.DataFrame(sentiment_df["offen_lang"].tolist())], axis=1)
    print("expanded to columns")

    # Save topic dataset
    sentiment_df.to_csv(output_path + "/text-sentiment-data.csv", encoding="utf-8", index=False, sep=";")
    print("exported data")
