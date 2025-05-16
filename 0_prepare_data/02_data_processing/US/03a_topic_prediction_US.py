
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
    topic_df = pd.concat([topic_df.drop('topic', axis=1),
                          pd.DataFrame(topic_df["topic"].tolist())], axis=1)
    print("expanded to columns")

    # Save topic dataset
    # topic_df.to_csv(output_path + "/topic-data.csv", encoding="utf-8", index=False, sep=";")
    print("exported data")
