
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

from dask.diagnostics import ProgressBar

pd.options.display.width = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Set directories
# ----------------------------------------------------------------------------------------------------------------------

input_path = 'PATH TO FOLDER'
output_path = 'PATH TO FOLDER'


# Import tweet data
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Import data
    #ddf = dd.read_csv(input_path + r"\tweet_data.csv", sep=";", on_bad_lines='warn')
    cols = ['id',
            'created_at',
            'user_id',
            'tweet',
            'en_translation',
            'mun_id_tweet',
            'mun_name_tweet',
            'mun_id_user',
            'mun_name_user']

    df = pd.read_csv(input_path + "/translation-data.csv",
                     lineterminator='\n',
                     sep=";",
                     usecols=cols)

    # Separate tweets with missing translations
    missing_df = df[df['en_translation'].isna()].reset_index(drop=True)
    df = df[~df['en_translation'].isna()].reset_index(drop=True)

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
        try:
            # print('Start partition', time.strftime("%H:%M:%S", time.localtime()))

            topics = model.topic(part_df['en_translation'].to_list(), return_probability=True)
            topics = [x['probability'] for x in topics]
            return part_df.assign(topic=topics)

        except Exception as e:
            print('translate_es_to_en')
            print(e)

            return part_df.assign(topic=np.nan)


    meta = [('mun_id_tweet', np.int64),
            ('mun_name_tweet', object),
            ('mun_id_user', np.int64),
            ('mun_name_user', object),
            ('id', np.int64),
            ('created_at', object),
            ('tweet', object),
            ('user_id', np.int64),
            ('en_translation', object),
            ('topic', object)]

    ddf = ddf.map_partitions(lambda part_df: compute_topics(part_df), meta=meta)

    ProgressBar().register()
    topic_df = ddf.compute(scheduler='threads', num_workers=3)
    # [########################################] | 100% Completed | 5hr 0m

    end = time.time()
    print('End topic computation')
    print(end - start)


if __name__ == '__main__':

    # Expand columns
    topic_export_df = pd.concat([topic_df.drop('topic', axis=1),
                                 pd.DataFrame(topic_df["topic"].tolist())], axis=1)

    print("expanded to columns")

    # Add the tweets with missing translations
    topic_export_df = pd.concat([topic_export_df, missing_df],
                                axis=0).reset_index(drop=True)
