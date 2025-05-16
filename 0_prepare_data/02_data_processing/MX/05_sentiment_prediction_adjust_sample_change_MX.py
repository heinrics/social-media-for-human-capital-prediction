# ######################################################################################################################
# Sentiment predictions
# ######################################################################################################################


if __name__ == '__main__':

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

    translation_df = pd.read_csv(input_path + "/translation-data.csv",
                                 lineterminator='\n',
                                 sep=";",
                                 usecols=cols)

    # Existing sentiment data
    existing_sentiment_df = pd.read_csv(output_path + "text-sentiment-data-mx-old.csv",
                                        encoding="utf-8",
                                        sep=";",
                                        lineterminator='\n')

    missing_tweets = translation_df[~translation_df['id'].isin(existing_sentiment_df['id'])]

    ddf = dd.from_pandas(missing_tweets, chunksize=500)
    print("data loaded")


if __name__ == '__main__':

    # Load sentiment models
    sent_model = tweetnlp.load_model('sentiment')
    # sent_multi_model = tweetnlp.load_model('sentiment', multilingual=True)
    hate_model = tweetnlp.load_model('hate')
    offl_model = tweetnlp.load_model('offensive')

    print('Begin sentiment computation')

    # Apply NLP predictions
    start = time.time()

    # Sentiment
    def compute_sentiment(part_df):

        try:
            # print('Start sentiment partition', time.strftime("%H:%M:%S", time.localtime()))

            sentiment = sent_model.sentiment(part_df['en_translation'].to_list(), return_probability=True)
            sentiment = [x['probability'] for x in sentiment]

            # sentiment_multi = sent_model.sentiment(part_df['tweet'].to_list(), return_probability=True)
            # sentiment_multi = [x['probability'] for x in sentiment_multi]

            hate_speech = hate_model.hate(part_df['en_translation'].to_list(), return_probability=True)
            hate_speech = [x['probability'] for x in hate_speech]

            offen_lang = offl_model.offensive(part_df['en_translation'].to_list(), return_probability=True)
            offen_lang = [x['probability'] for x in offen_lang]

            return part_df.assign(sentiment=sentiment,
                                  # sentiment_multi=sentiment_multi,
                                  hate_speech=hate_speech,
                                  offen_lang=offen_lang)

        except Exception as e:
            print('translate_es_to_en')
            print(e)

            return part_df.assign(sentiment=np.nan,
                                  # sentiment_multi=sentiment_multi,
                                  hate_speech=np.nan,
                                  offen_lang=np.nan)


    meta = [('mun_id_tweet', np.int64),
            ('mun_name_tweet', object),
            ('mun_id_user', np.int64),
            ('mun_name_user', object),
            ('id', np.int64),
            ('created_at', object),
            ('tweet', object),
            ('user_id', np.int64),
            ('en_translation', object),
            ('sentiment', object),
            # ('sentiment_multi', object),
            ('hate_speech', object),
            ('offen_lang', object)]

    ddf = ddf.map_partitions(lambda part_df: compute_sentiment(part_df), meta=meta)

    ProgressBar().register()
    sentiment_df = ddf.compute(scheduler='threads', num_workers=3)

    end = time.time()
    print('End NLP computation')
    print(end - start)


if __name__ == '__main__':

    # Expand columns
    sentiment_export_df = pd.concat([sentiment_df.drop(columns=['sentiment', 'hate_speech', 'offen_lang'], axis=1).reset_index(drop=True),
                                     pd.DataFrame(sentiment_df["sentiment"].tolist()),
                                     # pd.DataFrame(sentiment_df["sentiment_multi"].tolist()),
                                     pd.DataFrame(sentiment_df["hate_speech"].tolist()),
                                     pd.DataFrame(sentiment_df["offen_lang"].tolist())], axis=1)
    print("expanded to columns")


    updated_sentiment_df = pd.concat([existing_sentiment_df, sentiment_export_df])

    updated_sentiment_df = updated_sentiment_df[updated_sentiment_df['id'].isin(translation_df['id'])]
