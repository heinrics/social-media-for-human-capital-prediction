# Notes on machine translation approach to classification:
# Multilingual classification:  https://medium.com/elca-it/zero-shot-multilingual-classification-models-13c3c0f44ad7
#                               https://bilat.xyz/pdf/cltd.pdf
# Marian NMT: https://marian-nmt.github.io/
# Implementation: https://huggingface.co/docs/transformers/model_doc/marian
# Traind model: https://huggingface.co/Helsinki-NLP/opus-mt-es-en

import time

import pandas as pd
import numpy as np

import dask.dataframe as dd
from dask.diagnostics import ProgressBar

import torch
from transformers import MarianMTModel, MarianTokenizer

pd.options.display.width = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# Set directories ##############################################################
################################################################################

input_path = "PATH TO FOLDER"
output_path = "PATH TO FOLDER"


# Import tweet data ############################################################
################################################################################

if __name__ == '__main__':

    # Import data
    #ddf = dd.read_csv(input_path + r"\tweet_data.csv", sep=";", on_bad_lines='warn')
    cols = ['id',
            'created_at',
            'user_id',
            'tweet',
            'mun_id_tweet',
            'mun_name_tweet',
            'mun_id_user',
            'mun_name_user']

    df = pd.read_csv(input_path + "/tweet_data.csv",
                     sep=";",
                     usecols=cols)


    ddf = dd.from_pandas(df, chunksize=50)
    print("data loaded")


# Translation ##################################################################
################################################################################


if __name__ == '__main__':

    model_name = "Helsinki-NLP/opus-mt-es-en"

    model_0 = MarianMTModel.from_pretrained(model_name).to('cuda:0')
    model_1 = MarianMTModel.from_pretrained(model_name).to('cuda:1')
    model_2 = MarianMTModel.from_pretrained(model_name).to('cuda:2')
    model_list = [model_0, model_1, model_2]

    gpu_store = [0, 1, 2]

    start = time.time()

    def translate_es_to_en(part_df):

        try:

            # Checkout device
            torch.cuda.empty_cache()

            # Select device with most free memory
            # gpu_mem_list = []
            # for gpu_idx in [0, 1, 2]:
            #     gpu_mem = torch.cuda.mem_get_info(gpu_idx)
            #     gpu_mem_list.append(gpu_mem[0] / gpu_mem[1])

            selected_device_idx = gpu_store.pop() # gpu_mem_list.index(max(gpu_mem_list))
            selected_device = 'cuda:' + str(selected_device_idx)
            model = model_list[selected_device_idx]

            gpu_mem = torch.cuda.mem_get_info(selected_device)
            if (gpu_mem[0] / gpu_mem[1]) < 0.5:
                torch.cuda.empty_cache()
                time.sleep(3)
                print('sleeping: ' + selected_device)

            # Translate tweets
            tokenizer = MarianTokenizer.from_pretrained(model_name)

            translation_vec = model.generate(**tokenizer(part_df['tweet'].to_list(),
                                                         return_tensors="pt",
                                                         padding=True).to(selected_device),
                                             max_new_tokens=512)

            translation = [tokenizer.decode(t, skip_special_tokens=True) for t in translation_vec]

            del tokenizer
            del translation_vec
            del model
            torch.cuda.empty_cache()

            # Return device
            gpu_store.append(selected_device_idx)

            return part_df.assign(en_trans=translation)

        except Exception as e:
            print('translate_es_to_en')
            print(e)

            return part_df.assign(en_trans=np.nan)

    meta = [('mun_id_tweet', object),
            ('mun_name_tweet', object),
            ('mun_id_user', object),
            ('mun_name_user', object),
            ('id', np.int64),
            ('created_at', object),
            ('tweet', object),
            ('user_id', np.int64),
            ('en_trans', object)]


    ddf = ddf.map_partitions(lambda part_df: translate_es_to_en(part_df), meta=meta)


    ProgressBar().register()
    translation_df = ddf.compute(scheduler='threads', num_workers=3)

    end = time.time()
    print('End topic computation')
    print(end - start)


if __name__ == '__main__':

    # Expand columns
    translation_df = pd.concat([translation_df.drop('en_trans', axis=1),
                                pd.DataFrame(translation_df["en_trans"].tolist())], axis=1)
    print("expanded to columns")

    translation_df.rename(columns={0: 'en_translation'}, inplace=True)

    # Save topic dataset
    translation_df.to_csv(output_path + "/translation-data.csv", encoding="utf-8", index=False, sep=";")
    print("exported data")
