
# Run spell checking for all [ENGLISH/SPANISH] tweets

################################################################################
# Import modules
################################################################################
import language_tool_python

import dask
import dask.bag as db

from dask.diagnostics import ProgressBar
import json
import random
import glob
import emoji
import time


################################################################################
# Specify data location
################################################################################

base_path = "PATH TO FOLDER"
input_path = base_path + '/01-data/americas/americas_tweets-*.txt'  # .gz
output_dir = base_path + '/02-preprocessing/01-spell-checking/english/'

################################################################################
# Process data
################################################################################

# This list creates multiple language tool instances. In my understanding the
# instances are servers and the multiple instantiation should prevent bottle
# necks when processing the data in parallel. But maybe this is not necessary.

USED_LANGUAGE_TWEET = 'en'
USED_LANGUAGE_TOOL = 'en-US'


if __name__ == '__main__':


    def filter_language(tweet):

        if 'lang' in tweet:
            return tweet['lang'] == USED_LANGUAGE_TWEET
        else:
            return False

    # Process tweets with language tool - this function is applied by dask for
    # every tweet in the dataset

    def apply_lang_tool(tweet, lang_tool, file_path): # , file_path

        # time.sleep(random.uniform(0, 0.01))

        try:

            tweet['lang_tool'] = []
            tweet['emoji'] = []

           # Tweets can be stored in different places, depending on their length
            try:
                if "extended_tweet" in tweet:
                    tweet_text = tweet["extended_tweet"]['full_text']

                elif "text" in tweet:
                    tweet_text = tweet['text']

                else:
                    tweet_text = ''

            except Exception as e:

                tweet_text = ''

                # text error
                tweet['lang_tool_text_err'] = repr(e)

                print('extract tweet text exception')
                print(e)
                # print(file_path)


            # Emoji processing
            try:
                # Extract emoji list and store in tweet['emoji']
                emoji_list = emoji.emoji_list(tweet_text)

                if emoji_list:
                    tweet['emoji'] = emoji_list

                # Remove emoji from text
                tweet_text = emoji.replace_emoji(tweet_text, replace='', version=-1)

            except Exception as e:

                # emoji error
                tweet['emoji_err'] = repr(e)

                print('emoji_list exception')
                print(e)
                # print(file_path)


            # Run spellchecking
            try:

                spell_check = lang_tool.check(tweet_text) # tweet_spell_checking(tweet_text, lang_tool, 0)

                # Return complete output: tweet + spell checking result
                error_list = []

                for item in spell_check:
                    error_list.append({'ruleId': item.ruleId,
                                       'message': item.message,
                                       'replacements': item.replacements,
                                       'offsetInContext': item.offsetInContext,
                                       'context': item.context,
                                       'offset': item.offset,
                                       'errorLength': item.errorLength,
                                       'category': item.category,
                                       'ruleIssueType': item.ruleIssueType,
                                       'sentence': item.sentence})


                tweet['lang_tool'] = error_list

            except Exception as e:

                # parsing error
                tweet['lang_tool_parse_err'] = repr(e)

                print('error_parse')
                print(e)
                # print(file_path)
                raise

            return tweet

        except Exception as e:

            print('apply_lang_tool')
            print(e)
            # print(file_path)

            # general error
            tweet['lang_tool_general_err'] = repr(e)

            return tweet


    def spell_checking_full(file_path):

        time.sleep(random.uniform(0, 1))

        try:
            # Every file has own lang tool instance
            lang_tool = language_tool_python.LanguageTool(USED_LANGUAGE_TOOL)

            # for non-compressed
            db.read_text([file_path]) \
              .filter(lambda x: x != '\n') \
              .map(lambda x: x.replace('\n', '')) \
              .map(json.loads) \
              .filter(filter_language) \
              .map(apply_lang_tool, lang_tool=lang_tool, file_path=file_path) \
              .map(json.dumps) # \
              # TODO: remove comments to rerun
              # .to_textfiles([output_dir + file_path.split('/')[-1]],
              #               encoding='utf-8')

            # close lang tool
            lang_tool.close()


        except Exception as e:
            print('spell_checking_full')
            print(e)
            print(file_path)


    input_file_list = glob.glob(input_path)

    file_bag = db.from_sequence(input_file_list)

    with dask.config.set(scheduler='threads'):
        file_bag.map(spell_checking_full).compute()