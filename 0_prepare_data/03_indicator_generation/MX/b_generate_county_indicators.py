import sys
from pathlib import Path
import pandas as pd
import time

code_base_path = "PATH TO FOLDER"
base_dir = '/mnt/7adaf322-ecbb-4b5d-bc6f-4c54f7f808eb/Dropbox/mapping_mistakes/'

def generate_county_indicators(filter_date_start, filter_date_end, week, output_folder_name):

    output_folder_path = base_dir + '01_data/MX/03-indicators/' + output_folder_name

    # Time and Platform indicators #############################################
    print('a_time_platform_indicators_MX ')
    sys.path.insert(1, code_base_path + '03_indicator_generation/MX/01_time_platform_indicators')
    from a_time_platform_indicators_MX import compute_indicators as tp_indicators

    tp_out_dir = output_folder_path + 'time-platform/'
    Path(tp_out_dir).mkdir(parents=True, exist_ok=True)
    tp_indicators(filter_date_start, filter_date_end, week, tp_out_dir)

    # Mistakes indicators ######################################################
    print('b_mistakes_indicators_MX')
    sys.path.insert(1, code_base_path + '03_indicator_generation/MX/02_mistakes_indicators')
    from b_mistakes_indicators_MX import compute_indicators as err_indicators

    err_out_dir = output_folder_path + 'text-mistakes-data/'
    Path(err_out_dir).mkdir(parents=True, exist_ok=True)
    err_indicators(filter_date_start, filter_date_end, week, err_out_dir)

    # Topic indicators #########################################################
    print('a_topic_indicators_MX')
    sys.path.insert(1, code_base_path + '03_indicator_generation/MX/03_topic_indicators')
    from a_topic_indicators_MX import compute_indicators as topic_indicators

    topic_out_dir = output_folder_path + 'topics/'
    Path(topic_out_dir).mkdir(parents=True, exist_ok=True)
    topic_indicators(filter_date_start, filter_date_end, week, topic_out_dir)

    # Sentiment indicators ##########################################################
    print('b_sentiment_indicators_MX')
    sys.path.insert(1, code_base_path + '03_indicator_generation/MX/03_topic_indicators')
    from b_sentiment_indicators_MX import compute_indicators as sentiment_indicators

    semantic_out_dir = output_folder_path + 'sentiment/'
    Path(semantic_out_dir).mkdir(parents=True, exist_ok=True)
    sentiment_indicators(filter_date_start, filter_date_end, week, semantic_out_dir)

    # Social indicators ########################################################
    print('b_activity_and_social_indicators_MX')
    sys.path.insert(1, code_base_path + '03_indicator_generation/MX/04_social_indicators')
    from b_activity_and_social_indicators_MX import compute_indicators as social_indicators

    social_out_dir = output_folder_path + 'social/'
    Path(social_out_dir).mkdir(parents=True, exist_ok=True)
    social_indicators(filter_date_start, filter_date_end, week, social_out_dir)

    # Network indicators #######################################################
    print('b_compute_network_indicators_MX')
    sys.path.insert(1, code_base_path + '03_indicator_generation/MX/05_network_indicators')
    from b_compute_network_indicators_MX import compute_indicators as network_indicators

    network_out_dir = output_folder_path + 'network/'
    Path(network_out_dir).mkdir(parents=True, exist_ok=True)
    network_indicators(filter_date_start, filter_date_end, week, network_out_dir)


# Full dataset #################################################################
# Run function for total sample
# generate_county_indicators(None, None, 'total/')


# Weeks datasets ###############################################################
# Run function for cumulative time slices
selected_dates = pd.read_csv(base_dir + '01_data/MX/02-processed-data/selected_dates.csv',
                             sep=';')


for row in selected_dates.iterrows():

    if row[1]['week'] <= 9:

        min_date = 0
        max_date = 0

        if row[1]['min_date'] == row[1]['min_date']:
            min_date = time.strptime(row[1]['min_date'], '%Y-%m-%d')

        if row[1]['max_date'] == row[1]['max_date']:
            max_date = time.strptime(row[1]['max_date'], '%Y-%m-%d')

        output_folder = 'weeks/week_' + str(row[1]['week']).zfill(2) + '/'

        print(output_folder)
        generate_county_indicators(min_date, max_date, row[1]['week'], output_folder)


# Days datasets ################################################################

selected_dates = pd.read_csv(base_dir + '01_data/MX/02-processed-data/selected_dates-days.csv',
                             sep=';')


for row in selected_dates.iterrows():

    if row[1]['day'] <= 9:

        min_date = 0
        max_date = 0

        if row[1]['min_date'] == row[1]['min_date']:
            min_date = time.strptime(row[1]['min_date'], '%Y-%m-%d')

        if row[1]['max_date'] == row[1]['max_date']:
            max_date = time.strptime(row[1]['max_date'], '%Y-%m-%d')

        output_folder = 'days/day_' + str(row[1]['day']).zfill(2) + '/'

        print(output_folder)
        generate_county_indicators(min_date, max_date, row[1]['day'], output_folder)
