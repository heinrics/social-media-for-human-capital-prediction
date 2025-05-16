import pandas as pd
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt

################################################################################
# Set environment
################################################################################

pd.options.display.width = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

################################################################################
# Specify data location
################################################################################

from paths import paths_mx as paths
mx_shap_dir = paths["results"] + "/"
exhibits_dir = paths["exhibits"] + "/figures"

from paths import paths_us as paths
us_shap_dir = paths["results"] + "/"


################################################################################
# Load data
################################################################################

# Import feature names (for plots) #############################################
flabels = pd.read_excel(os.getcwd() + "/_modules/feature_labels.xlsx")
label_dict = dict(zip(flabels.col_name, flabels.label))

# Mexico #######################################################################

mx_yschooling_df = pd.read_csv(mx_shap_dir + 'yschooling_shapley.csv',
                               sep=';',
                               encoding='utf-8')

mx_yschooling_df.drop(columns=['GEOID', 'base_values'], inplace=True)


mx_pbasic_df = pd.read_csv(mx_shap_dir + 'pbasic_shapley.csv',
                           sep=';',
                           encoding='utf-8')

mx_pbasic_df.drop(columns=['GEOID', 'base_values'], inplace=True)


mx_primary_df = pd.read_csv(mx_shap_dir + 'primary_shapley.csv',
                            sep=';',
                            encoding='utf-8')

mx_primary_df.drop(columns=['GEOID', 'base_values'], inplace=True)


mx_secondary_df = pd.read_csv(mx_shap_dir + 'secondary_shapley.csv',
                              sep=';',
                              encoding='utf-8')

mx_secondary_df.drop(columns=['GEOID', 'base_values'], inplace=True)


# United States ################################################################
us_yschooling_df = pd.read_csv(us_shap_dir + 'yschooling_shapley.csv',
                               sep=';',
                               encoding='utf-8')
us_yschooling_df.drop(columns=['GEOID', 'base_values'], inplace=True)


us_bachelor_df = pd.read_csv(us_shap_dir + 'bachelor_shapley.csv',
                             sep=';',
                             encoding='utf-8')
us_bachelor_df.drop(columns=['GEOID', 'base_values'], inplace=True)


us_high_school_df = pd.read_csv(us_shap_dir + 'high_school_shapley.csv',
                                sep=';',
                                encoding='utf-8')
us_high_school_df.drop(columns=['GEOID', 'base_values'], inplace=True)


us_some_college_df = pd.read_csv(us_shap_dir + 'some_college_shapley.csv',
                                 sep=';',
                                 encoding='utf-8')
us_some_college_df.drop(columns=['GEOID', 'base_values'], inplace=True)


################################################################################
# Describe shapley values
################################################################################
mx_yschooling_df.apply(np.abs).sum().sort_values(ascending=False)
us_yschooling_df.apply(np.abs).sum().sort_values(ascending=False)

mx_pbasic_df.apply(np.abs).sum().sort_values(ascending=False)
us_bachelor_df.apply(np.abs).sum().sort_values(ascending=False)

mx_primary_df.apply(np.abs).sum().sort_values(ascending=False)
us_high_school_df.apply(np.abs).sum().sort_values(ascending=False)

mx_secondary_df.apply(np.abs).sum().sort_values(ascending=False)
us_some_college_df.apply(np.abs).sum().sort_values(ascending=False)


################################################################################
# Visualize shapley values
################################################################################

mx_color = (0, 0, 0.545, 0.3)
us_color = (0.545, 0, 0, 0.3)

def bar_shapley_values(shapley_df, top_n, color, label_dict, x_lable_text='', out_file=False):

    # Visualize model
    columns = shapley_df.apply(np.abs).sum() \
        .sort_values(ascending=False).head(top_n).index

    plt.grid(which='major', axis='x',
             linestyle='--', color="#e2e4e0", zorder=0)

    ax = sns.barplot(data=shapley_df[columns].apply(np.abs),
                     orient='h',
                     edgecolor="0",
                     facecolor=color,
                     # color=color,
                     # alpha=0.3,
                     # edgecolor='black',
                     errorbar=('ci', 99),
                     errwidth=1.2,
                     zorder=3)

    ax.set_axisbelow(True)
    ax.set_yticklabels([label_dict[col] for col in columns])
    plt.xlabel(x_lable_text)

    # plt.box(False)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.tight_layout()

    if out_file:
        plt.savefig(exhibits_dir + '/shap/' + out_file,
                    bbox_inches='tight')

    plt.show()


def box_shapley_values(shapley_df, top_n):

    # Visualize model
    columns = shapley_df.apply(np.abs).mean() \
        .sort_values(ascending=False).head(top_n).index

    sns.boxplot(data=shapley_df[columns], orient='h')

    plt.tight_layout()
    plt.show()


# Mexico and United States bar plots ###########################################
bar_shapley_values(mx_yschooling_df, 20,
                   mx_color, label_dict, 'Absolute SHAP values',
                   'mx-yschooling-top-20.pdf')
bar_shapley_values(us_yschooling_df, 20,
                   us_color, label_dict, 'Absolute SHAP values',
                   'us-yschooling-top-20.pdf')

bar_shapley_values(mx_pbasic_df, 20,
                   mx_color, label_dict, 'Absolute SHAP values',
                   'mx-pbasic-top-20.pdf')
bar_shapley_values(us_bachelor_df, 20,
                   us_color, label_dict, 'Absolute SHAP values',
                   'us-bachelor-top-20.pdf')

bar_shapley_values(mx_primary_df, 20,
                   mx_color, label_dict, 'Absolute SHAP values',
                   'mx-primary-top-20.pdf')
bar_shapley_values(us_high_school_df, 20,
                   us_color, label_dict, 'Absolute SHAP values',
                   'us-high-school-top-20.pdf')

bar_shapley_values(mx_secondary_df, 20,
                   mx_color, label_dict, 'Absolute SHAP values',
                   'mx-secondary-top-20.pdf')
bar_shapley_values(us_some_college_df, 20,
                   us_color, label_dict, 'Absolute SHAP values',
                   'us-some-collage-top-20.pdf')


# Mexico plot by feature group #################################################

def group_mx_features(df):
    # mx_id_cols = ['GEOID']
    # mx_geo_cols = ['geometry']

    mx_pop_cols = ['population', 'pop_density']
    mx_count_cols = ['nr_tweets', 'tweet_density', 'nr_users', 'user_density']
    mx_tweet_cols = list(df.loc[:, 'share_weekdays':'log_emoji_per_tweet'].columns)
    mx_topic_cols = list(df.loc[:, 'log_arts_&_culture':'log_youth_&_student_life'].columns)
    mx_error_cols = list(df.loc[:, 'log_error_per_char':'log_char_TYPOS'])
    mx_sentiment_cols = list(df.loc[:, 'negative':'log_offensive'].columns)
    mx_network_cols = list(df.loc[:, 'log_ref_in_deg':'log_ref_pagerank'].columns)
    mx_cluster_cols = ['cluster_zero_in_cluster', 'cluster_zero_in_cluster2']


    mx_pop_df = df.filter(regex='|'.join(mx_pop_cols))
    print(len(mx_pop_df.columns))
    # 4
    mx_group_df = pd.DataFrame(mx_pop_df[[col for col in mx_pop_df]].apply(np.abs).sum(axis=1))
    mx_group_df.columns = ['population']

    mx_count_df = df.filter(regex='|'.join(mx_count_cols))
    print(len(mx_count_df.columns))
    # 8
    mx_group_df['count'] = mx_count_df[[col for col in mx_count_df]].apply(np.abs).sum(axis=1)

    mx_tweet_df = df.filter(regex='|'.join(mx_tweet_cols))
    print(len(mx_tweet_df.columns))
    # 42
    mx_group_df['tweet'] = mx_tweet_df[[col for col in mx_tweet_df]].apply(np.abs).sum(axis=1)

    mx_topic_df = df.filter(regex='|'.join(mx_topic_cols))
    print(len(mx_topic_df.columns))
    # 38
    mx_group_df['topic'] = mx_topic_df[[col for col in mx_topic_df]].apply(np.abs).sum(axis=1)

    mx_error_df = df.filter(regex='|'.join(mx_error_cols))
    print(len(mx_error_df.columns))
    # 46
    mx_group_df['error'] = mx_error_df[[col for col in mx_error_df]].apply(np.abs).sum(axis=1)

    mx_sentiment_df = df.filter(regex='|'.join(mx_sentiment_cols))
    print(len(mx_sentiment_df.columns))
    # 8
    mx_group_df['sentiment'] = mx_sentiment_df[[col for col in mx_sentiment_df]].apply(np.abs).sum(axis=1)

    mx_network_df = mx_yschooling_df.filter(regex='|'.join(mx_network_cols))
    print(len(mx_network_df.columns))
    # 8
    mx_group_df['network'] = mx_network_df[[col for col in mx_network_df]].apply(np.abs).sum(axis=1)

    mx_cluster_df = mx_yschooling_df.filter(regex='|'.join(mx_cluster_cols))

    # Check if all columns are captured
    print(len(mx_pop_df.columns) + len(mx_count_df.columns) + len(mx_tweet_df.columns) +
          len(mx_topic_df.columns) + len(mx_error_df.columns) +
          len(mx_sentiment_df.columns) + len(mx_network_df.columns) +
          len(mx_cluster_df.columns))
    # 156

    print(len(mx_yschooling_df.columns))
    # 156

    return {'df': mx_group_df,
            'pop_cols': str(len(mx_pop_df.columns)),
            'count_cols': str(len(mx_count_df.columns)),
            'error_cols': str(len(mx_error_df.columns)),
            'tweet_cols': str(len(mx_tweet_df.columns)),
            'topic_cols': str(len(mx_topic_df.columns)),
            'sentiment_cols': str(len(mx_sentiment_df.columns)),
            'network_cols': str(len(mx_network_df.columns))}


for outcome in [['yschooling', mx_yschooling_df],
                ['pbasic', mx_pbasic_df],
                ['secondary', mx_secondary_df],
                ['primary', mx_primary_df]]:

    group_df = group_mx_features(outcome[1])

    feature_groups = {'population': 'Population (' + group_df['pop_cols'] + ')',
                      'count': 'Twitter penetration (' + group_df['count_cols'] + ')',
                      'error': 'Errors (' + group_df['error_cols'] + ')',
                      'tweet': 'Usage statistics (' + group_df['tweet_cols'] + ')',
                      'topic': 'Topics (' + group_df['topic_cols'] + ')',
                      'sentiment': 'Sentiments (' + group_df['sentiment_cols'] + ')',
                      'network': 'Networks (' + group_df['network_cols'] + ')'}

    bar_shapley_values(group_df['df'],
                       len(feature_groups),
                       mx_color,
                       feature_groups,
                       'Sum of absolute SHAP values',
                       'mx-' + outcome[0] + '-var-groups.pdf')



# United States plot by feature group ##########################################

def group_us_features(df):

    # us_id_cols = ['GEOID']
    # us_geo_cols = ['geometry']

    us_pop_cols = ['population', 'pop_density']
    us_count_cols = ['nr_tweets', 'tweet_density', 'nr_users', 'user_density']
    us_tweet_cols = list(df.loc[:, 'share_weekdays':'log_emoji_per_tweet'].columns)
    us_topic_cols = list(df.loc[:, 'log_arts_&_culture':'log_youth_&_student_life'].columns)
    us_error_cols = list(df.loc[:, 'log_error_per_char':'log_char_TYPOS'])
    us_sentiment_cols = list(df.loc[:, 'negative':'log_offensive'].columns)
    us_network_cols = list(df.loc[:, 'log_ref_in_deg':'log_ref_pagerank'].columns)
    us_cluster_cols = ['cluster_zero_in_cluster', 'cluster_zero_in_cluster2']


    us_pop_df = df.filter(regex='|'.join(us_pop_cols))
    print(len(us_pop_df.columns))
    # 4
    us_yschooling_group_df = pd.DataFrame(us_pop_df[[col for col in us_pop_df]].apply(np.abs).sum(axis=1))
    us_yschooling_group_df.columns = ['population']

    us_count_df = df.filter(regex='|'.join(us_count_cols))
    print(len(us_count_df.columns))
    # 8
    us_yschooling_group_df['count'] = us_count_df[[col for col in us_count_df]].apply(np.abs).sum(axis=1)

    us_tweet_df = df.filter(regex='|'.join(us_tweet_cols))
    print(len(us_tweet_df.columns))
    # 42
    us_yschooling_group_df['tweet'] = us_tweet_df[[col for col in us_tweet_df]].apply(np.abs).sum(axis=1)

    us_topic_df = df.filter(regex='|'.join(us_topic_cols))
    print(len(us_topic_df.columns))
    # 38
    us_yschooling_group_df['topic'] = us_topic_df[[col for col in us_topic_df]].apply(np.abs).sum(axis=1)

    us_error_df = df.filter(regex='|'.join(us_error_cols))
    print(len(us_error_df.columns))
    # 32
    us_yschooling_group_df['error'] = us_error_df[[col for col in us_error_df]].apply(np.abs).sum(axis=1)
    us_sentiment_df = df.filter(regex='|'.join(us_sentiment_cols))
    print(len(us_sentiment_df.columns))
    # 8
    us_yschooling_group_df['sentiment'] = us_sentiment_df[[col for col in us_sentiment_df]].apply(np.abs).sum(axis=1)

    us_network_df = df.filter(regex='|'.join(us_network_cols))
    print(len(us_network_df.columns))
    # 8
    us_yschooling_group_df['network'] = us_network_df[[col for col in us_network_df]].apply(np.abs).sum(axis=1)
    us_cluster_df = df.filter(regex='|'.join(us_cluster_cols))

    # Check if all columns are captured
    print(len(us_pop_df.columns) + len(us_count_df.columns) + len(us_tweet_df.columns) +
          len(us_topic_df.columns) + len(us_error_df.columns) +
          len(us_sentiment_df.columns) + len(us_network_df.columns) +
          len(us_cluster_df.columns))
    # 141

    print(len(df.columns))
    # 141

    return {'df': us_yschooling_group_df,
            'pop_cols': str(len(us_pop_df.columns)),
            'count_cols': str(len(us_count_df.columns)),
            'error_cols': str(len(us_error_df.columns)),
            'tweet_cols': str(len(us_tweet_df.columns)),
            'topic_cols': str(len(us_topic_df.columns)),
            'sentiment_cols': str(len(us_sentiment_df.columns)),
            'network_cols': str(len(us_network_df.columns))}


# Bar plots years of schooling
for outcome in [['yschooling', us_yschooling_df],
                ['bachelor', us_bachelor_df],
                ['high-school', us_high_school_df],
                ['some-collage', us_some_college_df]]:

    group_df = group_us_features(outcome[1])

    feature_groups = {'population': 'Population (' + group_df['pop_cols'] + ')',
                      'count': 'Twitter penetration (' + group_df['count_cols'] + ')',
                      'error': 'Errors (' + group_df['error_cols'] + ')',
                      'tweet': 'Usage statistics (' + group_df['tweet_cols'] + ')',
                      'topic': 'Topics (' + group_df['topic_cols'] + ')',
                      'sentiment': 'Sentiments (' + group_df['sentiment_cols'] + ')',
                      'network': 'Networks (' + group_df['network_cols'] + ')'}

    bar_shapley_values(group_df['df'],
                       len(feature_groups),
                       us_color,
                       feature_groups,
                       'Sum of absolute SHAP values',
                       'us-' + outcome[0] + '-var-groups.pdf')
