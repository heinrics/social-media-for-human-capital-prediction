################################################################################
# Load packages
################################################################################
import pandas as pd
import networkx as nx
import time

pd.options.display.width = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


################################################################################
# Specify data location
################################################################################
base_path_sebastian_linux_ssd = '/mnt/4e9e0a0f-64a6-4182-b861-0904a6a7d78d/Mapping_Mistakes'
base_path_sebastian_linux_dropbox = '/mnt/7adaf322-ecbb-4b5d-bc6f-4c54f7f808eb/Dropbox/mapping_mistakes'

input_files = base_path_sebastian_linux_dropbox + '/01_data/US/02-processed-data/social/'
output_files_indicator = base_path_sebastian_linux_dropbox + '/01_data/US/03-indicators/'


# Construct joint references (MENTION & QUOTED) graph ##########################
################################################################################

def load_data(filter_date_start, filter_date_end, out_dir):

    # Pooled network of mention and quotes references
    ment_edge_df = pd.read_csv(input_files + 'mentions-edge-list.csv',
                               dtype={'source_county_id': str,
                                      'target_county_id': str},
                               sep=';',
                               encoding='utf-8')

    # print(ment_edge_df.shape)
    # (2040994, 2)

    quotes_edge_df = pd.read_csv(input_files + 'quotes-edge-list.csv',
                                 dtype={'source_county_id': str,
                                        'target_county_id': str},
                                 sep=';',
                                 encoding='utf-8')
    # print(quotes_edge_df.shape)

    # (607622, 2)

    edge_df = pd.concat([ment_edge_df,
                         quotes_edge_df],
                        axis='index')

    # print(edge_df.shape)
    # (2648616, 2)

    edge_df['created_at_tm'] = edge_df['source_created_at'].apply(lambda x: time.strptime(x, '%a %b %d %H:%M:%S +0000 %Y'))

    # Filter tweets by date
    if filter_date_start:
        edge_df = edge_df[edge_df['created_at_tm'] >= filter_date_start]

    if filter_date_end:
        edge_df = edge_df[edge_df['created_at_tm'] <= filter_date_end]

    # Load and update correct user county data
    cols = ['user_id',
            'county_id_user']

    user_id_df = pd.read_csv('/'.join(out_dir.split('/')[0:-2]).replace('03-indicators/',
                                                                        '03-indicators/selected_con/') +
                             "/tweet_data_selection.csv",
                             sep=";",
                             low_memory=False,
                             lineterminator='\n',
                             encoding="utf-8",
                             usecols=cols,
                             dtype={'county_id_user': 'str'}).drop_duplicates('user_id')

    edge_df = edge_df.merge(user_id_df,
                            left_on='source_user_id',
                            right_on='user_id',
                            how='left')\
                     .rename(columns={'county_id_user': 'source_county_id'})\
                     .drop('user_id', axis=1)

    edge_df = edge_df.merge(user_id_df,
                            left_on='target_user_id',
                            right_on='user_id',
                            how='left')\
                     .rename(columns={'county_id_user': 'target_county_id'})\
                     .drop('user_id', axis=1)

    edge_df.dropna(subset=['target_county_id',
                           'source_county_id'], inplace=True)

    return edge_df[['source_county_id',
                    'target_county_id']]


def county_networks(df, out_dir):

    ref_G = nx.from_pandas_edgelist(df,
                                    source='source_county_id',
                                    target='target_county_id',
                                    edge_attr=None,
                                    create_using=nx.MultiDiGraph(),)

    # Degree centrality (N of ties to/from/with others)
    ref_in_deg = nx.in_degree_centrality(ref_G)
    ref_out_deg = nx.out_degree_centrality(ref_G)

    # Closeness centrality (sum of length of shortest paths to others)
    ref_clos_cent = nx.closeness_centrality(ref_G)

    # Page rank (ranking of the nodes based on the structure of the incoming links)
    ref_pagerank_cent = nx.pagerank(ref_G)



    # Store results in pandas DF ###################################################

    ref_in_deg_df = pd.DataFrame(list(zip(ref_in_deg.keys(),
                                           ref_in_deg.values())),
                                  columns=['county_id_user', 'ref_in_deg'])

    ref_out_deg_df = pd.DataFrame(list(zip(ref_out_deg.keys(),
                                            ref_out_deg.values())),
                                   columns=['county_id_user', 'ref_out_deg'])

    ref_clos_cent_df = pd.DataFrame(list(zip(ref_clos_cent.keys(),
                                              ref_clos_cent.values())),
                                     columns=['county_id_user', 'ref_clos_cent'])

    ref_pagerank_cent_df = pd.DataFrame(list(zip(ref_pagerank_cent.keys(),
                                                  ref_pagerank_cent.values())),
                                         columns=['county_id_user', 'ref_pagerank'])

    ref_results_df = ref_in_deg_df.merge(ref_out_deg_df, on='county_id_user')\
                                  .merge(ref_clos_cent_df, on='county_id_user')\
                                  .merge(ref_pagerank_cent_df, on='county_id_user')

    # print(ref_results_df.describe())
    #
    #
    # print(ref_results_df.describe())
    #
    #
    # ref_results_df.replace(0, np.nan).hist(bins=100)
    # plt.show()
    #
    # np.log(ref_results_df['ref_in_deg'] + 0.001).hist(bins=100)
    # plt.show()
    #
    # np.log(ref_results_df['ref_out_deg'] + 0.001).hist(bins=100)
    # plt.show()
    #
    # np.log(ref_results_df['ref_clos_cent'] + 0.001).hist(bins=100)
    # plt.show()
    #
    # np.log(ref_results_df['ref_pagerank'] + 0.001).hist(bins=100)
    # plt.show()



    # # Complete dataset with missing counties #######################################

    county_df = pd.read_csv('/'.join(out_dir.split('/')[0:-2]).replace('03-indicators/',
                            '03-indicators/selected_con/') +
                            "/tweet_data_selection.csv",
                            sep=";",
                            low_memory=False,
                            lineterminator='\n',
                            encoding="utf-8",
                            usecols=['county_id_user'],
                            dtype={'county_id_user': 'str'}).drop_duplicates()

    #
    # county_list_df = pd.read_csv(base_path_sebastian_linux_dropbox + '/01_data/US/02-processed-data/social/tweet-level-indicators-US.csv',
    #                              dtype={'county_id_user': str},
    #                              usecols=['county_id_user'],
    #                              sep=';',
    #                              encoding='utf-8').drop_duplicates()
#
#
    all_network_results = county_df.merge(ref_results_df,
                                          on='county_id_user',
                                          how='left')
#
    # # Set counties without any references to 0
    all_network_results = all_network_results.fillna(0)

    all_network_results.to_csv(out_dir + 'county-level-network-indicators-US.csv',
                               index=False,
                               sep=';',
                               encoding='utf-8')

def compute_indicators(filter_date_start, filter_date_end, out_dir):

    df = load_data(filter_date_start, filter_date_end, out_dir)

    county_networks(df, out_dir)
