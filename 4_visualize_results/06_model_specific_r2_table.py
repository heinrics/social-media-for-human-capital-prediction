################################################################################
# Import modules
################################################################################
import pandas as pd

pd.options.display.width = 0

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

pd.set_option('display.max_colwidth', None)


################################################################################
# Load
################################################################################

# US ###########################################################################

# Specify data location
from paths import paths_us as paths
input_path = paths["results"] + "/"
exhibits_path = paths["exhibits"] + "/tables/"


# Years of schooling
us_r2_years = pd.read_csv(input_path + 'yschooling_r2.csv',
                          sep=';',
                          encoding='utf-8') \
    .rename(columns={'Unnamed: 0': 'model'})

us_weights_years = pd.read_csv(input_path + 'yschooling_weights.csv',
                               sep=';',
                               encoding='utf-8') \
    .rename(columns={'Unnamed: 0': 'model'})

# Bachelor
us_r2_ba = pd.read_csv(input_path + 'bachelor_r2.csv',
                       sep=';',
                       encoding='utf-8') \
    .rename(columns={'Unnamed: 0': 'model'})

us_weights_ba = pd.read_csv(input_path + 'bachelor_weights.csv',
                            sep=';',
                            encoding='utf-8') \
    .rename(columns={'Unnamed: 0': 'model'})

# Some college
us_r2_col = pd.read_csv(input_path + 'some_college_r2.csv',
                        sep=';',
                        encoding='utf-8') \
    .rename(columns={'Unnamed: 0': 'model'})

us_weights_col = pd.read_csv(input_path + 'some_college_weights.csv',
                             sep=';',
                             encoding='utf-8') \
    .rename(columns={'Unnamed: 0': 'model'})

# High School
us_r2_high = pd.read_csv(input_path + 'high_school_r2.csv',
                         sep=';',
                         encoding='utf-8') \
    .rename(columns={'Unnamed: 0': 'model'})

us_weights_high = pd.read_csv(input_path + 'high_school_weights.csv',
                              sep=';',
                              encoding='utf-8') \
    .rename(columns={'Unnamed: 0': 'model'})

# MX ###########################################################################

# Specify data location
from paths import paths_mx as paths
input_path = paths["results"] + "/"

# Years of schooling
mx_r2_years = pd.read_csv(input_path + 'yschooling_r2.csv',
                          sep=';',
                          encoding='utf-8') \
    .rename(columns={'Unnamed: 0': 'model'})

mx_weights_years = pd.read_csv(input_path + 'yschooling_weights.csv',
                               sep=';',
                               encoding='utf-8') \
    .rename(columns={'Unnamed: 0': 'model'})

# Post basic education
mx_r2_pb = pd.read_csv(input_path + 'pbasic_r2.csv',
                       sep=';',
                       encoding='utf-8') \
    .rename(columns={'Unnamed: 0': 'model'})

mx_weights_pb = pd.read_csv(input_path + 'pbasic_weights.csv',
                            sep=';',
                            encoding='utf-8') \
    .rename(columns={'Unnamed: 0': 'model'})


# Secondary education
mx_r2_sec = pd.read_csv(input_path + 'secondary_r2.csv',
                        sep=';',
                        encoding='utf-8') \
    .rename(columns={'Unnamed: 0': 'model'})

mx_weights_sec = pd.read_csv(input_path + 'secondary_weights.csv',
                             sep=';',
                             encoding='utf-8') \
    .rename(columns={'Unnamed: 0': 'model'})


# Primary education
mx_r2_prim = pd.read_csv(input_path + 'primary_r2.csv',
                         sep=';',
                         encoding='utf-8') \
    .rename(columns={'Unnamed: 0': 'model'})

mx_weights_prim = pd.read_csv(input_path + 'primary_weights.csv',
                              sep=';',
                              encoding='utf-8') \
    .rename(columns={'Unnamed: 0': 'model'})

################################################################################
# Process and merge data
################################################################################

digits = 3

# US ###########################################################################
# Years of schooling
us_r2_years['avg_r2_us_years'] = us_r2_years.iloc[:, 1:].mean(axis=1)
us_r2_years = us_r2_years[['model', 'avg_r2_us_years']].round(digits)

us_weights_years['avg_weights_us_years'] = us_weights_years.iloc[:, 1:].mean(axis=1)
us_weights_years = us_weights_years[['model', 'avg_weights_us_years']].round(digits)

# Bachelor
us_r2_ba['avg_r2_us_years'] = us_r2_ba.iloc[:, 1:].mean(axis=1)
us_r2_ba = us_r2_ba[['model', 'avg_r2_us_years']].round(digits)

us_weights_ba['avg_weights_us_years'] = us_weights_ba.iloc[:, 1:].mean(axis=1)
us_weights_ba = us_weights_ba[['model', 'avg_weights_us_years']].round(digits)

# Some college
us_r2_col['avg_r2_us_years'] = us_r2_col.iloc[:, 1:].mean(axis=1)
us_r2_col = us_r2_col[['model', 'avg_r2_us_years']].round(digits)

us_weights_col['avg_weights_us_years'] = us_weights_col.iloc[:, 1:].mean(axis=1)
us_weights_col = us_weights_col[['model', 'avg_weights_us_years']].round(digits)

# High School
us_r2_high['avg_r2_us_years'] = us_r2_high.iloc[:, 1:].mean(axis=1)
us_r2_high = us_r2_high[['model', 'avg_r2_us_years']].round(digits)

us_weights_high['avg_weights_us_years'] = us_weights_high.iloc[:, 1:].mean(axis=1)
us_weights_high = us_weights_high[['model', 'avg_weights_us_years']].round(digits)


# MX ###########################################################################
# Years of schooling
mx_r2_years['avg_r2_mx_years'] = mx_r2_years.iloc[:, 1:].mean(axis=1)
mx_r2_years = mx_r2_years[['model', 'avg_r2_mx_years']].round(digits)

mx_weights_years['avg_weights_mx_years'] = mx_weights_years.iloc[:, 1:].mean(axis=1)
mx_weights_years = mx_weights_years[['model', 'avg_weights_mx_years']].round(digits)

# Post basic education
mx_r2_pb['avg_r2_mx_years'] = mx_r2_pb.iloc[:, 1:].mean(axis=1)
mx_r2_pb = mx_r2_pb[['model', 'avg_r2_mx_years']].round(digits)

mx_weights_pb['avg_weights_mx_years'] = mx_weights_pb.iloc[:, 1:].mean(axis=1)
mx_weights_pb = mx_weights_pb[['model', 'avg_weights_mx_years']].round(digits)

# Secondary education
mx_r2_sec['avg_r2_mx_years'] = mx_r2_sec.iloc[:, 1:].mean(axis=1)
mx_r2_sec = mx_r2_sec[['model', 'avg_r2_mx_years']].round(digits)

mx_weights_sec['avg_weights_mx_years'] = mx_weights_sec.iloc[:, 1:].mean(axis=1)
mx_weights_sec = mx_weights_sec[['model', 'avg_weights_mx_years']].round(digits)

# Primary education
mx_r2_prim['avg_r2_mx_years'] = mx_r2_prim.iloc[:, 1:].mean(axis=1)
mx_r2_prim = mx_r2_prim[['model', 'avg_r2_mx_years']].round(digits)

mx_weights_prim['avg_weights_mx_years'] = mx_weights_prim.iloc[:, 1:].mean(axis=1)
mx_weights_prim = mx_weights_prim[['model', 'avg_weights_mx_years']].round(digits)

# Merge ########################################################################

def cell_formatting(df_r2, df_weights):

    results = df_r2.merge(df_weights,
                          how='left',
                          on='model')

    results.iloc[:-1, -1:] = '(' + results.iloc[:-1, -1].apply(lambda x: '{:,.1f}'.format(x))  + '\%)'

    results['results'] = '\makecell{' + \
                         results.iloc[:, 1].apply(lambda x: '{:,.3f}'.format(x)) + ' \\\\' + \
                         results.iloc[:, 2] + '}'

    results.loc[results['results'].isna(), 'results'] = '\makecell{' + \
                                                        results.loc[results['results'].isna(), results.columns[1]].apply(lambda x: '{:,.3f}'.format(x)) + \
                                                        ' \\\\ }'

    return results[['results']]


mx_weights_years.iloc[:, 1] = (mx_weights_years.iloc[:, 1] * 100).round(2)
mx_weights_pb.iloc[:, 1] = (mx_weights_pb.iloc[:, 1] * 100).round(2)
mx_weights_sec.iloc[:, 1] = (mx_weights_sec.iloc[:, 1] * 100).round(2)
mx_weights_prim.iloc[:, 1] = (mx_weights_prim.iloc[:, 1] * 100).round(2)
us_weights_years.iloc[:, 1] = (us_weights_years.iloc[:, 1] * 100).round(2)
us_weights_ba.iloc[:, 1] = (us_weights_ba.iloc[:, 1] * 100).round(2)
us_weights_col.iloc[:, 1] = (us_weights_col.iloc[:, 1] * 100).round(2)
us_weights_high.iloc[:, 1] = (us_weights_high.iloc[:, 1] * 100).round(2)


mx_years = cell_formatting(mx_r2_years, mx_weights_years)
mx_pb = cell_formatting(mx_r2_pb, mx_weights_pb)
mx_sec = cell_formatting(mx_r2_sec, mx_weights_sec)
mx_prim = cell_formatting(mx_r2_prim, mx_weights_prim)

us_years = cell_formatting(us_r2_years, us_weights_years)
us_ba = cell_formatting(us_r2_ba, us_weights_ba)
us_col = cell_formatting(us_r2_col, us_weights_col)
us_high = cell_formatting(us_r2_high, us_weights_high)

model_names = [r'\makecell[l]{Elastic Net}',
               r'\makecell[l]{Gradient \\ Boosting}',
               r'\makecell[l]{Support Vector \\ Machine}',
               r'\makecell[l]{Nearest Neighbour \\ Matching}',
               r'\makecell[l]{Multi-layer \\ Perceptron}',
               r'\makecell[l]{Stacking}']

res_df = pd.concat([pd.DataFrame(model_names),
                    mx_years,
                    mx_pb,
                    mx_sec,
                    mx_prim,
                    us_years,
                    us_ba,
                    us_col,
                    us_high],
                   axis=1)


cols = [('', ''),
        (r'\makecell{Mexico}', r'\makecell{Years of \\ Schooling}'),
        (r'\makecell{Mexico}', r'\makecell{Post Basic \\ Education}'),
        (r'\makecell{Mexico}', r'\makecell{Secondary\\ Education}'),
        (r'\makecell{Mexico}', r'\makecell{Primary\\ Education}'),
        (r'\makecell{United States}', r'\makecell{Years of \\ Schooling}'),
        (r'\makecell{United States}', r'\makecell{Bachelor \\ Degree}'),
        (r'\makecell{United States}', r'\makecell{Some \\ College}'),
        (r'\makecell{United States}', r'\makecell{Only \\ High School}')]

# Create multi-level index
res_df.columns = pd.MultiIndex.from_tuples(cols)

n_rows, n_cols = res_df.shape

# Return latex table as list of strings, split by lines
tex_str = res_df.to_latex(column_format="@{}l" + "r" * (n_cols - 1) + '@{}',
                          multicolumn_format='c',
                          multirow=True,
                          escape=False,
                          index=False) \
    .splitlines()

tex_str.insert(3, '\cmidrule(lr){2-5}\cmidrule(l){6-9}')
tex_str.insert(7, '\midrule')
tex_str.insert(9, '\midrule')
tex_str.insert(11, '\midrule')
tex_str.insert(13, '\midrule')
tex_str.insert(15, '\midrule')
tex_str.insert(16, '\midrule')

tex_str = '\n'.join(tex_str)

print(tex_str)

with open(exhibits_path + 'main-model-results.tex', "w") as text_file:
    print(tex_str, file=text_file)
