import pandas as pd
from sklearn.metrics import r2_score

################################################################################
# Specify data location
################################################################################

from paths import paths_mx as paths
mx_input_path = paths["results"]
mx_train_path = paths["train_data"]
mx_geo_path = geo_path = paths["geo_data"]
exhibits_path = paths["exhibits"]

from paths import paths_us as paths
us_input_path = paths["results"]
us_train_path = paths["train_data"]
us_geo_path = us_path = paths["geo_data"]

################################################################################
# Load data
################################################################################

# Mexico #######################################################################
mx_pop = pd.read_csv(mx_train_path + "/features.csv",
                     sep=";",
                     usecols=["GEOID",
                              "population"])

pbasic_df = pd.read_csv(mx_input_path + '/pbasic_predictions.csv',
                        sep=';')
pbasic_df = pbasic_df.merge(mx_pop, on='GEOID')

primary_df = pd.read_csv(mx_input_path + '/primary_predictions.csv',
                         sep=';')
primary_df = primary_df.merge(mx_pop, on='GEOID')

secondary_df = pd.read_csv(mx_input_path + '/secondary_predictions.csv',
                           sep=';')
secondary_df = secondary_df.merge(mx_pop, on='GEOID')

yschooling_df = pd.read_csv(mx_input_path + '/yschooling_predictions.csv',
                            sep=';')
yschooling_df = yschooling_df.merge(mx_pop, on='GEOID')



# United States ################################################################

us_features = pd.read_csv(us_train_path + "/features.csv",
                          sep=";",
                          usecols=['GEOID', 'population'],
                          dtype={'GEOID': str})

us_edu_bach = pd.read_csv(us_input_path + '/bachelor_predictions.csv',
                          sep=';',
                          dtype={'GEOID': str})
us_edu_bach = us_edu_bach.merge(us_features, on='GEOID')

us_edu_some_col = pd.read_csv(us_input_path + '/some_college_predictions.csv',
                              sep=';',
                              dtype={'GEOID': str})
us_edu_some_col = us_edu_some_col.merge(us_features, on='GEOID')

us_edu_high_school = pd.read_csv(us_input_path + '/high_school_predictions.csv',
                                 sep=';',
                                 dtype={'GEOID': str})
us_edu_high_school = us_edu_high_school.merge(us_features, on='GEOID')

us_edu_y_schooling = pd.read_csv(us_input_path + '/yschooling_predictions.csv',
                                 sep=';',
                                 dtype={'GEOID': str})
us_edu_y_schooling = us_edu_y_schooling.merge(us_features, on='GEOID')


################################################################################
# R2
################################################################################

# Mexico #######################################################################
mx_pbasic_r2 = r2_score(pbasic_df['y'],
                        pbasic_df["y_hat"])
mx_pbasic_r2_w = r2_score(pbasic_df['y'],
                          pbasic_df["y_hat"],
                          sample_weight=pbasic_df["population"])

mx_primary_r2 = r2_score(primary_df['y'],
                         primary_df["y_hat"])
mx_primary_r2_w = r2_score(primary_df['y'],
                           primary_df["y_hat"],
                           sample_weight=primary_df["population"])

mx_secondary_r2 = r2_score(secondary_df['y'],
                           secondary_df["y_hat"])
mx_secondary_r2_w = r2_score(secondary_df['y'],
                             secondary_df["y_hat"],
                             sample_weight=secondary_df["population"])

mx_yschooling_r2 = r2_score(yschooling_df['y'],
                            yschooling_df["y_hat"])
mx_yschooling_r2_w = r2_score(yschooling_df['y'],
                              yschooling_df["y_hat"],
                              sample_weight=yschooling_df["population"])


# United States ################################################################

us_edu_bach_r2 = r2_score(us_edu_bach['y'],
                          us_edu_bach["y_hat"])
us_edu_bach_r2_w = r2_score(us_edu_bach['y'],
                            us_edu_bach["y_hat"],
                            sample_weight=us_edu_bach["population"])

us_edu_some_col_r2 = r2_score(us_edu_some_col['y'],
                              us_edu_some_col["y_hat"])
us_edu_some_col_r2_w = r2_score(us_edu_some_col['y'],
                                us_edu_some_col["y_hat"],
                                sample_weight=us_edu_some_col["population"])

us_edu_high_school_r2 = r2_score(us_edu_high_school['y'],
                                 us_edu_high_school["y_hat"])
us_edu_high_school_r2_w = r2_score(us_edu_high_school['y'],
                                   us_edu_high_school["y_hat"],
                                   sample_weight=us_edu_high_school["population"])

us_edu_y_schooling_r2 = r2_score(us_edu_y_schooling['y'],
                                 us_edu_y_schooling["y_hat"])
us_edu_y_schooling_r2_w = r2_score(us_edu_y_schooling['y'],
                                   us_edu_y_schooling["y_hat"],
                                   sample_weight=us_edu_y_schooling["population"])
# \multicolumn{{2}}{{l}}{{\textbf{{MX}}}}    \\ \hline
mx_tex = r"""
\begin{{tabular}}{{lc}}
\textbf{{MX}} & $r^2$   \\ \hline
\textit{{Years of schooling}} & {0:.3f} \\
\textit & ({1:.3f}) \\
\textit{{\% Post-basic}}      & {2:.3f} \\
\textit & ({3:.3f}) \\
\textit{{\% Secondary}}       & {4:.3f} \\
\textit & ({5:.3f}) \\
\textit{{\% Primary}}         & {6:.3f} \\
\textit & ({7:.3f}) \\
                            &      \\
""".format(mx_yschooling_r2, mx_yschooling_r2_w,
           mx_pbasic_r2, mx_pbasic_r2_w,
           mx_secondary_r2, mx_secondary_r2_w,
           mx_primary_r2, mx_primary_r2_w)

us_tex = r"""                            
\textbf{{US}}                 &   $r^2$    \\ \hline
\textit{{Years of schooling}} & {0:.3f} \\
\textit & ({1:.3f}) \\
\textit{{\% Bachelor}}      & {2:.3f} \\
\textit & ({3:.3f}) \\
\textit{{\% Some College}}       & {4:.3f} \\
\textit & ({5:.3f}) \\
\textit{{\% High School}}         & {6:.3f} \\
\textit & ({7:.3f}) \\
\end{{tabular}}
""".format(us_edu_y_schooling_r2, us_edu_y_schooling_r2_w,
           us_edu_bach_r2, us_edu_bach_r2_w,
           us_edu_some_col_r2, us_edu_some_col_r2_w,
           us_edu_high_school_r2, us_edu_high_school_r2_w)

print(mx_tex + us_tex)
with open(exhibits_path + '/tables/r2-comparison-table.tex', "w") as text_file:
    print(mx_tex + us_tex, file=text_file)
