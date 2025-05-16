
# ######################################################################################################################
# Prepare data on Mexican schools, test scores, years of schooling and population
# ######################################################################################################################

import geopandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

pd.set_option('display.max_columns', 500)


#%%

################################################################################
# Specify paths
################################################################################

input_path = "PATH TO FOLDER"
output_path = "PATH TO FOLDER"
geo_path = "PATH TO FOLDER"

################################################################################
# 1. Import geodata
################################################################################

# Read in shp-file for municipalities in Mexico
muns = geopandas.read_file(geo_path + "/mex_admbnda_adm2_govmex_20210618.shp")

# Change to same projection as points
muns = muns.to_crs("EPSG:5071")
muns.head()

################################################################################
# 2. Import educational attainment data
################################################################################

# Import, merge and check data
# -----------------------------------------------------------------------------
df = pd.read_csv(input_path + r"\population\population_municipality.csv", sep= ";", encoding="UTF8")

# Keep only relevant variables
df = pd.concat([df.loc[:,'ENTIDAD':'P_85YMAS_M'], df.loc[:,'P3A5_NOA':'GRAPROES_M']], axis=1)
df = df[df.columns[~df.columns.str.endswith('F') &
                      ~df.columns.str.endswith('M')]] # drop female/male vars
df.drop(["LOC", "NOM_LOC", "LONGITUD", "LATITUD", "ALTITUD"], axis=1, inplace=True)

# Create municipality code
df["code"] = ("MX" +
               df.ENTIDAD.astype("str").str.zfill(2) +
               df.MUN.astype("str").str.zfill(3))

# Merge with geodata
muns[["ADM2_PCODE", "geometry"]].merge(df,
                                       left_on = "ADM2_PCODE",
                                       right_on="code",
                                       how="outer",
                                       indicator=True)["_merge"].value_counts() # Only 12 from right not merged

df = muns[["ADM2_PCODE", "geometry"]].merge(df,
                                             left_on= "ADM2_PCODE",
                                             right_on="code",
                                             how="inner")


# Consistency checks
df.POBTOT.sum() # Population Mexico: 130.3 million (2021)
df[["NOM_ENT", "NOM_MUN", "ADM2_PCODE"]].value_counts()
len(df)
df.head(20)


# Inspect and create education and population variables
# -----------------------------------------------------------------------------

# Population
df["POBTOT"].describe() # Población total
df.rename({"POBTOT": "population"}, inplace=True, axis=1)
df["P_15YMAS"].describe() # Población de 15 años y más
df["P_18YMAS"].describe() # Población de 18 años y más

# Years of schooling
df["GRAPROES"].describe() # Grado promedio de escolaridad
df.rename({"GRAPROES": "edu_years_schooling"}, inplace=True, axis=1)

# Inspect schooling categories
(df["P15YM_SE"]/df["P_15YMAS"]).describe() # no schooling
(df["P15PRI_IN"]/df["P_15YMAS"]).describe() # incomplete primary
(df["P15PRI_CO"]/df["P_15YMAS"]).describe() # complete primary
(df["P15SEC_IN"]/df["P_15YMAS"]).describe() # incomplete secondary
(df["P15SEC_CO"]/df["P_15YMAS"]).describe() # complete secondary
(df["P18YM_PB"]/df["P_18YMAS"]).describe() # postbasic degree (college/high school, university)

((df["P15YM_SE"]
  + df["P15PRI_IN"]
  + df["P15PRI_CO"]
  + df["P15SEC_IN"]
  + df["P15SEC_CO"]
  + df["P18YM_PB"]) # defined for 18+ instead of 15+ (15+ can be obtained as residual)
 /df["P_15YMAS"]).describe()

# Postbasic degree (bachillerato o más)
df["edu_postbasic"] = df["P18YM_PB"]/df["P_18YMAS"]
df["edu_postbasic"].describe() # % with high school (bachillerato) or university

# Secondary degree
df["edu_secondary"] = (df["P_15YMAS"]
                       -df["P15YM_SE"]
                       -df["P15PRI_IN"]
                       -df["P15PRI_CO"]
                       -df["P15SEC_IN"])/df["P_15YMAS"]
df["edu_secondary"].describe() # % with completed secondary or more


# Primary degree
df["edu_primary"] = (df["edu_secondary"]
                     + df["P15SEC_IN"]/df["P_15YMAS"]
                     + df["P15PRI_CO"]/df["P_15YMAS"])
df["edu_primary"].describe() # % with completed primary or more


# Export dataframe with years of schooling and population data
edu = df[["ADM2_PCODE", "NOM_MUN", "edu_years_schooling",
           "edu_postbasic", "edu_secondary", "edu_primary"]]
edu.rename({"ADM2_PCODE": "GEOID",
            "NOM_MUN": "NAME"}, inplace=True, axis=1)
edu.head()

edu.to_excel(output_path + "/education-data.xlsx", index=False)


