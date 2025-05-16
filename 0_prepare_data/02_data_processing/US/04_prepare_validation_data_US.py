
#-------------------------------------------------------------------------------
################################################################################
# Prepare and check validation data for the US
#-------------------------------------------------------------------------------
# - Imports, cleans, inspects and maps county level data on schooling
# - (and population) to export file for data analysis
################################################################################
#-------------------------------------------------------------------------------

################################################################################
# Import modules
################################################################################

import pandas as pd
import numpy as np
import geopandas
from matplotlib import pyplot as plt


################################################################################
# Specify paths to folders
################################################################################

input_path = "PATH TO FOLDER"
output_path = "PATH TO FOLDER"
geo_path = "PATH TO FOLDER"


################################################################################
# Import, clean and inspect schooling data
################################################################################

df = pd.read_excel(input_path + r"\Education_US.xlsx",
                   converters={"Federal Information Processing Standards (FIPS) Code":str})
df.head()



# Keep only relevant columns
df = df.iloc[: , np.r_[0:3, 5:7, -8:-0]]

# Rename columns
df.columns = ["GEOID", "state", "name", "rural_urban_2013", "urban_influence2013",
              "total_no_high", "total_only_high", "total_some_college", "total_bachelor",
              "percent_no_high", "percent_only_high", "percent_some_college", "percent_bachelor"]
df.head()

# Keep only county-level information
df["isstate"] = df["GEOID"].apply(lambda x: x[2:] == "000")
states = df[df["isstate"]]
df = df[df["isstate"] == False]
df.drop("isstate", axis=1, inplace=True)
len(df)

# Inspect educational attainment variables
print(df.percent_bachelor.describe())
print(df.percent_some_college.describe())
print(df.percent_only_high.describe())
print(df.percent_no_high.describe())


# Create outcome variables
df["edu_years_schooling"] = (df.percent_no_high * 8.4 +
                    df.percent_only_high * 12 +
                    df.percent_some_college * 13.9 +
                    df.percent_bachelor * 17.1)/100 # Years of schooling (based on nation averages per category)
df["edu_years_schooling"].describe()


df["edu_bachelor"] = df["percent_bachelor"]
df["edu_bachelor"].describe()

df["edu_some_college"] = df.percent_bachelor + df.percent_some_college # At least some college
df["edu_some_college"].describe()

df["edu_high"] = df.percent_only_high + df.percent_bachelor + df.percent_some_college # At least high school
df["edu_high"].describe()

# Create population variable
df["population"] = df.total_no_high + df.total_only_high + df.total_some_college + df.total_bachelor


# Keep only relevant variables
df.columns
df = df[["GEOID", "name", "edu_years_schooling", "edu_bachelor",
         "edu_some_college", "edu_high"]]



################################################################################
# Combine with SHP file
################################################################################

# Read in shp-file on us counties
counties = geopandas.read_file(geo_path + r"\cb_2018_us_county_500k.shp")
counties.head()

# Check if merge is correct
gdf = counties.merge(df, on="GEOID", how="outer")
print(gdf.loc[gdf["STATEFP"].isna(), ["name", "GEOID"]]) # Not in geodata
print(gdf.loc[gdf["name"].isna(), ["NAME", "GEOID"]]) # Not in census data

# Merge
gdf = counties.merge(df, on="GEOID", how="inner")
len(gdf)

# Plot to check if merge is correct
gdf = gdf.to_crs("EPSG:5071") # change projection
ax = gdf.plot(figsize=(11, 5), edgecolor="black")
gdf.plot(column="edu_bachelor",
         ax=ax,
         cmap="OrRd",
         edgecolor="black",
         linewidth=0.2,
         legend=True,
         missing_kwds={"color": "lightgrey", "label": "Missing values"},
         legend_kwds={'label': "% with bachelor"})
plt.axis("off")

ax.set_xlim(-2400000, 2300000)
ax.set_ylim(200000, 3200000)
plt.show()


################################################################################
# Export cleaned dataframe on education levels
################################################################################

df.rename(columns={"name": "NAME"}).to_csv(output_path + r"\education-data.csv",
                                           index=False,
                                           sep=";",
                                           encoding="UTF8")