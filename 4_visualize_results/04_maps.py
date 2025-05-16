

#-------------------------------------------------------------------------------
################################################################################
# Maps
# -----------------------------------------------------------
# - Maps with true vss predicted for MX
# - Maps with true vss predicted for US
# - Difference maps for MX and US
################################################################################
#-------------------------------------------------------------------------------



################################################################################
# Import modules
################################################################################

import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm
import matplotlib.colors
import geopandas
import matplotlib
from matplotlib.patches import Rectangle

pd.set_option('display.max_columns', 20)

################################################################################
# Specify paths to folders
################################################################################

from paths import paths_mx as paths
mx_input_path = paths["results"]
mx_geo_path = geo_path = paths["geo_data"]
exhibits_path = paths["exhibits"] + r"/figures/analysis/"
module_path = os.getcwd() + paths["modules"]

from paths import paths_us as paths
us_input_path = paths["results"]
us_geo_path = us_path = paths["geo_data"]

################################################################################
# Import functions and labels
################################################################################

# Import functions
sys.path.insert(1, module_path)


################################################################################
# Prepare data and set parameters
################################################################################

# Colors
mx_color = "#00008B"
us_color = "#8B0000"

mcmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#fcfffb", "#d0eeca",
                                                                 "#389acb", "#00008B"])
uscmap = "OrRd"
diff_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#f7f6f6", "#e58268",
                                                                    "#8B0000"])

diff_map = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#00008B", "#64a8cf",
                                                                    "#f7f6f6", "#e58268",
                                                                    "#8B0000"])

# Prediction data
mx_df = pd.read_csv(mx_input_path + r"\yschooling_predictions.csv",
                    sep=";",
                    index_col="GEOID")

us_df = pd.read_csv(us_input_path + r"\yschooling_predictions.csv",
                    sep=";", dtype={"GEOID": str},
                    index_col="GEOID")


# Import and add geodata
muns = geopandas.read_file(mx_geo_path + "\mex_admbnda_adm2_govmex_20210618.shp")
muns.rename({"ADM2_ES" : "NAME",
             "ADM2_PCODE" : "GEOID"},
            axis=1,
            inplace=True)
muns = muns.set_index("GEOID")
print(muns.crs)
muns = muns.to_crs("EPSG:5071")
mx_df = mx_df.merge(muns["geometry"],
                    left_index=True,
                    right_index=True)
mx_df = geopandas.GeoDataFrame(mx_df)
len(mx_df)


cts = geopandas.read_file(us_geo_path + r"\cb_2018_us_county_500k.shp",
                          dtype={"GEOID": str})
cts = cts.set_index("GEOID")
print(cts.crs)
cts = cts.to_crs("EPSG:5071")

us_df = us_df.merge(cts["geometry"],
                    left_index=True,
                    right_index=True)
us_df = geopandas.GeoDataFrame(us_df)
len(us_df)


# Set fontsize for all figures
matplotlib.rcParams.update({'font.size': 20})




# #######################################################################################
# Plot predictions and true values for MX
# #######################################################################################


# Set data and define colorbar
data = mx_df
norm = mpl.colors.Normalize(vmin=3.8, vmax=14.2)
sm = ScalarMappable(cmap=mcmap, norm=norm)

# Define circle around MX city
circle_mcity = data.centroid.loc["MX09002"].buffer(200000)
mx_clipped = data.clip(circle_mcity)

# Initialize figure
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

for i, col in enumerate(["y", "y_hat"]):

    # Plot borders
    data.plot(edgecolor="black", ax=axes[i], linewidth=2)

    # Plot education data
    data.plot(ax=axes[i],
              column=col,
              cmap=mcmap,
              linewidth=0.1,
              norm=norm,
              missing_kwds={"color": "black"})

    # Customize axis
    axes[i].set_xlim(-2000000, 1050000)
    axes[i].axis("off")

    # Add circle around MX city
    geopandas.geoseries.GeoSeries(circle_mcity).plot(ax=axes[i],
                                                     edgecolor="black",
                                                     facecolor="none",
                                                     linewidth=1)

# Create new axes for zoomed part
zoom_axes = fig.add_axes([0.262, 0.5, 0.33, 0.33]), \
    fig.add_axes([0.66, 0.5, 0.33, 0.33])

# Add circle map with zoomed part
for i, col in enumerate(["y", "y_hat"]):

    # Plot borders
    mx_clipped.plot(edgecolor="black", ax=zoom_axes[i], linewidth=1)

    # Plot education data
    zoom1_map = mx_clipped.plot(ax=zoom_axes[i],
                                column=col,
                                cmap=mcmap,
                                linewidth=0.1,
                                norm=norm,
                                missing_kwds={"color": "black"})
    # Turn axis off
    zoom_axes[i].axis("off")

    # Add circle
    geopandas.geoseries.GeoSeries(circle_mcity).plot(ax=zoom_axes[i],
                                                     edgecolor="black",
                                                     facecolor="none",
                                                     linewidth=1)
# Set titles
axes[0].set_title("Years of schooling", y=-0.1)
axes[1].set_title("Predicted years of schooling", y=-0.1)

# Add colorbar
cbaxes = fig.add_axes([0.93, 0.07, 0.02, 0.76])
cb = fig.colorbar(sm, cax=cbaxes)

# Adjust and export
fig.subplots_adjust(wspace=0.05)
fig.savefig(exhibits_path + r'\mx_yschooling_map.png', bbox_inches='tight')
fig.show()




# #######################################################################################
# Plot predictions and true values for the US
# #######################################################################################

# Set dataset and colormap
data = us_df
data["y"].min()
data["y"].nsmallest(3)
data["y"].max()

norm = mpl.colors.Normalize(vmin=10.6,
                            vmax=16.8)
sm = ScalarMappable(cmap=uscmap,
                    norm=norm)

# Initalize plot
fig, axes = plt.subplots(1, 2,
                         figsize=(24, 10))

# Define axes
for i, col in enumerate(["y", "y_hat"]):

    # Plot borders
    data.plot(edgecolor="black", ax=axes[i], linewidth=2)

    # Plot education data
    data.plot(ax=axes[i],
              column=col,
              cmap=uscmap,
              linewidth=0.1,
              norm=norm,
              missing_kwds={"color": "black"})

    # Customize axis
    axes[i].set_xlim(-2500000, 2300000)
    axes[i].set_ylim(300000, 3200000)
    axes[i].axis("off")


# Create new axes for Alaska and Hawaii
al_axes = fig.add_axes([0.1, 0.18, 0.15, 0.15]), \
    fig.add_axes([0.5, 0.18, 0.15, 0.15])

ha_axes = fig.add_axes([0.2, 0.18, 0.08, 0.08]), \
    fig.add_axes([0.6, 0.18, 0.08, 0.08])


# Define axes
for i, col in enumerate(["y", "y_hat"]):
    # Plot borders
    data.plot(edgecolor="black", ax=al_axes[i], linewidth=2)
    data.plot(edgecolor="black", ax=ha_axes[i], linewidth=2)

    # Plot education data
    for sub_axes in [al_axes, ha_axes]:
        data.plot(ax=sub_axes[i],
                  column=col,
                  cmap=uscmap,
                  linewidth=0.1,
                  norm=norm,
                  missing_kwds={"color": "black"})

        sub_axes[i].tick_params(left=False, bottom=False)
        sub_axes[i].xaxis.set_ticklabels([])
        sub_axes[i].yaxis.set_ticklabels([])

        sub_axes[i].spines['bottom'].set_color('lightgrey')
        sub_axes[i].spines['top'].set_color('lightgrey')
        sub_axes[i].spines['right'].set_color('lightgrey')
        sub_axes[i].spines['left'].set_color('lightgrey')

    # Customize axis
    al_axes[i].set_xlim(-4800000, -2100000)
    al_axes[i].set_ylim(3850000, 6300000)

    ha_axes[i].set_xlim(-6400000, -5900000)
    ha_axes[i].set_ylim(1450000, 2150000)


# Add titles
axes[0].set_title("Years of schooling", y=-0.2)
axes[1].set_title("Predicted years of schooling", y=-0.2)

# Add colorbar
cbaxes = fig.add_axes([0.93, 0.12, 0.02, 0.645])
cb = fig.colorbar(sm, cax=cbaxes)

# Adjust and export
fig.subplots_adjust(wspace=0.05)
fig.savefig(exhibits_path + r'\us_yschooling_map.png', bbox_inches='tight')
fig.show()



# #######################################################################################
# Plot differences for MX and US
# #######################################################################################


# Compute and inspect differences
mx_df["diff"] = mx_df["y_hat"] - mx_df["y"]
mx_df["diff"].nsmallest(3)
mx_df["diff"].nlargest(3)

us_df["diff"] = us_df["y_hat"] - us_df["y"]
us_df["diff"].nsmallest(3)
us_df["diff"].nlargest(3)

# Set parameters for figure
#cmap = cm.get_cmap('RdBu_r')
cmap = diff_map
norm = mpl.colors.Normalize(vmin=-3,
                            vmax=3)
sm = ScalarMappable(cmap=cmap,
                    norm=norm)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

# Map for Mexico
#-------------------------------------------------------------------------------------------
mx_df.plot(edgecolor="black",
           ax=ax1,
           linewidth=2)
map1 = mx_df.plot(ax=ax1,
                  column="diff",
                  cmap=cmap,
                  linewidth=0.1,
                  norm=norm,
                  missing_kwds={"color": "black"})
ax1.axis("off")
ax1.set_title("MX: $\hat{y} - y$", y=-0.1)

# Map for USA
#-------------------------------------------------------------------------------------------
us_df.plot(edgecolor="black", ax=ax2, linewidth=2)
map2 = us_df.plot(ax=ax2,
                  column="diff",
                  cmap=cmap,
                  linewidth=0.1,
                  norm=norm,
                  missing_kwds={"color": "black"})
ax2.axis("off")
ax2.set_xlim(-2500000, 2300000)
ax2.set_ylim(300000, 3200000)
ax2.set_title("US: $\hat{y} - y$", y=-0.18)

# Zoom map for MX
#-------------------------------------------------------------------------------------------

# Add ax for zoom around MX city
mx_zoom = fig.add_axes([0.262, 0.5, 0.33, 0.33])
mx_clipped = mx_df.clip(circle_mcity)
mx_clipped.plot(edgecolor="black", ax=mx_zoom, linewidth=1)

# Plot education data
mx_clipped.plot(ax=mx_zoom,
                column="diff",
                cmap=cmap,
                linewidth=0.1,
                norm=norm,
                missing_kwds={"color": "black"})
# Turn axis off
mx_zoom.axis("off")

# Add circles
geopandas.geoseries.GeoSeries(circle_mcity).plot(ax=mx_zoom,
                                                 edgecolor="black",
                                                 facecolor="none",
                                                 linewidth=1)
geopandas.geoseries.GeoSeries(circle_mcity).plot(ax=ax1,
                                                 edgecolor="black",
                                                 facecolor="none",
                                                 linewidth=1)


# Add Alaska and Hawaii
#-------------------------------------------------------------------------------------------


# Create new axes for Alaska and Hawaii
sub_axes = fig.add_axes([0.5, 0.18, 0.15, 0.15]), \
    fig.add_axes([0.6, 0.18, 0.08, 0.08])

# Define axes
for i in range(2):
    # Plot borders
    us_df.plot(edgecolor="black", ax=sub_axes[i], linewidth=2)

    # Plot education data
    us_df.plot(ax=sub_axes[i],
               column="diff",
               cmap=cmap,
               linewidth=0.1,
               norm=norm,
               missing_kwds={"color": "black"})

    sub_axes[i].tick_params(left=False, bottom=False)
    sub_axes[i].xaxis.set_ticklabels([])
    sub_axes[i].yaxis.set_ticklabels([])

    sub_axes[i].spines['bottom'].set_color('lightgrey')
    sub_axes[i].spines['top'].set_color('lightgrey')
    sub_axes[i].spines['right'].set_color('lightgrey')
    sub_axes[i].spines['left'].set_color('lightgrey')

# Customize axis
sub_axes[0].set_xlim(-4800000, -2100000)
sub_axes[0].set_ylim(3850000, 6300000)

sub_axes[1].set_xlim(-6400000, -5900000)
sub_axes[1].set_ylim(1450000, 2150000)


# Add colorbar and finalize
#-------------------------------------------------------------------------------------------
cbaxes = fig.add_axes([0.93, 0.1, 0.02, 0.7])
cb = fig.colorbar(sm,
                  cax=cbaxes,
                  norm=norm)
fig.subplots_adjust(wspace=0)
fig.savefig(exhibits_path + r'\yschooling_map_diff.png',
            bbox_inches='tight')
fig.show()

