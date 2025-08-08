# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 09:26:05 2025

@author: tchen
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 14:33:59 2025

@author: tchen


Groundwater, Tidal, Rain, and Pumpage Station Plotting
Through Folium Library

Creates "east_coast_grouped_stations.html" in same directory

"""

from datetime import datetime
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import getpass
import pyodbc
import folium
from folium import FeatureGroup, CircleMarker, Popup
from folium.plugins import Search
import random
import time
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import LineString, MultiLineString
from shapely import union_all
from shapely import wkt
from pyproj import Transformer
import folium
import branca.colormap as cm
from folium.plugins import FastMarkerCluster, MarkerCluster, Search
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.patches as mpatches
from sklearn.metrics import mean_squared_error
from matplotlib.dates import DateFormatter
import itertools
username = "pub"
password = "pub"
tns_name = "GENP"

try:
    reg_engine = create_engine(
        f"oracle+cx_oracle://{username}:{password}@{tns_name}")
    print('Database connection established')
except Exception as err:
    print(err)

username = "pub"
password = "pub"
tns_name = "WREP"

try:
    dbhydro_engine = create_engine(
        f"oracle+cx_oracle://{username}:{password}@{tns_name}")
    print('Database connection established')
except Exception as err:
    print(err)


# Pull groundwater data from DBHydro

aquifers = ['BISCAYNE', 'SURFICIAL AQUIFER SYSTEM']

dbcoords_df = pd.DataFrame()
coords_aquifers = []

for Aquifer in aquifers:
    dbkey_coords = f"""SELECT k.DBKEY, k.STATION, k.XCOORD, k.YCOORD, mv.AQUIFER
            FROM dmdbase.keyword_tab k JOIN
            dmdbase.well_inventory_mv mv ON
            k.STATION = mv.STATION
            and k.DBKEY = mv.SFWMD_DBKEY
            WHERE k.data_type LIKE 'WEL%'
            AND UPPER(mv.AQUIFER) LIKE '{Aquifer}'
            ORDER BY k.DBKEY"""
    coords_df = pd.read_sql(dbkey_coords, dbhydro_engine)
    coords_aquifers.append(coords_df)

dbcoords_df = pd.concat(coords_aquifers, ignore_index=True)

shapefile_paths = [
    "East_Coast_Surficial.shp",
    "West_Coast_Lower_Tamiami.shp",
    "West_Coast_Mid-Hawthorn.shp",
    "West_Coast_Sandstone.shp",
    "West_Coast_Water_Table.shp",
    "East_Coast_Surficial2019.shp",
    "West_Coast_Lower_Tamiami2019.shp",
    "West_Coast_Mid-Hawthorn2019.shp",
    "West_Coast_Sandstone2019.shp",
    "West_Coast_Water_Table2019.shp",
    "East_Coast_Surficial2009.shp",
    "West_Coast_Lower_Tamiami2009.shp",
    "West_Coast_Mid-Hawthorn2009.shp",
    "West_Coast_Sandstone2009.shp",
    "West_Coast_Water_Table2009.shp"
]

prefix = "shapefiles"

isochlor_buffers = []
buffer_distance_meters = 8046.72

for shapefile in shapefile_paths:
    path = os.path.join(prefix, shapefile)
    label = os.path.splitext(shapefile)[0].replace("_", " ")

    # Load and buffer
    isochlor_gdf = gpd.read_file(path)
    if isochlor_gdf.crs is None:
        isochlor_gdf.set_crs(epsg=2236, inplace=True)
    else:
        isochlor_gdf = isochlor_gdf.to_crs(epsg=2236)

    buffered = isochlor_gdf.buffer(buffer_distance_meters)
    isochlor_buffers.extend(buffered)

combined_buffer = gpd.GeoSeries(union_all(isochlor_buffers), crs="EPSG:2236")

dbcoords_gdf = gpd.GeoDataFrame(dbcoords_df, geometry=gpd.points_from_xy(
    dbcoords_df['xcoord'], dbcoords_df['ycoord']), crs="EPSG:2881")
dbcoords_gdf.to_crs(epsg=2236)

buffered_wells = dbcoords_gdf[dbcoords_gdf.geometry.intersects(
    combined_buffer.iloc[0])]

buffered_wells.to_crs(epsg=2881)

groundwater_df = pd.DataFrame()
start_date = '01/01/1960'

# daily_dfs = []
# for DBKEY in buffered_wells["dbkey"]:
#     dbhydro_sql = f"""SELECT DAILY_DATE, VALUE, CODE
#                   FROM DMDBASE.DM_DAILY_DATA
#                  WHERE DBKEY = '{DBKEY}'
#                    AND DAILY_DATE > TO_DATE('{start_date}', 'MM/DD/YYYY')
#                  ORDER BY DAILY_DATE"""
#     daily = pd.read_sql(dbhydro_sql, dbhydro_engine)
#     daily_dfs.append(daily)

# groundwater_df = pd.concat(daily_dfs, ignore_index=True)


def get_groundwater_daily_from_dbkey(dbkey, dbhydro_engine):
    dbhydro_sql = f"""SELECT DAILY_DATE, VALUE, CODE
                  FROM DMDBASE.DM_DAILY_DATA
                 WHERE DBKEY = '{dbkey}'
                   AND DAILY_DATE > TO_DATE('01/01/1980', 'MM/DD/YYYY')
                 ORDER BY DAILY_DATE"""
    query = dbhydro_sql.format(dbkey=dbkey)
    data = pd.read_sql(query, dbhydro_engine)

    if len(data) < 20:
        return None

    return data


def safe_sjoin_nearest(left_gdf, right_gdf, how="left", distance_col="distance", max_distance=None):     # Rain #
    # A safe sjoin for geopandas dataframes
    # Drop any conflicting columns in advance
    conflict_cols = ['index_right', 'index_left']
    right_gdf = right_gdf.drop(
        columns=[col for col in conflict_cols if col in right_gdf.columns], errors='ignore')
    left_gdf = left_gdf.drop(
        columns=[col for col in conflict_cols if col in left_gdf.columns], errors='ignore')

    if not all(left_gdf.geometry.geom_type == 'Point'):
        left_gdf = left_gdf.copy()
        left_gdf['geometry'] = left_gdf.geometry.centroid

    if not all(right_gdf.geometry.geom_type == 'Point'):
        right_gdf = right_gdf.copy()
        right_gdf['geometry'] = right_gdf.geometry.centroid

    # Ensure CRS matches
    if left_gdf.crs != right_gdf.crs:
        right_gdf = right_gdf.to_crs(left_gdf.crs)

    # Perform join
    result = gpd.sjoin_nearest(
        left_gdf, right_gdf, how=how, distance_col=distance_col
    )

    # Filter out rows with distance greater than max_distance
    if max_distance is not None:
        result = result[result[distance_col] <= max_distance].copy()

    return result


# Pull tidal station stage data
tidal_stations = pd.read_csv(
    r"\\ad.sfwmd.gov\dfsroot\data\wsd\SUP\devel\source\Python\LookAtTheWells\look_at_wells_TSC\CHD_STG_pnts.csv")
tidal_gdf = gpd.GeoDataFrame(tidal_stations, geometry=gpd.points_from_xy(
    tidal_stations["centx"], tidal_stations["centy"]), crs="EPSG:2881")
stage_data = pd.read_csv(
    r"\\ad.sfwmd.gov\dfsroot\data\wsd\SUP\devel\source\Python\LookAtTheWells\look_at_wells_TSC\CHD_STG_timeseries.csv")

buffered_tidal_stations = safe_sjoin_nearest(
    # This number is probably too high but I don't want to lose too many stations
    buffered_wells, tidal_gdf, max_distance=48280)
# print("BUFF TIDAL STATIONS: ", buffered_tidal_stations)
matched_station_ids = buffered_tidal_stations['STATION'].unique()
buffered_stage_data = stage_data[stage_data['station'].isin(
    matched_station_ids)]

# Pull pixels from DBHydro

pixel_sql = """
    SELECT pixel_id, pixel_centroid_x, pixel_centroid_y
        FROM nrd_pixel
    WHERE pixel_id <> 99999999
        AND pixel_centroid_x != 0
        """
pixels_df = pd.read_sql(pixel_sql, dbhydro_engine)
pixels_gdf = gpd.GeoDataFrame(pixels_df, geometry=gpd.points_from_xy(
    pixels_df["pixel_centroid_x"], pixels_df["pixel_centroid_y"]), crs="EPSG:2881")
buffered_pixels = safe_sjoin_nearest(
    buffered_wells, pixels_gdf, max_distance=4828)
# print("BUFF PIXELS: ", buffered_pixels)

# buffered_pixel_data = rain_data_from_pixels(buffered_pixels, 2024, 2025)


# Novewmber to May 20 - DRY

# Pull pumpage from REG
pumpage_sql = """
WITH RankedApps AS (
  SELECT
     admin.app_no,
     TO_CHAR (apfac.FACINV_ID, '9999999') FACINV_ID,
     fac.FACINV_TYPE,
     fac.NAME fac_Name,
     fac.facwlsts_code facility_status,
     fac.CASED_DEPTH,
     fac.WELL_DEPTH,
     fac.PUMP_COORDX,
     fac.PUMP_COORDY,
     fac.PUMP_INTAKE_DEPTH,
     fac.TOP_OF_CASING,
     fac.MEAS_PT_ELEV,
     src.id source_id,
     admin.FINAL_ACTION_DATE,
     ROW_NUMBER() OVER (
       PARTITION BY apfac.FACINV_ID
       ORDER BY
         CASE WHEN admin.FINAL_ACTION_DATE IS NULL THEN 1 ELSE 0 END,
         admin.FINAL_ACTION_DATE DESC
     ) AS rn
  FROM REG.admin admin
     INNER JOIN REG.WU_APP_FACILITY apfac ON apfac.ADMIN_APP_NO = admin.APP_NO
     INNER JOIN REG.APP_COUNTIES cnty ON cnty.ADMIN_APP_NO = admin.APP_NO
     INNER JOIN REG.APP_LC_DEFS lc ON lc.ADMIN_APP_NO = admin.APP_NO
     INNER JOIN REG.APP_LANDUSES lu ON lu.ADMIN_APP_NO = admin.APP_NO
     INNER JOIN REG.WU_FAC_INV fac ON fac.ID = apfac.FACINV_ID
     INNER JOIN REG.WUC_APP_LC_REQMTS req ON req.APPLC_ADMIN_APP_NO = admin.APP_no
     INNER JOIN REG.tl_counties tl_counties ON tl_counties.COUNTY_CODE = cnty.CNTY_CODE
     INNER JOIN REG.tl_sources src ON src.ID = apfac.SOURCE_ID
     INNER JOIN REG.TL_REQUIREMENTS tlreq ON tlreq.ID = req.TL_LC_REQM_ID
     INNER JOIN REG.WU_FAC_STS_TRK sts ON sts.facinv_id = apfac.facinv_id
  WHERE lc.ADMIN_APP_NO = req.APPLC_ADMIN_APP_NO
     AND TO_CHAR(apFac.FACINV_ID) = req.REQM_ENT_KEY1
     AND lc.ID = req.APPLC_ID
     AND cnty.PRIORITY LIKE 1
     AND lu.USE_PRIORITY LIKE 1
     AND tlreq.NAME LIKE 'Water Use Report%'
     AND admin.APP_STATUS = 'COMPLETE'
     AND lu.LU_CODE IN (
        'PWS','NUR','LIV','AQU','AGR','LAN','GOL','REC','IND','COM',
        'PPG','PPO','PPR','PPM','DIV','DI2'
     )
     AND cnty.CNTY_CODE IN (6, 8, 11, 13, 22, 26, 28, 36, 43, 44, 47, 48, 49, 50, 53, 56)
)
SELECT
  app_no,
  facinv_id,
  fac_name,
  facility_status,
  cased_depth,
  well_depth,
  pump_coordx,
  pump_coordy,
  pump_intake_depth,
  top_of_casing,
  meas_pt_elev,
  source_id
FROM RankedApps
  WHERE rn = 1 """
pumpage_df = pd.read_sql(pumpage_sql, reg_engine)
pumpage_gdf = gpd.GeoDataFrame(pumpage_df, geometry=gpd.points_from_xy(
    pumpage_df['pump_coordx'], pumpage_df['pump_coordy']), crs="EPSG:2881")

buffered_pumpage_df = safe_sjoin_nearest(
    buffered_wells, pumpage_gdf, max_distance=4828)
# print("BUFFERED PUMPAGE: ", buffered_pumpage_df)

merged1 = buffered_wells.merge(buffered_tidal_stations, on=[
                               'dbkey', 'station', 'xcoord', 'ycoord', 'aquifer', 'geometry'], how='inner')
merged2 = merged1.merge(buffered_pixels, on=[
    'dbkey', 'station', 'xcoord', 'ycoord', 'aquifer', 'geometry'], how='inner')
well_tidal_rain_pumpage = merged2.merge(buffered_pumpage_df, on=[
    'dbkey', 'station', 'xcoord', 'ycoord', 'aquifer', 'geometry'], how='inner')
well_tidal_rain_pumpage.to_csv("well_tidal_rain_pumpage.csv", index=False,
                               quoting=1, encoding="utf-8")


def plot_folium_map_from_buffered_wells(df, rain_df, tidal_df, pumpage_df, output_html="florida_wells_by_depth.html"):
    transformer = Transformer.from_crs(
        "EPSG:2236", "EPSG:4326", always_xy=True)

    m = folium.Map(location=[27.8, -81.7], zoom_start=7)

    # Group isochlors by year
    group_2024 = folium.FeatureGroup(name="Isochlors 2024")
    group_2019 = folium.FeatureGroup(name="Isochlors 2019")
    group_2009 = folium.FeatureGroup(name="Isochlors 2009")

    shapefile_paths = [
        "East_Coast_Surficial.shp",
        "West_Coast_Lower_Tamiami.shp",
        "West_Coast_Mid-Hawthorn.shp",
        "West_Coast_Sandstone.shp",
        "West_Coast_Water_Table.shp",
        "East_Coast_Surficial2019.shp",
        "West_Coast_Lower_Tamiami2019.shp",
        "West_Coast_Mid-Hawthorn2019.shp",
        "West_Coast_Sandstone2019.shp",
        "West_Coast_Water_Table2019.shp",
        "East_Coast_Surficial2009.shp",
        "West_Coast_Lower_Tamiami2009.shp",
        "West_Coast_Mid-Hawthorn2009.shp",
        "West_Coast_Sandstone2009.shp",
        "West_Coast_Water_Table2009.shp"
    ]

    # Assign colors by aquifer group
    aquifer_colors = {
        "Surficial": "red",
        "Lower_Tamiami": "blue",
        "Mid-Hawthorn": "green",
        "Sandstone": "orange",
        "Water_Table": "purple"
    }

    # Assign line dash patterns by year
    year_styles = {
        "2024": "0",          # Solid
        "2019": "5, 5",       # Dashed
        "2009": "1, 6"        # Dotted
    }

    colors = ['red', 'blue', 'green', 'orange', 'purple'] * 3
    prefix = "shapefiles"

    # Add isochlor lines

    for shapefile in shapefile_paths:
        path = os.path.join(prefix, shapefile)
        label = os.path.splitext(shapefile)[0].replace("_", " ")

        # Extract aquifer and year from filename
        name = os.path.splitext(shapefile)[0]
        match = re.search(r'(.*?)(2009|2019)?$', name)

        aquifer_key = match.group(1).replace(
            "East_Coast_", "").replace("West_Coast_", "").rstrip("_")
        year_key = match.group(2) if match.group(2) else "2024"

        color = aquifer_colors.get(aquifer_key, "black")
        dash = year_styles.get(year_key, "0")

        # Load and buffer
        isochlor_gdf = gpd.read_file(path)
        if isochlor_gdf.crs is None:
            isochlor_gdf.set_crs(epsg=2236, inplace=True)
        else:
            isochlor_gdf = isochlor_gdf.to_crs(epsg=2236)

        isochlor_gdf = isochlor_gdf.to_crs(epsg=4326)

        # Format datetime columns
        for col in isochlor_gdf.select_dtypes(include=['datetime64[ns]']).columns:
            isochlor_gdf[col] = isochlor_gdf[col].dt.strftime(
                '%Y-%m-%d %H:%M:%S')

        # Add styled GeoJson
        layer = folium.GeoJson(
            isochlor_gdf,
            name=label,
            style_function=make_style(color=color, dash_array=dash)
        )

        if year_key == '2019':
            layer.add_to(group_2019)
        elif year_key == '2009':
            layer.add_to(group_2009)
        else:
            layer.add_to(group_2024)

    # Add groups to map
    group_2024.add_to(m)
    group_2019.add_to(m)
    group_2009.add_to(m)

    wells_near_isochlors = gpd.GeoDataFrame(
        df, geometry='geometry', crs="EPSG:2881")

    # Reproject to WGS84 for Folium
    wells_near_isochlors = wells_near_isochlors.to_crs(epsg=4326)
    print(wells_near_isochlors)
    rain_df = gpd.GeoDataFrame(rain_df, geometry=gpd.points_from_xy(
        rain_df['pixel_centroid_x'], rain_df['pixel_centroid_y']), crs="EPSG:2881")
    rain_pixels = rain_df.to_crs(epsg=4326)
    print(rain_pixels)
    tidal_df = gpd.GeoDataFrame(tidal_df, geometry=gpd.points_from_xy(
        tidal_df['centx'], tidal_df['centy']), crs="EPSG:2881")
    tidal_stations = tidal_df.to_crs(epsg=4326)
    print(tidal_stations)
    pumpage_df = gpd.GeoDataFrame(pumpage_df, geometry=gpd.points_from_xy(
        pumpage_df['pump_coordx'], pumpage_df['pump_coordy']), crs="EPSG:2881")
    pump_stations = pumpage_df.to_crs(epsg=4326)
    print(pump_stations)

    # Extract lat/lon for plotting
    wells_near_isochlors['LAT'] = wells_near_isochlors.geometry.y
    wells_near_isochlors['LON'] = wells_near_isochlors.geometry.x

    rain_pixels['LAT'] = rain_pixels.geometry.y
    rain_pixels['LON'] = rain_pixels.geometry.x

    tidal_stations['LAT'] = tidal_stations.geometry.y
    tidal_stations['LON'] = tidal_stations.geometry.x

    pump_stations['LAT'] = pump_stations.geometry.y
    pump_stations['LON'] = pump_stations.geometry.x

    well_group = folium.FeatureGroup(name="Wells (Searchable)")
    feature_list = []

    for _, row in wells_near_isochlors.iterrows():
        if pd.notnull(row['LAT']) and pd.notnull(row['LON']):
            marker = folium.CircleMarker(
                location=[row['LAT'], row['LON']],
                radius=4,
                popup=f"{row['station']}",
                color="blue",
                fill=True,
                fill_opacity=0.8
            )

        feature = folium.GeoJson(
            data={
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row['LON'], row['LAT']],
                },
                "properties": {
                    "name": row['station'],  # Use 'WELL' as searchable name
                },
            },
            name=row['station'],
            marker=marker,
            tooltip=row['station']
        )
        feature.add_to(well_group)
        feature_list.append(feature)

    well_group.add_to(m)

    rain_group = folium.FeatureGroup(name="Rain")

    for _, row in rain_pixels.iterrows():
        if pd.notnull(row['LAT']) and pd.notnull(row['LON']):
            marker = folium.CircleMarker(
                location=[row['LAT'], row['LON']],
                radius=4,
                popup=f"{row['pixel_id']}",
                color="red",
                fill=True,
                fill_opacity=0.8
            )

        feature = folium.GeoJson(
            data={
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row['LON'], row['LAT']],
                },
                "properties": {
                    "name": row['pixel_id'],
                },
            },
            name=row['pixel_id'],
            marker=marker,
            tooltip=row['pixel_id']
        )
        feature.add_to(rain_group)

    rain_group.add_to(m)

    tidal_group = folium.FeatureGroup(name="Tidal Stations")

    for _, row in tidal_stations.iterrows():
        if pd.notnull(row['LAT']) and pd.notnull(row['LON']):
            marker = folium.CircleMarker(
                location=[row['LAT'], row['LON']],
                radius=4,
                popup=f"{row['STATION']}",
                color="green",
                fill=True,
                fill_opacity=0.8
            )

        feature = folium.GeoJson(
            data={
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row['LON'], row['LAT']],
                },
                "properties": {
                    "name": row['STATION'],
                },
            },
            name=row['STATION'],
            marker=marker,
            tooltip=row['STATION']
        )
        feature.add_to(tidal_group)

    tidal_group.add_to(m)

    pump_group = folium.FeatureGroup(name="Pumpage Stations")

    for _, row in pump_stations.iterrows():
        if pd.notnull(row['LAT']) and pd.notnull(row['LON']):
            marker = folium.CircleMarker(
                location=[row['LAT'], row['LON']],
                radius=4,
                popup=f"{row['fac_name']}",
                color="orange",
                fill=True,
                fill_opacity=0.8
            )

        feature = folium.GeoJson(
            data={
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row['LON'], row['LAT']],
                },
                "properties": {
                    "name": row['fac_name'],
                },
            },
            name=row['fac_name'],
            marker=marker,
            tooltip=row['fac_name']
        )
        feature.add_to(pump_group)

    pump_group.add_to(m)

    Search(
        layer=well_group,
        search_label="name",
        placeholder="Search for a well...",
        collapsed=False,
    ).add_to(m)

    # Add layer toggle
    folium.LayerControl(collapsed=False).add_to(m)

    # Save map
    try:
        print("Map ready, saving...")
        m.save(output_html)
    except Exception as e:
        print("Failed to save Folium map: ", e)


def make_style(color, dash_array="0"):
    return lambda feature: {
        'color': color,
        'weight': 3,
        'opacity': 0.8,
        'dashArray': dash_array
    }


# def plot_folium_map_from_grouped_stations(df, output_html="east_coast_grouped_stations.html"):
#     """
#     Plot a Folium map where each row represents a group of related stations.
#     When you click on any station, it will show information about all associated stations.
#     """
#     from pyproj import Transformer
#     import folium
#     from folium.plugins import Search
#     import geopandas as gpd
#     import pandas as pd
#     import os
#     import re

#     transformer = Transformer.from_crs(
#         "EPSG:2881", "EPSG:4326", always_xy=True)

#     m = folium.Map(location=[27.8, -81.7], zoom_start=7)

#     # Add isochlor groups
#     group_2024 = folium.FeatureGroup(name="Isochlors 2024")
#     group_2019 = folium.FeatureGroup(name="Isochlors 2019")
#     group_2009 = folium.FeatureGroup(name="Isochlors 2009")

#     # Create station groups
#     well_group = folium.FeatureGroup(name="Wells (Primary)")
#     rain_group = folium.FeatureGroup(name="Rain Stations")
#     tidal_group = folium.FeatureGroup(name="Tidal Stations")
#     pump_group = folium.FeatureGroup(name="Pumpage Stations")

#     # Process each row (group of related stations)
#     for idx, row in df.iterrows():
#         group_id = f"group_{idx}"

#         # Create a comprehensive popup with all station info for this group
#         popup_content = f"""
#         <div style='font-family: Arial; max-width: 300px;'>
#             <h4>Station Group {idx + 1}</h4>
#             <hr>
#         """

#         # Well information (primary station)
#         well_marker = None
#         if pd.notnull(row.get('station')):
#             if hasattr(row['geometry'], 'x') and hasattr(row['geometry'], 'y'):
#                 well_lon, well_lat = transformer.transform(
#                     row['geometry'].x, row['geometry'].y)
#                 popup_content += f"""
#                 <b>Well:</b> {row.get('station', 'N/A')}<br>
#                 """

#         # Rain station information
#         rain_marker = None
#         rain_x = row.get('pixel_centroid_x')
#         rain_y = row.get('pixel_centroid_y')
#         if pd.notnull(rain_x) and pd.notnull(rain_y):
#             rain_lon, rain_lat = transformer.transform(rain_x, rain_y)
#             popup_content += f"""
#             <b>Rain Station:</b> {row.get('pixel_id', 'N/A')}<br>
#             """

#         # Tidal station information
#         tidal_marker = None
#         tidal_x = row.get('centx')
#         tidal_y = row.get('centy')
#         if pd.notnull(tidal_x) and pd.notnull(tidal_y):
#             tidal_lon, tidal_lat = transformer.transform(tidal_x, tidal_y)
#             popup_content += f"""
#             <b>Tidal Station:</b> {row.get('STATION', 'N/A')}<br>
#             """

#         # Pumpage station information
#         pump_marker = None
#         pump_x = row.get('pump_coordx')
#         pump_y = row.get('pump_coordy')
#         if pd.notnull(pump_x) and pd.notnull(pump_y):
#             pump_lon, pump_lat = transformer.transform(pump_x, pump_y)
#             popup_content += f"""
#             <b>Pumpage Station:</b> {row.get('fac_name', 'N/A')}<br>
#             """

#         popup_content += "</div>"

#         # Create and add markers with the complete popup content
#         if pd.notnull(row.get('station')) and 'well_lat' in locals():
#             well_marker = folium.CircleMarker(
#                 location=[well_lat, well_lon],
#                 radius=6,
#                 popup=popup_content,
#                 color="blue",
#                 fill=True,
#                 fillColor="blue",
#                 fillOpacity=0.8,
#                 tooltip=f"Well: {row.get('station', 'N/A')} (Group {idx + 1})"
#             )
#             well_marker.add_to(well_group)

#         if pd.notnull(rain_x) and pd.notnull(rain_y):
#             rain_marker = folium.CircleMarker(
#                 location=[rain_lat, rain_lon],
#                 radius=5,
#                 popup=popup_content,
#                 color="red",
#                 fill=True,
#                 fillColor="red",
#                 fillOpacity=0.8,
#                 tooltip=f"Rain: {row.get('pixel_id', 'N/A')} (Group {idx + 1})"
#             )
#             rain_marker.add_to(rain_group)

#         if pd.notnull(tidal_x) and pd.notnull(tidal_y):
#             tidal_marker = folium.CircleMarker(
#                 location=[tidal_lat, tidal_lon],
#                 radius=5,
#                 popup=popup_content,
#                 color="green",
#                 fill=True,
#                 fillColor="green",
#                 fillOpacity=0.8,
#                 tooltip=f"Tidal: {row.get('STATION', 'N/A')} (Group {idx + 1})"
#             )
#             tidal_marker.add_to(tidal_group)

#         if pd.notnull(pump_x) and pd.notnull(pump_y):
#             pump_marker = folium.CircleMarker(
#                 location=[pump_lat, pump_lon],
#                 radius=5,
#                 popup=popup_content,
#                 color="orange",
#                 fill=True,
#                 fillColor="orange",
#                 fillOpacity=0.8,
#                 tooltip=f"Pump: {row.get('fac_name', 'N/A')} (Group {idx + 1})"
#             )
#             pump_marker.add_to(pump_group)

#     # Add all groups to map
#     well_group.add_to(m)
#     rain_group.add_to(m)
#     tidal_group.add_to(m)
#     pump_group.add_to(m)

#     Search(
#         layer=well_group,
#         search_label="tooltip",
#         placeholder="Search for a well...",
#         collapsed=False,
#     ).add_to(m)

#     # Add layer control
#     folium.LayerControl(collapsed=False).add_to(m)

#     # Save map
#     try:
#         print("Map ready, saving...")
#         m.save(output_html)
#         print(f"Map saved to {output_html}")
#     except Exception as e:
#         print("Failed to save Folium map: ", e)


def plot_folium_map_from_grouped_stations(df, output_html="east_coast_grouped_stations.html"):
    """
    Plot a Folium map where each row represents a group of related stations.
    When you click on any station, it will show information about all associated stations.
    """
    from pyproj import Transformer
    import folium
    from folium.plugins import Search
    import geopandas as gpd
    import pandas as pd
    import os
    import re

    transformer = Transformer.from_crs(
        "EPSG:2881", "EPSG:4326", always_xy=True)

    m = folium.Map(location=[27.8, -81.7], zoom_start=7)

    # Add isochlor
    group_2024 = folium.FeatureGroup(name="Isochlors 2024")
    group_2019 = folium.FeatureGroup(name="Isochlors 2019")
    group_2009 = folium.FeatureGroup(name="Isochlors 2009")

    # Create station groups
    well_group = folium.FeatureGroup(name="Wells (Primary)")
    rain_group = folium.FeatureGroup(name="Rain Stations")
    tidal_group = folium.FeatureGroup(name="Tidal Stations")
    pump_group = folium.FeatureGroup(name="Pumpage Stations")

    # Store marker references for group highlighting
    all_markers = []

    # Process each row (group of related stations)
    for idx, row in df.iterrows():
        group_id = idx  # Use simple index as group ID

        # Create a comprehensive popup with all station info for this group
        popup_content = f"""
        <div style='font-family: Arial; max-width: 300px;'>
            <h4>Station Group {idx + 1}</h4>
            <hr>
        """

        group_markers = []  # Store markers for this specific group

        # Well information (primary station)
        if pd.notnull(row.get('station')):
            # Convert well geometry (assuming it's in EPSG:2881)
            if hasattr(row['geometry'], 'x') and hasattr(row['geometry'], 'y'):
                well_lon, well_lat = transformer.transform(
                    row['geometry'].x, row['geometry'].y)
                popup_content += f"""
                <b>Well:</b> {row.get('station', 'N/A')}<br>
                """

        # Rain station information
        rain_x = row.get('pixel_centroid_x')
        rain_y = row.get('pixel_centroid_y')
        if pd.notnull(rain_x) and pd.notnull(rain_y):
            rain_lon, rain_lat = transformer.transform(rain_x, rain_y)
            popup_content += f"""
            <b>Rain Station:</b> {row.get('pixel_id', 'N/A')}<br>
            """

        # Tidal station information
        tidal_x = row.get('centx')
        tidal_y = row.get('centy')
        if pd.notnull(tidal_x) and pd.notnull(tidal_y):
            tidal_lon, tidal_lat = transformer.transform(tidal_x, tidal_y)
            popup_content += f"""
            <b>Tidal Station:</b> {row.get('STATION', 'N/A')}<br>
            """

        # Pumpage station information
        pump_x = row.get('pump_coordx')
        pump_y = row.get('pump_coordy')
        if pd.notnull(pump_x) and pd.notnull(pump_y):
            pump_lon, pump_lat = transformer.transform(pump_x, pump_y)
            popup_content += f"""
            <b>Pumpage Station:</b> {row.get('fac_name', 'N/A')}<br>
            """

        popup_content += "</div>"

        # Create markers and add them to groups, storing references for highlighting
        if pd.notnull(row.get('station')) and 'well_lat' in locals():
            well_marker = folium.CircleMarker(
                location=[well_lat, well_lon],
                radius=6,
                popup=popup_content,
                color="blue",
                fill=True,
                fillColor="blue",
                fillOpacity=0.8,
                tooltip=f"Well: {row.get('station', 'N/A')} (Group {idx + 1})"
            )
            well_marker.add_to(well_group)
            group_markers.append(well_marker)

        if pd.notnull(rain_x) and pd.notnull(rain_y):
            rain_marker = folium.CircleMarker(
                location=[rain_lat, rain_lon],
                radius=5,
                popup=popup_content,
                color="red",
                fill=True,
                fillColor="red",
                fillOpacity=0.8,
                tooltip=f"Rain: {row.get('pixel_id', 'N/A')} (Group {idx + 1})"
            )
            rain_marker.add_to(rain_group)
            group_markers.append(rain_marker)

        if pd.notnull(tidal_x) and pd.notnull(tidal_y):
            tidal_marker = folium.CircleMarker(
                location=[tidal_lat, tidal_lon],
                radius=5,
                popup=popup_content,
                color="green",
                fill=True,
                fillColor="green",
                fillOpacity=0.8,
                tooltip=f"Tidal: {row.get('STATION', 'N/A')} (Group {idx + 1})"
            )
            tidal_marker.add_to(tidal_group)
            group_markers.append(tidal_marker)

        if pd.notnull(pump_x) and pd.notnull(pump_y):
            pump_marker = folium.CircleMarker(
                location=[pump_lat, pump_lon],
                radius=5,
                popup=popup_content,
                color="orange",
                fill=True,
                fillColor="orange",
                fillOpacity=0.8,
                tooltip=f"Pump: {row.get('fac_name', 'N/A')} (Group {idx + 1})"
            )
            pump_marker.add_to(pump_group)
            group_markers.append(pump_marker)

        # Store this group's markers with group ID for later JavaScript access
        all_markers.append({
            'group_id': group_id,
            'markers': group_markers
        })

    # Add all groups to map
    well_group.add_to(m)
    rain_group.add_to(m)
    tidal_group.add_to(m)
    pump_group.add_to(m)

    # Add CSS and JavaScript for group highlighting
    highlight_script = """
    <style>
        .marker.highlighted {
            stroke: #ffff00 !important;
            stroke-width: 4 !important;
            stroke-opacity: 1 !important;
            fill-opacity: 1 !important;
        }
        .marker.highlighted.well-marker {
            fill: #0066ff !important;
        }
        .marker.highlighted.rain-marker {
            fill: #ff3333 !important;
        }
        .marker.highlighted.tidal-marker {
            fill: #00cc66 !important;
        }
        .marker.highlighted.pump-marker {
            fill: #ff9900 !important;
        }
    </style>
    
    <script>
        function highlightGroup(groupId) {
            // Remove previous highlights
            document.querySelectorAll('.marker').forEach(function(marker) {
                marker.classList.remove('highlighted');
            });
            
            // Highlight all markers in the same group
            document.querySelectorAll('.' + groupId).forEach(function(marker) {
                marker.classList.add('highlighted');
            });
        }
        
        function clearHighlights() {
            document.querySelectorAll('.marker').forEach(function(marker) {
                marker.classList.remove('highlighted');
            });
        }
        
        // Add event listeners after the map loads
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(function() {
                // Add click listeners to all markers
                document.querySelectorAll('.marker').forEach(function(marker) {
                    var groupClass = Array.from(marker.classList).find(cls => cls.startsWith('group_'));
                    if (groupClass) {
                        marker.addEventListener('click', function() {
                            highlightGroup(groupClass);
                        });
                        
                        // Optional: clear highlights when clicking elsewhere
                        marker.addEventListener('mouseleave', function() {
                            setTimeout(clearHighlights, 3000); // Clear after 3 seconds
                        });
                    }
                });
                
                // Clear highlights when clicking on the map background
                document.querySelector('.folium-map').addEventListener('click', function(e) {
                    if (e.target.classList.contains('leaflet-container')) {
                        clearHighlights();
                    }
                });
            }, 1000); // Wait for markers to be rendered
        });
    </script>
    """

    # Add the highlight script to the map
    m.get_root().html.add_child(folium.Element(highlight_script))

    # Add search functionality (searches wells by default)
    Search(
        layer=well_group,
        search_label="tooltip",
        placeholder="Search for a well...",
        collapsed=False,
    ).add_to(m)

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Save map
    try:
        print("Map ready, saving...")
        m.save(output_html)
        print(f"Map saved to {output_html}")
    except Exception as e:
        print("Failed to save Folium map: ", e)


# plot_folium_map_from_buffered_wells(
#      well_tidal_rain_pumpage.iloc[:, 0:6], well_tidal_rain_pumpage.iloc[:, 15:18], well_tidal_rain_pumpage.iloc[:, 7:13], well_tidal_rain_pumpage.iloc[:, 20:32])
plot_folium_map_from_grouped_stations(well_tidal_rain_pumpage)
