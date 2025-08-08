# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 14:33:59 2025

@author: tchen


Groundwater Level Regression Analysis
For Wells on the East Coast Surficial and Biscayne Aquifers

Parameters include tidal stage data, rain data, and pumpage
to create additive multilinear regression model

Several functions taken frmo RodbergReport

Look into upgrading through multiprocessing in the future

Improved models could include more factors, dimensions, interaction terms

"""
import os
from multiprocessing import Pool
from sqlalchemy import create_engine, text
from sqlalchemy import inspect, Table, MetaData
from sqlalchemy.types import (
    String, Date, DateTime, Numeric, Integer, Float, NullType
)
from datetime import datetime
import pandas as pd
import numpy as np
import getpass

import pyodbc
import time
import os
import sys
import pandas as pd
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

# config
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

# Pull pixel rain data from DBHydro


def rain_data_from_pixels(geo_df, start_year, end_year):      # Rain #
    pixel_list = geo_df['pixel_id'].astype(int).tolist()
    all_years_data = []

    for year in range(start_year, end_year+1):
        pixel_tuple = tuple(pixel_list)
        if (len(pixel_tuple) == 1):
            pixel_tuple = f"({pixel_list[0]})"
        sqlQ = f'''WITH selected_pixels AS (
           SELECT pixel_id, pixel_centroid_x AS x, pixel_centroid_y AS y
             FROM nrd_pixel
            WHERE pixel_id
               IN {pixel_tuple}
           ),
           date_series AS (
               SELECT TO_CHAR(DATE '{year}-01-01'+LEVEL-1, 'yyyy-mm-dd') AS da
                 FROM dual
              CONNECT BY DATE '{year}-01-01'+LEVEL-1 < DATE '{year + 1}-01-01'
           ),
           base_grid AS (
               SELECT p.pixel_id, p.x, p.y, d.da
                 FROM selected_pixels p
                CROSS JOIN date_series d
           ),
           ts_agg AS (
               SELECT nts.featureid, TRUNC(nts.tsdatetime) AS ts_date,
                      SUM(nts.tsvalue) AS sumtsvalue
                 FROM nrd_time_series nts
                WHERE nts.featureid
                   IN {pixel_tuple}
                  AND nts.tstypeid = 3
                  AND nts.tsdatetime >= DATE '{year}-01-01'
                  AND nts.tsdatetime < DATE '{year + 1}-01-01'
                GROUP BY nts.featureid, TRUNC(nts.tsdatetime)
           )
           SELECT bg.pixel_id, bg.x, bg.y, bg.da,
                 COALESCE(ta.sumtsvalue, 0) AS value
             FROM base_grid bg
             LEFT JOIN ts_agg ta
               ON bg.pixel_id = ta.featureid
              AND bg.da = TO_CHAR(ta.ts_date, 'yyyy-mm-dd')
            ORDER BY bg.pixel_id, bg.da
            '''
        chunk_df = pd.read_sql(sqlQ, dbhydro_engine)
        all_years_data.append(chunk_df)

    full_df = []
    if all_years_data:
        full_df = pd.concat(all_years_data, ignore_index=True)

    return full_df


def rain_data_from_pixel_id(pixel_id, start_year, end_year):      # Rain #
    pixel_list = [pixel_id]
    all_years_data = []

    for year in range(start_year, end_year+1):
        pixel_tuple = tuple(pixel_list)
        if (len(pixel_tuple) == 1):
            pixel_tuple = f"({pixel_list[0]})"
        sqlQ = f'''WITH selected_pixels AS (
           SELECT pixel_id, pixel_centroid_x AS x, pixel_centroid_y AS y
             FROM nrd_pixel
            WHERE pixel_id
               IN {pixel_tuple}
           ),
           date_series AS (
               SELECT TO_CHAR(DATE '{year}-01-01'+LEVEL-1, 'yyyy-mm-dd') AS da
                 FROM dual
              CONNECT BY DATE '{year}-01-01'+LEVEL-1 < DATE '{year + 1}-01-01'
           ),
           base_grid AS (
               SELECT p.pixel_id, p.x, p.y, d.da
                 FROM selected_pixels p
                CROSS JOIN date_series d
           ),
           ts_agg AS (
               SELECT nts.featureid, TRUNC(nts.tsdatetime) AS ts_date,
                      SUM(nts.tsvalue) AS sumtsvalue
                 FROM nrd_time_series nts
                WHERE nts.featureid
                   IN {pixel_tuple}
                  AND nts.tstypeid = 3
                  AND nts.tsdatetime >= DATE '{year}-01-01'
                  AND nts.tsdatetime < DATE '{year + 1}-01-01'
                GROUP BY nts.featureid, TRUNC(nts.tsdatetime)
           )
           SELECT bg.pixel_id, bg.x, bg.y, bg.da,
                 COALESCE(ta.sumtsvalue, 0) AS value
             FROM base_grid bg
             LEFT JOIN ts_agg ta
               ON bg.pixel_id = ta.featureid
              AND bg.da = TO_CHAR(ta.ts_date, 'yyyy-mm-dd')
            ORDER BY bg.pixel_id, bg.da
            '''
        chunk_df = pd.read_sql(sqlQ, dbhydro_engine)
        all_years_data.append(chunk_df)

    full_df = []
    if all_years_data:
        full_df = pd.concat(all_years_data, ignore_index=True)

    return full_df


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


def get_pumpage_data_from_facinv_id(facinv_id, reg_engine):
    pumpage_sql = """
    SELECT
         admin.app_no,
         TO_CHAR(apfac.FACINV_ID, '9999999') AS FACINV_ID,
         sub.APPLIES_TO_DATE,
         sub.SUBM_VALUE AS data_value,
         sub.TMU_CODE AS data_value_units,
         req.TRE_ID AS subm_type,
         tlreq.NAME AS req_name,
         req.WALR_TYPE AS report_type,
         admin.permit_no AS site_id
      FROM REG.admin admin
         INNER JOIN REG.WU_APP_FACILITY apfac ON apfac.ADMIN_APP_NO = admin.APP_NO
         INNER JOIN REG.APP_COUNTIES cnty ON cnty.ADMIN_APP_NO = admin.APP_NO
         INNER JOIN REG.APP_LC_DEFS lc ON lc.ADMIN_APP_NO = admin.APP_NO
         INNER JOIN REG.APP_LANDUSES lu ON lu.ADMIN_APP_NO = admin.APP_NO
         INNER JOIN REG.WU_FAC_INV fac ON fac.ID = apfac.FACINV_ID
         INNER JOIN REG.WUC_APP_LC_REQMTS req ON req.APPLC_ADMIN_APP_NO = admin.APP_NO
         INNER JOIN REG.TL_REQUIREMENTS tlreq ON tlreq.ID = req.TL_LC_REQM_ID
         INNER JOIN REG.WUC_APP_SIM_SUBMS sub ON sub.WALR_REQM_ID = req.REQM_ID
      WHERE lc.ADMIN_APP_NO = req.APPLC_ADMIN_APP_NO
         AND TO_CHAR(apfac.FACINV_ID) = req.REQM_ENT_KEY1
         AND req.TRE_ID IN (3, 4)
         AND lc.ID = req.APPLC_ID
         AND sub.SUBM_VALUE IS NOT NULL
         AND cnty.PRIORITY LIKE 1
         AND lu.USE_PRIORITY LIKE 1
         AND sub.APPLIES_TO_DATE BETWEEN TO_DATE('01/01/1980','MM/DD/YYYY')
                                    AND TO_DATE('12/31/2024','MM/DD/YYYY')
         AND tlreq.NAME LIKE 'Water Use Report%'
         AND lu.LU_CODE IN ('PWS','NUR','LIV','AQU','AGR','LAN','GOL','REC','IND','COM',
                            'PPG','PPO','PPR','PPM','DIV','DI2')
         AND cnty.CNTY_CODE IN (6, 8, 11, 13, 22, 26, 28, 36, 43, 44, 47, 48, 49, 50, 53, 56)
         AND req.WALR_TYPE LIKE 'SIMPLE'
         AND admin.APP_STATUS = 'COMPLETE'
         AND sub.tmu_code LIKE '%MG/MONTH%'
         AND facinv_id = '{facinv_id}'"""
    query = pumpage_sql.format(facinv_id=facinv_id)
    data = pd.read_sql(query, reg_engine)

    # If not enough data is returned
    if len(data) < 20:
        return None

    return data


def get_tidal_data_from_station(station, buffered_stage_data):
    station_data = buffered_stage_data[buffered_stage_data['station'] == station]

    # Make sure there is enough data
    if len(station_data) < 20:
        return None

    return station_data


def lmg_r2_parts(X, y):
    """
    Compute Lindeman (LMG) contributions of columns in X to the R^2 of y.
    Returns a dict {col_name: contribution}, which sums to full R^2.
    """
    cols = list(X.columns)
    p = len(cols)
    if p == 0:
        return {}

    full_model = LinearRegression().fit(X, y)
    full_r2 = full_model.score(X, y)

    # If only one predictor, trivially it's the full R2
    if p == 1:
        return {cols[0]: full_r2}

    # Sum incremental contributions over permutations
    contrib_sum = {c: 0.0 for c in cols}
    perms = list(itertools.permutations(cols))
    n_perms = len(perms)

    for perm in perms:
        prev_cols = []
        prev_r2 = 0.0
        for col in perm:
            # fit model with prev_cols + [col]
            cols_now = prev_cols + [col]
            model = LinearRegression().fit(X[cols_now], y)
            r2_now = model.score(X[cols_now], y)
            incremental = r2_now - prev_r2
            contrib_sum[col] += incremental
            prev_r2 = r2_now
            prev_cols = cols_now

    # average contributions across permutations
    contrib_avg = {c: contrib_sum[c] / n_perms for c in cols}

    # numerical cleanup: small negatives due to numerical errors -> clip tiny negatives
    for c in contrib_avg:
        if abs(contrib_avg[c]) < 1e-12:
            contrib_avg[c] = 0.0

    # sanity: they should sum to full_r2 (within rounding)
    total = sum(contrib_avg.values())
    # small adjustment optionally to force exact equality (distribute residual)
    residual = full_r2 - total
    if abs(residual) > 1e-12:
        # Add residual to the largest contributor to preserve relative sizes
        max_c = max(contrib_avg, key=lambda k: contrib_avg[k])
        contrib_avg[max_c] += residual

    return contrib_avg


def groundwater_regression(well_tidal_rain_pumpage):
    results = []

    # the first ~200 DBKEYS dont have data
    for _, row in well_tidal_rain_pumpage.iloc[215:].iterrows():
        dbkey = row['dbkey']
        tidal_station = row['STATION']
        pixel_id = row['pixel_id']
        pumpage_fac = row['facinv_id']

        groundwater_data = get_groundwater_daily_from_dbkey(
            dbkey, dbhydro_engine)
        if groundwater_data is None:
            continue
        tidal_data = get_tidal_data_from_station(
            tidal_station, buffered_stage_data)
        if tidal_data is None:
            continue
        rain_data = rain_data_from_pixel_id(pixel_id, 1980, 2024)
        if rain_data is None:
            continue
        pumpage_data = get_pumpage_data_from_facinv_id(pumpage_fac, reg_engine)
        if pumpage_data is None:
            continue

        # Remove any rows with missing or invalid stage values
        tidal_data['stage'] = pd.to_numeric(
            tidal_data['stage'], errors='coerce')
        tidal_data.dropna(subset=['stage'], inplace=True)

        print("groundwater_data: ", groundwater_data)
        print("tidal_data: ", tidal_data)
        print("rain_data: ", rain_data)
        print("pumpage_data: ", pumpage_data)
        # Ensure all queries returned something
        if any(x is None or x.empty for x in [groundwater_data, tidal_data, rain_data, pumpage_data]):
            continue

        # Convert dates
        groundwater_data['DATE'] = pd.to_datetime(
            groundwater_data['daily_date'])
        tidal_data['DATE'] = pd.to_datetime(tidal_data['date'])
        rain_data['DATE'] = pd.to_datetime(rain_data['da'])
        pumpage_data['DATE'] = pd.to_datetime(pumpage_data['applies_to_date'])

        # Format tidal, rain, and groundwater to monthly
        rain_monthly = rain_data.set_index('DATE')['value'].resample(
            'M').sum().to_frame(name='rain')
        tidal_monthly = tidal_data.set_index(
            'DATE')['stage'].resample('M').mean().to_frame(name='tide')
        pumpage_monthly = pumpage_data.set_index(
            'DATE')[['data_value']].rename(columns={'data_value': 'pumpage'})
        groundwater_monthly = groundwater_data.set_index(
            'DATE')['value'].resample('M').mean().to_frame(name='gw')

        print("groundwater_monthly: ", groundwater_monthly)
        # Combine dfs on shared dates
        merged_df = groundwater_monthly \
            .join([rain_monthly, tidal_monthly, pumpage_monthly], how='inner')

        # Add wet/dry factor
        merged_df['season'] = merged_df.index.month.map(
            lambda m: 'wet' if 5 <= m <= 10 else 'dry')
        merged_df = pd.get_dummies(
            merged_df, columns=['season'], drop_first=True)
        print("merged: ", merged_df)

        merged_df.dropna(inplace=True)
        if merged_df.empty:
            continue

        # Prepare predictors
        predictors = ['rain', 'tide', 'pumpage']
        if 'season_wet' in merged_df.columns:
            predictors.append('season_wet')

        X = merged_df[predictors]
        y = merged_df['gw']

        model = LinearRegression()
        model.fit(X, y)

        full_r2 = model.score(X, y)
        rmse = np.sqrt(mean_squared_error(y, model.predict(X)))

        # Partial r2 values
        partial_r2 = lmg_r2_parts(X, y)

        # Predict based on model
        merged_df['predicted_gw'] = model.predict(X)
        merged_df['dbkey'] = dbkey

        # Save to results
        results.append({
            'dbkey': dbkey,
            'r_squared': full_r2,
            'coefficients': dict(zip(X.columns, model.coef_)),
            'intercept': model.intercept_,
            'n_obs': len(merged_df),
            'partial_r2': partial_r2,
            'timeseries': merged_df.reset_index(),
            'RMSE': rmse
        })

    return pd.DataFrame(results)


def plot_well_timeseries(results_df, output_dir="groundwater_plots"):
    os.makedirs(output_dir, exist_ok=True)

    for _, row in results_df.iterrows():
        df = row['timeseries']
        dbkey = row['dbkey']
        partial_r2 = row['partial_r2']
        rmse = row['RMSE']
        r2 = row['r_squared']

        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot actual vs predicted groundwater
        ax.plot(df['DATE'], df['gw'], label='Observed', color='black')
        ax.plot(df['DATE'], df['predicted_gw'],
                label='Predicted', color='blue', linestyle='--')

        # Shade wet season (May–October)
        for year in df['DATE'].dt.year.unique():
            wet_start = pd.Timestamp(f"{year}-05-01")
            wet_end = pd.Timestamp(f"{year}-10-31")
            ax.axvspan(wet_start, wet_end, color='skyblue',
                       alpha=0.15, label='_nolegend_')

        # Title with performance metrics
        ax.set_title(f"DBKEY {dbkey} | R² = {r2:.2f} | RMSE = {rmse:.2f} ft")
        ax.set_xlabel("Date")
        ax.set_ylabel("Groundwater Level (ft)")

        # Legend for proportional R²s
        r2_patches = [
            mpatches.Patch(label=f"{factor}: {value:.2f}")
            for factor, value in partial_r2.items()
        ]

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + r2_patches,
                  loc='upper right', frameon=True)

        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

        plt.tight_layout()
        plt.savefig(f"{output_dir}/gw_plot_{dbkey}.png", dpi=300)
        plt.close()


results = groundwater_regression(well_tidal_rain_pumpage)
print(results)

plot_well_timeseries(results)


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


# plot_folium_map_from_buffered_wells(
#     well_tidal_rain_pumpage.iloc[:, 0:6], well_tidal_rain_pumpage.iloc[:, 15:18], well_tidal_rain_pumpage.iloc[:, 7:13], well_tidal_rain_pumpage.iloc[:, 20:32])
