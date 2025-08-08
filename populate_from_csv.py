# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 10:57:34 2025

@author: tchen

A program to insert csv entries to staging table.

Current login through pub/pub@WRED

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

# config, change for user
username = "pub"
password = getpass.getpass("Enter your password: ")
tns_name = "WRED"

try:
    engine = create_engine(
        f"oracle+cx_oracle://{username}:{password}@{tns_name}")
    print('Database connection established')
except Exception as err:
    print(err)


# Maps column names from reg csv output to GWIS column names, could be done dynamically
column_mappings = {
    "STG_GWIS_WATER_USE": {
        "app_no": "app_no",
        "applies_to_date": "applies_to_date",
        "subm_value": "data_value",
        "tmu_code": "data_value_units",
        "tre_id": "subm_type",
        "req_name": "req_name",
        "walr_type": "report_type",
        "permit_no": "site_id"
    },
    "STG_GWIS_PERMIT_INFO": {
        "permit_no": "permit_no",
        "project_name": "project_name",
        "app_no": "app_no",
        "final_action_date": "final_action_date",
        "expiration_date": "expiration_date",
        "permit_duration": "permit_duration",
        "acres_served": "acres_served",
        "lu_code": "landuse_code",
        "fee_category": "fee_category"
    },
    "STG_GWIS_WU_MONITORING": {
        "app_no": "app_no",
        "applies_to_date": "applies_to_date",
        "data_value": "data_value",
        "data_value_units": "data_value_units",
        "subm_type": "subm_type",
        "data_type": "data_type",
        "report_type": "report_type",
        "site_id": "site_id"
    },
    "STG_GWIS_WU_FACILITIES": {
        "app_no": "app_no",
        "facinv_id": "facinv_id",
        "facinv_type": "facinv_type",
        "fac_name": "fac_name",
        "facility_status": "facility_status",
        "cased_depth": "cased_depth",
        "well_depth": "well_depth",
        "pump_coordx": "pump_coordx",
        "pump_coordy": "pump_coordy",
        "pump_intake_depth": "pump_intake_depth",
        "top_of_casing": "top_of_casing",
        "meas_pt_elev": "meas_pt_elev",
        "source_id": "source_id"
    },
    "STG_GWIS_WU_APP_LU": {
        "app_no": "app_no",
        "landuse_code": "landuse_code",
        "priority": "priority"
    },
    "STG_GWIS_APP_COUNTIES": {
        "app_no": "app_no",
        "county_code": "county_code",
        "priority": "priority"
    },
    "STG_GWIS_WU_ALLOCATION": {
        "app_no": "app_no",
        "max_day_alloc": "max_day_alloc",
        "max_mon_alloc": "max_mon_alloc",
        "annual_alloc": "annual_alloc"
    },
    "STG_GWIS_AG_IRR_ACRES": {
        "app_no": "app_no",
        "field_id": "field_id",
        "irr_eff": "irr_eff",
        "landuse_code": "landuse_code",
        "field_acres": "field_acres",
        "crop_name": "crop_name",
        "use_priority": "use_priority",
        "irr_type": "irr_type"
    },
    "STG_GWIS_WU_APP_TYPE": {
        "app_no": "app_no",
        "fee_category": "fee_category",
        "fee_category_desc": "fee_category_desc"
    },
    "STG_WU_FACILITY_GROUP": {
        "facgrp_id": "facgrp_id",
        "app_no": "app_no",
        "permit_no": "permit_no",
        "name": "name",
        "process_text": "process_text",
        "comments": "comment",
        "active_flag": "active_flag"
    },
    "STG_WU_APP_FAC_GROUP": {
        "facinv_id": "facinv_id",
        "facgrp_id": "facgrp_id",
        "seq_no": "seq_no"
    }
}


#   Likely not using this function in finished script
# def create_empty_table(engine, original_table, new_table):
#     with engine.connect() as conn:
#         create_sql = f"""
#         CREATE TABLE {new_table} AS
#         SELECT * FROM {original_table} WHERE 1 = 0
#         """
#         try:
#             conn.execute(text(create_sql))
#             print(f"Table {new_table} created successfully from {
#                   original_table}.")
#         except Exception as err:
#             print(err)

#   Using batch insert now
# def insert_into_table(df, engine, target_table):
#     df.to_sql(target_table, engine, if_exists='append',
#               index=False)
#     print(f"Inserted {len(df)} rows into {target_table}.")

# Batch insert


def insert_batch(engine, df, target_table, batch_size=1000, schema="WSUP"):
    metadata = MetaData()
    metadata.reflect(bind=engine, schema=schema, only=[target_table.lower()])
    full_table_name = f"{schema}.{target_table.lower()}"
    table = metadata.tables[full_table_name]

    # Convert df to list of dictionaries
    records = df.to_dict(orient='records')

    if not records:
        print(f"No records to insert into {target_table}.")
        return

    with engine.begin() as conn:
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            conn.execute(table.insert(), batch)

    print(f"Inserted {len(records)} rows into {target_table}.")


def table_has_data(engine, table_name, schema="WSUP"):
    schema_prefix = f"{schema}." if schema else ""
    query = text(f"SELECT COUNT(*) FROM {schema_prefix}{table_name}")
    with engine.connect() as conn:
        result = conn.execute(query).scalar()
        return result > 0


def prep_tables(df, engine, table_name, schema="WSUP"):
    inspector = inspect(engine)
    columns_info = inspector.get_columns(table_name, schema=schema)

    # dictionary of {column_name: column_type}
    col_types = {
        col['name'].lower(): col['type']
        for col in columns_info
    }

    for col in df.columns:
        col_lc = col.lower()

        if col_lc not in col_types:
            continue  # skip if not found in DB table

        sql_type = col_types[col_lc]

        # handle datetime fields
        if isinstance(sql_type, (Date, DateTime)):
            df[col] = pd.to_datetime(df[col], errors='coerce')

        # convert non-numeric fields to string
        elif not isinstance(sql_type, (Numeric, Integer, Float)):
            df[col] = df[col].astype(str)

    return df


# Map options to a table and path
table_options = {
    1: ("STG_GWIS_WATER_USE", "upd_GWIS_WATER_USE.csv"),
    2: ("STG_GWIS_PERMIT_INFO", "upd_GWIS_PERMIT_INFO.csv"),
    3: ("STG_GWIS_WU_MONITORING", "upd_GWIS_WU_MONITORING.csv"),
    4: ("STG_GWIS_WU_FACILITIES", "upd_GWIS_WU_FACILITIES.csv"),
    5: ("STG_GWIS_APP_COUNTIES", "upd_GWIS_APP_COUNTIES.csv"),
    6: ("STG_GWIS_WU_ALLOCATION", "upd_GWIS_WU_ALLOCATION.csv"),    # check
    7: ("STG_GWIS_AG_IRR_ACRES", "upd_GWIS_AG_IRR_ACRES.csv"),
    8: ("STG_GWIS_WU_APP_TYPE", "upd_GWIS_WU_APP_TYPE.csv"),
    9: ("STG_WU_FACILITY_GROUP", "upd_REG_WU_FACILITY_GROUP.csv"),
    10: ("STG_WU_APP_FAC_GROUP", "upd_REG_WU_APP_FAC_GROUP.csv"),
    11: ("STG_GWIS_WU_APP_LU", "upd_GWIS_WU_APP_LU.csv")
    # Add more
}


def display_menu():
    print("Select the table to load:")
    for key, (name, _) in table_options.items():
        print(f"{key}. {name}")


if __name__ == "__main__":
    display_menu()
    selection = int(input("Select the table to populate: ").strip())

    if selection not in table_options:
        raise ValueError(f"Invalid selection: {selection}")

    table_name, csv_path = table_options[selection]

    upd_tables = []
    upd_tables.append(table_name)

    csvs = []
    csvs.append(csv_path)

    df = pd.read_csv(csv_path)

    prep_tables(df, engine, table_name)

    for upd_table in upd_tables:
        mapping = column_mappings[upd_table]
        df_mapped = df.rename(columns=mapping)

        # add missing cols with null values (temp solution)
        for col in mapping.values():
            if col not in df_mapped.columns:
                df_mapped[col] = pd.NA

        # subset df to only relevant columns
        df_subset = df_mapped[list(mapping.values())]

        # Deduplicate by primary key
        # if 'app_no' in df_subset.columns:
        #     df_subset = df_subset.drop_duplicates(subset='app_no')

        df_subset.replace({pd.NaT: None}, inplace=True)
        df_subset.replace({"nan": None}, inplace=True)
        df_subset.replace({np.NaN: None}, inplace=True)

        print(df_subset)

        if table_has_data(engine, upd_table):
            user_input = input(
                f"Table '{upd_table}' has existing data. Clear` it? [y/N]: ")
            if user_input.strip().lower() == 'y':
                try:
                    with engine.begin() as conn:
                        conn.execute(text(f"DELETE FROM WSUP.{upd_table}"))
                    print(f"Table '{upd_table}' was cleared.")
                    insert_batch(engine, df_subset, upd_table)
                    print("Insert completed successfully.")
                except Exception as e:
                    print(f"Insert failed for {upd_table}: {e}")
            else:
                print("Insert canceled.")
        else:
            try:
                insert_batch(engine, df_subset, upd_table)
                print("Insert completed successfully.")
            except Exception as e:
                print(f"Insert failed for {upd_table}: {e}")
