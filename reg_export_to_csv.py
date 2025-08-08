# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 14:08:04 2025

@author: tchen
"""
from multiprocessing import Pool
from sqlalchemy import create_engine, text
import pandas as pd
import os

# Shared engine and base_query will be initialized per process
base_query = None
engine = None
table_name = None


def init_worker(shared_query, conn_str, table_name_arg):
    global base_query, engine, table_name
    base_query = shared_query
    table_name = table_name_arg
    engine = create_engine(conn_str)


def fetch_and_export(cnty_code):
    query = base_query.format(cnty_code=cnty_code)
    df = pd.read_sql(query, engine)
    filename = f"export_cnty_{cnty_code}_{table_name}.csv"
    df.to_csv(filename, index=False, quoting=1, encoding="utf-8")
    print(f"Exported cnty_code {cnty_code} to {filename}")
    return filename


sql_options = {
    1: ("GWIS_WATER_USE", "../sql/GWIS/water_use_from_reg.sql"),
    2: ("GWIS_PERMIT_INFO", "../sql/GWIS/permit_info_from_reg.sql"),
    3: ("GWIS_WU_MONITORING", "../sql/GWIS/wu_monitoring_from_reg.sql"),
    4: ("GWIS_WU_APP_LU", "../sql/GWIS/wu_app_lu_from_reg.sql"),
    5: ("GWIS_APP_COUNTIES", "../sql/GWIS/app_counties_from_reg.sql"),
    6: ("GWIS_WU_ALLOCATION", "../sql/GWIS/wu_allocation_from_reg.sql"),
    # look further into query
    8: ("GWIS_AG_IRR_ACRES", "../sql/GWIS/ag_irr_acres_from_reg.sql"),
    7: ("GWIS_WU_APP_TYPE", "../sql/GWIS/wu_app_type_from_reg.sql"),
    9: ("REG_WU_FACILITY_GROUP", "../sql/GWIS/wu_facility_group_from_reg.sql"),
    10: ("REG_WU_APP_FAC_GROUP", "../sql/GWIS/wu_app_fac_group_from_reg.sql"),
    11: ("GWIS_WU_FACILITIES", "../sql/GWIS/wu_facilities_from_reg.sql")
    # Add more
}


def display_menu():
    print("Select the table to load:")
    for key, (name, _) in sql_options.items():
        print(f"{key}. {name}")


if __name__ == '__main__':
    display_menu()
    selection = int(input(
        "Enter the number of the table you want to export: ").strip())

    if selection not in sql_options:
        raise ValueError(f"Invalid selection: {selection}")

    table_name, sql_path = sql_options[selection]

    if not os.path.isfile(sql_path):
        raise FileNotFoundError(f"SQL file not found at: {sql_path}")

    with open(sql_path, "r") as file:
        base_query_text = file.read()

    print(f"Loaded SQL for table: {table_name}")

    # TNS connection
    username = "pub"
    password = "pub"
    tns_name = "GENP"
    conn_str = f"oracle+cx_oracle://{username}:{password}@{tns_name}"

    cnty_codes = [6, 8, 11, 13, 22, 26, 28, 36, 43, 44, 47, 48, 49, 50, 53, 56]

    if (selection <= 7):
        with Pool(processes=min(len(cnty_codes), os.cpu_count()),
                  initializer=init_worker,
                  initargs=(base_query_text, conn_str, table_name)) as pool:
            pool.map(fetch_and_export, cnty_codes)

        # Export
        dfs = [pd.read_csv(f"export_cnty_{code}_{table_name}.csv")
               for code in cnty_codes]
        combined_df = pd.concat(dfs, ignore_index=True)
        export_name = f"upd_{table_name}.csv"
        combined_df.to_csv(export_name, index=False,
                           quoting=1, encoding="utf-8")
        print(f"Completed export to {export_name}")

        # Delete county by county files
        for code in cnty_codes:
            filename = f"export_cnty_{code}_{table_name}.csv"
            try:
                os.remove(filename)
                print(f"Deleted {filename}")
            except FileNotFoundError:
                print(f"{filename} not found for deletion.")
            except Exception as e:
                print(f"Error deleting {filename}: {e}")
    else:   # Continue without multiprocessing
        engine = create_engine(conn_str)
        df = pd.read_sql(base_query_text, engine)
        export_name = f"upd_{table_name}.csv"
        df.to_csv(export_name, index=False, quoting=1, encoding="utf-8")
        print(f"Completed export to {export_name}")
