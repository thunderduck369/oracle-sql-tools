# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 09:11:02 2025

@author: tchen


A script to insert, update, and delete entries from staging to live tables.

Update functionality is shaky for some tables without reliable primary keys.

Current login through pub/pub@WRED

In terminal (navigated to directory), run:
streamlit run merge_differences.py

Ensure library dependencies are installed
Anaconda: in Anaconda prompt, run: conda activate {your_environment}

"""

import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy import inspect, Table, MetaData
from sqlalchemy.types import (
    String, Date, DateTime, Numeric, Integer, Float, NullType
)
from datetime import datetime

# --- Table Configuration ---
table_options = {
    1: ("UPD_GWIS_WATER_USE", "STG_GWIS_WATER_USE"),
    2: ("UPD_GWIS_PERMIT_INFO", "STG_GWIS_PERMIT_INFO"),
    3: ("UPD_GWIS_WU_MONITORING", "STG_GWIS_WU_MONITORING"),
    4: ("UPD_GWIS_WU_APP_LU", "STG_GWIS_WU_APP_LU"),
    5: ("UPD_GWIS_APP_COUNTIES", "STG_GWIS_APP_COUNTIES"),
    6: ("UPD_GWIS_WU_ALLOCATION", "STG_GWIS_WU_ALLOCATION"),
    7: ("UPD_GWIS_AG_IRR_ACRES", "STG_GWIS_AG_IRR_ACRES"),
    8: ("UPD_GWIS_WU_APP_TYPE", "STG_GWIS_WU_APP_TYPE"),
    9: ("REG_WU_FACILITY_GROUP", "STG_WU_FACILITY_GROUP"),
    10: ("REG_WU_APP_FAC_GROUP", "STG_WU_APP_FAC_GROUP"),
    11: ("UPD_GWIS_WU_FACILITIES", "STG_GWIS_WU_FACILITIES")
}

key_cols_dict = {
    # update unreliable for GWIS_WATER_USE and GWIS_WU_MONITORING
    "UPD_GWIS_WATER_USE": ["app_no", "applies_to_date", "site_id", "req_name", "data_value_units", "report_type"],
    "UPD_GWIS_PERMIT_INFO": ["app_no"],
    "UPD_GWIS_WU_MONITORING": ["app_no", "applies_to_date", "site_id", "data_type", "data_value_units", "report_type"],
    "UPD_GWIS_WU_APP_LU": ["app_no", "landuse_code"],
    "UPD_GWIS_APP_COUNTIES": ["app_no", "county_code"],
    "UPD_GWIS_WU_ALLOCATION": ["app_no"],
    "UPD_GWIS_AG_IRR_ACRES": ["app_no"],
    "UPD_GWIS_WU_APP_TYPE": ["app_no"],
    "REG_WU_FACILITY_GROUP": ["app_no"],
    "REG_WU_APP_FAC_GROUP": ["facinv_id"],
    "UPD_GWIS_WU_FACILITIES": ["facinv_id"]
}

# --- Sidebar Inputs ---
st.title("Staging Table To Final Merge")

# Sidebar setup
st.sidebar.header("Database Setup")

# Password input
if "password" not in st.session_state:
    st.session_state["password"] = ""

password_input = st.sidebar.text_input(
    "Oracle Password", type="password", key="password_input")
if password_input:
    st.session_state["password"] = password_input

if not st.session_state["password"]:
    st.warning("Please enter your password to continue.")
    st.stop()

# Table dropdown
table_labels = [f"{i}. {name}" for i, (name, _) in table_options.items()]
selected_label = st.sidebar.selectbox(
    "Select Table", table_labels, key="table_selector")

# Extract and persist selection
selected_number = int(selected_label.split(".")[0])
st.session_state["table_selection"] = selected_number

selection = st.session_state["table_selection"]
table_old, table_new = table_options[selection]
key_columns = key_cols_dict[table_old]

# --- Database Connection ---
username = "pub"
tns_name = "WRED"
engine = create_engine(
    f"oracle+cx_oracle://{username}:{st.session_state.password}@{tns_name}")

# --- Load Tables ---


def load_table(table_name, schema="WSUP"):
    query = f"SELECT * FROM {schema}.{table_name}"
    return pd.read_sql(query, engine)


df_old = load_table(table_old)
df_new = load_table(table_new)

for col in df_old.columns:
    if 'date' in col.lower():
        df_old[col] = pd.to_datetime(df_old[col], errors='coerce')
        df_new[col] = pd.to_datetime(df_new[col], errors='coerce')

# --- Compare Logic ---


def normalize_dataframe(df, engine, table_name, schema="WSUP"):
    df_normalized = df.copy()

    # Get database column information
    inspector = inspect(engine)
    columns_info = inspector.get_columns(table_name, schema=schema)

    # Create mapping of column names to types (case-insensitive)
    col_types = {
        col['name'].lower(): col['type']
        for col in columns_info
    }

    for col in df_normalized.columns:
        col_lower = col.lower()

        if col_lower not in col_types:
            continue

        sql_type = col_types[col_lower]

        # Handle different data types
        if isinstance(sql_type, (Date, DateTime)):
            # Normalize datetime columns
            df_normalized[col] = pd.to_datetime(
                df_normalized[col], errors='coerce')

        elif isinstance(sql_type, (Numeric, Integer, Float)):
            # Normalize numeric columns
            df_normalized[col] = pd.to_numeric(
                df_normalized[col], errors='coerce')

        else:
            # String columns - convert to string but preserve nulls
            df_normalized[col] = df_normalized[col].astype('object')
            df_normalized[col] = df_normalized[col].apply(
                lambda x: str(x).strip() if pd.notnull(
                    x) and str(x).strip() != '' else np.nan
            )

    return df_normalized


def standardize_nulls(df):
    df_clean = df.copy()

    # Replace various null representations with pandas NA
    null_values = [None, 'None', 'NULL', 'null', '', 'NaN', 'nan', 'NaT']

    for col in df_clean.columns:
        # Replace string representations of null
        df_clean[col] = df_clean[col].replace(null_values, np.nan)

        # Handle numeric columns that might have string nulls
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(
                lambda x: np.nan if (pd.isna(x) or str(x).strip() in [
                                     '', 'None', 'NULL', 'null']) else x
            )

    return df_clean


def compare_data_improved(df_old, df_new, key_columns, engine, table_name):
    old_normalized = df_old.copy()
    new_normalized = df_new.copy()

    # Normalize both dataframes
    old_normalized = normalize_dataframe(old_normalized, engine, table_name)
    new_normalized = normalize_dataframe(new_normalized, engine, table_name)

    # Standardize nulls
    old_normalized = standardize_nulls(old_normalized)
    new_normalized = standardize_nulls(new_normalized)

    # Ensure key columns exist in both dataframes
    missing_keys_old = [
        key for key in key_columns if key not in old_normalized.columns]
    missing_keys_new = [
        key for key in key_columns if key not in new_normalized.columns]

    if missing_keys_old:
        print(f"Warning: Key columns missing in old table: {missing_keys_old}")
    if missing_keys_new:
        print(f"Warning: Key columns missing in new table: {missing_keys_new}")

    # Use only available key columns
    available_keys = [
        key for key in key_columns if key in old_normalized.columns and key in new_normalized.columns]

    if not available_keys:
        raise ValueError("No valid key columns found for comparison")

    # Create composite keys for comparison
    old_normalized['__composite_key__'] = old_normalized[available_keys].apply(
        lambda x: '|||'.join([str(val) if pd.notnull(val) else '__NULL__' for val in x]), axis=1
    )
    new_normalized['__composite_key__'] = new_normalized[available_keys].apply(
        lambda x: '|||'.join([str(val) if pd.notnull(val) else '__NULL__' for val in x]), axis=1
    )

    # Set composite key as index
    old_indexed = old_normalized.set_index('__composite_key__')
    new_indexed = new_normalized.set_index('__composite_key__')

    # Find new entries (in new but not in old)
    new_entries = new_indexed[~new_indexed.index.isin(
        old_indexed.index)].copy()

    # Find deleted entries (in old but not in new)
    deleted_entries = old_indexed[~old_indexed.index.isin(
        new_indexed.index)].copy()

    # Find potentially updated entries (common keys)
    common_keys = new_indexed.index.intersection(old_indexed.index)

    if len(common_keys) == 0:
        updated_entries = pd.DataFrame()
    else:
        # Compare rows with same keys
        old_common = old_indexed.loc[common_keys]
        new_common = new_indexed.loc[common_keys]

        # Align columns
        common_columns = old_common.columns.intersection(new_common.columns)
        old_common = old_common[common_columns]
        new_common = new_common[common_columns]

        # Find rows where any column has changed
        # Use a more robust comparison method
        updated_mask = pd.Series(False, index=common_keys)

        for col in common_columns:
            # Compare column by column, handling nulls properly
            old_vals = old_common[col]
            new_vals = new_common[col]

            # Both null - no change
            both_null = pd.isna(old_vals) & pd.isna(new_vals)

            # One null, other not - change detected
            null_diff = pd.isna(old_vals) != pd.isna(new_vals)

            # Both not null - check if values different
            both_not_null = pd.notnull(old_vals) & pd.notnull(new_vals)
            value_diff = both_not_null & (old_vals != new_vals)

            # Update mask where changes detected
            col_changed = null_diff | value_diff
            updated_mask = updated_mask | col_changed

        updated_entries = new_common[updated_mask].copy()

    # Reset index to get key columns back
    new_entries = new_entries.reset_index(drop=True)
    updated_entries = updated_entries.reset_index(drop=True)
    deleted_entries = deleted_entries.reset_index(drop=True)

    return new_entries, updated_entries, deleted_entries


def prepare_for_oracle(df, engine, table_name, schema="WSUP"):
    if df.empty:
        return df

    df_prep = df.copy()

    # Get database column information for proper type conversion
    inspector = inspect(engine)
    columns_info = inspector.get_columns(table_name, schema=schema)

    col_types = {
        col['name'].lower(): col['type']
        for col in columns_info
    }

    for col in df_prep.columns:
        col_lower = col.lower()

        if col_lower not in col_types:
            continue

        sql_type = col_types[col_lower]

        # Handle datetime fields
        if isinstance(sql_type, (Date, DateTime)):
            df_prep[col] = pd.to_datetime(df_prep[col], errors='coerce')

        # Handle numeric fields
        elif isinstance(sql_type, (Numeric, Integer, Float)):
            df_prep[col] = pd.to_numeric(df_prep[col], errors='coerce')
            # Convert numpy types to native Python types for Oracle
            df_prep[col] = df_prep[col].apply(
                lambda x: None if pd.isna(x) else (
                    int(x) if isinstance(
                        sql_type, Integer) and x == int(x) else float(x)
                )
            )

        # Handle string fields
        else:
            df_prep[col] = df_prep[col].apply(
                lambda x: None if pd.isna(x) else str(x)
            )

    return df_prep


def merge_entries_improved(df, operation, engine, table_name, key_columns=None, schema="WSUP"):
    if df.empty:
        return

    try:
        with engine.begin() as conn:
            for _, row in df.iterrows():
                data = {}
                for col in df.columns:
                    val = row[col]

                    # Handle different null representations
                    if pd.isna(val) or val is None:
                        data[col] = None
                    elif isinstance(val, (np.integer, np.floating)):
                        if np.isnan(val):
                            data[col] = None
                        else:
                            # Convert numpy type to Python type
                            data[col] = val.item()
                    elif isinstance(val, str) and val.strip() in ['', 'NULL', 'None', 'null']:
                        data[col] = None
                    else:
                        data[col] = val

                if operation == "insert":
                    # Exclude identity columns
                    # Add other identity columns as needed
                    identity_columns = ["record_id"]
                    cols = [col for col in df.columns if col.lower(
                    ) not in [ic.lower() for ic in identity_columns]]

                    insert_data = {col: data[col]
                                   for col in cols if col in data}

                    placeholders = ', '.join([f':{col}' for col in cols])
                    columns_str = ', '.join(cols)

                    sql = f"INSERT INTO {schema}.{
                        table_name} ({columns_str}) VALUES ({placeholders})"
                    conn.execute(text(sql), insert_data)

                elif operation == "update":
                    if not key_columns:
                        raise ValueError(
                            "Key columns required for update operation")

                    non_keys = [
                        col for col in df.columns if col not in key_columns]

                    set_clause = ", ".join(
                        [f"{col} = :{col}" for col in non_keys])
                    where_clause = " AND ".join(
                        [f"{col} = :{col}" for col in key_columns])

                    sql = f"UPDATE {schema}.{table_name} SET {
                        set_clause} WHERE {where_clause}"
                    conn.execute(text(sql), data)

                elif operation == "delete":
                    if not key_columns:
                        raise ValueError(
                            "Key columns required for delete operation")

                    where_clause = " AND ".join(
                        [f"{col} = :{col}" for col in key_columns])
                    key_data = {col: data[col] for col in key_columns}

                    sql = f"DELETE FROM {schema}.{
                        table_name} WHERE {where_clause}"
                    conn.execute(text(sql), key_data)

    except Exception as e:
        raise Exception(f"{operation.title()} operation failed: {str(e)}")


def get_comparison_results(df_old, df_new, key_columns, engine, table_old):
    try:
        new_entries, updated_entries, deleted_entries = compare_data_improved(
            df_old, df_new, key_columns, engine, table_old
        )

        # Prepare dataframes for Oracle
        new_entries = prepare_for_oracle(new_entries, engine, table_old)
        updated_entries = prepare_for_oracle(
            updated_entries, engine, table_old)
        deleted_entries = prepare_for_oracle(
            deleted_entries, engine, table_old)

        return new_entries, updated_entries, deleted_entries

    except Exception as e:
        print(f"Error in comparison: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# def compare_data(df_old, df_new, key_columns):
#     df_old.set_index(key_columns, inplace=True)
#     df_new.set_index(key_columns, inplace=True)
#     # Fill NA's to properly compare data
#     df_old.fillna("<<NULL>>", inplace=True)
#     df_new.fillna("<<NULL>>", inplace=True)

#     new = df_new[~df_new.index.isin(df_old.index)]
#     common = df_new.index.intersection(df_old.index)
#     updated_mask = (df_new.loc[common] != df_old.loc[common]).any(axis=1)
#     updated = df_new.loc[common][updated_mask]
#     deleted = df_old.loc[df_old.index.difference(df_new.index)]

#     new.replace({"<<NULL>>": None}, inplace=True)
#     updated.replace({"<<NULL>>": None}, inplace=True)
#     deleted.replace({"<<NULL>>": None}, inplace=True)

#     return new.reset_index(), \
#         updated.reset_index(), \
#         deleted.reset_index()


# def prep_tables(df, engine, table_name, schema="WSUP"):
#     inspector = inspect(engine)
#     columns_info = inspector.get_columns(table_name, schema=schema)

#     # dictionary of {column_name: column_type}
#     col_types = {
#         col['name'].lower(): col['type']
#         for col in columns_info
#     }

#     for col in df.columns:
#         col_lc = col.lower()

#         if col_lc not in col_types:
#             continue  # skip if not found in DB table

#         sql_type = col_types[col_lc]

#         # handle datetime fields
#         if isinstance(sql_type, (Date, DateTime)):
#             df[col] = pd.to_datetime(df[col], errors='coerce')

#         # convert non-numeric fields to string
#         elif not isinstance(sql_type, (Numeric, Integer, Float)):
#             df[col] = df[col].apply(
#                 lambda x: str(x) if pd.notnull(x) else None)

#         elif isinstance(sql_type, (Numeric, Integer, Float)):
#             # Convert numpy numeric types to native Python types
#             df[col] = df[col].apply(
#                 lambda x: float(x) if pd.notnull(x) else None)

#     return df


new_entries, updated_entries, deleted_entries = get_comparison_results(
    df_old, df_new, key_columns, engine, table_old
)

# new_entries, updated_entries, deleted_entries = compare_data(
#     df_old, df_new, key_columns)

# new_entries = prep_tables(new_entries, engine, table_new)
# updated_entries = prep_tables(updated_entries, engine, table_new)
# deleted_entries = prep_tables(deleted_entries, engine, table_new)


# --- Streamlit UI ---
st.markdown(f"### Merging from `{table_new}` → `{table_old}`")

st.info(f"""
**Comparison Summary:**
- Records in staging table: {len(df_new)}
- Records in production table: {len(df_old)}
- Key columns used: {', '.join(key_columns)}
""")

st.subheader(f"New Entries ({len(new_entries)})")
if not new_entries.empty:
    new_to_merge = st.data_editor(
        new_entries,
        use_container_width=True,
        num_rows="dynamic",
        key=f"new_editor_{table_old}"
    )
else:
    st.write("No new entries found.")
    new_to_merge = pd.DataFrame()

st.subheader(f"Updated Entries ({len(updated_entries)})")
if not updated_entries.empty:
    updated_to_merge = st.data_editor(
        updated_entries,
        use_container_width=True,
        num_rows="dynamic",
        key=f"update_editor_{table_old}"
    )
else:
    st.write("No updated entries found.")
    updated_to_merge = pd.DataFrame()

st.subheader(f"Deleted Entries ({len(deleted_entries)})")
if not deleted_entries.empty:
    deleted_to_merge = st.data_editor(
        deleted_entries,
        use_container_width=True,
        num_rows="dynamic",
        key=f"delete_editor_{table_old}"
    )
else:
    st.write("No deleted entries found.")
    deleted_to_merge = pd.DataFrame()

# --- Action Buttons---
col1, col2, col3 = st.columns(3)

with col1:
    if st.button(f"Insert New Entries ({len(new_to_merge) if not new_to_merge.empty else 0})"):
        if not new_to_merge.empty:
            try:
                merge_entries_improved(pd.DataFrame(
                    new_to_merge), "insert", engine, table_old, key_columns)
                st.success(f"Successfully inserted {
                           len(new_to_merge)} new entries.")
            except Exception as e:
                st.error(f"Insert failed: {str(e)}")
        else:
            st.warning("No new entries to insert.")

with col2:
    if st.button(f"Update Entries ({len(updated_to_merge) if not updated_to_merge.empty else 0})"):
        if not updated_to_merge.empty:
            try:
                merge_entries_improved(pd.DataFrame(
                    updated_to_merge), "update", engine, table_old, key_columns)
                st.success(f"Successfully updated {
                           len(updated_to_merge)} entries.")
            except Exception as e:
                st.error(f"Update failed: {str(e)}")
        else:
            st.warning("No entries to update.")

with col3:
    if st.button(f"Delete Entries ({len(deleted_to_merge) if not deleted_to_merge.empty else 0})"):
        if not deleted_to_merge.empty:
            try:
                merge_entries_improved(pd.DataFrame(
                    deleted_to_merge), "delete", engine, table_old, key_columns)
                st.success(f"Successfully deleted {
                           len(deleted_to_merge)} entries.")
            except Exception as e:
                st.error(f"Delete failed: {str(e)}")
        else:
            st.warning("No entries to delete.")

st.markdown("---")
if st.button("Process All Changes", type="primary"):
    total_changes = 0

    try:
        # Insert new entries
        if not new_to_merge.empty:
            merge_entries_improved(pd.DataFrame(
                new_to_merge), "insert", engine, table_old, key_columns)
            total_changes += len(new_to_merge)
            st.success(f"Inserted {len(new_to_merge)} new entries")

        # Update existing entries
        if not updated_to_merge.empty:
            merge_entries_improved(pd.DataFrame(
                updated_to_merge), "update", engine, table_old, key_columns)
            total_changes += len(updated_to_merge)
            st.success(f"Updated {len(updated_to_merge)} entries")

        # Delete removed entries
        if not deleted_to_merge.empty:
            merge_entries_improved(pd.DataFrame(
                deleted_to_merge), "delete", engine, table_old, key_columns)
            total_changes += len(deleted_to_merge)
            st.success(f"Deleted {len(deleted_to_merge)} entries")

        if total_changes > 0:
            st.success(f"🎉 All operations completed successfully! Total changes: {
                       total_changes}")
        else:
            st.info("No changes to process.")

    except Exception as e:
        st.error(f"Batch operation failed: {str(e)}")

# --- Database Merge Functions ---


# def merge_entries(df, operation, engine, table_name, key_columns=None, schema="WSUP"):
#     if df.empty:
#         return

#     try:
#         with engine.begin() as conn:
#             for _, row in df.iterrows():
#                 data = {
#                     col: (
#                         None if pd.isna(row[col]) or row[col] is None else
#                         float(row[col]) if isinstance(row[col], (np.float64, np.float32)) else
#                         int(row[col]) if isinstance(row[col], (np.int64, np.int32)) else
#                         str(row[col]) if isinstance(row[col], str) else
#                         row[col]
#                     )
#                     for col in df.columns
#                 }

#                 if operation == "insert":
#                     identity_column = "record_id"
#                     cols = [col for col in df.columns if col != identity_column]
#                     data = {col: None if pd.isna(
#                         row[col]) else row[col] for col in cols}

#                     sql = f"""INSERT INTO {schema}.{table_name} ({', '.join(cols)})
#                               VALUES ({', '.join([f':{col}' for col in cols])})"""
#                     conn.execute(text(sql), data)

#                 elif operation == "update":
#                     non_keys = [
#                         col for col in df.columns if col not in key_columns]
#                     set_clause = ", ".join(
#                         [f"{col} = :{col}" for col in non_keys])
#                     where_clause = " AND ".join(
#                         [f"{col} = :{col}" for col in key_columns])
#                     sql = f"""UPDATE {schema}.{table_name} SET {
#                         set_clause} WHERE {where_clause}"""
#                     conn.execute(text(sql), data)

#                 elif operation == "delete":
#                     where_clause = " AND ".join(
#                         [f"{col} = :{col}" for col in key_columns])
#                     key_data = {col: data[col] for col in key_columns}
#                     sql = f"""DELETE FROM {table_name} WHERE {where_clause}"""
#                     conn.execute(text(sql), key_data)

#     except Exception as e:
#         st.error(
#             f"{operation.title()} failed: transaction rolled back.\nError: {e}")
