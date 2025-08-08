# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 09:18:19 2025

@author: tchen

A Python script to create a GUI which allows
users to insert, update, or delete entries
in the GWIS database.

"""

import pandas as pd
import itertools
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox, scrolledtext
from sqlalchemy import create_engine, text, inspect
from tkinter import ttk
import pandas as pd
import datetime
from decimal import Decimal
import getpass

# ==== DATABASE CONFIG ====
username = "tchen"
password = password = getpass.getpass("Enter your password: ")
tns_name = "WRED"

# TNS connection string
engine = create_engine(f"oracle+cx_oracle://{username}:{password}@{tns_name}")

# ==== GUI ROOT ====
root = tk.Tk()
root.title("Oracle Table GUI")
root.geometry("900x900")

notebook = ttk.Notebook(root)
notebook.pack(expand=1, fill="both")

# ==== TABS ====
entry_tab = ttk.Frame(notebook)
view_tab = ttk.Frame(notebook)

top_frame = tk.Frame(entry_tab)
top_frame.pack(pady=5, fill="x")

form_frame = tk.Frame(entry_tab)
form_frame.pack(pady=10, fill="x")

button_frame = tk.Frame(entry_tab)
button_frame.pack(pady=10)

notebook.add(entry_tab, text="Edit Records")
notebook.add(view_tab, text="Browse Records")

# ==== TABLE DROPDOWN ====
tables = [
    "GWIS_WATER_USE",
    "EXPORT_TABLE",
    "GWIS_WU_MONITORING"
    "UPD_GWIS_APP_COUNTIES",
    "UPD_GWIS_WU_APP_TYPE",
    "UPD_GWIS_WU_FACILITY_GROUP",
    "UPD_GWIS_TL_SOURCES",
    "UPD_GWIS_TL_WSP_REGIONS",
    "UPD_GWIS_WU_APP_FAC_GROUP",
    "UPD_GWIS_WU_APP_LU",
    "UPD_GWIS_WU_ALLOCATION",
    "UPD_GWIS_TL_COUNTIES",
    "UPD_GWIS_WU_MONITORING",
    "UPD_GWIS_PERMIT_INFO",
    "UPD_GWIS_TL_LAND_USE",
    "UPD_GWIS_WU_FACILITIES",
    "UPD_GWIS_DATA_INDEX"   # add more tables
]

pks = {"UPD_GWIS_WATER_USE": ["record_id"],
       "UPD_GWIS_APP_COUNTIES": ["app_no"],
       "UPD_GWIS_WU_APP_TYPE": ["app_no"],
       "UPD_GWIS_WU_FACILITY_GROUP": ["record_id"],
       "UPD_GWIS_TL_SOURCES": ["source_id"],
       "UPD_GWIS_TL_WSP_REGIONS": ["app_no"],
       "UPD_GWIS_WU_APP_FAC_GROUP": ["app_fac_id", "facgrp_id"],
       "UPD_GWIS_WU_APP_LU": ["app_no", "landuse_code"],
       "UPD_GWIS_WU_ALLOCATION": ["app_no"],
       "UPD_GWIS_TL_COUNTIES": ["county_code"],
       "UPD_GWIS_WU_MONITORING": ["record_id"],
       "UPD_GWIS_PERMIT_INFO": ["app_no"],
       "UPD_GWIS_TL_LAND_USE": ["landuse_code"],
       "UPD_GWIS_WU_FACILITIES": ["facinv_id"],
       "UPD_GWIS_DATA_INDEX": ["gwtid"]}

selected_table = tk.StringVar(value=tables[0])
selected_table.trace_add("write", lambda *args: on_table_change())
tk.Label(top_frame, text="Select Table:").pack(side="left", padx=5)
table_menu = tk.OptionMenu(top_frame, selected_table, *tables)
table_menu.pack(side="left")


# ==== FORM FIELDS ====

def build_entry_form(table_name):
    global entry_vars
    for widget in form_frame.winfo_children():
        widget.destroy()

    columns = get_column_names(table_name)
    entry_vars = {col: tk.StringVar() for col in columns}

    for idx, col in enumerate(columns):
        tk.Label(form_frame, text=col, anchor="w", width=20).grid(
            row=idx, column=0, sticky="w")
        tk.Entry(form_frame, textvariable=entry_vars[col], width=40).grid(
            row=idx, column=1, pady=2)


# ==== ACTION FUNCTIONS ====


def insert_entry():
    data = {col: var.get() for col, var in entry_vars.items()}
    col_names = ", ".join(data.keys())
    placeholders = ", ".join([f":{col}" for col in data])
    query = f"INSERT INTO {selected_table.get()} ({col_names}) VALUES ({
        placeholders})"
    try:
        with engine.begin() as conn:
            conn.execute(text(query), data)
        messagebox.showinfo("Success", "Entry inserted.")
        load_table_data()
    except Exception as e:
        messagebox.showerror("Insert Error", str(e))


def update_entry():
    table = selected_table.get()
    print(table)
    data = {col: var.get() for col, var in entry_vars.items()}
    print("data", data)
    pk_fields = pks.get(table, [])
    print(pk_fields)

    missing_keys = [pk for pk in pk_fields if not data.get(pk)]
    print(missing_keys)
    if missing_keys:
        messagebox.showwarning("Missing Key", f"Missing primary key(s): {
                               ', '.join(missing_keys)}")
        return

    # Build SET clause without PKs
    set_data = {k: v for k, v in data.items() if k not in pk_fields}
    set_clause = ", ".join([f"{col} = :{col}" for col in set_data])

    # Build WHERE clause with PKs
    where_clause = " AND ".join([f"{pk} = :{pk}" for pk in pk_fields])
    query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"

    # Merge data for binding
    params = {**set_data, **{pk: data[pk] for pk in pk_fields}}

    try:
        with engine.begin() as conn:
            conn.execute(text(query), params)
        messagebox.showinfo("Success", "Entry updated.")
        load_table_data()
    except Exception as e:
        messagebox.showerror("Update Error", str(e))


def delete_entry():
    table = selected_table.get()
    pk_fields = pks.get(table, [])
    params = {}

    for pk in pk_fields:
        value = entry_vars.get(pk)
        if not value or not value.get():
            messagebox.showwarning(
                "Missing Key", f"{pk} is required for deletion.")
            return
        params[pk] = value.get()

    where_clause = " AND ".join([f"{pk} = :{pk}" for pk in pk_fields])
    query = f"DELETE FROM {table} WHERE {where_clause}"

    try:
        with engine.begin() as conn:
            conn.execute(text(query), params)
        messagebox.showinfo("Success", "Entry deleted.")
        load_table_data()
    except Exception as e:
        messagebox.showerror("Delete Error", str(e))


# ==== BUTTONS ====
button_frame = tk.Frame(entry_tab)
button_frame.pack(pady=10)

tk.Button(button_frame, text="Insert", command=insert_entry,
          width=15).grid(row=0, column=0, padx=5)
tk.Button(button_frame, text="Update", command=update_entry,
          width=15).grid(row=0, column=1, padx=5)
tk.Button(button_frame, text="Delete", command=delete_entry,
          width=15).grid(row=0, column=2, padx=5)

# ==== TABLE VIEW ====
tree = ttk.Treeview(view_tab, show='headings')
tree.pack(expand=True, fill='both')

scrollbar_y = ttk.Scrollbar(view_tab, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=scrollbar_y.set)
scrollbar_y.pack(side="right", fill="y")

scrollbar_x = ttk.Scrollbar(view_tab, orient="horizontal", command=tree.xview)
tree.configure(xscrollcommand=scrollbar_x.set)
scrollbar_x.pack(side="bottom", fill="x")


def load_table_data():
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT * FROM {selected_table.get()} FETCH FIRST 100 ROWS ONLY"))
            rows = result.fetchall()
            print(rows[0])
            cols = result.keys()
            cols = list(result.keys())
            print(cols)

            tree.delete(*tree.get_children())  # Clear existing
            tree["columns"] = cols

            for col in cols:
                tree.heading(col, text=col)
                tree.column(col, anchor="w", width=120)

            for row in rows:
                display_row = []
                for val in row:
                    if isinstance(val, datetime.datetime):
                        display_row.append(val.strftime('%Y-%m-%d'))
                    elif isinstance(val, Decimal):
                        display_row.append(str(val))
                    elif val is None:
                        # Optional: replace None with blank
                        display_row.append('')
                    else:
                        display_row.append(str(val))
                tree.insert("", "end", values=display_row)

    except Exception as e:
        messagebox.showerror("Load Error", str(e))


def on_tab_change(event):
    if notebook.index(notebook.select()) == 1:  # Browse Records tab
        load_table_data()


notebook.bind("<<NotebookTabChanged>>", on_tab_change)


# ==== HELPER FUNCTIONS ====
def get_column_names(table_name):
    inspector = inspect(engine)
    try:
        return [col['name'] for col in inspector.get_columns(table_name)]
    except Exception as e:
        messagebox.showerror("Error", f"Could not fetch columns: {e}")
        return []


def on_table_change():
    table = selected_table.get()
    build_entry_form(table)
    load_table_data()  # If you want to refresh the table view as well


# ==== MAIN LOOP ====
root.mainloop()
