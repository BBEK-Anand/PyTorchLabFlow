"""
create or use uor lab
"""
from pathlib import Path
import os
import json
from datetime import datetime
from typing import Optional
import pandas as pd

from .context import set_shared_data, get_caller, register_libs_path, get_shared_data
from .utils import Db

__all__ = ["lab_setup", "create_project"]


def lab_setup(settings_path: Optional[str]) -> None:
    if settings_path and os.path.exists(settings_path):
        with open(settings_path, encoding="utf-8") as sp:
            settings = json.load(sp)
    else:
        raise ValueError("Provide either settings_path or settings for lab setup")


    caller = get_caller()

    log_path = os.path.join(settings["data_path"], "logs.db")
    db = Db(db_path=log_path)

    # Get current number of logs
    cursor = db.execute("SELECT COUNT(*) FROM logs")
    row_count = cursor.fetchone()[0]
    logid = f"log{row_count}"
    # Insert new log
    db.execute(
        "INSERT INTO logs (logid, called_at) VALUES (?, ?)",
        (logid,  caller)
    )


    set_shared_data(settings, logid)
    register_libs_path(settings["component_dir"])
    
def export_settigns():
    settings = get_shared_data()
    # Dump settings JSON
    pth = os.path.join(Path(settings['project_path']).parent, settings["project_name"]+".json")
    with open(pth, "w", encoding="utf-8") as out_file:
        json.dump(settings, out_file, indent=4)
    return pth

def create_project(settings: dict) -> str:


    settings['data_path'] = settings['project_dir']+"/"+settings['project_name']
    settings["setting_path"] = settings["data_path"]+"/"+settings['project_name']+".json"

    for key in ["data_path", "setting_path","component_dir"]:
        if not os.path.isabs(settings[key]):
            settings[key] = os.path.abspath(settings[key])

    for key in ["data_path", "setting_path", "component_dir"]:
        path = settings[key]
        # Infer it's a file if it has an extension
        dir_to_create = os.path.dirname(path) if os.path.splitext(path)[1] else path
        os.makedirs(dir_to_create, exist_ok=True)


    # Dump settings JSON
    with open(settings["setting_path"], "w", encoding="utf-8") as out_file:
        json.dump(settings, out_file, indent=4)

    # Create subdirectories under data_path
    os.makedirs(os.path.join(settings["data_path"], "Configs"), exist_ok=True)
    
    # Archived and Transfer folders
    for parent_dir in ["Archived", "Transfer"]:
        os.makedirs(
                os.path.join(settings["data_path"], parent_dir, "Configs"), exist_ok=True
            )
    
    db_path = f"{settings['data_path']}/logs.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    db_path = f"{settings['data_path']}/ppls.db"
    if os.path.exists(db_path):
        os.remove(db_path)



    setup_databases(settings)
    set_shared_data(settings)
    return export_settigns()

def setup_databases(settings: dict):
    """
    tytjf
    """
    # settings = get_shared_data()
    db_path = f"{settings['data_path']}/logs.db"

    # Connect to the database (creates the file if it doesn't exist)
    db = Db(db_path=db_path)
    table = """
        CREATE TABLE IF NOT EXISTS logs (
        logid TEXT PRIMARY KEY,
        called_at TEXT NOT NULL,
        created_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """

    # Create a table if it doesn't already exist
    db.execute(table)

    db.execute("INSERT INTO logs (logid, called_at) VALUES (?, ?)",
                ('log0', get_caller())
            )
    db.close()


    db_path = f"{settings['data_path']}/ppls.db"

    # Connect to the database (creates the file if it doesn't exist)
    db = Db(db_path=db_path)
    table = """
        CREATE TABLE IF NOT EXISTS ppls (
        pplid TEXT PRIMARY KEY,
        args_hash TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'init' CHECK(status IN ('init', 'running', 'frozen', 'cleaned')),
        created_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """

    # Create a table if it doesn't already exist
    db.execute(table)

    table = """
        CREATE TABLE IF NOT EXISTS edges (
            edgid INTEGER PRIMARY KEY AUTOINCREMENT,
            prev TEXT NOT NULL,
            next TEXT NOT NULL,
            desc TEXT,
            directed BOOL DEFAULT TRUE,
            FOREIGN KEY(prev) REFERENCES ppls(pplid),
            FOREIGN KEY(next) REFERENCES ppls(pplid)
        )
        """
    db.execute(table)
    runnings = """
        CREATE TABLE IF NOT EXISTS runnings (
            runid INTEGER PRIMARY KEY AUTOINCREMENT,
            pplid NOT NULL,
            logid TEXT DEFAULT NULL,
            parity TEXT DEFAULT NULL,
            started_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(pplid) REFERENCES ppls(pplid)
        )
        """
    db.execute(runnings)
    db.close()
    db_path = f"{settings['data_path']}/Archived/ppls.db"

    # Connect to the database (creates the file if it doesn't exist)
    db = Db(db_path=db_path)
    table = """
        CREATE TABLE IF NOT EXISTS ppls (
        pplid TEXT PRIMARY KEY,
        args_hash TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'init' CHECK(status IN ('init', 'running', 'frozen', 'cleaned')),
        created_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """

    # Create a table if it doesn't already exist
    db.execute(table)
    db.close()

def transfer_lab(settings, transfer_type: str = "export"):
    pass