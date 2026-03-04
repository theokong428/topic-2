"""
Data loading and preprocessing module for university course timetabling.
Loads all Excel data files and provides clean DataFrames and utility lookups.

Data files:
  - Rooms and Room Types.xlsx   -> room inventory (649 rooms, campus/capacity/type)
  - 2024-5 Event Module Room.xlsx -> event-module-room (~32,757 rows)
  - 2024-5 Student Programme Module Event.xlsx -> student enrollment (~930,174 rows)
  - Programme-Course.xlsx       -> programme-course mapping
  - 2024-5 DPT Data.xlsx        -> degree programme structure
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Data directory (note spelling: "timtabling", not "timetabling")
DATA_DIR = Path(__file__).resolve().parent.parent / "course_timtabling"


# Room Data
def load_rooms(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Load room inventory with campus, capacity, type, and allocation info.
    Key columns: Id, Capacity, Campus, Specialist room type, Central/Local.
    """
    fp = data_dir / "Rooms and Room Types.xlsx"
    df = pd.read_excel(fp, sheet_name="Room", engine="openpyxl")
    # Original data has two Building columns: code and name, rename for clarity
    df = df.rename(columns={"Building.1": "BuildingName"})
    df["Capacity"] = df["Capacity"].astype(int)
    return df


def get_gt_rooms(rooms: pd.DataFrame, campus: Optional[str] = None) -> pd.DataFrame:
    """
    Filter for General Teaching (GT) rooms, optionally by campus.
    NOTE: Use 'Specialist room type' (not 'Room Type'). Room Type = physical layout;
    Specialist room type = functional (General Teaching, Laboratory, etc.).
    """
    mask = rooms["Specialist room type"] == "General Teaching"
    if campus:
        mask &= rooms["Campus"] == campus
    return rooms[mask].copy()


def get_central_gt_rooms(rooms: pd.DataFrame) -> pd.DataFrame:
    """Get Central campus GT rooms (232 rooms, 10,091 capacity)."""
    return get_gt_rooms(rooms, campus="Central")


def get_holyrood_gt_rooms(rooms: pd.DataFrame) -> pd.DataFrame:
    """Get Holyrood campus GT rooms (69 rooms, 2,241 capacity)."""
    return get_gt_rooms(rooms, campus="Holyrood")


def get_lauriston_gt_rooms(rooms: pd.DataFrame) -> pd.DataFrame:
    """Get Lauriston campus GT rooms (6 rooms, 399 capacity)."""
    return get_gt_rooms(rooms, campus="Lauriston")


def get_newcollege_gt_rooms(rooms: pd.DataFrame) -> pd.DataFrame:
    """Get New College campus GT rooms (8 rooms, 622 capacity)."""
    return get_gt_rooms(rooms, campus="New College")


# Travel Time Matrix
def load_travel_times(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Load inter-campus travel time matrix from the 'Room Constraints' sheet.
    Returns DataFrame: CampusFrom, CampusTo, TravelMins.
    Key times: same campus 0 min; Central↔Holyrood/Lauriston/NC 10 min;
    Central↔Kings Buildings 30 min; most others 60 min.
    """
    fp = data_dir / "Rooms and Room Types.xlsx"
    df = pd.read_excel(fp, sheet_name="Room Constraints", engine="openpyxl")
    # Travel matrix is in specific columns
    travel_cols = ["Campus From", "Campus To", "Travel time (mins)"]
    travel = df[travel_cols].dropna(subset=["Campus From", "Campus To"]).copy()
    travel.columns = ["CampusFrom", "CampusTo", "TravelMins"]
    travel["TravelMins"] = travel["TravelMins"].astype(int)
    return travel


def build_travel_dict(travel_df: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    """
    Convert travel DataFrame to lookup dict: (from, to) -> minutes.
    """
    return {
        (row.CampusFrom, row.CampusTo): row.TravelMins
        for row in travel_df.itertuples()
    }


# Teaching Hours Policy (Current 2024/25)
# Format: {Day: (start_hour, end_hour)}
CURRENT_TEACHING_HOURS = {
    "Monday":    (9, 18),
    "Tuesday":   (9, 18),
    "Wednesday": (9, 18),   # Wed 9-13 whole-class, 13-18 group only
    "Thursday":  (9, 18),
    "Friday":    (9, 18),
    "Saturday":  (9, 18),   # Some events exist on Saturday
}

# Wednesday: whole-class events must end by 1pm
WEDNESDAY_WHOLE_CLASS_END = 13

# Q2 Scenario A: Mon-Fri 9am-5pm (1 hour shorter each day)
SCENARIO_A_HOURS = {
    "Monday":    (9, 17),
    "Tuesday":   (9, 17),
    "Wednesday": (9, 17),
    "Thursday":  (9, 17),
    "Friday":    (9, 17),
}

# Q2 Scenario B: Eliminate Fri 12pm-6pm, Mon-Thu stays the same
SCENARIO_B_HOURS = {
    "Monday":    (9, 18),
    "Tuesday":   (9, 18),
    "Wednesday": (9, 18),
    "Thursday":  (9, 18),
    "Friday":    (9, 12),   # Friday only 9am-12pm
}


# Event Data

def load_events(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Load event-module-room data with parsed timeslot info.
    Each row is an "event" (a scheduled teaching session).
    Derived columns: Day, Hour, Minute, WeekList, EndHour.
    """
    fp = data_dir / "2024-5 Event Module Room.xlsx"
    df = pd.read_excel(fp, sheet_name="2024-5 Event Module Room", engine="openpyxl")

    # Parse timeslot string into Day, Hour, Minute
    parsed = df["Timeslot"].apply(parse_timeslot)
    df["Day"] = parsed.apply(lambda x: x[0] if x else None)
    df["Hour"] = parsed.apply(lambda x: x[1] if x else None)
    df["Minute"] = parsed.apply(lambda x: x[2] if x else None)

    # Parse week numbers string into list
    df["WeekList"] = df["Weeks"].apply(parse_weeks)

    # Ensure numeric types
    df["Duration (minutes)"] = pd.to_numeric(df["Duration (minutes)"], errors="coerce")
    df["Event Size"] = pd.to_numeric(df["Event Size"], errors="coerce").fillna(0).astype(int)

    # Compute end time in fractional hours
    df["EndHour"] = df["Hour"] + df["Minute"] / 60 + df["Duration (minutes)"] / 60

    return df


def load_week_numbers(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Load week number to date mapping (1-52 weeks -> dates and semester labels).
    """
    fp = data_dir / "2024-5 Event Module Room.xlsx"
    return pd.read_excel(fp, sheet_name="Week Numbers", engine="openpyxl")


def get_holyrood_gt_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    Get events currently assigned to Holyrood General Teaching rooms.
    NOTE: Events use 'Room type 2' column (matches 'Specialist room type' in rooms).
    """
    mask = (events["Campus"] == "Holyrood") & (events["Room type 2"] == "General Teaching")
    return events[mask].copy()


def parse_timeslot(ts: str) -> Optional[Tuple[str, int, int]]:
    """
    Parse timeslot string into structured tuple. Example: 'Monday 09:00' -> ('Monday', 9, 0)
    """
    if pd.isna(ts) or not isinstance(ts, str):
        return None
    parts = ts.strip().split()
    if len(parts) != 2:
        return None
    day = parts[0]
    time_parts = parts[1].split(":")
    if len(time_parts) != 2:
        return None
    return (day, int(time_parts[0]), int(time_parts[1]))


def parse_weeks(weeks_str) -> List[int]:
    """
    Parse weeks string into list of integers. Example: '10, 11, 12, 13, 15' -> [10, 11, 12, 13, 15]
    """
    if pd.isna(weeks_str):
        return []
    if isinstance(weeks_str, (int, float)):
        return [int(weeks_str)]
    return [int(w.strip()) for w in str(weeks_str).split(",") if w.strip().isdigit()]


# Student Enrolment Data
def load_students(data_dir: Path = DATA_DIR, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load student-programme-module-event data (~930k rows, ~25k students).
    Use nrows to limit for prototyping. Key columns: AnonID, Programme, Event ID, Semester.
    """
    fp = data_dir / "2024-5 Student Programme Module Event.xlsx"
    df = pd.read_excel(
        fp,
        sheet_name="2024-5 Student Programme Module",
        engine="openpyxl",
        nrows=nrows,
    )
    return df


def build_student_event_map(students: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build dict: student_id -> list of event_ids.
    """
    return students.groupby("AnonID")["Event ID"].apply(list).to_dict()


def build_event_student_map(students: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build dict: event_id -> list of student_ids.
    """
    return students.groupby("Event ID")["AnonID"].apply(list).to_dict()


# Programme-Course Data
def load_programme_course(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Load programme-course relationships. Columns: CourseId, ModuleId, Compulsory.
    """
    fp = data_dir / "Programme-Course.xlsx"
    return pd.read_excel(fp, sheet_name="CourseModule", engine="openpyxl")


# DPT Data
def load_dpt(data_dir: Path = DATA_DIR, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load Degree Programme Table (DPT) data (~950k rows).
    """
    fp = data_dir / "2024-5 DPT Data.xlsx"
    return pd.read_excel(fp, sheet_name="Sheet1", engine="openpyxl", nrows=nrows)


# Timeslot Utilities

ALL_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

def generate_timeslots(hours_policy: Dict[str, Tuple[int, int]],
                       granularity_min: int = 30) -> List[Tuple[str, int, int]]:
    """
    Generate all valid (day, hour, minute) timeslots under a given hours policy.
    Parameters: hours_policy {Day: (start_h, end_h)}, granularity_min (default 30).
    Example: generate_timeslots({"Monday": (9, 11)}, 30) -> [("Monday", 9, 0), ...]
    """
    slots = []
    for day, (start_h, end_h) in hours_policy.items():
        h, m = start_h, 0
        while h + m / 60 < end_h:
            slots.append((day, h, m))
            m += granularity_min
            if m >= 60:
                h += 1
                m = 0
    return slots


def events_overlap(day1, hour1, min1, dur1, weeks1,
                   day2, hour2, min2, dur2, weeks2) -> bool:
    """
    Check if two events overlap in time (same day, overlapping hours, shared weeks).
    Overlap: start1 < end2 AND start2 < end1.
    """
    if day1 != day2:
        return False
    # Check week overlap
    if not set(weeks1) & set(weeks2):
        return False
    # Check time overlap
    start1 = hour1 + min1 / 60
    end1 = start1 + dur1 / 60
    start2 = hour2 + min2 / 60
    end2 = start2 + dur2 / 60
    return start1 < end2 and start2 < end1


# Convenience: Load Everything
def load_all(data_dir: Path = DATA_DIR,
             student_nrows: Optional[int] = None) -> dict:
    """
    Load all datasets into a single dict. Returns: rooms, events, travel,
    travel_dict, students, weeks.
    """
    print("Loading rooms...")
    rooms = load_rooms(data_dir)
    print("Loading events...")
    events = load_events(data_dir)
    print("Loading travel times...")
    travel = load_travel_times(data_dir)
    print("Loading students...")
    students = load_students(data_dir, nrows=student_nrows)
    print("Loading week numbers...")
    weeks = load_week_numbers(data_dir)
    print("All data loaded.")

    return {
        "rooms": rooms,
        "events": events,
        "travel": travel,
        "travel_dict": build_travel_dict(travel),
        "students": students,
        "weeks": weeks,
    }
