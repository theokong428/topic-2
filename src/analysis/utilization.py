"""
Utilization metrics for rooms and timeslots.

This module computes key space-utilization metrics following HEFCE
(Higher Education Funding Council for England) standards, widely used
by UK universities.

Key metrics (from HEFCE/industry standards):
  - Frequency Rate = hours_used / hours_available
  - Occupancy Rate = avg(students / capacity) when occupied
  - Utilization Rate = Frequency * Occupancy
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


def frequency_rate(
    room_id: str,
    events: pd.DataFrame,
    available_hours_per_week: float,
) -> float:
    """
    Fraction of available hours the room is in use per week (averaged across weeks).

    The frequency rate measures how often a room is scheduled relative to
    its total available teaching time. A value of 1.0 means the room is
    fully booked every available hour.

    Parameters
    ----------
    room_id : str
        Room identifier (matches events["Room"]).
    events : DataFrame
        Events data with columns: Room, Duration (minutes), WeekList.
    available_hours_per_week : float
        Total available teaching hours per week for this room.
    """
    # Filter events for this specific room
    room_events = events[events["Room"] == room_id]
    if len(room_events) == 0 or available_hours_per_week <= 0:
        return 0.0

    # Total hours used across all weeks
    total_event_hours = 0
    total_week_count = 0
    for _, ev in room_events.iterrows():
        # Number of weeks this event spans
        n_weeks = len(ev["WeekList"])
        # Convert duration from minutes to hours
        dur_hours = ev["Duration (minutes)"] / 60
        # Accumulate: duration per week × number of weeks = total hours
        total_event_hours += dur_hours * n_weeks
        # Track max week number to determine semester span
        total_week_count = max(total_week_count, max(ev["WeekList"]) if ev["WeekList"] else 0)

    if total_week_count == 0:
        return 0.0

    # Formula: average hours per week = total event hours / total week count
    avg_hours_per_week = total_event_hours / total_week_count
    # Cap at 1.0 to prevent exceeding 100%
    return min(avg_hours_per_week / available_hours_per_week, 1.0)


def occupancy_rate(
    room_id: str,
    room_capacity: int,
    events: pd.DataFrame,
) -> float:
    """
    Average (students / capacity) when the room is occupied.

    This metric measures how efficiently space is used when the room is
    actually in use. A low occupancy rate indicates rooms are over-sized
    for the classes they host.

    Parameters
    ----------
    room_id : str
        Room identifier.
    room_capacity : int
        Total seating capacity of the room.
    events : DataFrame
        Events data, must contain "Event Size" column.
    """
    # Filter events for this room
    room_events = events[events["Room"] == room_id]
    if len(room_events) == 0 or room_capacity <= 0:
        return 0.0

    # Occupancy per event = actual students / room capacity, then average
    occupancies = room_events["Event Size"] / room_capacity
    return occupancies.mean()


def utilization_rate(
    room_id: str,
    room_capacity: int,
    events: pd.DataFrame,
    available_hours_per_week: float,
) -> float:
    """
    Combined utilization = frequency * occupancy.

    This is the HEFCE standard composite metric that captures both
    how often a room is used and how full it is when used.
    """
    freq = frequency_rate(room_id, events, available_hours_per_week)
    occ = occupancy_rate(room_id, room_capacity, events)
    return freq * occ


def campus_utilization_summary(
    rooms: pd.DataFrame,
    events: pd.DataFrame,
    campus: Optional[str] = None,
    hours_policy: Optional[Dict[str, Tuple[int, int]]] = None,
) -> pd.DataFrame:
    """
    Compute utilization metrics for all rooms on a campus (or all campuses).

    Iterates over every target room, computes frequency, occupancy, and
    combined utilization, and returns a summary DataFrame.

    Parameters
    ----------
    rooms : DataFrame
        Room data with Id, Capacity, Campus columns.
    events : DataFrame
        Event data.
    campus : str, optional
        Filter to a specific campus.
    hours_policy : dict, optional
        Day -> (start_h, end_h) for computing available hours.
        Defaults to Mon-Fri 9-18, Wed 9-18 (full current policy).
    """
    # If no hours policy specified, use default: Mon-Fri 9am-6pm
    if hours_policy is None:
        hours_policy = {
            "Monday": (9, 18), "Tuesday": (9, 18), "Wednesday": (9, 18),
            "Thursday": (9, 18), "Friday": (9, 18),
        }

    # Total available hours per week = sum of (end - start) for each day
    available_per_week = sum(end - start for start, end in hours_policy.values())

    target_rooms = rooms.copy()
    # Filter by campus if specified
    if campus:
        target_rooms = target_rooms[target_rooms["Campus"] == campus]

    # Compute metrics for each room
    results = []
    for _, rm in target_rooms.iterrows():
        rid = rm["Id"]
        cap = rm["Capacity"]
        freq = frequency_rate(rid, events, available_per_week)
        occ = occupancy_rate(rid, cap, events)
        # Combined utilization = frequency × occupancy
        util = freq * occ
        results.append({
            "Room": rid,
            "Campus": rm["Campus"],
            "Capacity": cap,
            "Room Type": rm["Room Type"],
            "Frequency": freq,
            "Occupancy": occ,
            "Utilization": util,
        })

    return pd.DataFrame(results)


def timeslot_utilization_heatmap(
    events: pd.DataFrame,
    rooms: Optional[pd.DataFrame] = None,
    campus: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute events per (day, hour) grid for weekly distribution analysis.
    Returns a pivot table: rows=hours, columns=days, values=event count.

    This heatmap is useful for identifying peak and off-peak teaching
    periods, and for assessing whether teaching load is evenly spread.
    """
    df = events.copy()
    # Filter by campus if specified
    if campus:
        df = df[df["Campus"] == campus]

    # Drop records missing Day or Hour
    df = df.dropna(subset=["Day", "Hour"])

    # Count events per (Day, Hour)
    counts = df.groupby(["Day", "Hour"]).size().reset_index(name="EventCount")

    # Convert long-format to pivot table
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    pivot = counts.pivot(index="Hour", columns="Day", values="EventCount").fillna(0)

    # Reorder columns by standard weekday order
    ordered_cols = [d for d in day_order if d in pivot.columns]
    pivot = pivot[ordered_cols]

    return pivot.astype(int)


def room_utilization_by_timeslot(
    events: pd.DataFrame,
    rooms: pd.DataFrame,
    campus: Optional[str] = None,
) -> pd.DataFrame:
    """
    For each (day, hour), compute:
      - rooms_in_use: number of rooms occupied
      - total_rooms: total available rooms
      - pct_rooms_used: percentage

    This is the room-level utilization view, as opposed to the
    event-count view in timeslot_utilization_heatmap.
    """
    target_rooms = rooms.copy()
    # Filter by campus if specified
    if campus:
        target_rooms = target_rooms[target_rooms["Campus"] == campus]
    # Count total target rooms
    total_rooms = len(target_rooms)
    # Build set of target room IDs for fast lookup
    room_ids = set(target_rooms["Id"])

    # Keep only events in target rooms with valid time information
    df = events[events["Room"].isin(room_ids)].dropna(subset=["Day", "Hour"])

    # Count distinct rooms in use per (Day, Hour) combination
    counts = df.groupby(["Day", "Hour"])["Room"].nunique().reset_index(name="RoomsInUse")
    counts["TotalRooms"] = total_rooms
    # Calculate percentage of rooms used: rooms in use / total rooms × 100
    counts["PctUsed"] = (counts["RoomsInUse"] / total_rooms * 100).round(1)

    return counts


def compare_utilization(
    before: pd.DataFrame,
    after: pd.DataFrame,
    label_before: str = "Before",
    label_after: str = "After",
) -> pd.DataFrame:
    """
    Compare two utilization summaries side by side.
    Both should be outputs of campus_utilization_summary().

    Uses outer merge so rooms appearing in only one scenario are preserved.
    """
    # Merge on Room key, combining Frequency/Occupancy/Utilization columns side-by-side
    merged = before[["Room", "Frequency", "Occupancy", "Utilization"]].merge(
        after[["Room", "Frequency", "Occupancy", "Utilization"]],
        on="Room",
        suffixes=(f"_{label_before}", f"_{label_after}"),
        how="outer",
    )
    return merged
