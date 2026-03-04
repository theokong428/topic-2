"""
Clash detection utilities for student and room conflicts.
Detects: 1) Student clashes (overlapping events), 2) Room clashes (double-booked),
3) Programme-level lecture clashes. Overlap checked on time and teaching weeks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


def detect_student_clashes(
    events: pd.DataFrame,
    student_event_map: Dict[str, List[str]],
    sample_size: int = 10000,
) -> pd.DataFrame:
    """
    Find cases where a student has two events at the same timeslot
    (same day, overlapping hours, overlapping weeks).
    Groups events by day, then checks pairs for time and week overlap.

    Parameters
    ----------
    events : DataFrame
        Events with Day, Hour, Minute, Duration (minutes), WeekList.
    student_event_map : dict
        student_id -> list of event_ids.
    sample_size : int
        Max students to check (for performance).

    Returns
    -------
    DataFrame of clashes with columns:
        StudentID, EventID_1, EventID_2, Day, SharedWeeks, OverlapStart, OverlapEnd.
    """
    # Build fast lookup indexed by Event ID
    event_lookup = events.set_index("Event ID")[[
        "Day", "Hour", "Minute", "Duration (minutes)", "WeekList"
    ]]

    students = list(student_event_map.keys())
    # Random sample if students exceed limit
    if len(students) > sample_size:
        students = np.random.choice(students, sample_size, replace=False).tolist()

    clashes = []
    for sid in students:
        eids = student_event_map[sid]
        # Group by day
        # Group this student's events by day
        day_events = defaultdict(list)
        for eid in eids:
            # Skip event IDs not found in event table
            if eid not in event_lookup.index:
                continue
            ev = event_lookup.loc[eid]
            # Handle case where .loc returns a DataFrame (duplicate index)
            if isinstance(ev, pd.DataFrame):
                ev = ev.iloc[0]
            # Skip events without a scheduled day
            if pd.isna(ev["Day"]):
                continue
            # Compute event start and end time (decimal hours)
            start = ev["Hour"] + ev["Minute"] / 60
            end = start + ev["Duration (minutes)"] / 60
            day_events[ev["Day"]].append({
                "eid": eid, "start": start, "end": end,
                "weeks": set(ev["WeekList"]),
            })

        # Check pairwise within each day
        # Pairwise comparison within each day
        for day, evts in day_events.items():
            for i in range(len(evts)):
                for j in range(i + 1, len(evts)):
                    ei, ej = evts[i], evts[j]
                    # Check time overlap
                    # Time overlap: A.start < B.end AND B.start < A.end
                    if ei["start"] < ej["end"] and ej["start"] < ei["end"]:
                        # Check week overlap
                        # Week overlap: intersection of week sets
                        shared_weeks = ei["weeks"] & ej["weeks"]
                        if shared_weeks:
                            clashes.append({
                                "StudentID": sid,
                                "EventID_1": ei["eid"],
                                "EventID_2": ej["eid"],
                                "Day": day,
                                # Number of weeks with clash
                                "SharedWeeks": len(shared_weeks),
                                # Start and end of overlapping period
                                "OverlapStart": max(ei["start"], ej["start"]),
                                "OverlapEnd": min(ei["end"], ej["end"]),
                            })

    return pd.DataFrame(clashes)


def detect_room_clashes(events: pd.DataFrame) -> pd.DataFrame:
    """
    Find cases where two events are in the same room at overlapping times and weeks.
    Groups by room, then by day; pairwise overlap checks (time and week).

    Parameters
    ----------
    events : DataFrame
        Events with Room, Day, Hour, Minute, Duration (minutes), WeekList, Event ID.

    Returns
    -------
    DataFrame of clashes with columns:
        Room, EventID_1, EventID_2, Day, SharedWeeks.
    """
    clashes = []
    # Group by room
    # Group by room
    for room_id, group in events.groupby("Room"):
        # Skip events with no room assignment
        if pd.isna(room_id):
            continue
        evts = []
        for _, ev in group.iterrows():
            # Skip events without a scheduled day
            if pd.isna(ev["Day"]):
                continue
            # Compute start and end time (decimal hours)
            # Compute start and end time (decimal hours)
            start = ev["Hour"] + ev["Minute"] / 60
            end = start + ev["Duration (minutes)"] / 60
            evts.append({
                "eid": ev["Event ID"],
                "day": ev["Day"],
                "start": start,
                "end": end,
                "weeks": set(ev["WeekList"]),
            })

        # Pairwise check within same day
        # Pairwise comparison within each day
        # Group by day first to reduce unnecessary pairwise comparisons
        day_groups = defaultdict(list)
        for e in evts:
            day_groups[e["day"]].append(e)

        for day, day_evts in day_groups.items():
            for i in range(len(day_evts)):
                for j in range(i + 1, len(day_evts)):
                    ei, ej = day_evts[i], day_evts[j]
                    # Time overlap check
                    if ei["start"] < ej["end"] and ej["start"] < ei["end"]:
                        # Teaching week overlap check
                        shared = ei["weeks"] & ej["weeks"]
                        if shared:
                            clashes.append({
                                "Room": room_id,
                                "EventID_1": ei["eid"],
                                "EventID_2": ej["eid"],
                                "Day": day,
                                "SharedWeeks": len(shared),
                            })

    return pd.DataFrame(clashes)


def detect_lecture_clashes_for_programme(
    events: pd.DataFrame,
    students: pd.DataFrame,
    programme: str,
) -> pd.DataFrame:
    """
    For a specific programme, find lecture clashes among compulsory modules.
    Useful for Q2: checking if lectures can still be delivered without clashes.
    For a specific programme, detect lecture clashes among compulsory modules.
    Approach: get programme students, collect event IDs, filter to lectures,
    pairwise overlap check.

    Parameters
    ----------
    events : DataFrame
        Events (must include "Event Type").
    students : DataFrame
        Student enrolment (Programme, AnonID, Event ID).
    programme : str
        Target programme name (supports partial match).

    Returns
    -------
    DataFrame of lecture clashes with columns:
        Programme, Day, Module_1, Module_2, EventID_1, EventID_2, SharedWeeks.
    """
    # Get students in this programme
    # Get all students in this programme (partial match)
    prog_students = students[students["Programme"].str.contains(programme, na=False)]
    student_ids = set(prog_students["AnonID"])

    # Get their events
    # Collect all event IDs these students are enrolled in
    prog_events = students[students["AnonID"].isin(student_ids)]
    event_ids = set(prog_events["Event ID"])

    # Filter to lectures
    # Keep only Lecture-type events
    lectures = events[
        (events["Event ID"].isin(event_ids)) &
        (events["Event Type"] == "Lecture")
    ]

    # Check pairwise
    # Group by day then pairwise lecture clash detection
    clashes = []
    for day, day_group in lectures.groupby("Day"):
        evts = []
        for _, ev in day_group.iterrows():
            # Compute start and end time (decimal hours)
            start = ev["Hour"] + ev["Minute"] / 60
            end = start + ev["Duration (minutes)"] / 60
            evts.append({
                "eid": ev["Event ID"],
                "module": ev["Module Name"],
                "start": start,
                "end": end,
                "weeks": set(ev["WeekList"]),
            })

        # Pairwise comparison of lectures on the same day
        for i in range(len(evts)):
            for j in range(i + 1, len(evts)):
                ei, ej = evts[i], evts[j]
                # Time overlap check
                if ei["start"] < ej["end"] and ej["start"] < ei["end"]:
                    # Teaching week overlap check
                    shared = ei["weeks"] & ej["weeks"]
                    if shared:
                        clashes.append({
                            "Programme": programme,
                            "Day": day,
                            "Module_1": ei["module"],
                            "Module_2": ej["module"],
                            "EventID_1": ei["eid"],
                            "EventID_2": ej["eid"],
                            "SharedWeeks": len(shared),
                        })

    return pd.DataFrame(clashes)
