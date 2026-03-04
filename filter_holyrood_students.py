#!/usr/bin/env python3
"""
Filter students affected by Holyrood GT room closure — extract only the
transitions around Holyrood GT events (prev + Holyrood GT + next).

Core insight:
  In Q1; only rooms change (Holyrood -> Central etc).
  Travel impact occurs only at transitions between Holyrood GT events and
  their prev/next events. No need to traverse full timetable; just extract
  prev and next event for each Holyrood GT event.

  Key fix (v2): Must check teaching week overlap when finding prev/next.
  Two events on the same day but with no shared weeks never occur
  consecutively — no actual travel happens.

Output:
  results/filter/holyrood_gt_transitions.csv — one row per transition-week:
      StudentID, HolyroodEventID, Day, Week,
      PrevEventID, PrevCampus, PrevEnd,
      NextEventID, NextCampus, NextStart,
      HolyroodStart, HolyroodEnd, HolyroodDuration, HolyroodCampus
  results/filter/holyrood_gt_student_ids.csv — affected student ID summary

Usage:
    python filter_holyrood_students.py [--semester "Semester 1"]
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Project root (script dir); ensures output always goes to topic/results/
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_events, load_students, get_holyrood_gt_events


def build_event_info(events: pd.DataFrame) -> dict:
    """
    Pre-build event info lookup dict: event_id -> {Day, Start, End, Campus, WeekList, ...}.
    Start/End in fractional hours (e.g. 9:30 -> 9.5, end = start + duration/60).
    WeekList is the list of teaching weeks; WeekSet for fast lookup.
    """
    info = {}
    for _, ev in events.iterrows():
        eid = ev["Event ID"]
        if pd.isna(ev.get("Day")):
            continue
        hour = ev["Hour"]
        minute = ev["Minute"]
        start = hour + minute / 60
        dur = ev["Duration (minutes)"]
        week_list = ev["WeekList"] if isinstance(ev["WeekList"], list) else []
        info[eid] = {
            "Day": ev["Day"],
            "Start": start,
            "End": start + dur / 60,
            "Campus": ev["Campus"],
            "Duration": dur,
            "Timeslot": ev.get("Timeslot", ""),
            "WeekList": week_list,
            "WeekSet": set(week_list),  # Set for fast week lookup
        }
    return info


def extract_transitions(
    affected_student_ids: set,
    student_event_map: dict,
    event_info: dict,
    holyrood_gt_event_ids: set,
) -> pd.DataFrame:
    """
    For each affected student, for each Holyrood GT event, for each teaching
    week that event runs, find the immediately preceding and following events
    on the same day IN THAT WEEK.

    v2 Logic:
      1. Get all Holyrood GT events for this student
      2. For each Holyrood GT event, for each teaching week:
         a. Collect same-day events that run in that week
         b. Sort by start time
         c. Find Holyrood GT event position
         d. Take prev and next (if exist)
      3. Output records with Week column

    Why: Two same-day events may run in different weeks (e.g. A week 10 only,
    B week 11 only). Old version ignored weeks and produced phantom transitions
    (87.5% were spurious). New version builds per-week schedule.

    Returns DataFrame with one row per (student, holyrood_event, week) tuple.
    """
    records = []
    n_students = len(affected_student_ids)

    for idx, sid in enumerate(sorted(affected_student_ids)):
        if idx % 1000 == 0 and idx > 0:
            print(f"  Processed {idx}/{n_students} students...")

        eids = student_event_map.get(sid, [])
        if not eids:
            continue

        # Get all events with info for this student
        student_events = []
        for eid in eids:
            ev = event_info.get(eid)
            if ev is None:
                continue
            student_events.append((eid, ev))

        # Group by day
        day_events = defaultdict(list)
        for eid, ev in student_events:
            day_events[ev["Day"]].append((eid, ev))

        # Find this student's Holyrood GT events
        holyrood_events_for_student = [
            (eid, ev) for eid, ev in student_events
            if eid in holyrood_gt_event_ids
        ]

        # For each Holyrood GT event
        for h_eid, h_ev in holyrood_events_for_student:
            h_day = h_ev["Day"]
            h_weeks = h_ev["WeekSet"]

            if not h_weeks:
                continue

            # Get all same-day events
            same_day = day_events.get(h_day, [])

            # For each teaching week of this Holyrood GT event
            for week in sorted(h_weeks):
                # Filter same-day events that run in this week
                week_events = [
                    (eid, ev) for eid, ev in same_day
                    if week in ev["WeekSet"]
                ]

                # Sort by start time (tie-break by event ID for determinism)
                week_events.sort(key=lambda x: (x[1]["Start"], x[0]))

                # Find position of Holyrood GT event
                h_pos = None
                for i, (eid, _) in enumerate(week_events):
                    if eid == h_eid:
                        h_pos = i
                        break

                if h_pos is None:
                    continue  # Should not happen

                # Build transition record
                record = {
                    "StudentID": sid,
                    "HolyroodEventID": h_eid,
                    "Day": h_day,
                    "Week": week,
                    "HolyroodCampus": h_ev["Campus"],
                    "HolyroodStart": h_ev["Start"],
                    "HolyroodEnd": h_ev["End"],
                    "HolyroodDuration": h_ev["Duration"],
                    # Previous event info
                    "PrevEventID": None,
                    "PrevCampus": None,
                    "PrevEnd": None,
                    # Next event info
                    "NextEventID": None,
                    "NextCampus": None,
                    "NextStart": None,
                }

                # Previous event (if exists)
                if h_pos > 0:
                    prev_eid, prev_ev = week_events[h_pos - 1]
                    record["PrevEventID"] = prev_eid
                    record["PrevCampus"] = prev_ev["Campus"]
                    record["PrevEnd"] = prev_ev["End"]

                # Next event (if exists)
                if h_pos < len(week_events) - 1:
                    next_eid, next_ev = week_events[h_pos + 1]
                    record["NextEventID"] = next_eid
                    record["NextCampus"] = next_ev["Campus"]
                    record["NextStart"] = next_ev["Start"]

                records.append(record)

    return pd.DataFrame(records)


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="Filter Holyrood GT transitions (week-aware)"
    )
    parser.add_argument(
        "--semester", default=None,
        help="Filter events by semester (e.g., 'Semester 1')"
    )
    args = parser.parse_args()

    # Step 1: Load data
    print("=" * 60)
    print("FILTER HOLYROOD GT TRANSITIONS (Week-Aware v2)")
    print("=" * 60)

    print("\nLoading events...")
    events = load_events()

    if args.semester:
        events = events[events["Semester"] == args.semester].copy()
        print(f"Filtered to {args.semester}: {len(events)} events")

    print("Loading students...")
    students = load_students()
    print(f"Total student enrollment rows: {len(students):,}")
    print(f"Total unique students: {students['AnonID'].nunique():,}")

    # Step 2: Identify Holyrood GT events
    print("\n--- Identifying Holyrood GT events ---")
    holyrood_gt_events = get_holyrood_gt_events(events)
    holyrood_gt_event_ids = set(holyrood_gt_events["Event ID"])
    print(f"Holyrood GT events: {len(holyrood_gt_event_ids)}")

    # Summarize week distribution
    week_counts = holyrood_gt_events["WeekList"].apply(len)
    print(f"  Weeks per event: min={week_counts.min()}, max={week_counts.max()}, "
          f"median={week_counts.median():.0f}, mean={week_counts.mean():.1f}")
    single_week = (week_counts == 1).sum()
    print(f"  Single-week events: {single_week} ({single_week/len(holyrood_gt_events)*100:.1f}%)")

    # Step 3: Find affected students
    print("\n--- Finding affected students ---")
    students_in_holyrood = students[students["Event ID"].isin(holyrood_gt_event_ids)]
    affected_student_ids = set(students_in_holyrood["AnonID"])
    print(f"Affected students: {len(affected_student_ids):,} "
          f"({len(affected_student_ids)/students['AnonID'].nunique()*100:.1f}%)")

    # Step 4: Build lookup structures
    print("\n--- Building lookup structures ---")
    event_info = build_event_info(events)
    print(f"Event info dict: {len(event_info)} events")

    # Build student -> event list map (only for affected students)
    affected_students_df = students[students["AnonID"].isin(affected_student_ids)]
    student_event_map = affected_students_df.groupby("AnonID")["Event ID"].apply(list).to_dict()
    print(f"Student-event map: {len(student_event_map)} students")

    # Step 5: Extract transitions (week-aware)
    print("\n--- Extracting week-aware transitions around Holyrood GT events ---")
    print("  (For each Holyrood GT event × each teaching week, find prev and next events)")
    transitions = extract_transitions(
        affected_student_ids, student_event_map,
        event_info, holyrood_gt_event_ids,
    )
    print(f"\nTotal transition-week records: {len(transitions):,}")

    # Statistics
    n_unique_pairs = transitions.groupby(["StudentID", "HolyroodEventID"]).ngroups if len(transitions) > 0 else 0
    print(f"Unique (student, event) pairs: {n_unique_pairs:,}")
    if n_unique_pairs > 0:
        print(f"Avg weeks per (student, event): {len(transitions)/n_unique_pairs:.1f}")

    has_prev = transitions["PrevEventID"].notna().sum()
    has_next = transitions["NextEventID"].notna().sum()
    has_both = ((transitions["PrevEventID"].notna()) & (transitions["NextEventID"].notna())).sum()
    has_neither = ((transitions["PrevEventID"].isna()) & (transitions["NextEventID"].isna())).sum()
    print(f"\n  With prev event:     {has_prev:,} ({has_prev/len(transitions)*100:.1f}%)")
    print(f"  With next event:     {has_next:,} ({has_next/len(transitions)*100:.1f}%)")
    print(f"  With both:           {has_both:,} ({has_both/len(transitions)*100:.1f}%)")
    print(f"  With neither (isolated): {has_neither:,} ({has_neither/len(transitions)*100:.1f}%)")

    # Compare: records with neighbors vs isolated (genuine transition ratio)
    has_neighbor = has_prev + has_next - has_both  # at least one neighbor
    genuine_travel = len(transitions) - has_neither
    print(f"\n  Records with at least one neighbor: {genuine_travel:,} ({genuine_travel/len(transitions)*100:.1f}%)")
    print(f"  Isolated records (no neighbor):     {has_neither:,} ({has_neither/len(transitions)*100:.1f}%)")

    # Cross-campus transition statistics
    cross_campus_prev = transitions[
        (transitions["PrevCampus"].notna()) &
        (transitions["PrevCampus"] != transitions["HolyroodCampus"])
    ]
    cross_campus_next = transitions[
        (transitions["NextCampus"].notna()) &
        (transitions["NextCampus"] != transitions["HolyroodCampus"])
    ]
    print(f"\n  Cross-campus transitions (prev→Holyrood): {len(cross_campus_prev):,}")
    print(f"  Cross-campus transitions (Holyrood→next): {len(cross_campus_next):,}")

    # Step 6: Save results
    output_dir = PROJECT_ROOT / "results" / "filter"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save transition records
    trans_path = output_dir / "holyrood_gt_transitions.csv"
    transitions.to_csv(trans_path, index=False)
    print(f"\nSaved transitions to: {trans_path}")

    # Save student ID summary
    student_summary = []
    for sid in sorted(affected_student_ids):
        sid_trans = transitions[transitions["StudentID"] == sid]
        n_holyrood_events = sid_trans["HolyroodEventID"].nunique()
        n_total_events = len(student_event_map.get(sid, []))
        programme = ""
        sid_rows = students[students["AnonID"] == sid]
        if len(sid_rows) > 0 and "Programme" in sid_rows.columns:
            programme = sid_rows["Programme"].iloc[0]
        student_summary.append({
            "AnonID": sid,
            "Programme": programme,
            "HolyroodGTEvents": n_holyrood_events,
            "TotalEvents": n_total_events,
            "TransitionRecords": len(sid_trans),
        })

    summary_df = pd.DataFrame(student_summary)
    summary_path = output_dir / "holyrood_gt_student_ids.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved student summary to: {summary_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Holyrood GT events:         {len(holyrood_gt_event_ids):,}")
    print(f"Affected students:          {len(affected_student_ids):,}")
    print(f"Transition-week records:    {len(transitions):,}")
    print(f"Unique (student, event):    {n_unique_pairs:,}")
    if len(affected_student_ids) > 0:
        print(f"Avg transition-weeks/student: {len(transitions)/len(affected_student_ids):.1f}")

    # Distribution by day
    if len(transitions) > 0:
        print("\n--- Transition-weeks by day ---")
        day_counts = transitions["Day"].value_counts().sort_index()
        for day, cnt in day_counts.items():
            print(f"  {day:12s}: {cnt:,}")

        # Distribution by week
        print("\n--- Transition-weeks by teaching week (top 15) ---")
        week_dist = transitions["Week"].value_counts().sort_index()
        for wk, cnt in week_dist.head(15).items():
            print(f"  Week {wk:2d}: {cnt:,}")
        if len(week_dist) > 15:
            print(f"  ... ({len(week_dist)} total weeks)")

    # Top 10 affected programmes
    print("\n--- Top 10 affected programmes ---")
    prog_counts = summary_df["Programme"].value_counts().head(10)
    for prog, count in prog_counts.items():
        print(f"  {count:4d} students — {prog}")

    print("\nDone!")


if __name__ == "__main__":
    main()
