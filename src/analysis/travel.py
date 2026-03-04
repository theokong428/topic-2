"""
Student travel impact analysis — focused on transitions around Holyrood GT events.

Core insight:
  In Q1 timeslots are fixed; only room campus changes (Holyrood -> Central etc).
  Travel impact occurs only at transitions between Holyrood GT events and
  their prev/next events.

  For each transition-week record:
    Before: prev_campus → Holyrood,   Holyrood → next_campus
    After:  prev_campus → new_campus, new_campus → next_campus
    Δtravel = travel_after - travel_before

  v2: Each record corresponds to a (student, holyrood_event, week) tuple,
      containing only transitions that genuinely occur in that teaching week
      (phantom transitions eliminated).

  Full traversal of all affected students (no sampling).
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def compute_transition_travel(
    transitions: pd.DataFrame,
    reassignment_map: Dict[str, str],
    travel_dict: Dict[Tuple[str, str], int],
) -> pd.DataFrame:
    """
    Compute before/after travel time for each transition around a Holyrood GT event.
    For each row: Before: prev→Holyrood, Holyrood→next; After: prev→new_campus, new→next.
    Also checks travel feasibility: if gap insufficient, marks as infeasible.

    Parameters
    ----------
    transitions : DataFrame
        Output from filter_holyrood_students.py: holyrood_gt_transitions.csv (week-aware v2).
        Columns: StudentID, HolyroodEventID, Day, Week, HolyroodCampus,
                 HolyroodStart, HolyroodEnd, HolyroodDuration,
                 PrevEventID, PrevCampus, PrevEnd,
                 NextEventID, NextCampus, NextStart
    reassignment_map : dict
        event_id -> new_campus. Maps each Holyrood GT event to its new campus
        after MIP reassignment (e.g., "Central", "Lauriston", "New College").
    travel_dict : dict
        (campus_from, campus_to) -> travel_minutes.

    Returns
    -------
    DataFrame with added columns: NewCampus, TravelBefore_Prev, TravelBefore_Next,
    TravelAfter_Prev, TravelAfter_Next, TravelBefore, TravelAfter, TravelDelta,
    InfeasibleBefore_Prev, InfeasibleBefore_Next, InfeasibleAfter_Prev, InfeasibleAfter_Next.
    """
    result = transitions.copy()

    # Lookup new campus for each Holyrood GT event
    result["NewCampus"] = result["HolyroodEventID"].map(reassignment_map)

    # Unplaced events have NaN new campus, mark as "UNPLACED"
    result["NewCampus"] = result["NewCampus"].fillna("UNPLACED")

    def _get_travel(campus_from, campus_to):
        """Look up travel time between campuses; 0 for same campus, 60 default for unknown."""
        if pd.isna(campus_from) or pd.isna(campus_to):
            return 0
        if campus_from == campus_to:
            return 0
        return travel_dict.get((campus_from, campus_to), 60)

    def _check_infeasible(campus_from, campus_to, gap_hours):
        """Check feasibility: infeasible if gap (minutes) < travel time."""
        if pd.isna(campus_from) or pd.isna(campus_to):
            return False
        if campus_from == campus_to:
            return False
        travel_mins = travel_dict.get((campus_from, campus_to), 60)
        gap_mins = gap_hours * 60
        return gap_mins < travel_mins

    # Compute Before travel (original: event at Holyrood)
    travel_before_prev = []
    travel_before_next = []
    infeasible_before_prev = []
    infeasible_before_next = []

    # Compute After travel (new: event moved to NewCampus)
    travel_after_prev = []
    travel_after_next = []
    infeasible_after_prev = []
    infeasible_after_next = []

    for _, row in result.iterrows():
        holyrood_campus = row["HolyroodCampus"]
        new_campus = row["NewCampus"]
        prev_campus = row["PrevCampus"]
        next_campus = row["NextCampus"]

        # ── Prev → Holyrood GT event ──
        # Before: prev → Holyrood
        tb_prev = _get_travel(prev_campus, holyrood_campus)
        # After: prev → new_campus
        ta_prev = _get_travel(prev_campus, new_campus) if new_campus != "UNPLACED" else _get_travel(prev_campus, holyrood_campus)

        travel_before_prev.append(tb_prev)
        travel_after_prev.append(ta_prev)

        # Check feasibility
        if pd.notna(row["PrevEnd"]):
            gap_prev = row["HolyroodStart"] - row["PrevEnd"]
            infeasible_before_prev.append(_check_infeasible(prev_campus, holyrood_campus, gap_prev))
            infeasible_after_prev.append(
                _check_infeasible(prev_campus, new_campus, gap_prev)
                if new_campus != "UNPLACED"
                else _check_infeasible(prev_campus, holyrood_campus, gap_prev)
            )
        else:
            infeasible_before_prev.append(False)
            infeasible_after_prev.append(False)

        # ── Holyrood GT event → Next ──
        # Before: Holyrood → next
        tb_next = _get_travel(holyrood_campus, next_campus)
        # After: new_campus → next
        ta_next = _get_travel(new_campus, next_campus) if new_campus != "UNPLACED" else _get_travel(holyrood_campus, next_campus)

        travel_before_next.append(tb_next)
        travel_after_next.append(ta_next)

        # Check feasibility
        if pd.notna(row["NextStart"]):
            gap_next = row["NextStart"] - row["HolyroodEnd"]
            infeasible_before_next.append(_check_infeasible(holyrood_campus, next_campus, gap_next))
            infeasible_after_next.append(
                _check_infeasible(new_campus, next_campus, gap_next)
                if new_campus != "UNPLACED"
                else _check_infeasible(holyrood_campus, next_campus, gap_next)
            )
        else:
            infeasible_before_next.append(False)
            infeasible_after_next.append(False)

    # Assign columns
    result["TravelBefore_Prev"] = travel_before_prev
    result["TravelBefore_Next"] = travel_before_next
    result["TravelAfter_Prev"] = travel_after_prev
    result["TravelAfter_Next"] = travel_after_next

    # Total travel time
    result["TravelBefore"] = result["TravelBefore_Prev"] + result["TravelBefore_Next"]
    result["TravelAfter"] = result["TravelAfter_Prev"] + result["TravelAfter_Next"]
    result["TravelDelta"] = result["TravelAfter"] - result["TravelBefore"]

    # Infeasibility flags
    result["InfeasibleBefore_Prev"] = infeasible_before_prev
    result["InfeasibleBefore_Next"] = infeasible_before_next
    result["InfeasibleAfter_Prev"] = infeasible_after_prev
    result["InfeasibleAfter_Next"] = infeasible_after_next

    return result


def travel_impact_summary(impact_df: pd.DataFrame) -> dict:
    """
    Aggregate travel impact statistics from transition-week-level data.
    Each record represents one genuine travel occurrence for a
    (student, holyrood_event, week) tuple.

    Parameters
    ----------
    impact_df : DataFrame
        Output from compute_transition_travel().

    Returns
    -------
    dict with keys: total_transition_weeks, unique_student_event_pairs,
    avg_travel_before/after/delta, total_travel_*, infeasible_*_count,
    students_affected, students_with_*_travel, pct_transitions_*.
    """
    if len(impact_df) == 0:
        return {"total_transition_weeks": 0}

    # Overall statistics
    total = len(impact_df)

    # Unique (student, event) pairs
    unique_pairs = impact_df.groupby(["StudentID", "HolyroodEventID"]).ngroups

    # Infeasibility analysis (either prev or next infeasible counts)
    infeasible_before = (
        impact_df["InfeasibleBefore_Prev"] | impact_df["InfeasibleBefore_Next"]
    ).sum()
    infeasible_after = (
        impact_df["InfeasibleAfter_Prev"] | impact_df["InfeasibleAfter_Next"]
    ).sum()

    was_feasible_before = ~(impact_df["InfeasibleBefore_Prev"] | impact_df["InfeasibleBefore_Next"])
    is_infeasible_after = (impact_df["InfeasibleAfter_Prev"] | impact_df["InfeasibleAfter_Next"])
    new_infeasible = (was_feasible_before & is_infeasible_after).sum()

    was_infeasible_before = (impact_df["InfeasibleBefore_Prev"] | impact_df["InfeasibleBefore_Next"])
    is_feasible_after = ~(impact_df["InfeasibleAfter_Prev"] | impact_df["InfeasibleAfter_Next"])
    resolved_infeasible = (was_infeasible_before & is_feasible_after).sum()

    # Per-student summary (aggregate travel changes across transition-weeks)
    student_agg = impact_df.groupby("StudentID").agg(
        total_before=("TravelBefore", "sum"),
        total_after=("TravelAfter", "sum"),
        any_new_infeasible_prev=("InfeasibleAfter_Prev", "any"),
        any_new_infeasible_next=("InfeasibleAfter_Next", "any"),
    )
    student_agg["delta"] = student_agg["total_after"] - student_agg["total_before"]

    # Per-student weekly average travel change
    student_weekly = impact_df.groupby("StudentID").agg(
        n_weeks=("Week", "nunique"),
        total_delta=("TravelDelta", "sum"),
    )
    student_weekly["avg_weekly_delta"] = student_weekly["total_delta"] / student_weekly["n_weeks"]

    return {
        "total_transition_weeks": total,
        "unique_student_event_pairs": int(unique_pairs),
        # Travel times
        "avg_travel_before": impact_df["TravelBefore"].mean(),
        "avg_travel_after": impact_df["TravelAfter"].mean(),
        "avg_travel_delta": impact_df["TravelDelta"].mean(),
        "total_travel_before": impact_df["TravelBefore"].sum(),
        "total_travel_after": impact_df["TravelAfter"].sum(),
        "total_travel_delta": impact_df["TravelDelta"].sum(),
        # Infeasibility stats
        "infeasible_before_count": int(infeasible_before),
        "infeasible_after_count": int(infeasible_after),
        "new_infeasible_count": int(new_infeasible),
        "resolved_infeasible_count": int(resolved_infeasible),
        # Student-level
        "students_affected": len(student_agg),
        "students_with_increased_travel": int((student_agg["delta"] > 0).sum()),
        "students_with_decreased_travel": int((student_agg["delta"] < 0).sum()),
        "students_no_change": int((student_agg["delta"] == 0).sum()),
        "students_with_new_infeasible": int(
            (student_agg["any_new_infeasible_prev"] | student_agg["any_new_infeasible_next"]).sum()
        ),
        # Weekly average travel change (student level)
        "avg_weekly_delta_per_student": student_weekly["avg_weekly_delta"].mean(),
        "median_weekly_delta_per_student": student_weekly["avg_weekly_delta"].median(),
        # Transition-week-level change distribution
        "pct_transitions_improved": (impact_df["TravelDelta"] < 0).sum() / total * 100,
        "pct_transitions_worsened": (impact_df["TravelDelta"] > 0).sum() / total * 100,
        "pct_transitions_unchanged": (impact_df["TravelDelta"] == 0).sum() / total * 100,
    }


def print_travel_summary(summary: dict):
    """
    Pretty-print the travel impact summary.
    """
    if summary.get("total_transition_weeks", 0) == 0:
        print("  No transition data available.")
        return

    print(f"  Total transition-weeks analyzed:    {summary['total_transition_weeks']:,}")
    print(f"  Unique (student, event) pairs:      {summary['unique_student_event_pairs']:,}")
    print(f"  Affected students:                  {summary['students_affected']:,}")
    print()
    print(f"  Avg travel per transition (before):  {summary['avg_travel_before']:.1f} min")
    print(f"  Avg travel per transition (after):   {summary['avg_travel_after']:.1f} min")
    print(f"  Avg travel change per transition:    {summary['avg_travel_delta']:+.1f} min")
    print(f"  Semester total travel change:        {summary['total_travel_delta']:+,.0f} min")
    print()
    print(f"  Avg weekly travel change/student:    {summary['avg_weekly_delta_per_student']:+.1f} min")
    print(f"  Median weekly travel change/student: {summary['median_weekly_delta_per_student']:+.1f} min")
    print()
    print(f"  Transition-weeks improved (less):    {summary['pct_transitions_improved']:.1f}%")
    print(f"  Transition-weeks worsened (more):    {summary['pct_transitions_worsened']:.1f}%")
    print(f"  Transition-weeks unchanged:          {summary['pct_transitions_unchanged']:.1f}%")
    print()
    print(f"  Infeasible transitions (before):     {summary['infeasible_before_count']}")
    print(f"  Infeasible transitions (after):      {summary['infeasible_after_count']}")
    print(f"  NEW infeasible (was OK, now bad):    {summary['new_infeasible_count']}")
    print(f"  RESOLVED infeasible (was bad, now OK): {summary['resolved_infeasible_count']}")
    print()
    print(f"  Students with increased travel:      {summary['students_with_increased_travel']}")
    print(f"  Students with decreased travel:      {summary['students_with_decreased_travel']}")
    print(f"  Students with no change:             {summary['students_no_change']}")
    print(f"  Students with new infeasible:        {summary['students_with_new_infeasible']}")
