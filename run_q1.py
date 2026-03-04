#!/usr/bin/env python3
"""
Run Q1 SPACE analysis: Holyrood GT room closure scenarios.

Usage:
    python run_q1.py [--semester "Semester 1"] [--time-limit 300]
"""

import argparse
import sys
from pathlib import Path

# Project root; ensures output always goes to topic/results/
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_all, get_holyrood_gt_events, get_holyrood_gt_rooms,
    get_central_gt_rooms, get_lauriston_gt_rooms, get_newcollege_gt_rooms,
    build_student_event_map, build_event_student_map,
)
from src.models.q1_space import SpaceScenarioModel
from src.analysis.utilization import (
    campus_utilization_summary, timeslot_utilization_heatmap,
    compare_utilization,
)
from src.analysis.travel import compute_transition_travel, travel_impact_summary
from src.analysis.clash_detection import detect_room_clashes
from src.visualization.plots import (
    timeslot_heatmap, utilization_comparison_bar,
    capacity_fit_histogram, travel_impact_boxplot,
    travel_change_distribution, scenario_comparison_table,
)


def main():
    parser = argparse.ArgumentParser(description="Q1 SPACE Analysis")
    parser.add_argument("--semester", default=None, help="Filter by semester (e.g., 'Semester 1')")
    parser.add_argument("--time-limit", type=int, default=300, help="Solver time limit in seconds")
    parser.add_argument("--student-sample", type=int, default=5000, help="Student sample size for travel analysis")
    args = parser.parse_args()

    # ── Load Data ──
    print("=" * 60)
    print("Q1 SPACE ANALYSIS")
    print("=" * 60)

    data = load_all()
    rooms = data["rooms"]
    events = data["events"]
    students = data["students"]
    travel_dict = data["travel_dict"]

    # Optional semester filter
    if args.semester:
        events = events[events["Semester"] == args.semester].copy()
        print(f"Filtered to {args.semester}: {len(events)} events")

    # ── Baseline Statistics ──
    print("\n--- Baseline Statistics ---")
    holyrood_gt_rooms = get_holyrood_gt_rooms(rooms)
    central_gt_rooms = get_central_gt_rooms(rooms)
    lauriston_gt_rooms = get_lauriston_gt_rooms(rooms)
    nc_gt_rooms = get_newcollege_gt_rooms(rooms)

    print(f"Holyrood GT rooms: {len(holyrood_gt_rooms)} (capacity {holyrood_gt_rooms['Capacity'].sum()})")
    print(f"Central GT rooms:  {len(central_gt_rooms)} (capacity {central_gt_rooms['Capacity'].sum()})")
    print(f"Lauriston GT rooms: {len(lauriston_gt_rooms)} (capacity {lauriston_gt_rooms['Capacity'].sum()})")
    print(f"New College GT rooms: {len(nc_gt_rooms)} (capacity {nc_gt_rooms['Capacity'].sum()})")

    displaced = get_holyrood_gt_events(events)
    print(f"\nEvents to displace: {len(displaced)}")
    print(f"  Unique timeslots: {displaced['Timeslot'].nunique()}")
    print(f"  Event size range: {displaced['Event Size'].min()} - {displaced['Event Size'].max()}")
    print(f"  Event size median: {displaced['Event Size'].median()}")

    # ── Baseline Utilization ──
    print("\n--- Baseline Utilization (before closure) ---")
    util_before = campus_utilization_summary(rooms, events)
    for campus in ["Holyrood", "Central", "Lauriston", "New College"]:
        cu = util_before[util_before["Campus"] == campus]
        if len(cu) > 0:
            print(f"  {campus}: freq={cu['Frequency'].mean():.2%}, "
                  f"occ={cu['Occupancy'].mean():.2%}, "
                  f"util={cu['Utilization'].mean():.2%}")

    heatmap_before = timeslot_utilization_heatmap(events)
    print("\nTimeslot heatmap (all campuses, before):")
    print(heatmap_before)

    # ── Scenario 1a: Central-only ──
    print("\n" + "=" * 60)
    print("SCENARIO 1a: Central GT rooms only")
    print("=" * 60)

    central_room_ids = set(central_gt_rooms["Id"])
    existing_central = events[events["Room"].isin(central_room_ids)]

    # Load transition data (if exists) for MIP travel optimisation
    trans_path = PROJECT_ROOT / "results" / "filter" / "holyrood_gt_transitions.csv"
    transitions_df = pd.read_csv(trans_path) if trans_path.exists() else None
    if transitions_df is not None:
        print(f"Loaded {len(transitions_df):,} transition records for travel optimisation")

    model_a = SpaceScenarioModel(
        displaced_events=displaced,
        candidate_rooms=central_gt_rooms,
        existing_events=existing_central,
        travel_dict=travel_dict,
        rooms_full=rooms,
        transitions_df=transitions_df,
    )
    model_a.build_model()
    model_a.solve(time_limit=args.time_limit)
    summary_a = model_a.get_results_summary()
    model_a.export_results(str(PROJECT_ROOT / "results" / "q1" / "scenario_1a"))

    # ── Scenario 1b: Central + Lauriston + New College ──
    print("\n" + "=" * 60)
    print("SCENARIO 1b: Central + Lauriston + New College GT rooms")
    print("=" * 60)

    extra_rooms = pd.concat([
        central_gt_rooms, lauriston_gt_rooms, nc_gt_rooms
    ], ignore_index=True)
    extra_room_ids = set(extra_rooms["Id"])
    existing_extra = events[events["Room"].isin(extra_room_ids)]

    model_b = SpaceScenarioModel(
        displaced_events=displaced,
        candidate_rooms=extra_rooms,
        existing_events=existing_extra,
        travel_dict=travel_dict,
        rooms_full=rooms,
        transitions_df=transitions_df,
    )
    model_b.build_model()
    model_b.solve(time_limit=args.time_limit)
    summary_b = model_b.get_results_summary()
    model_b.export_results(str(PROJECT_ROOT / "results" / "q1" / "scenario_1b"))

    # ── Stage 2 pending: export unplaced events list ──
    if model_b.solution and model_b.solution["unplaced_count"] > 0:
        unplaced = model_b.solution["unplaced"]
        stage2_path = PROJECT_ROOT / "results" / "q1" / "stage2_unplaced_events.csv"
        stage2_path.parent.mkdir(parents=True, exist_ok=True)
        unplaced.to_csv(stage2_path, index=False)
        print(f"\nStage 2 pending: {len(unplaced)} unplaced events exported to {stage2_path}")

    # ── Comparison ──
    print("\n" + "=" * 60)
    print("SCENARIO COMPARISON")
    print("=" * 60)
    scenario_comparison_table({
        "1a (Central only)": summary_a,
        "1b (Central+Lauriston+NC)": summary_b,
    })

    # ── Visualizations ──
    if model_a.solution and len(model_a.solution["assignments"]) > 0:
        capacity_fit_histogram(
            model_a.solution["assignments"],
            title="Scenario 1a: Capacity Waste",
            save_path=str(PROJECT_ROOT / "results" / "q1" / "scenario_1a" / "capacity_waste.png"),
        )

    if model_b.solution and len(model_b.solution["assignments"]) > 0:
        capacity_fit_histogram(
            model_b.solution["assignments"],
            title="Scenario 1b: Capacity Waste",
            save_path=str(PROJECT_ROOT / "results" / "q1" / "scenario_1b" / "capacity_waste.png"),
        )

    # ── Travel Impact (for scenario 1b) ──
    if model_b.solution and len(model_b.solution["assignments"]) > 0:
        print("\n--- Travel Impact Analysis (Scenario 1b) ---")
        # Use transition data from filter_holyrood_students.py
        trans_path = PROJECT_ROOT / "results" / "filter" / "holyrood_gt_transitions.csv"
        if trans_path.exists():
            transitions = pd.read_csv(trans_path)
            # Build event_id -> new_campus mapping
            asg = model_b.solution["assignments"]
            reassignment_map = dict(zip(asg["Event ID"], asg["New Campus"]))
            impact = compute_transition_travel(transitions, reassignment_map, travel_dict)
        else:
            print("  Skipping travel impact: run filter_holyrood_students.py first to generate results/filter/holyrood_gt_transitions.csv")
            impact = pd.DataFrame()

        if len(impact) > 0:
            ti_summary = travel_impact_summary(impact)
            print(f"  Avg travel before: {ti_summary.get('avg_travel_before', 0):.1f} min/day")
            print(f"  Avg travel after:  {ti_summary.get('avg_travel_after', 0):.1f} min/day")
            print(f"  Students with increased travel: {ti_summary.get('students_with_increased_travel', 0)}")
            print(f"  New infeasible transitions: {ti_summary.get('students_with_new_infeasible', 0)}")

            impact.to_csv(PROJECT_ROOT / "results" / "q1" / "scenario_1b" / "travel_impact.csv", index=False)

            travel_impact_boxplot(
                impact,
                title="Scenario 1b: Student Travel Impact",
                save_path=str(PROJECT_ROOT / "results" / "q1" / "scenario_1b" / "travel_boxplot.png"),
            )
            travel_change_distribution(
                impact,
                title="Scenario 1b: Change in Travel Time",
                save_path=str(PROJECT_ROOT / "results" / "q1" / "scenario_1b" / "travel_change.png"),
            )

    # ── Post-reassignment Utilization ──
    print("\n--- Post-Reassignment Utilization ---")
    if model_b.solution and len(model_b.solution["assignments"]) > 0:
        events_after = events.copy()
        for _, row in model_b.solution["assignments"].iterrows():
            mask = events_after["Event ID"] == row["Event ID"]
            events_after.loc[mask, "Room"] = row["New Room"]
            events_after.loc[mask, "Campus"] = row["New Campus"]
    else:
        events_after = events
    heatmap_after = timeslot_utilization_heatmap(events_after)
    timeslot_heatmap(
        heatmap_before,
        title="Before: Events per Timeslot",
        save_path=str(PROJECT_ROOT / "results" / "q1" / "heatmap_before.png"),
    )
    timeslot_heatmap(
        heatmap_after,
        title="After (1b): Events per Timeslot",
        save_path=str(PROJECT_ROOT / "results" / "q1" / "heatmap_after.png"),
    )

    print(f"\nQ1 analysis complete. Results saved to {PROJECT_ROOT / 'results' / 'q1'}/")


if __name__ == "__main__":
    import pandas as pd
    main()
