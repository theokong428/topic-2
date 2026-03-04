#!/usr/bin/env python3
"""
Baseline analysis of the original 2024/25 timetable — data description
and pre-intervention metrics for comparison with Q1/Q2 scenarios.

This script computes:
  1. Data overview (events, rooms, students, campuses)
  2. Room inventory by campus, Room Type, Specialist room type
  3. Event distribution (type, campus, day, duration, semester, weeks)
  4. Holyrood GT events: detailed profiling
  5. Capacity analysis: waste, overcrowding in original assignments
  6. Room Type matching baseline for Holyrood GT events
  7. HEFCE utilization metrics (frequency, occupancy, utilization)
  8. Timeslot distribution heatmap

All outputs saved to results/baseline/.

Usage:
    python baseline_analysis.py [--skip-students]
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_rooms, load_events, load_students, load_travel_times,
    build_travel_dict, get_holyrood_gt_events,
    get_gt_rooms, CURRENT_TEACHING_HOURS,
)
from src.analysis.utilization import (
    campus_utilization_summary, timeslot_utilization_heatmap,
)


def section(title: str):
    """Print a section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def subsection(title: str):
    """Print a subsection header"""
    print(f"\n--- {title} ---")


def main():
    parser = argparse.ArgumentParser(description="Baseline timetable analysis")
    parser.add_argument("--skip-students", action="store_true",
                        help="Skip loading student data (faster)")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "results" / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    #  1. LOAD DATA
    # ================================================================
    section("1. DATA LOADING")

    print("Loading rooms...")
    rooms = load_rooms()
    print("Loading events...")
    events = load_events()
    print("Loading travel times...")
    travel_df = load_travel_times()
    travel_dict = build_travel_dict(travel_df)

    students = None
    if not args.skip_students:
        print("Loading students (this may take a minute)...")
        students = load_students()
        print(f"  Student rows: {len(students):,}")
        print(f"  Unique students: {students['AnonID'].nunique():,}")
    else:
        print("  Skipped student data loading.")

    # ================================================================
    #  2. DATA OVERVIEW
    # ================================================================
    section("2. DATA OVERVIEW")

    n_events = len(events)
    n_unique_events = events["Event ID"].nunique()
    n_modules = events["Module Code"].nunique() if "Module Code" in events.columns else 0
    n_rooms_total = len(rooms)
    n_campuses = rooms["Campus"].nunique()

    print(f"Total event-room records:  {n_events:,}")
    print(f"Unique Event IDs:          {n_unique_events:,}")
    print(f"Unique modules:            {n_modules:,}")
    print(f"Total rooms:               {n_rooms_total}")
    print(f"Campuses:                  {n_campuses}")
    print(f"Semesters:                 {events['Semester'].nunique()}")

    # Save overview table
    overview = pd.DataFrame([{
        "Metric": "Event-room records", "Value": n_events,
    }, {
        "Metric": "Unique Event IDs", "Value": n_unique_events,
    }, {
        "Metric": "Unique modules", "Value": n_modules,
    }, {
        "Metric": "Total rooms", "Value": n_rooms_total,
    }, {
        "Metric": "Campuses", "Value": n_campuses,
    }])
    if students is not None:
        overview = pd.concat([overview, pd.DataFrame([{
            "Metric": "Unique students", "Value": students["AnonID"].nunique(),
        }, {
            "Metric": "Student enrollment rows", "Value": len(students),
        }])], ignore_index=True)
    overview.to_csv(output_dir / "data_overview.csv", index=False)

    # ================================================================
    #  3. ROOM INVENTORY
    # ================================================================
    section("3. ROOM INVENTORY")

    subsection("3.1 Rooms by Campus")
    room_by_campus = rooms.groupby("Campus").agg(
        Count=("Id", "count"),
        TotalCapacity=("Capacity", "sum"),
        MeanCapacity=("Capacity", "mean"),
        MedianCapacity=("Capacity", "median"),
        MinCapacity=("Capacity", "min"),
        MaxCapacity=("Capacity", "max"),
    ).sort_values("Count", ascending=False)
    print(room_by_campus.to_string())
    room_by_campus.to_csv(output_dir / "rooms_by_campus.csv")

    subsection("3.2 GT Rooms by Campus")
    gt_rooms = get_gt_rooms(rooms)
    gt_by_campus = gt_rooms.groupby("Campus").agg(
        Count=("Id", "count"),
        TotalCapacity=("Capacity", "sum"),
        MeanCapacity=("Capacity", "mean"),
        MedianCapacity=("Capacity", "median"),
    ).sort_values("Count", ascending=False)
    print(gt_by_campus.to_string())
    gt_by_campus.to_csv(output_dir / "gt_rooms_by_campus.csv")

    subsection("3.3 Room Type (physical layout) by Campus")
    rt_campus = rooms.groupby(["Campus", "Room Type"]).size().unstack(fill_value=0)
    print(rt_campus.to_string())
    rt_campus.to_csv(output_dir / "room_type_by_campus.csv")

    subsection("3.4 Specialist room type by Campus")
    srt_campus = rooms.groupby(["Campus", "Specialist room type"]).size().unstack(fill_value=0)
    print(srt_campus.to_string())
    srt_campus.to_csv(output_dir / "specialist_type_by_campus.csv")

    # ================================================================
    #  4. EVENT DISTRIBUTION
    # ================================================================
    section("4. EVENT DISTRIBUTION")

    subsection("4.1 Events by Type")
    evt_type = events["Event Type"].value_counts()
    print(evt_type.to_string())
    evt_type.to_frame("Count").to_csv(output_dir / "events_by_type.csv")

    subsection("4.2 Events by Campus")
    evt_campus = events["Campus"].value_counts()
    print(evt_campus.to_string())
    evt_campus.to_frame("Count").to_csv(output_dir / "events_by_campus.csv")

    subsection("4.3 Events by Semester")
    evt_sem = events["Semester"].value_counts()
    print(evt_sem.to_string())

    subsection("4.4 Events by Day")
    evt_day = events["Day"].value_counts()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    for d in day_order:
        if d in evt_day.index:
            print(f"  {d:12s}: {evt_day[d]:,}")

    subsection("4.5 Duration Distribution")
    dur = events["Duration (minutes)"].dropna()
    print(f"  Mean:   {dur.mean():.1f} min")
    print(f"  Median: {dur.median():.0f} min")
    print(f"  Min:    {dur.min():.0f} min")
    print(f"  Max:    {dur.max():.0f} min")
    dur_counts = dur.value_counts().sort_index()
    print("\n  Duration (min)  Count")
    for d_val, cnt in dur_counts.items():
        print(f"  {d_val:>8.0f}       {cnt:>6,}")
    dur_counts.to_frame("Count").to_csv(output_dir / "duration_distribution.csv")

    subsection("4.6 Teaching Weeks Distribution")
    week_lens = events["WeekList"].apply(len)
    print(f"  Mean weeks per event:   {week_lens.mean():.1f}")
    print(f"  Median weeks per event: {week_lens.median():.0f}")
    print(f"  Single-week events:     {(week_lens == 1).sum()} ({(week_lens == 1).mean()*100:.1f}%)")
    print(f"  ≥10 week events:        {(week_lens >= 10).sum()} ({(week_lens >= 10).mean()*100:.1f}%)")

    subsection("4.7 Event Size Distribution")
    esize = events["Event Size"]
    print(f"  Mean:   {esize.mean():.1f}")
    print(f"  Median: {esize.median():.0f}")
    print(f"  Min:    {esize.min()}")
    print(f"  Max:    {esize.max()}")
    print(f"  Events with size 0:  {(esize == 0).sum()}")
    print(f"  Events with size >200: {(esize > 200).sum()}")

    # ================================================================
    #  5. HOLYROOD GT EVENTS — DETAILED PROFILE
    # ================================================================
    section("5. HOLYROOD GT EVENTS — DETAILED PROFILE")

    hgt = get_holyrood_gt_events(events)
    print(f"Total Holyrood GT event records: {len(hgt)}")
    print(f"Unique Holyrood GT Event IDs:    {hgt['Event ID'].nunique()}")

    subsection("5.1 By Event Type")
    hgt_type = hgt["Event Type"].value_counts()
    print(hgt_type.to_string())
    hgt_type.to_frame("Count").to_csv(output_dir / "holyrood_gt_by_type.csv")

    subsection("5.2 By Day")
    hgt_day = hgt["Day"].value_counts()
    for d in day_order:
        if d in hgt_day.index:
            print(f"  {d:12s}: {hgt_day[d]:,}")

    subsection("5.3 Size Distribution")
    hgt_size = hgt["Event Size"]
    print(f"  Mean:   {hgt_size.mean():.1f}")
    print(f"  Median: {hgt_size.median():.0f}")
    print(f"  Min:    {hgt_size.min()}")
    print(f"  Max:    {hgt_size.max()}")
    # Bucketed counts
    bins = [0, 20, 50, 100, 200, 500]
    labels = ["1-20", "21-50", "51-100", "101-200", "201+"]
    hgt_size_cut = pd.cut(hgt_size.clip(lower=1), bins=bins, labels=labels, right=True)
    print("\n  Size range   Count")
    for label in labels:
        cnt = (hgt_size_cut == label).sum()
        print(f"  {label:>10s}   {cnt:>5}")

    subsection("5.4 Duration Distribution")
    hgt_dur = hgt["Duration (minutes)"].dropna()
    hgt_dur_counts = hgt_dur.value_counts().sort_index()
    print(f"  Mean: {hgt_dur.mean():.1f} min, Median: {hgt_dur.median():.0f} min")
    for d_val, cnt in hgt_dur_counts.items():
        print(f"  {d_val:>6.0f} min: {cnt:>5}")

    subsection("5.5 Teaching Weeks")
    hgt_wk = hgt["WeekList"].apply(len)
    print(f"  Mean: {hgt_wk.mean():.1f}, Median: {hgt_wk.median():.0f}")
    print(f"  Single-week: {(hgt_wk == 1).sum()} ({(hgt_wk == 1).mean()*100:.1f}%)")

    subsection("5.6 Room Type of Current Rooms")
    hgt_with_rt = hgt.merge(
        rooms[["Id", "Room Type", "Capacity"]].rename(columns={"Id": "Room"}),
        on="Room", how="left", suffixes=("", "_room"),
    )
    hgt_rt = hgt_with_rt["Room Type"].value_counts()
    print(hgt_rt.to_string())
    hgt_rt.to_frame("Count").to_csv(output_dir / "holyrood_gt_room_types.csv")

    subsection("5.7 Room Type by Event Type")
    hgt_rt_et = hgt_with_rt.groupby(["Event Type", "Room Type"]).size().unstack(fill_value=0)
    print(hgt_rt_et.to_string())
    hgt_rt_et.to_csv(output_dir / "holyrood_gt_roomtype_by_eventtype.csv")

    # ================================================================
    #  6. CAPACITY ANALYSIS — ORIGINAL ASSIGNMENTS
    # ================================================================
    section("6. CAPACITY ANALYSIS")

    subsection("6.1 All Events — Capacity Waste")
    ev_cap = events.merge(
        rooms[["Id", "Capacity"]].rename(columns={"Id": "Room"}),
        on="Room", how="left",
    )
    ev_cap["Waste"] = ev_cap["Capacity"] - ev_cap["Event Size"]
    ev_cap["Overcrowded"] = ev_cap["Waste"] < 0

    valid_cap = ev_cap.dropna(subset=["Capacity"])
    print(f"  Events with room capacity info: {len(valid_cap):,} / {len(events):,}")
    print(f"  Waste (Capacity - EventSize):")
    print(f"    Mean:    {valid_cap['Waste'].mean():.1f}")
    print(f"    Median:  {valid_cap['Waste'].median():.0f}")
    print(f"    Std:     {valid_cap['Waste'].std():.1f}")
    print(f"    Min:     {valid_cap['Waste'].min():.0f} (most overcrowded)")
    print(f"    Max:     {valid_cap['Waste'].max():.0f} (most wasted)")
    print(f"  Overcrowded events (size > capacity): {valid_cap['Overcrowded'].sum()}")
    print(f"  Perfect fit (waste = 0):               {(valid_cap['Waste'] == 0).sum()}")

    subsection("6.2 Capacity Waste by Campus")
    waste_by_campus = valid_cap.groupby("Campus").agg(
        Events=("Waste", "count"),
        MeanWaste=("Waste", "mean"),
        MedianWaste=("Waste", "median"),
        Overcrowded=("Overcrowded", "sum"),
    ).sort_values("Events", ascending=False)
    print(waste_by_campus.round(1).to_string())
    waste_by_campus.to_csv(output_dir / "capacity_waste_by_campus.csv")

    subsection("6.3 Holyrood GT — Capacity Waste")
    hgt_cap = hgt_with_rt.copy()
    hgt_cap["Waste"] = hgt_cap["Capacity"] - hgt_cap["Event Size"]
    hgt_cap["Overcrowded"] = hgt_cap["Waste"] < 0
    valid_hgt = hgt_cap.dropna(subset=["Capacity"])
    print(f"  Mean waste:     {valid_hgt['Waste'].mean():.1f}")
    print(f"  Median waste:   {valid_hgt['Waste'].median():.0f}")
    print(f"  Overcrowded:    {valid_hgt['Overcrowded'].sum()}")

    subsection("6.4 Holyrood GT — Waste by Room Type")
    waste_by_rt = valid_hgt.groupby("Room Type").agg(
        Events=("Waste", "count"),
        MeanWaste=("Waste", "mean"),
        MedianWaste=("Waste", "median"),
        Overcrowded=("Overcrowded", "sum"),
    )
    print(waste_by_rt.round(1).to_string())
    waste_by_rt.to_csv(output_dir / "holyrood_gt_waste_by_roomtype.csv")

    # ================================================================
    #  7. ROOM TYPE MATCHING FEASIBILITY
    # ================================================================
    section("7. ROOM TYPE MATCHING")

    subsection("7.1 Holyrood GT Room Type Demand vs Central Supply")
    print("Demand (Holyrood GT events' current Room Type):")
    demand_rt = hgt_with_rt["Room Type"].value_counts()
    for rt, cnt in demand_rt.items():
        print(f"  {rt}: {cnt}")

    central_gt = get_gt_rooms(rooms, campus="Central")
    supply_rt = central_gt["Room Type"].value_counts()
    print("\nSupply (Central GT rooms' Room Type):")
    for rt, cnt in supply_rt.items():
        print(f"  {rt}: {cnt}")

    # Lauriston + New College
    for campus_name in ["Lauriston", "New College"]:
        campus_gt = get_gt_rooms(rooms, campus=campus_name)
        if len(campus_gt) > 0:
            s = campus_gt["Room Type"].value_counts()
            print(f"\nSupply ({campus_name} GT rooms' Room Type):")
            for rt, cnt in s.items():
                print(f"  {rt}: {cnt}")

    # Combined comparison table
    all_rts = set(demand_rt.index)
    for s in [supply_rt]:
        all_rts.update(s.index)
    comparison = pd.DataFrame({
        "Holyrood_Demand": demand_rt,
        "Central_Supply": supply_rt,
        "Lauriston_Supply": get_gt_rooms(rooms, campus="Lauriston")["Room Type"].value_counts(),
        "NC_Supply": get_gt_rooms(rooms, campus="New College")["Room Type"].value_counts(),
    }).fillna(0).astype(int)
    print("\n  Combined comparison:")
    print(comparison.to_string())
    comparison.to_csv(output_dir / "room_type_demand_supply.csv")

    # ================================================================
    #  8. HEFCE UTILIZATION
    # ================================================================
    section("8. HEFCE UTILIZATION")

    subsection("8.1 GT Rooms Utilization by Campus")
    gt_util = campus_utilization_summary(gt_rooms, events)
    gt_util_summary = gt_util.groupby("Campus").agg(
        Rooms=("Room", "count"),
        MeanFrequency=("Frequency", "mean"),
        MeanOccupancy=("Occupancy", "mean"),
        MeanUtilization=("Utilization", "mean"),
    ).sort_values("Rooms", ascending=False)
    print(gt_util_summary.round(4).to_string())
    gt_util_summary.to_csv(output_dir / "gt_utilization_by_campus.csv")

    # Save full GT room utilization table
    gt_util.to_csv(output_dir / "gt_room_utilization_full.csv", index=False)

    subsection("8.2 Holyrood GT Rooms — Detailed Utilization")
    hgt_util = gt_util[gt_util["Campus"] == "Holyrood"]
    print(f"  Rooms: {len(hgt_util)}")
    print(f"  Mean Frequency:    {hgt_util['Frequency'].mean():.2%}")
    print(f"  Mean Occupancy:    {hgt_util['Occupancy'].mean():.2%}")
    print(f"  Mean Utilization:  {hgt_util['Utilization'].mean():.2%}")
    print(f"  Rooms with Freq=0: {(hgt_util['Frequency'] == 0).sum()}")
    # Utilization buckets
    util_bins = [0, 0.05, 0.10, 0.20, 0.50, 1.0]
    util_labels = ["0-5%", "5-10%", "10-20%", "20-50%", "50-100%"]
    hgt_util_cut = pd.cut(hgt_util["Utilization"], bins=util_bins, labels=util_labels)
    print("\n  Utilization distribution (Holyrood GT rooms):")
    for label in util_labels:
        cnt = (hgt_util_cut == label).sum()
        print(f"    {label:>8s}: {cnt} rooms")

    subsection("8.3 Central GT Rooms — Utilization")
    cgt_util = gt_util[gt_util["Campus"] == "Central"]
    print(f"  Rooms: {len(cgt_util)}")
    print(f"  Mean Frequency:    {cgt_util['Frequency'].mean():.2%}")
    print(f"  Mean Occupancy:    {cgt_util['Occupancy'].mean():.2%}")
    print(f"  Mean Utilization:  {cgt_util['Utilization'].mean():.2%}")

    # ================================================================
    #  9. TIMESLOT DISTRIBUTION
    # ================================================================
    section("9. TIMESLOT DISTRIBUTION")

    subsection("9.1 All Campuses")
    heatmap_all = timeslot_utilization_heatmap(events)
    print(heatmap_all.to_string())
    heatmap_all.to_csv(output_dir / "timeslot_heatmap_all.csv")

    subsection("9.2 Holyrood Only")
    heatmap_hol = timeslot_utilization_heatmap(events, campus="Holyrood")
    print(heatmap_hol.to_string())
    heatmap_hol.to_csv(output_dir / "timeslot_heatmap_holyrood.csv")

    subsection("9.3 Central Only")
    heatmap_cen = timeslot_utilization_heatmap(events, campus="Central")
    print(heatmap_cen.to_string())
    heatmap_cen.to_csv(output_dir / "timeslot_heatmap_central.csv")

    # ================================================================
    #  10. INTER-CAMPUS TRAVEL MATRIX
    # ================================================================
    section("10. INTER-CAMPUS TRAVEL MATRIX")

    travel_pivot = travel_df.pivot(index="CampusFrom", columns="CampusTo", values="TravelMins")
    print(travel_pivot.to_string())
    travel_pivot.to_csv(output_dir / "travel_time_matrix.csv")

    # ================================================================
    #  11. PLOTS
    # ================================================================
    section("11. GENERATING PLOTS")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 11.1 Event Size Distribution (all events vs Holyrood GT)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(events["Event Size"].clip(upper=300), bins=60, edgecolor="black", alpha=0.7)
        axes[0].set_xlabel("Event Size (students)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("All Events: Event Size Distribution")
        axes[0].axvline(events["Event Size"].median(), color="red", ls="--", label=f"Median={events['Event Size'].median():.0f}")
        axes[0].legend()

        axes[1].hist(hgt["Event Size"].clip(upper=300), bins=40, edgecolor="black", alpha=0.7, color="orange")
        axes[1].set_xlabel("Event Size (students)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Holyrood GT: Event Size Distribution")
        axes[1].axvline(hgt["Event Size"].median(), color="red", ls="--", label=f"Median={hgt['Event Size'].median():.0f}")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(output_dir / "event_size_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved event_size_distribution.png")

        # 11.2 Capacity Waste Histogram (all events)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        waste_data = valid_cap["Waste"].clip(-50, 300)
        axes[0].hist(waste_data, bins=70, edgecolor="black", alpha=0.7)
        axes[0].axvline(0, color="red", ls="--", label="Perfect fit")
        axes[0].set_xlabel("Capacity Waste (Room Capacity - Event Size)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("All Events: Capacity Waste Distribution")
        axes[0].legend()

        hgt_waste = valid_hgt["Waste"].clip(-50, 300)
        axes[1].hist(hgt_waste, bins=40, edgecolor="black", alpha=0.7, color="orange")
        axes[1].axvline(0, color="red", ls="--", label="Perfect fit")
        axes[1].set_xlabel("Capacity Waste (Room Capacity - Event Size)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Holyrood GT: Capacity Waste Distribution")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(output_dir / "capacity_waste_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved capacity_waste_distribution.png")

        # 11.3 Events by Campus (bar chart)
        fig, ax = plt.subplots(figsize=(10, 5))
        campus_counts = events["Campus"].value_counts().sort_values(ascending=True)
        campus_counts.plot(kind="barh", ax=ax, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Number of Event Records")
        ax.set_title("Events by Campus")
        plt.tight_layout()
        plt.savefig(output_dir / "events_by_campus.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved events_by_campus.png")

        # 11.4 Room Type pie chart for Holyrood GT
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        hgt_rt.plot(kind="pie", ax=axes[0], autopct="%1.0f%%", startangle=90)
        axes[0].set_ylabel("")
        axes[0].set_title("Holyrood GT: Current Room Types")

        # Central GT room types
        supply_rt.plot(kind="pie", ax=axes[1], autopct="%1.0f%%", startangle=90)
        axes[1].set_ylabel("")
        axes[1].set_title("Central GT: Available Room Types")
        plt.tight_layout()
        plt.savefig(output_dir / "room_type_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved room_type_comparison.png")

        # 11.5 Timeslot heatmap
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, (title, hm) in zip(axes, [
            ("All Campuses", heatmap_all),
            ("Holyrood Only", heatmap_hol),
        ]):
            im = ax.imshow(hm.values, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(len(hm.columns)))
            ax.set_xticklabels([d[:3] for d in hm.columns], rotation=45)
            ax.set_yticks(range(len(hm.index)))
            ax.set_yticklabels([f"{int(h):02d}:00" for h in hm.index])
            ax.set_title(title)
            plt.colorbar(im, ax=ax, label="Events")
        plt.tight_layout()
        plt.savefig(output_dir / "timeslot_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved timeslot_heatmap.png")

    except Exception as e:
        print(f"  Warning: Could not generate plots: {e}")

    # ================================================================
    #  FINAL SUMMARY
    # ================================================================
    section("BASELINE ANALYSIS COMPLETE")
    print(f"All outputs saved to: {output_dir}/")
    print(f"Files generated:")
    for f in sorted(output_dir.glob("*")):
        print(f"  {f.name}")
    print("\nDone!")


if __name__ == "__main__":
    main()
