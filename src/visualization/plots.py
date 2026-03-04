"""
Visualization functions for timetabling analysis results.
Produces heat maps, bar charts, and comparison plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Tuple


# Style defaults
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# Day ordering: Monday through Saturday
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def timeslot_heatmap(
    pivot: pd.DataFrame,
    title: str = "Events per Timeslot",
    save_path: Optional[str] = None,
    cmap: str = "YlOrRd",
):
    """
    Plot a heatmap of events per (day, hour).

    Parameters
    ----------
    pivot : DataFrame
        Output of utilization.timeslot_utilization_heatmap().
        Rows = hours, columns = days.
    title : str
        Chart title.
    save_path : str, optional
        File path to save the figure.
    cmap : str
        Colormap name for the heatmap.
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw heatmap: annotated with integer values, white gridlines
    sns.heatmap(
        pivot, annot=True, fmt="d", cmap=cmap, ax=ax,
        linewidths=0.5, linecolor="white",
    )
    ax.set_title(title)
    ax.set_ylabel("Hour")
    ax.set_xlabel("Day")
    plt.tight_layout()

    # Save figure to disk if a path is provided
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def utilization_comparison_bar(
    before: pd.DataFrame,
    after: pd.DataFrame,
    metric: str = "Frequency",
    title: str = "Room Utilization Comparison",
    save_path: Optional[str] = None,
):
    """
    Side-by-side bar chart comparing a utilization metric before/after.

    Parameters
    ----------
    before : DataFrame
        Utilization data before reassignment.
    after : DataFrame
        Utilization data after reassignment.
    metric : str
        Metric column to compare.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Aggregate by campus
    b_agg = before.groupby("Campus")[metric].mean()
    a_agg = after.groupby("Campus")[metric].mean()

    # Collect and sort all campus names from both datasets
    campuses = sorted(set(b_agg.index) | set(a_agg.index))
    x = np.arange(len(campuses))
    width = 0.35

    # Draw two groups of bars: "Before" and "After"
    bars1 = ax.bar(x - width / 2, [b_agg.get(c, 0) for c in campuses],
                   width, label="Before", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, [a_agg.get(c, 0) for c in campuses],
                   width, label="After", color="#DD8452")

    ax.set_xlabel("Campus")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(campuses, rotation=30, ha="right")
    ax.legend()

    # Format y-axis as percentage (base value 1.0)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    plt.tight_layout()

    # Optionally save the figure
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def capacity_fit_histogram(
    assignments: pd.DataFrame,
    title: str = "Capacity Waste Distribution",
    save_path: Optional[str] = None,
):
    """
    Histogram of capacity waste (room capacity - event size) for assignments.

    Parameters
    ----------
    assignments : DataFrame
        Assignment results; must contain a "Capacity Waste" column.
    """
    # Validate: check for required column and non-empty data
    if "Capacity Waste" not in assignments.columns or len(assignments) == 0:
        print("No assignment data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    # Draw histogram with 30 bins
    ax.hist(assignments["Capacity Waste"], bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Capacity Waste (seats)")
    ax.set_ylabel("Number of Events")
    ax.set_title(title)

    # Mark the median with a red dashed line
    ax.axvline(assignments["Capacity Waste"].median(), color="red",
               linestyle="--", label=f"Median: {assignments['Capacity Waste'].median():.0f}")
    ax.legend()
    plt.tight_layout()

    # Optionally save the figure
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def travel_impact_boxplot(
    impact_df: pd.DataFrame,
    title: str = "Student Travel Time Impact",
    save_path: Optional[str] = None,
):
    """
    Box plot comparing travel time before and after reassignment.

    Parameters
    ----------
    impact_df : DataFrame
        Travel impact data with "TravelBefore" and "TravelAfter" columns.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Build a comparison DataFrame with before/after travel times
    data = pd.DataFrame({
        "Before": impact_df["TravelBefore"],
        "After": impact_df["TravelAfter"],
    })
    data.boxplot(ax=ax)
    ax.set_ylabel("Daily Travel Time (minutes)")
    ax.set_title(title)
    plt.tight_layout()

    # Optionally save the figure
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def travel_change_distribution(
    impact_df: pd.DataFrame,
    title: str = "Change in Daily Travel Time",
    save_path: Optional[str] = None,
):
    """
    Histogram of per-student change in travel time.
    """
    # Calculate the change in travel time (after - before)
    change = impact_df["TravelAfter"] - impact_df["TravelBefore"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(change, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Change in Daily Travel (minutes)")
    ax.set_ylabel("Count (student-days)")
    ax.set_title(title)

    # Draw a red dashed line at x=0 as the "no change" reference
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    plt.tight_layout()

    # Optionally save the figure
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def lunch_break_chart(
    lunch_data: dict,
    title: str = "Lunch Break Availability (12-2pm)",
    save_path: Optional[str] = None,
):
    """
    Bar chart of % students with lunch break by day.

    Parameters
    ----------
    lunch_data : dict
        Dictionary keyed by day name, each value has a "pct_with_lunch" key.
    """
    # Weekday list
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    # Extract lunch break percentage for each day
    pcts = [lunch_data.get(d, {}).get("pct_with_lunch", 0) for d in days]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(days, pcts, color="#55A868", edgecolor="black", alpha=0.8)
    ax.set_ylabel("% Students with Lunch Break")
    ax.set_title(title)
    ax.set_ylim(0, 105)

    # 50% threshold reference line
    ax.axhline(50, color="red", linestyle="--", alpha=0.5, label="50% threshold")
    ax.legend()

    # Add value labels on top of each bar
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%", ha="center", fontsize=10)

    plt.tight_layout()

    # Optionally save the figure
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def scenario_comparison_table(
    summaries: Dict[str, dict],
    title: str = "Scenario Comparison",
) -> pd.DataFrame:
    """
    Create a comparison table from multiple scenario summaries.
    Returns a formatted DataFrame.

    Parameters
    ----------
    summaries : Dict[str, dict]
        Keys are scenario names, values are summary dictionaries.
    """
    # Flatten each scenario summary into a table row
    rows = []
    for name, summary in summaries.items():
        rows.append({"Scenario": name, **summary})
    df = pd.DataFrame(rows)

    # Print the formatted comparison table
    print(f"\n{title}")
    print("=" * 60)
    print(df.to_string(index=False))
    return df


def room_usage_by_campus(
    rooms: pd.DataFrame,
    events: pd.DataFrame,
    title: str = "Room Usage by Campus",
    save_path: Optional[str] = None,
):
    """
    Stacked bar: rooms with events vs rooms without, by campus.

    Parameters
    ----------
    rooms : DataFrame
        Room inventory data with Id and Campus columns.
    events : DataFrame
        Event data with a Room column to identify used rooms.
    """
    # Get the set of all room IDs that have at least one event assigned
    rooms_with_events = set(events["Room"].dropna().unique())

    # Count used and unused rooms per campus
    campus_stats = []
    for campus, group in rooms.groupby("Campus"):
        total = len(group)
        used = group["Id"].isin(rooms_with_events).sum()
        campus_stats.append({
            "Campus": campus,
            "Used": used,
            "Unused": total - used,
        })

    # Create stacked bar chart
    df = pd.DataFrame(campus_stats).set_index("Campus")
    fig, ax = plt.subplots(figsize=(10, 5))
    df[["Used", "Unused"]].plot(kind="bar", stacked=True, ax=ax,
                                 color=["#4C72B0", "#C44E52"])
    ax.set_ylabel("Number of Rooms")
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()

    # Optionally save the figure
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
