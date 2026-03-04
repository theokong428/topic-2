## Quick Start

### 1. Install dependencies

```bash
cd topic
pip install -r requirements.txt
```

### 2. Configure Xpress solver

- A valid FICO Xpress licence is required
- Licence path is configured via `xp.init()` in `src/models/q1_space.py`

### 3. Run Q1 space analysis

```bash
# First run filter_holyrood_students.py to generate transition data for travel analysis
python filter_holyrood_students.py 

# Run Q1 analysis
python run_q1.py
```

## Directory Structure

```
topic/
├── README.md                           # This document
├── requirements.txt                    # Python dependencies
│
│  ── Entry scripts ──
├── run_q1.py                           # Q1 space analysis (Stage 1: fixed timeslots, room reassignment)
├── run_q1_stage2.py                    # Q1 Stage 2 placeholder (timeslot + room joint reassignment)
├── filter_holyrood_students.py         # Holyrood student transition data (week-aware)
├── baseline_analysis.py                # Baseline data analysis (11 metrics + visualisation)
│
│  ── Raw data (read-only) ──
├── course_timtabling/                  # Note spelling: "timtabling"
│   ├── Rooms and Room Types.xlsx           # 649 rooms, inter-campus travel times
│   ├── 2024-5 Event Module Room.xlsx       # 32,757 event-room assignments
│   ├── 2024-5 Student Programme Module Event.xlsx  # 930,173 student enrolments
│   ├── Programme-Course.xlsx               # 53,963 programme-course mappings
│   ├── 2024-5 DPT Data.xlsx                 # 949,920 degree programme structure
│   └── TT Modelling Scenario - Optimization.docx  # Project brief
│
│  ── Source code ──
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Data loading and preprocessing (pandas + openpyxl)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── q1_space.py                 # Q1 room reassignment MIP model (Xpress solver)
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── utilization.py              # HEFCE utilisation metrics (frequency/occupancy/combined)
│   │   ├── travel.py                   # Student travel impact (week-aware)
│   │   └── clash_detection.py          # Student/room conflict detection
│   │
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                    # Heatmaps, bar charts, comparison plots
│
│  ── Output results (generated) ──
├── results/
│   ├── filter/                             # filter_holyrood_students.py output
│   │   ├── holyrood_gt_transitions.csv     # Week-aware transition records
│   │   └── holyrood_gt_student_ids.csv     # Affected student IDs
│   │
│   ├── q1/                                 # run_q1.py output
│   │   ├── stage2_unplaced_events.csv      # Stage 2: unplaced events list
│   │   ├── heatmap_before.png              # Timeslot heatmap before reassignment
│   │   ├── heatmap_after.png               # Timeslot heatmap after reassignment
│   │   ├── holyrood_gt_transitions.csv     # Copy for travel analysis
│   │   ├── holyrood_gt_student_ids.csv     # Copy for travel analysis
│   │   ├── holyrood_gt_students.csv        # Affected students summary
│   │   │
│   │   ├── scenario_1a/                    # Scenario 1a: Central only
│   │   │   ├── assignments.csv
│   │   │   ├── unplaced_events.csv
│   │   │   ├── travel_impact.csv
│   │   │   └── capacity_waste.png
│   │   │
│   │   └── scenario_1b/                    # Scenario 1b: Central + Lauriston + NC
│   │       ├── assignments.csv
│   │       ├── unplaced_events.csv
│   │       ├── travel_impact.csv
│   │       ├── capacity_waste.png
│   │       ├── travel_boxplot.png
│   │       └── travel_change.png
│   │
│   └── baseline/                           # baseline_analysis.py output
│       ├── data_overview.csv
│       ├── rooms_by_campus.csv
│       ├── gt_rooms_by_campus.csv
│       ├── room_type_by_campus.csv
│       ├── specialist_type_by_campus.csv
│       ├── events_by_type.csv
│       ├── events_by_campus.csv
│       ├── duration_distribution.csv
│       ├── holyrood_gt_by_type.csv
│       ├── holyrood_gt_room_types.csv
│       ├── holyrood_gt_roomtype_by_eventtype.csv
│       ├── capacity_waste_by_campus.csv
│       ├── holyrood_gt_waste_by_roomtype.csv
│       ├── room_type_demand_supply.csv
│       ├── gt_utilization_by_campus.csv
│       ├── gt_room_utilization_full.csv
│       ├── timeslot_heatmap_all.csv
│       ├── timeslot_heatmap_holyrood.csv
│       ├── timeslot_heatmap_central.csv
│       ├── travel_time_matrix.csv
│       ├── event_size_distribution.png
│       ├── capacity_waste_distribution.png
│       ├── events_by_campus.png
│       ├── room_type_comparison.png
│       └── timeslot_heatmap.png
```


