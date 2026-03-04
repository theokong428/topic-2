"""
Q1 SPACE: Improved Room Reassignment Model (Stage 1) using Xpress.

Stage 1 reassigns Holyrood GT events to candidate rooms with FIXED timeslots.

Improvements over baseline:
  1. Room Type matching soft constraint (W3=50)
  2. Tiered capacity waste with τ_e buffer (W2=1)
  3. Gap-aware travel penalty for fixed/coupled transitions (α=γ=0.1)
  4. Travel feasibility hard constraint C5 for coupled transitions
  5. Pre-filtering F(e) includes travel feasibility with fixed neighbours

Mathematical formulation: see model_formulations.tex, Section 1.
"""

import os
import xpress as xp
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from itertools import combinations
from collections import defaultdict
import math

# Initialize Xpress solver license. Prefer XPAUTH_PATH env var; otherwise use platform default.
_license_path = os.environ.get("XPAUTH_PATH")
if _license_path:
    xp.init(_license_path)
else:
    xp.init()  # Auto-detects default path (e.g. Windows: C:\xpressmp\bin\xpauth.xpr)


class SpaceScenarioModel:
    """
    Improved MIP model for reassigning Holyrood GT events to other campus rooms.

    Objective (5 terms):
      (a) W1 × Σu_e                    — unplaced penalty
      (b) W2 × Σx_{e,r}·ω_{e,r}       — tiered capacity waste
      (c) W3 × Σx_{e,r}·(1-Match)     — Room Type mismatch
      (d) α × fixed-neighbour travel   — fixed-neighbour gap-aware travel
      (e) γ × coupled travel penalty   — coupled gap-aware travel penalty

    Constraints:
      C1: Assignment
      C2: Room-time conflict
      C3: Campus linking
      C4: McCormick linearisation
      C5: Travel feasibility (hard, with β buffer)
    """

    # Objective weights (model_formulations.tex Section 1.6)
    W1 = 10_000   # Unplaced event penalty
    W2 = 1         # Tiered waste per excess seat
    W3 = 50        # Room Type mismatch penalty
    ALPHA = 0.1    # Fixed-neighbour travel coefficient
    GAMMA = 0.1    # Coupled travel coefficient
    BETA = 10      # Travel buffer (minutes)

    def __init__(
        self,
        displaced_events: pd.DataFrame,
        candidate_rooms: pd.DataFrame,
        existing_events: pd.DataFrame,
        travel_dict: Dict[Tuple[str, str], int],
        rooms_full: pd.DataFrame = None,
        transitions_df: pd.DataFrame = None,
    ):
        """
        Parameters
        ----------
        displaced_events : DataFrame
            Events from Holyrood GT rooms that need reassignment.
        candidate_rooms : DataFrame
            Available rooms (Central GT, optionally + Lauriston/NC).
        existing_events : DataFrame
            Events already scheduled in candidate rooms (for conflict detection).
        travel_dict : dict
            (campus_from, campus_to) -> travel_minutes.
        rooms_full : DataFrame, optional
            Full rooms inventory (for Room Type lookup of original rooms).
            If None, Room Type matching is disabled.
        transitions_df : DataFrame, optional
            Week-aware transition records from filter_holyrood_students.py.
            Columns: StudentID, HolyroodEventID, Day, Week, HolyroodCampus,
                HolyroodStart, HolyroodEnd, HolyroodDuration,
                PrevEventID, PrevCampus, PrevEnd,
                NextEventID, NextCampus, NextStart.
            If None, travel optimisation is disabled.
        """
        # Reset index for contiguous integer indexing
        self.displaced = displaced_events.reset_index(drop=True)
        self.rooms = candidate_rooms.reset_index(drop=True)
        self.existing = existing_events.reset_index(drop=True)
        self.travel_dict = travel_dict
        self.rooms_full = rooms_full
        self.transitions_df = transitions_df

        # Model and variable storage
        self.model: Optional[xp.problem] = None
        self.x = {}          # x[e_idx, r_idx]: binary assignment
        self.unplaced = {}   # u[e_idx]: binary unplaced indicator
        self.y = {}          # y[e_idx, c]: continuous campus indicator
        self.z = {}          # z[(i_idx, j_idx, a, b)]: McCormick linearisation
        self.solution = None

        # Precomputed data (populated in build_model)
        self.feasible = {}       # feasible (e, r) pairs
        self.n_events = 0
        self.n_rooms = 0
        self.campuses = []       # candidate campus name list
        self.room_campus = {}    # r_idx -> campus name
        self.event_room_type = {}   # e_idx -> Room Type string
        self.room_room_type = {}    # r_idx -> Room Type string
        self.tau = {}            # e_idx -> capacity buffer τ_e
        self.omega = {}          # (e_idx, r_idx) -> tiered waste ω_{e,r}
        self.fixed_transitions = []    # [(e_idx, c_k, n_ek, G_ek), ...]
        self.coupled_transitions = []  # [(i_idx, j_idx, n_ij, G_ij), ...]

    # Pre-computation helpers
    # ══════════════════════════════════════════════════════════════════

    def _precompute_conflicts(self) -> dict:
        """
        Precompute existing room occupancy for conflict checking.
        Returns: room_id -> [(day, hour, minute, dur, weeks), ...]
        """
        occupied = {}
        for _, row in self.existing.iterrows():
            rid = row["Room"]
            day, hour, minute = row["Day"], row["Hour"], row["Minute"]
            dur = row["Duration (minutes)"]
            weeks = row["WeekList"]
            if rid not in occupied:
                occupied[rid] = []
            occupied[rid].append((day, hour, minute, dur, weeks))
        return occupied

    def _events_conflict_with_room_schedule(
        self, e_day, e_hour, e_min, e_dur, e_weeks, room_schedule
    ) -> bool:
        """
        Check if a displaced event conflicts with any existing event in a room.
        Conflict condition (all must hold): same day, overlapping weeks, overlapping time.
        """
        e_start = e_hour + e_min / 60
        e_end = e_start + e_dur / 60
        for (day, hour, minute, dur, weeks) in room_schedule:
            if day != e_day:
                continue
            if not set(e_weeks) & set(weeks):
                continue
            s_start = hour + minute / 60
            s_end = s_start + dur / 60
            if e_start < s_end and s_start < e_end:
                return True
        return False

    def _displaced_pair_conflict(self, i: int, j: int) -> bool:
        """
        Check if two displaced events conflict in time (same day + overlapping weeks + time).
        """
        ei = self.displaced.iloc[i]
        ej = self.displaced.iloc[j]
        if ei["Day"] != ej["Day"]:
            return False
        if not set(ei["WeekList"]) & set(ej["WeekList"]):
            return False
        si = ei["Hour"] + ei["Minute"] / 60
        ei_end = si + ei["Duration (minutes)"] / 60
        sj = ej["Hour"] + ej["Minute"] / 60
        ej_end = sj + ej["Duration (minutes)"] / 60
        return si < ej_end and sj < ei_end

    def _build_room_type_maps(self):
        """
        Build Room Type maps for events (from original rooms) and candidate rooms.
        Room Type = physical layout (Classroom Style, Lecture Theatre, etc.);
        Specialist room type = functional (General Teaching, Laboratory, etc.).
        """
        # Build room_id -> Room Type lookup from full rooms inventory
        room_type_lookup = {}
        if self.rooms_full is not None and "Room Type" in self.rooms_full.columns:
            for _, row in self.rooms_full.iterrows():
                rt = row.get("Room Type", "")
                if pd.isna(rt):
                    rt = "Unknown"
                room_type_lookup[row["Id"]] = rt

        # Event Room Type: look up the original room's Room Type
        for e_idx in range(self.n_events):
            ev = self.displaced.iloc[e_idx]
            orig_room = ev.get("Room", "")
            self.event_room_type[e_idx] = room_type_lookup.get(orig_room, "Unknown")

        # Candidate room Room Type
        for r_idx in range(self.n_rooms):
            rm = self.rooms.iloc[r_idx]
            rt = rm.get("Room Type", "")
            if pd.isna(rt):
                rt = "Unknown"
            self.room_room_type[r_idx] = rt

    def _build_campus_map(self):
        """
        Build room index -> campus mapping and collect campus set.
        """
        campus_set = set()
        for r_idx in range(self.n_rooms):
            c = self.rooms.iloc[r_idx]["Campus"]
            self.room_campus[r_idx] = c
            campus_set.add(c)
        self.campuses = sorted(campus_set)

    def _compute_tiered_waste(self):
        """
        Precompute tiered capacity waste: τ_e = max(⌈0.2×Size_e⌉, 20),
        ω_{e,r} = max(0, (Cap_r - Size_e) - τ_e).
        Three zones: insufficient (excluded by F(e)), acceptable buffer (ω=0),
        excess waste (ω>0, penalised).
        """
        for e_idx in range(self.n_events):
            size = int(self.displaced.iloc[e_idx]["Event Size"])
            self.tau[e_idx] = max(math.ceil(0.2 * size), 20)

        for (e_idx, r_idx) in self.feasible:
            size = int(self.displaced.iloc[e_idx]["Event Size"])
            cap = int(self.rooms.iloc[r_idx]["Capacity"])
            excess = cap - size  # F(e) guarantees >= 0
            self.omega[(e_idx, r_idx)] = max(0, excess - self.tau[e_idx])

    def _build_transitions(self):
        """
        Parse transitions CSV into fixed and coupled transition lists.
        Fixed: neighbour not in displaced set -> campus known.
        Coupled: both events displaced -> both campuses are decision variables.
        Aggregation: group by (e, c_k) or (i, j); n = student×week count; G = min gap.
        """
        if self.transitions_df is None or len(self.transitions_df) == 0:
            print("  No transition data provided; travel optimisation disabled.")
            return

        trans = self.transitions_df

        # Build event_id -> e_idx lookup
        eid_to_idx = {}
        for e_idx in range(self.n_events):
            eid = self.displaced.iloc[e_idx]["Event ID"]
            eid_to_idx[eid] = e_idx
        displaced_eids = set(eid_to_idx.keys())

        # Collect raw transition records (per student×week)
        raw_fixed = []
        raw_coupled = []

        for _, row in trans.iterrows():
            h_eid = row["HolyroodEventID"]
            if h_eid not in eid_to_idx:
                continue
            e_idx = eid_to_idx[h_eid]
            h_start = row["HolyroodStart"]   # fractional hours
            h_end = row["HolyroodEnd"]

            # Previous neighbour
            prev_eid = row.get("PrevEventID")
            if pd.notna(prev_eid) and prev_eid != "":
                prev_end = row["PrevEnd"]
                gap_min = (h_start - prev_end) * 60  # convert to minutes

                if prev_eid in displaced_eids:
                    # Coupled: both events are displaced
                    j_idx = eid_to_idx[prev_eid]
                    pair = (min(e_idx, j_idx), max(e_idx, j_idx))
                    raw_coupled.append((pair[0], pair[1], gap_min))
                else:
                    # Fixed: neighbour campus is known
                    prev_campus = row["PrevCampus"]
                    if pd.notna(prev_campus):
                        raw_fixed.append((e_idx, str(prev_campus), gap_min))

            # Next neighbour
            next_eid = row.get("NextEventID")
            if pd.notna(next_eid) and next_eid != "":
                next_start = row["NextStart"]
                gap_min = (next_start - h_end) * 60  # convert to minutes

                if next_eid in displaced_eids:
                    j_idx = eid_to_idx[next_eid]
                    pair = (min(e_idx, j_idx), max(e_idx, j_idx))
                    raw_coupled.append((pair[0], pair[1], gap_min))
                else:
                    next_campus = row["NextCampus"]
                    if pd.notna(next_campus):
                        raw_fixed.append((e_idx, str(next_campus), gap_min))

        # Aggregate fixed transitions: group by (e_idx, c_k) -> n_ek, G_ek
        fix_agg = defaultdict(lambda: {"count": 0, "min_gap": float("inf")})
        for (e_idx, c_k, gap_min) in raw_fixed:
            key = (e_idx, c_k)
            fix_agg[key]["count"] += 1
            fix_agg[key]["min_gap"] = min(fix_agg[key]["min_gap"], gap_min)

        self.fixed_transitions = [
            (e_idx, c_k, agg["count"], agg["min_gap"])
            for (e_idx, c_k), agg in fix_agg.items()
        ]

        # Aggregate coupled transitions: group by (i_idx, j_idx), canonical order
        coup_agg = defaultdict(lambda: {"count": 0, "min_gap": float("inf")})
        for (i_idx, j_idx, gap_min) in raw_coupled:
            key = (i_idx, j_idx)
            coup_agg[key]["count"] += 1
            coup_agg[key]["min_gap"] = min(coup_agg[key]["min_gap"], gap_min)

        self.coupled_transitions = [
            (i_idx, j_idx, agg["count"], agg["min_gap"])
            for (i_idx, j_idx), agg in coup_agg.items()
        ]

        print(f"  Fixed transitions: {len(self.fixed_transitions)} groups "
              f"({sum(t[2] for t in self.fixed_transitions):,} student×week records)")
        print(f"  Coupled transitions: {len(self.coupled_transitions)} pairs "
              f"({sum(t[2] for t in self.coupled_transitions):,} student×week records)")

        # Count tight coupled pairs (G < d+β); for d=10, threshold = 20 min
        tight = sum(1 for (_, _, _, G) in self.coupled_transitions if G < 20)
        print(f"  Tight coupled pairs (G < d+β=20 min): {tight}")

    def _get_travel_time(self, campus_a: str, campus_b: str) -> float:
        """
        Look up travel time between two campuses (default 60 if not found).
        """
        if campus_a == campus_b:
            return 0
        return self.travel_dict.get(
            (campus_a, campus_b),
            self.travel_dict.get((campus_b, campus_a), 60)
        )

    # Model building
    # ══════════════════════════════════════════════════════════════════

    def build_model(self):
        """
        Build the improved Xpress MIP model.
        Steps: precompute occupancy/campus/RT maps; build transitions;
        enumerate feasible pairs; precompute waste; enumerate conflicts;
        create variables x,u,y,z; add constraints C1–C5; set objective.
        """
        print("=" * 60)
        print("Building improved Q1 SPACE model...")
        print("=" * 60)

        self.n_events = len(self.displaced)
        self.n_rooms = len(self.rooms)

        # Step 1: Precompute basics
        print("\n[Step 1] Precomputing room occupancy, campus map, Room Type map...")
        room_occupied = self._precompute_conflicts()
        self._build_campus_map()
        self._build_room_type_maps()
        print(f"  Campuses: {self.campuses}")

        # Step 2: Build transitions
        print("\n[Step 2] Building transition data...")
        self._build_transitions()

        # Step 3: Enumerate feasible (event, room) pairs
        # F(e) pre-filtering: capacity + room-time conflict + fixed-neighbour travel
        print("\n[Step 3] Enumerating feasible (event, room) pairs with travel pre-filtering...")

        # Build per-event fixed-transition constraints for pre-filtering
        event_fixed_constraints = defaultdict(list)
        for (e_idx, c_k, n_ek, G_ek) in self.fixed_transitions:
            event_fixed_constraints[e_idx].append((c_k, G_ek))

        feasible = {}
        travel_filtered_count = 0

        for e_idx in range(self.n_events):
            ev = self.displaced.iloc[e_idx]
            for r_idx in range(self.n_rooms):
                rm = self.rooms.iloc[r_idx]

                # Condition 1: capacity sufficient
                if rm["Capacity"] < ev["Event Size"]:
                    continue

                # Condition 2: no room-time conflict
                rid = rm["Id"]
                if rid in room_occupied:
                    if self._events_conflict_with_room_schedule(
                        ev["Day"], ev["Hour"], ev["Minute"],
                        ev["Duration (minutes)"], ev["WeekList"],
                        room_occupied[rid],
                    ):
                        continue

                # Condition 3: fixed-neighbour travel feasibility (with buffer β)
                # d(c_k, campus(r)) + β ≤ G_ek for all fixed transitions of event e
                r_campus = self.room_campus[r_idx]
                travel_ok = True
                for (c_k, G_ek) in event_fixed_constraints.get(e_idx, []):
                    d_travel = self._get_travel_time(c_k, r_campus)
                    if d_travel + self.BETA > G_ek:
                        travel_ok = False
                        travel_filtered_count += 1
                        break
                if not travel_ok:
                    continue

                feasible[(e_idx, r_idx)] = True

        self.feasible = feasible
        print(f"  Events to reassign: {self.n_events}")
        print(f"  Candidate rooms: {self.n_rooms}")
        print(f"  Feasible (event, room) pairs: {len(feasible)}")
        print(f"  Pairs filtered by travel feasibility: {travel_filtered_count}")

        # Check events with no feasible room
        events_no_room = sum(
            1 for e in range(self.n_events)
            if not any((e, r) in feasible for r in range(self.n_rooms))
        )
        if events_no_room > 0:
            print(f"  WARNING: {events_no_room} events have NO feasible room (will be unplaced)")

        # Step 4: Precompute tiered waste
        print("\n[Step 4] Precomputing tiered capacity waste coefficients...")
        self._compute_tiered_waste()

        # Step 5: Enumerate conflicting event pairs
        print("\n[Step 5] Enumerating conflicting displaced event pairs...")
        conflict_pairs = []
        for i, j in combinations(range(self.n_events), 2):
            if self._displaced_pair_conflict(i, j):
                conflict_pairs.append((i, j))
        print(f"  Conflicting displaced event pairs: {len(conflict_pairs)}")

        # Build Xpress Model
        m = xp.problem("Q1_Space_Improved")

        # Step 6a: x[e,r] binary assignment variables
        print("\n[Step 6] Creating decision variables...")
        x = {}
        for (e_idx, r_idx) in feasible:
            x[e_idx, r_idx] = m.addVariable(
                vartype=xp.binary, name=f"x_{e_idx}_{r_idx}")

        # Step 6b: u[e] unplaced indicator
        u = {}
        for e_idx in range(self.n_events):
            u[e_idx] = m.addVariable(
                vartype=xp.binary, name=f"u_{e_idx}")

        # Step 6c: y[e,c] campus indicator (continuous; binary via C3)
        # Campus indicator (continuous; automatically binary via C3 when x is binary)
        y = {}
        for e_idx in range(self.n_events):
            for c in self.campuses:
                y[e_idx, c] = m.addVariable(
                    vartype=xp.continuous, lb=0, ub=1,
                    name=f"y_{e_idx}_{c[:3]}")

        # Step 6d: z[(i,j,a,b)] McCormick linearisation (coupled only, a!=b)
        # Only for coupled transitions, only for cross-campus pairs a != b
        z = {}
        if self.coupled_transitions:
            for (i_idx, j_idx, n_ij, G_ij) in self.coupled_transitions:
                for a in self.campuses:
                    for b in self.campuses:
                        if a != b:
                            z[(i_idx, j_idx, a, b)] = m.addVariable(
                                vartype=xp.continuous, lb=0, ub=1,
                                name=f"z_{i_idx}_{j_idx}_{a[:3]}_{b[:3]}")

        print(f"  x variables: {len(x)}")
        print(f"  u variables: {len(u)}")
        print(f"  y variables: {len(y)}")
        print(f"  z variables: {len(z)}")
        print(f"  Total variables: {len(x) + len(u) + len(y) + len(z)}")

        # Constraints
        print("\n[Step 7] Adding constraints...")

        # C1: Assignment — each event assigned to exactly one room or unplaced
        # sum_{r in F(e)} x[e,r] + u[e] = 1, for all e
        for e_idx in range(self.n_events):
            feasible_rooms = [r for (e, r) in feasible if e == e_idx]
            if feasible_rooms:
                m.addConstraint(
                    xp.Sum(x[e_idx, r] for r in feasible_rooms) + u[e_idx] == 1
                )
            else:
                m.addConstraint(u[e_idx] == 1)
        print(f"  C1 assignment constraints: {self.n_events}")

        # C2: Room-time conflict — conflicting displaced events cannot share a room
        # x[i,r] + x[j,r] <= 1, for conflicting (i,j), shared feasible r
        n_c2 = 0
        for (i, j) in conflict_pairs:
            for r_idx in range(self.n_rooms):
                if (i, r_idx) in feasible and (j, r_idx) in feasible:
                    m.addConstraint(x[i, r_idx] + x[j, r_idx] <= 1)
                    n_c2 += 1
        print(f"  C2 room-time conflict constraints: {n_c2}")

        # C3: Campus linking — y[e,c]=1 iff event e assigned to campus c
        n_c3 = 0
        for e_idx in range(self.n_events):
            for c in self.campuses:
                rooms_on_c = [r for (e, r) in feasible
                              if e == e_idx and self.room_campus[r] == c]
                if rooms_on_c:
                    m.addConstraint(
                        y[e_idx, c] == xp.Sum(x[e_idx, r] for r in rooms_on_c)
                    )
                else:
                    m.addConstraint(y[e_idx, c] == 0)
                n_c3 += 1
        print(f"  C3 campus linking constraints: {n_c3}")

        # C4: McCormick linearisation for z = y[i,a] × y[j,b]
        n_c4 = 0
        for (i_idx, j_idx, a, b) in z:
            zvar = z[(i_idx, j_idx, a, b)]
            ya = y[i_idx, a]
            yb = y[j_idx, b]
            m.addConstraint(zvar <= ya)            # MC1: z ≤ y_{i,a}
            m.addConstraint(zvar <= yb)            # MC2: z ≤ y_{j,b}
            m.addConstraint(zvar >= ya + yb - 1)   # MC3: z ≥ y_{i,a} + y_{j,b} - 1
            # MC4: z >= 0 handled by lb=0
            n_c4 += 3
        print(f"  C4 McCormick constraints: {n_c4}")

        # C5: Travel feasibility for coupled transitions (with β buffer)
        n_c5 = 0
        for (i_idx, j_idx, n_ij, G_ij) in self.coupled_transitions:
            for a in self.campuses:
                for b in self.campuses:
                    if a != b:
                        d_ab = self._get_travel_time(a, b)
                        # φ_{ij,a,b} = 1 when d(a,b) + β > G_ij
                        if d_ab + self.BETA > G_ij:
                            m.addConstraint(y[i_idx, a] + y[j_idx, b] <= 1)
                            n_c5 += 1
        print(f"  C5 travel feasibility constraints (β={self.BETA}): {n_c5}")

        total_constraints = self.n_events + n_c2 + n_c3 + n_c4 + n_c5
        print(f"  Total constraints: {total_constraints}")

        # Objective Function (5 terms)
        print("\n[Step 8] Setting objective function...")

        # Term (a): Unplaced penalty
        # W1 × Σ u_e
        obj_unplaced = self.W1 * xp.Sum(u[e_idx] for e_idx in range(self.n_events))

        # Term (b): Tiered capacity waste
        # W2 × Σ x_{e,r} · ω_{e,r}
        obj_waste = self.W2 * xp.Sum(
            x[e_idx, r_idx] * self.omega[(e_idx, r_idx)]
            for (e_idx, r_idx) in feasible
        )

        # Term (c): Room Type mismatch
        # W3 × Σ x_{e,r} · (1 - Match(e,r))
        # Match(e,r) = 1 if RT(e) == RT(r), else 0
        obj_type_terms = []
        for (e_idx, r_idx) in feasible:
            ev_rt = self.event_room_type.get(e_idx, "Unknown")
            rm_rt = self.room_room_type.get(r_idx, "Unknown")
            if ev_rt != rm_rt:
                obj_type_terms.append(x[e_idx, r_idx])
        obj_type = self.W3 * xp.Sum(obj_type_terms) if obj_type_terms else 0

        # Term (d): Fixed-neighbour gap-aware travel
        # α × Σ_{(e,k,n_ek)} n_ek × Σ_{r in F(e)} x[e,r] · δ^fix_{e,k,r}
        # δ^fix_{e,k,r} = max(0, d(c_k, campus(r)) + β - G_ek)
        obj_fix_terms = []
        for (e_idx, c_k, n_ek, G_ek) in self.fixed_transitions:
            for r_idx in [r for (e, r) in feasible if e == e_idx]:
                r_campus = self.room_campus[r_idx]
                d_travel = self._get_travel_time(c_k, r_campus)
                delta_fix = max(0, d_travel + self.BETA - G_ek)
                if delta_fix > 0:
                    obj_fix_terms.append(n_ek * delta_fix * x[e_idx, r_idx])
        obj_fix_travel = self.ALPHA * xp.Sum(obj_fix_terms) if obj_fix_terms else 0

        # Term (e): Coupled gap-aware travel penalty
        # γ × Σ_{(i,j,n_ij)} n_ij × Σ_{a≠b} δ^coup_{ij,a,b} · z[i,j,a,b]
        # δ^coup_{ij,a,b} = max(0, d(a,b) + β - G_ij)
        obj_coup_terms = []
        for (i_idx, j_idx, n_ij, G_ij) in self.coupled_transitions:
            for a in self.campuses:
                for b in self.campuses:
                    if a != b:
                        d_ab = self._get_travel_time(a, b)
                        delta_coup = max(0, d_ab + self.BETA - G_ij)
                        if delta_coup > 0:
                            zvar = z.get((i_idx, j_idx, a, b))
                            if zvar is not None:
                                obj_coup_terms.append(n_ij * delta_coup * zvar)
        obj_coup_travel = self.GAMMA * xp.Sum(obj_coup_terms) if obj_coup_terms else 0

        # Total objective
        total_obj = obj_unplaced + obj_waste + obj_type + obj_fix_travel + obj_coup_travel
        m.setObjective(total_obj, sense=xp.minimize)

        print(f"  Objective: W1={self.W1}×unplaced + W2={self.W2}×tiered_waste"
              f" + W3={self.W3}×type_mismatch"
              f" + α={self.ALPHA}×fix_travel + γ={self.GAMMA}×coup_travel"
              f" (β={self.BETA} buffer in δ coefficients)")
        print(f"  Fixed travel terms: {len(obj_fix_terms)}")
        print(f"  Coupled travel terms: {len(obj_coup_terms)}")

        # Store model and variable references
        self.model = m
        self.x = x
        self.unplaced = u
        self.y = y
        self.z = z
        print("\nModel built successfully.")

    # Solving
    # ══════════════════════════════════════════════════════════════════

    def solve(self, time_limit: int = 300, mip_gap: float = 0.01):
        """
        Solve the model.
        Parameters: time_limit (seconds, default 300), mip_gap (default 0.01).
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        print(f"\nSolving (time limit={time_limit}s, gap={mip_gap})...")
        self.model.controls.maxtime = time_limit
        self.model.controls.miprelstop = mip_gap
        self.model.solve()

        # Get solve status
        solve_status = self.model.attributes.solvestatus
        sol_status = self.model.attributes.solstatus
        print(f"  Solve status: {solve_status}, Sol status: {sol_status}")

        if self.model.attributes.mipsols > 0:
            obj_val = self.model.attributes.objval
            print(f"  Objective value: {obj_val:.1f}")
            self._extract_solution()
        else:
            print("  No feasible solution found.")
            self.solution = None

        return str(solve_status)

    # Solution extraction
    # ══════════════════════════════════════════════════════════════════

    def _extract_solution(self):
        """
        Extract solution: event-to-room assignment with detailed metrics.
        """
        assignments = []
        unplaced_events = []

        for e_idx in range(self.n_events):
            u_val = self.model.getSolution(self.unplaced[e_idx])
            if u_val > 0.5:
                # Event is unplaced
                unplaced_events.append(self.displaced.iloc[e_idx])
                continue

            # Find the assigned room
            for r_idx in range(self.n_rooms):
                if (e_idx, r_idx) in self.feasible:
                    val = self.model.getSolution(self.x[e_idx, r_idx])
                    if val > 0.5:
                        ev = self.displaced.iloc[e_idx]
                        rm = self.rooms.iloc[r_idx]

                        # Room Type info
                        ev_rt = self.event_room_type.get(e_idx, "Unknown")
                        rm_rt = self.room_room_type.get(r_idx, "Unknown")

                        assignments.append({
                            "Event ID": ev["Event ID"],
                            "Event Name": ev.get("Event Name", ""),
                            "Event Size": ev["Event Size"],
                            "Day": ev["Day"],
                            "Timeslot": ev["Timeslot"],
                            "Duration": ev["Duration (minutes)"],
                            "Original Room": ev["Room"],
                            "Original Campus": ev["Campus"],
                            "New Room": rm["Id"],
                            "New Room Capacity": rm["Capacity"],
                            "New Campus": rm["Campus"],
                            "Capacity Waste": rm["Capacity"] - ev["Event Size"],
                            "Tiered Waste": self.omega.get((e_idx, r_idx), 0),
                            "Original Room Type": ev_rt,
                            "New Room Type": rm_rt,
                            "Room Type Match": 1 if ev_rt == rm_rt else 0,
                        })
                        break  # At most one room per event

        # Build solution summary
        self.solution = {
            "assignments": pd.DataFrame(assignments),
            "unplaced": pd.DataFrame(unplaced_events),
            "total_events": self.n_events,
            "placed": len(assignments),
            "unplaced_count": len(unplaced_events),
        }

        # Print summary
        print(f"\n  ── Solution Summary ──")
        print(f"  Placed: {self.solution['placed']}/{self.n_events} "
              f"({self.solution['placed']/max(self.n_events,1):.1%})")
        print(f"  Unplaced: {self.solution['unplaced_count']}/{self.n_events}")

        if len(assignments) > 0:
            asg = self.solution["assignments"]
            rt_match = asg["Room Type Match"].mean()
            avg_waste = asg["Capacity Waste"].mean()
            avg_tiered = asg["Tiered Waste"].mean()
            median_waste = asg["Capacity Waste"].median()
            print(f"  Room Type match rate: {rt_match:.1%}")
            print(f"  Avg capacity waste: {avg_waste:.1f} seats "
                  f"(median: {median_waste:.0f})")
            print(f"  Avg tiered waste: {avg_tiered:.1f} seats "
                  f"(beyond τ buffer)")
            print(f"  Rooms used: {asg['New Room'].nunique()}")

            # Per-campus breakdown
            campus_counts = asg["New Campus"].value_counts()
            for campus, count in campus_counts.items():
                print(f"    {campus}: {count} events")

    def get_results_summary(self) -> dict:
        """
        Return a summary dict of results.
        """
        if self.solution is None:
            return {"status": "no_solution"}

        asg = self.solution["assignments"]
        summary = {
            "total_displaced": self.solution["total_events"],
            "placed": self.solution["placed"],
            "unplaced": self.solution["unplaced_count"],
            "placement_rate": self.solution["placed"] / max(self.solution["total_events"], 1),
            "avg_capacity_waste": asg["Capacity Waste"].mean() if len(asg) > 0 else None,
            "median_capacity_waste": asg["Capacity Waste"].median() if len(asg) > 0 else None,
            "rooms_used": asg["New Room"].nunique() if len(asg) > 0 else 0,
        }
        if len(asg) > 0 and "Room Type Match" in asg.columns:
            summary["room_type_match_rate"] = asg["Room Type Match"].mean()
            summary["avg_tiered_waste"] = asg["Tiered Waste"].mean()
        return summary

    def export_results(self, output_dir: str = "results/q1"):
        """
        Save results to CSV files.
        """
        if self.solution is None:
            print("No solution to export.")
            return
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.solution["assignments"].to_csv(out / "assignments.csv", index=False)
        self.solution["unplaced"].to_csv(out / "unplaced_events.csv", index=False)
        print(f"Results exported to {out}/")
