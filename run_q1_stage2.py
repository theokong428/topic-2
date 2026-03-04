#!/usr/bin/env python3
"""
Q1 SPACE Stage 2: Timeslot Reassignment for Unplaced Events (TODO)

Stage 1 reassigns rooms with fixed timeslots; some events remain unplaced (see results/q1/stage2_unplaced_events.csv).
Stage 2 will jointly reassign timeslot + room for these unplaced events.

Input:
  - results/q1/stage2_unplaced_events.csv  (Stage 1 unplaced events)
  - Candidate room pool (same as Stage 1: Central + Lauriston + New College)
  - Student enrolment data (for clash detection)

Decision:
  - x[e, r, t] = 1: event e assigned to room r, timeslot t (timeslot is variable, unlike Stage 1)

Constraints:
  - No student clash (same student cannot have two events in same timeslot)
  - No room clash
  - Capacity sufficient
  - Travel feasibility (with prev/next event transitions)
  - Teaching hours policy (e.g. Wednesday whole-class before 13:00)

Objective:
  - Minimise unplaced count
  - Secondarily minimise capacity waste, room type mismatch

Mathematical formulation: see latex/model_formulations.tex Section 2 (Stage 2).

Usage (when implemented):
    python run_q1_stage2.py
"""

# TODO: Implement Stage 2 model and solve logic
# TODO: Read stage2_unplaced_events.csv
# TODO: Build x[e,r,t] decision variables
# TODO: Add student clash, room clash, travel feasibility constraints
# TODO: Solve and export results

if __name__ == "__main__":
    print("Stage 2 TODO. Unplaced events list: results/q1/stage2_unplaced_events.csv")
