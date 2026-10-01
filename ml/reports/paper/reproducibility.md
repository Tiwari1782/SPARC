# SPARC Reproducibility Report

- **Python Version**: 3.14.0
- **OS**: Windows 11
- **Random Seed**: 42 (fixed everywhere)
- **Execution Sequence**:
  1. `python ml/process_races.py` (Stage 1)
  2. `python ml/build_labels.py` (Stage 2: Adaptive labelling & blind audit)
  3. `python ml/calibrate_teams.py` (Stage 3: Energy calibration)
  4. `python ml/diagnostics.py` (Part A diagnostics)
  5. `python ml/evaluate_paper.py` (Part C-F full suite)
- **Total Suite Runtime**: 1345.0 seconds