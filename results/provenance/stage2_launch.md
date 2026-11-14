# Phase 3 stage 2: launch provenance

- Launched: 2026-10-28 21:55:45 (scripts/phase3_stage2.py; first line of logs/phase3_stage2.log)
- Launch commit: 6bb65f5ded6718bf36a4a4754679803e00ee24bf (committed 21:55:45, immediately before launch)
- Tree at launch: clean. The first stage-2 train.py run logged git_dirty=False at commit 6bb65f5.
- Tracked code/config diff vs the launch commit, excluding results/: empty.
  File: results/provenance/stage2_launch_code.patch (0 bytes), SHA-256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- Training code fingerprint (train.py, src/, configs/, scripts/phase3_stage2.py):
  working tree 074acf7bab0b4b79763db4573762fd82a7b6acbf326d81e296f6f7745884c847
  commit 6bb65f5 074acf7bab0b4b79763db4573762fd82a7b6acbf326d81e296f6f7745884c847
  (identical means the running code is exactly the launch commit)

## Why later records say git_dirty=True

After the first command, every stage-2 record has git_dirty=True. The runs
themselves rewrite tracked result files: the stage-1 records in results/runs/,
the ensembles in results/ensembles/ and the CSVs in results/metrics/ get their
test metrics added. src/utils/provenance.git_dirty counts any modified tracked
file, results included. No code file was modified (see the empty patch above).
A later fix should exclude results/ from the dirty check.

## Commits recorded in stage-2 records

Analysis scripts (stats.py, final_test_report.py, compare_duplicate_neural.py,
test_stats.py) were committed during the run as 90fd2ba (new files only, not
imported by train.py), followed by two provenance commits (74ac8b4 and the one
that adds this text). No further commits are made until stage 2 ends.
Stage-2 records therefore carry one of 6bb65f5, 90fd2ba, 74ac8b4 or the
commit containing this note. None changes training code:
  git diff --stat 6bb65f5 <commit> -- train.py src configs scripts/phase3_stage2.py
shows only the new src/evaluation/stats.py. The stage-2 report lists the exact
commits found in the records and repeats this check.

## Result (checked after stage 2 finished, 2026-10-29 03:30)

Commits found in stage-2 records: 6bb65f5 (60 seed-1 test evaluations and
early seed-2/3 records) and 70be18d (144 later records). In both, the training
path (train.py, src/, configs/, scripts/phase3_stage2.py) differs from 6bb65f5
only by the new src/evaluation/stats.py, which train.py does not import.
