"""Synthetic-observation experiments on a scanned mesh (occupancy GT available).

MILUV provides no occupancy ground truth (RESULTS_SUMMARY.md §10), so IoU /
precision / recall / false-free can only be reported in absolute terms on a
simulated space. This package renders a scanned room mesh into the SAME
observation formats the real pipeline consumes (raw local-frame ``vio.csv``,
MILUV-style ``uwb_range.csv``, depth + sigma_Z per frame), so ``covor.fusion``
and ``covor.occupancy`` run UNMODIFIED on it.

IMPORT ORDER TRAP: open3d ships its own libtbb and, if it is imported before
gtsam, gtsam's import fails with an undefined TBB symbol. Every module here
imports gtsam (or covor.fusion) first; keep it that way.
"""
