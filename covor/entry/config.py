"""Frozen parameters of the entry-map stage (DESIGN_entry_map.md §4).

One rule governs this file, the same one covor/synth/config.py states: anything
that already has a value elsewhere is NOT redefined here. The occupancy grid
resolution, the 3D classification thresholds and the fusion parameters are
imported or passed in; what lives here is only what the entry map itself has to
decide -- human body dimensions, the two bands, and the clearance convention.

Nothing here is tuned against a result. Three of the values were not fixed by
the design document and are marked as such below, with the reason for the value
chosen.
"""
from dataclasses import dataclass

# 2D band labels. Same numbering as covor.synth.mesh_gt so the two label spaces
# can never be confused by a reader: UNKNOWN is 0 in both.
UNKNOWN, FREE, OCCUPIED = 0, 1, 2

# Width grades (DESIGN §4). These are anthropometry, not a function of w: 0.45 m
# is half-shoulder plus margin for walking forward, 0.30 m is half chest depth
# for sidling through. `w` parameterises PASSABILITY (r = w/2 + k*sigma), which
# is a different question -- "does a body of this width fit" vs "how does a
# rescuer get through". Both are reported.
BLOCKED, NARROW, WALK = 0, 1, 2

# Merged classes across the two bands (DESIGN §1, the entry_grid.npz label set).
# FREE_BLOCKED is the case the design's five-value list leaves implicit: band-free
# but too tight for even a sideways pass in either band.
CLS_UNKNOWN, CLS_OCCUPIED = 0, 1
CLS_FREE_BLOCKED, CLS_FREE_CRAWL_ONLY, CLS_FREE_NARROW, CLS_FREE_WALK = 2, 3, 4, 5

CLASS_NAMES = {
    CLS_UNKNOWN: "unknown", CLS_OCCUPIED: "occupied",
    CLS_FREE_BLOCKED: "free_blocked", CLS_FREE_CRAWL_ONLY: "free_crawl_only",
    CLS_FREE_NARROW: "free_narrow", CLS_FREE_WALK: "free_walk",
}


@dataclass(frozen=True)
class EntryCfg:
    # --- grid -------------------------------------------------------------
    res: float = 0.10            # judgement grid; = OccCfg.resolution, NOT a
                                 # free choice (PREREG_RESOLUTION.md: metrics are
                                 # only ever compared inside one resolution)

    # --- human dimensions (DESIGN §4) -------------------------------------
    w: float = 0.70              # shoulder width of a kitted adult
    clear_walk: float = 0.45     # clearance >= this -> walk
    clear_narrow: float = 0.30   # clearance in [this, clear_walk) -> narrow

    # --- bands (DESIGN §2-1) ----------------------------------------------
    z_min: float = 0.10          # band starts one voxel above the floor plane
    H_walk: float = 1.90
    H_crawl: float = 0.90
    # NOTE (unresolved, reported not patched): the design's own regression case
    # "0.72 m desk, empty underneath -> crawl band passable" cannot hold at
    # H_crawl = 0.90, because the desktop at z = 0.72 lies INSIDE [0.10, 0.90]
    # and the band rule makes that column occupied. It holds at H_crawl <= 0.70.
    # The design value is kept here and the contradiction is pinned by value in
    # tests/test_entry_map.py::test_desk_band_membership_by_height rather than
    # silently resolved in either direction.

    # --- pose uncertainty as geometric margin (DESIGN §3-3) ---------------
    k_sigma: float = 1.0         # r = w/2 + k*sigma_xy. FIXED, never tuned.
                                 # Part 2 runs at k_sigma = 0; PREREG_entry_sigma.md
                                 # is what licenses k_sigma = 1.

    # --- 3D cleanup (DESIGN §2-0, §3-5) -----------------------------------
    close_iter: int = 1          # occupied-only closing; fills wall pinholes
    close_structure: str = "3d"  # "3d"  = 6-neighbourhood in 3D (DESIGN's reading
                                 #         of "3D cleanup", and the default)
                                 # "2d"  = 4-neighbourhood inside each z slice
                                 # MEASURED SIDE EFFECT of "3d": the floor-wall
                                 # junction is a 90 deg concave corner, so closing
                                 # fills the ring of free cells sitting on the
                                 # floor against every wall. The band projection
                                 # takes ANY occupied cell in the band, so that
                                 # ring thickens EVERY wall by one cell in plan
                                 # view and costs 0.10 m of clearance along all of
                                 # them. On room909's GT walk band: free
                                 # 1556 -> 1484, occupied 4430 -> 4583. A 0.6 /
                                 # 0.8 / 1.0 m door each drop one width grade.
                                 # "2d" closes wall pinholes without touching the
                                 # floor junction. Reported, not silently chosen:
                                 # the design value stands until it is decided.
    cube_min_voxels: int = 1     # occupied components SMALLER than this are
                                 # demoted to unknown. 1 = filter off.
                                 # NOT a "current value": no floating-cube filter
                                 # has ever existed in this pipeline
                                 # (scripts/gt_pose_control.py only COUNTS them).
                                 # Off is the default because demoting occupied
                                 # evidence is the direction that manufactures
                                 # false-passable cells, and RESULTS_SUMMARY §9-4
                                 # says report the sweep instead of picking a
                                 # threshold. Part 2 sweeps it.
    cube_connectivity: int = 3   # 26-neighbourhood for "is this blob attached to
                                 # anything": the most generous notion of
                                 # attachment, so the fewest blobs count as
                                 # floating.

    # --- routing (DESIGN §2-4), used in Part 4 -----------------------------
    lambda_route: float = 0.5

    # --- render / reporting -------------------------------------------------
    n_min_obs: int = 3           # columns observed by fewer frames than this are
                                 # drawn with a dotted outline (DESIGN §5). A
                                 # display threshold only: it enters no metric
                                 # and no passability decision.

    # --- entry point (DESIGN §8: "user-specified first") -------------------
    entry_xy: tuple = None       # (x, y) in metres, or None for the default rule
                                 # in reach.default_entry: the passable cell with
                                 # the smallest x. room909's scan is open around
                                 # x < -4 m (covor/synth/mesh_gt.build_gt_grid),
                                 # so that is the cell nearest the real opening.
                                 # There is no door in this scan to read a GT
                                 # entry point from, and no drone start position
                                 # is passable in the walk band at w = 0.70.

    def half_width(self):
        """w/2 -- the base of r before the sigma margin."""
        return 0.5 * self.w


# CLEARANCE CONVENTION (fixed here, before any number is produced).
#
#   clearance(cell) = max(EDT(cell) - 0.5, 0) * res
#
# EDT is scipy's exact Euclidean transform in CELL units, i.e. the distance from
# this cell's centre to the nearest obstacle cell's CENTRE. Half a voxel of that
# is inside the obstacle, so the distance to the obstacle's SURFACE -- which is
# what a body has to fit through -- is half a voxel less.
#
# This is not cosmetic: it decides the design's own regression case. A doorway of
# n free cells has a best interior cell at EDT = ceil(n/2), so
#
#   door   raw EDT*res   grade        (EDT-0.5)*res   grade
#   0.6 m     0.30       narrow           0.25        blocked
#   0.8 m     0.40       narrow           0.35        narrow
#   1.0 m     0.50       walk             0.45        walk
#
# DESIGN §6 requires blocked / narrow / walk at w = 0.70. Only the corrected form
# gives that with the §4 grade thresholds unchanged, so the convention is fixed
# rather than the thresholds. The 1.0 m door lands exactly on clear_walk, which
# is why the comparison is >= and why the test asserts the VALUE 0.45, not the
# grade alone.
