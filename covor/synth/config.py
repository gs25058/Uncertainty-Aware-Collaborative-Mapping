"""Frozen parameters of the synthetic experiment.

Two rules govern this file:

  * Anything that already has a value in RESULTS_SUMMARY.md appendix B is NOT
    redefined here -- ``covor.fusion.Cfg`` and ``covor.occupancy.OccCfg`` are
    imported and used as they are. What lives here is only what the real dataset
    supplied and a simulator has to invent: the trajectories, the front-end error
    process, and the UWB observation model.
  * Every invented number is either measured on MILUV (and says so) or is a
    stated design choice. None of them is tuned against a result.
"""
from dataclasses import dataclass, field
import os

ROOT = "/src/gs25058/cr_RNE/covor_slam"
ROBOTS = ("ifo001", "ifo002", "ifo003")
# UWB tag ids per robot (MILUV config/uwb/tags.yaml). data.robot_of_tag maps
# id // 10 -> robot, so these ids must keep that structure.
TAGS = {"ifo001": (10, 11), "ifo002": (20, 21), "ifo003": (30, 31)}


@dataclass
class SynthCfg:
    name: str = "room909"
    # The sequence name's trailing token indexes MILUV's anchor constellation
    # (data.load_anchors is called unconditionally by CoVOR.__init__), so it ends
    # in "_0". No anchor RANGES are generated and anchor_robots=() is passed, so
    # the constellation is never used -- this only keeps the loader happy.
    seq: str = "synth_room909_0"
    res: float = 0.10

    # --- trajectory ---------------------------------------------------------
    rate_hz: float = 20.0        # raw VIO/GT rate; graph nodes at rate/vins_stride
    duration_s: float = 90.0     # task brief: 60-120 s per robot
    speed: float = 0.35          # m/s. MILUV p95 speed is 0.80 m/s (data.py);
                                 # a mapping pass is flown slower than the peak.
    clearance: float = 0.35      # m; drone half-size + margin (= OccCfg.dyn_radius)
    # One cruise height per robot inside the brief's 1.0-1.5 m band, plus a slow
    # +-0.08 m bob. Different heights per robot are deliberate: they give the
    # 3-drone condition vertical diversity the 1-drone condition cannot have.
    heights: tuple = (1.10, 1.25, 1.40)
    bob_amp: float = 0.08
    bob_period_s: float = 11.0
    # Yaw sweeps around the heading so a lawnmower pass still sees the side walls.
    yaw_amp_deg: float = 50.0
    yaw_period_s: float = 12.0
    # GT body tilt: small, so the gravity prior has something non-degenerate to
    # act on without the camera pitching off the walls.
    tilt_amp_deg: float = 1.5
    tilt_period_s: float = 7.0
    row_step: float = 0.9        # m; lawnmower row spacing
    zone_axis: str = "x"         # axis the footprint is split into per-robot
                                 # zones along. "x" is the original (room909)
                                 # behaviour; "y" for a corridor whose long axis
                                 # is y -- see trajectory.zone_split.

    # --- front-end (VIO) error process -------------------------------------
    # See covor.synth.vio for the derivation. sigma_odo_* are appendix B values
    # and are NOT free here: they are the target the process is calibrated to.
    yaw_drift_deg_per_min: tuple = (1.5, -2.5, 4.0)   # per robot; measured VINS
                                                      # range was -4.48..+0.17
    tilt_tau_s: float = 2.0      # OU correlation time of the roll/pitch error

    # --- UWB ----------------------------------------------------------------
    uwb_rate_hz: float = 1.4     # per TAG PAIR; measured on MILUV (1.37-1.43 Hz)
    uwb_sigma: float = 0.05      # m; matches Cfg.range_sigma_floor
    uwb_bias: float = 0.004      # m; MILUV-measured inter-agent bias (+0.0035)
    uwb_nlos: bool = False       # optional: extra positive bias when the mesh
    uwb_nlos_bias: float = 0.30  # blocks the line of sight
    uwb_nlos_sigma: float = 0.15
    use_moment_arm: bool = True  # antenna offsets from MILUV tags.yaml

    # --- misc ---------------------------------------------------------------
    seed: int = 0
    # A non-zero clock offset, in MILUV's own two-field form, so the
    # timeshift_s + timeshift_ns/1e9 path (appendix A-8) is exercised rather
    # than bypassed by a zero.
    timeshift_s: int = 1706823924
    timeshift_ns: int = 255880117

    def outdir(self):
        return os.path.join(ROOT, "results", "synth_" + self.name)

    # Per-seed roots so seeds are independent processes: a shared root would
    # have one worker's --regen overwrite the csv another worker is reading.
    def dataroot(self):
        return os.path.join(self.outdir(), "data_seed%d" % self.seed)

    def vinsroot(self):
        return os.path.join(self.outdir(), "vins_seed%d" % self.seed)

    def gt_voxel(self):
        return os.path.join(self.outdir(), "gt_voxel.npz")

    def mesh_config(self):
        return os.path.join(self.outdir(), "mesh_config.json")

    @property
    def timeshift(self):
        return self.timeshift_s + self.timeshift_ns / 1e9
