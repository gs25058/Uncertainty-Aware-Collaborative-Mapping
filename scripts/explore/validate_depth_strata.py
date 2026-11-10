#!/usr/bin/env python
"""Post-run sensor audit by incidence/edge; never changes the frozen profile."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.depth_observation import RayDepthSensor
from covor.explore.runner import csv_write
from scripts.explore.validate_depth_model import Fixture


class TiltedPlane:
    def __init__(self, distance, degrees):
        import open3d as o3d
        a = np.deg2rad(degrees)
        rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        vertices = np.array([[-40, -40, 0], [40, -40, 0], [40, 40, 0], [-40, 40, 0]]) @ rotation.T
        vertices += [0, 0, distance]
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(np.array([[0, 1, 2], [0, 2, 3]], np.uint32)))

    def render_scene(self, textured=False):
        return self.scene, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/depth_model_20261106.json')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind in ('plane', 'floor', 'step', 'door', 'tilted60', 'tilted80'):
        for distance in (1., 2., 4., 6.):
            fixture = TiltedPlane(distance, int(kind[6:])) if kind.startswith('tilted') else Fixture(kind, distance)
            sensor = RayDepthSensor(fixture, cfg, 11000, 'clean')
            pose = np.eye(4)
            if kind == 'floor':
                pose[:3, :3] = np.diag([1., -1., -1.])
                pose[2, 3] = distance
            buckets = {}
            for frame in range(4):
                (z, _, valid), audit = sensor.measure(pose, frame * .25, 0, audit=True)
                for lo, hi in ((0, 30), (30, 60), (60, 75), (75, 90)):
                    for edge in (False, True):
                        mask = audit['geometric'] & (audit['incidence'] >= lo) & (audit['incidence'] < hi) & (audit['edge'] == edge)
                        if not mask.any():
                            continue
                        b = buckets.setdefault((lo, hi, edge), dict(n=0, valid=0, errors=[]))
                        b['n'] += int(mask.sum())
                        b['valid'] += int((mask & valid).sum())
                        b['errors'].extend((z[mask & valid] - audit['true_z'][mask & valid]).tolist())
            for (lo, hi, edge), b in sorted(buckets.items()):
                e = np.asarray(b.pop('errors'))
                rows.append(dict(kind=kind, nominal_z_m=distance, incidence_lo_deg=lo, incidence_hi_deg=hi, edge=edge, n_geometry=b['n'], n_valid=b['valid'], conditional_valid=b['valid']/b['n'], mean_m=float(e.mean()) if len(e) else None, median_m=float(np.median(e)) if len(e) else None, std_m=float(e.std()) if len(e) else None, abs_p95_m=float(np.percentile(abs(e), 95)) if len(e) else None, over_10cm=float((abs(e)>.1).mean()) if len(e) else None))
    csv_write(out/'sensor_error_strata.csv', rows)
    # Actual combined normal branch, rather than only the common field component.
    sensor = RayDepthSensor(Fixture('plane', 2.), cfg, 11000, 'clean')
    def normal(t, drone):
        _, a = sensor.measure(np.eye(4), t, drone, audit=True)
        return a['normal_error'] / a['sigma_true']
    x = normal(0., 0)
    correlations = [dict(kind='normal_branch_temporal', offset=t, correlation=float(np.corrcoef(x, normal(t, 0))[0, 1])) for t in (.25, .5, .75, 1.5)]
    correlations.append(dict(kind='cross_drone_same_surface', offset=1, correlation=float(np.corrcoef(x, normal(0., 1))[0, 1])))
    csv_write(out/'combined_correlations.csv', correlations)
    print(f'{len(rows)} strata; combined spatial/temporal error audit complete')


if __name__ == '__main__':
    main()
