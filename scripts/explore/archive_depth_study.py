#!/usr/bin/env python
"""Archive frozen sources and normalize replay metadata for direct-depth inputs."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from covor.explore.runner import write_json, csv_write
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results/depth_model_20261106')
    parser.add_argument('--report', default='reports/depth_model_20261106')
    args = parser.parse_args()
    root, report = Path(args.root), Path(args.report)
    frozen = json.loads((root/'main/source_manifest.json').read_text())
    assert all(sha(path) == expected for path, expected in frozen.items()), 'a frozen main-run source changed'
    normalized = 0
    for group in ('main', 'rate1'):
        for run in sorted((root/group).glob('scene*')):
            if not (run/'summary.json').exists():
                continue
            cfg = json.loads((run/'config.json').read_text())['configuration']
            target = run/'replay_manifest.json'
            manifest = json.loads(target.read_text())
            manifest.update(observation_backend=cfg['sensor']['backend'], generated_stereo_images=False,
                            depth_units='metres, optical camera Z', pose_direction='world_from_left_camera',
                            integration_hz=1/cfg['sensor_period'], camera=cfg['sensor'])
            for frame in manifest['inputs']:
                frame.pop('left', None)
                frame.pop('right', None)
                assert (run/frame['npz']).exists()
            write_json(target, manifest)
            normalized += 1
    snapshot = root/'source_snapshot'
    paths = set(frozen)
    paths.update(str(p) for p in Path('scripts/explore').glob('*depth*.py'))
    paths.update(str(p) for p in Path('configs').glob('depth_model*.json'))
    paths.add('configs/sensors/d455_inspired_optimistic.json')
    paths.add('tests/test_depth_observation.py')
    for path in sorted(paths):
        target = snapshot/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    external = Path('/src/gs25058/cr_RNE/external_observation_20261105')
    packages = {}
    for name in ('numpy', 'scipy', 'open3d', 'opencv-python', 'matplotlib'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'not registered under this distribution name'
    def git(*args):
        return subprocess.check_output(['git', *args], text=True).strip()
    native = external/'standalone/libfuel_bridge.so'
    metadata = dict(base_commit=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
                    python=sys.version, executable=sys.executable, packages=packages,
                    sources={path: sha(path) for path in sorted(paths)},
                    external_audit='reports/closed_loop_20261105/EXTERNAL_AUDIT.md',
                    native_library=str(native), native_library_sha256=sha(native),
                    normalized_replay_manifests=normalized, frozen_main_sources_unchanged=True)
    write_json(snapshot/'manifest.json', metadata)
    (snapshot/'tracked_changes.patch').write_text(git('diff', '--binary')+'\n')
    # Copy small measurement tables alongside the report; raw NPZ remain in results.
    for sub in ('validation', 'validation_strata', 'replay_full'):
        destination = report/sub
        destination.mkdir(parents=True, exist_ok=True)
        for p in (root/sub).glob('*.csv'):
            shutil.copy2(p, destination/p.name)
        for p in (root/sub).glob('checks.json'):
            shutil.copy2(p, destination/p.name)
    sensor_rows = []
    for run in sorted((root/'main').glob('scene*')):
        if not (run/'summary.json').exists():
            continue
        summary = json.loads((run/'summary.json').read_text())
        frames = list(csv.DictReader((run/'simulator_audit.csv').open()))
        def total(key):
            return sum(int(row[key]) for row in frames)
        n = total('n_geometry')
        sensor_rows.append(dict(run=run.name, condition=summary['condition'], policy=summary['policy'], frames=len(frames),
                                n_grid=total('n_grid'), n_left_geometry=total('n_left_geometry'), n_binocular_geometry=n,
                                n_valid=total('n_valid'), conditional_valid=total('n_valid')/n,
                                normal_all_Z_valid=total('n_normal_valid')/total('n_normal_geometry'),
                                outlier_fraction=total('n_outlier')/n, random_drop_fraction=total('n_random_drop')/n))
    csv_write(report/'sensor_observations.csv', sensor_rows)
    rate_rows = []
    for hz, group in ((4, 'main'), (1, 'rate1')):
        for policy in ('unknown', 'entry'):
            run = root/group/f'scene11_seed0_clean_{policy}'
            s = json.loads((run/'summary.json').read_text())
            metrics = list(csv.DictReader((run/'metrics_time.csv').open()))
            def first(key, threshold):
                return next((float(row['time']) for row in metrics if int(row[key]) >= threshold), None)
            integral = np.trapz([float(row['balanced_score']) for row in metrics], [float(row['time']) for row in metrics])/48
            rate_rows.append(dict(hz=hz, policy=policy, valid=s['valid_confirmed'], correct_blocked=s['correct_blocked'],
                                  unknown=s['unknown'], invalid=s['invalid_paths'], captures=s['captures'], distance=s['travel_distance'],
                                  first_valid_s=first('valid_confirmed', 1), all_four_valid_s=first('valid_confirmed', 4),
                                  first_correct_blocked_s=first('correct_blocked', 1), balanced_time_average=float(integral)))
    csv_write(report/'rate_comparison.csv', rate_rows)
    area_path = root/'analysis/legacy_area_metrics.csv'
    if area_path.exists():
        area = list(csv.DictReader(area_path.open()))
        for row in area:
            # Fixed v4 scene-specific XY denominator; source traverse.py is unchanged.
            row['nonfeasible_xy_denominator'] = 8000 if row['run'].startswith('scene11_') else 7982
        csv_write(report/'data/legacy_area_metrics.csv', area)
    print(f'{normalized} replay manifests; {len(paths)} exact source snapshots archived')


if __name__ == '__main__':
    main()
