"""Capture original legacy calculations without changing source files.
Usage: python capture_original_baseline.py SOURCE_DIR OUTPUT_DIR
"""
from __future__ import annotations
import dataclasses
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import sys
import numpy as np

source_dir = Path(sys.argv[1]).resolve()
output_dir = Path(sys.argv[2]).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(source_dir))
v = importlib.import_module('run_dmd_2d_validation')


def plain(value):
    if dataclasses.is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in dataclasses.fields(value)
                if not isinstance(getattr(value, field.name), np.ndarray)}
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, complex):
        return [value.real, value.imag]
    return value


def image_scalars(result):
    return {name: float(getattr(result, name)) for name in
        ('input_energy', 'output_energy', 'energy_transmission', 'frequency_cutoff_cyc_per_um')}


def scalar_dataclass(result, excluded):
    return {field.name: plain(getattr(result, field.name))
            for field in dataclasses.fields(result) if field.name not in excluded}


tc, sc, cc = v.TEXTBOOK_EXPERIMENT_CONFIG, v.SAMPLING_EXPERIMENT_CONFIG, v.CROSS_EXPERIMENT_CONFIG
checks = v.run_internal_validations()
base_pattern, base = v._make_base_image(samples_per_mirror=tc.samples_per_mirror,
    numerical_aperture=tc.numerical_aperture, experiment_config=tc)
base_profile = v._extract_center_horizontal_profile(base)
base_metrics = v.calculate_periodic_profile_metrics(base.x_um, base_profile,
    pitch_um=tc.pitch_um, on_width_um=tc.on_width_um, on_center_um=tc.phase_offset_um)
textbook_path = source_dir / 'textbook_dense_lines_aerial_image(1).py'
if not textbook_path.exists():
    textbook_path = source_dir / 'textbook_dense_lines_aerial_image.py'
# Old loader has only a filename constant; temporarily selecting exact existing file
# changes neither source nor the imported textbook computation.
original_name = v.TEXTBOOK_MODULE_FILENAME
v.TEXTBOOK_MODULE_FILENAME = textbook_path.name
try:
    parameters = inspect.signature(v.run_textbook_cross_validation).parameters
    extra = {'textbook_path': textbook_path} if 'textbook_path' in parameters else {}
    textbook = v.run_textbook_cross_validation(source_dir, base, tc, **extra)
finally:
    v.TEXTBOOK_MODULE_FILENAME = original_name
convergence = v.run_convergence_check(base, tc)
na = v.run_na_sweep(base_pattern, tc)
sampling = v.run_sampling_ratio_experiment(sc)
cross = v.run_cross_pattern_experiment(cc)
rows = v.build_metric_rows(base_pattern, base, base_metrics, textbook, convergence,
                          na, sampling, cross, tc, sc, cc)
v.write_metrics_csv(output_dir / 'baseline_legacy_metrics.csv', rows)
results = {
    'checks': plain(checks),
    'configurations': {'textbook': plain(tc), 'sampling': plain(sc), 'cross': plain(cc)},
    'base': {'metrics': plain(base_metrics), 'energy': image_scalars(base)},
    'textbook': plain(textbook),
    'convergence': plain(convergence),
    'na': [{'numerical_aperture': c.numerical_aperture,
            'maximum_geometric_order': c.maximum_geometric_order,
            'metrics': plain(c.metrics), 'energy': image_scalars(c.result)} for c in na],
    'sampling': [scalar_dataclass(c, {'pattern','ideal_field','ideal_image','dmd_image',
                                     'ideal_profile','dmd_profile'}) for c in sampling],
    'cross': scalar_dataclass(cross, {'pattern','ideal_field','image','threshold_mask'}),
    'legacy_metric_rows': plain(rows),
}
(output_dir / 'baseline_numeric.json').write_text(json.dumps(results, ensure_ascii=False,
    indent=2, allow_nan=False), encoding='utf-8')
arrays = {'base_x_um': base.x_um, 'base_profile': base_profile,
          'textbook_x_um': textbook.x_um, 'textbook_intensity': textbook.textbook_intensity,
          'textbook_dmd_intensity': textbook.dmd_2d_intensity,
          'cross_x_um': cross.image.x_um,
          'cross_center_horizontal_raw': v._extract_center_horizontal_profile(cross.image)}
for index, case in enumerate(na):
    arrays[f'na_{index}_profile'] = case.profile
for index, case in enumerate(sampling):
    arrays[f'sampling_{index}_x_um'] = case.dmd_image.x_um
    arrays[f'sampling_{index}_ideal_profile'] = case.ideal_profile
    arrays[f'sampling_{index}_dmd_profile'] = case.dmd_profile
np.savez_compressed(output_dir / 'baseline_profiles.npz', **arrays)
provenance = {'source_directory':str(source_dir), 'textbook_path':str(textbook_path),
    'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(source_dir.glob('*.py'))}}
(output_dir / 'baseline_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
print('Internal checks:', sum(c.passed for c in checks), '/', len(checks))
print('Textbook:', textbook.status, textbook.reason)
print('Textbook RMSE:', textbook.raw_rmse)
print('Convergence maximum relative change:', convergence.maximum_metric_relative_change)
print('Sampling N_CD / raster error:', [(c.n_cd,c.raster_cd_error_um) for c in sampling])
print('Baseline saved:', output_dir)
