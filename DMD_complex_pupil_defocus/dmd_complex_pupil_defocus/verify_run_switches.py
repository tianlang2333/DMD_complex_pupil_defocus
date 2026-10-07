"""回归验收辅助：仅在内存中替换main开关，产品源文件不发生变化。
Usage: MPLBACKEND=Agg python verify_run_switches.py SOURCE_DIR OUTPUT_DIR MODE
MODE: default / legacy / phase-only / dose-only / disabled / cross-only
      defocus-only / combined / defocus-disabled / defocus-cases
“default”为历史固定验收工况（ratio+phase+dose），并非上传版本main默认值。
旧模式仅在内存中显式固定历史rho/dose列表；combined原样运行当前main默认值。
combined可追加第四个参数DEFOCUS_ONLY_OUTPUT_DIR，比较两次实际main产生的同焦位结果。
"""
from __future__ import annotations
import ast
import csv
import hashlib
import importlib
import json
import math
import re
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

source_dir = Path(sys.argv[1]).resolve()
output_dir = Path(sys.argv[2]).resolve()
mode = sys.argv[3]
assert mode in {'default','legacy','phase-only','dose-only','disabled','cross-only',
                'defocus-only','combined','defocus-disabled','defocus-cases'}
output_dir.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(source_dir))
v = importlib.import_module('run_dmd_2d_validation')
source_file = source_dir / 'run_dmd_2d_validation.py'
source_sha = hashlib.sha256(source_file.read_bytes()).hexdigest()


def run_defocus_main_mode() -> None:
    """实际main调度验收；不替换物理计算，也不改磁盘中的main源文件。"""
    parsed = ast.parse(source_file.read_text(encoding='utf-8'), filename=str(source_file))
    main_node = next(node for node in parsed.body
                     if isinstance(node, ast.FunctionDef) and node.name == 'main')
    flag_names = {
        'run_textbook_validation', 'run_na_experiment', 'run_sampling_experiment',
        'run_original_cross', 'run_convergence', 'run_leakage_sweep', 'run_phase_sweep',
        'run_dose_check', 'run_cross_leakage', 'run_defocus_experiment',
    }
    original_flags, active_flags = {}, {}
    sentinel_set = False
    for node in ast.walk(main_node):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if name in flag_names:
            assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool)
            original_flags[name] = node.value.value
            enabled = (node.value.value if mode == 'combined'
                       else mode == 'defocus-only' and name == 'run_defocus_experiment')
            active_flags[name] = enabled
            if mode != 'combined':
                node.value = ast.copy_location(ast.Constant(enabled), node.value)
        elif name == 'defocus_values_um' and mode == 'defocus-disabled':
            # bool不是合法离焦；关闭后不能进入该实验的参数校验。
            node.value = ast.copy_location(ast.Tuple(elts=[ast.Constant(True)], ctx=ast.Load()),
                                           node.value)
            sentinel_set = True
    assert set(active_flags) == flag_names, active_flags
    assert mode != 'defocus-disabled' or sentinel_set
    exec(compile(ast.fix_missing_locations(ast.Module(body=[main_node], type_ignores=[])),
                 filename=str(source_file), mode='exec'), v.__dict__)

    # 真实旧文件标记：必须保持字节不变，并从本次manifest排除。
    stale_names = ['figure_05_na_sweep.png', 'validation_metrics.csv', 'figure_leakage_phase.png']
    if mode in {'defocus-only', 'defocus-disabled'}:
        stale_names += ['leakage_metrics.csv', 'leakage_representative_data.npz',
                        'figure_leakage_ratio.png', 'figure_leakage_dose.png',
                        'figure_leakage_cross.png']
    if mode == 'defocus-disabled':
        stale_names += ['defocus_metrics.csv', 'defocus_representative_data.npz',
                        'figure_defocus_comparison.png', 'figure_defocus_pupil.png']
    stale = {}
    for name in stale_names:
        marker = ('STALE OUTPUT SENTINEL: ' + name).encode()
        (output_dir / name).write_bytes(marker)
        stale[name] = hashlib.sha256(marker).hexdigest()

    calls = []
    original_sweep = v.run_defocus_sweep

    def checked_sweep(*args, **kwargs):
        calls.append('run_defocus_sweep')
        if mode == 'defocus-disabled':
            raise AssertionError('Disabled defocus branch executed.')
        # 默认独立纯离焦响应应为eta=1,rho=0,phi=0，不能继承旧漏光rho=.01。
        for key, expected in [('on_intensity_scale', 1.0),
                              ('off_to_on_intensity_ratio', 0.0),
                              ('off_relative_phase_rad', 0.0)]:
            assert kwargs[key] == expected, (key, kwargs[key])
        return original_sweep(*args, **kwargs)

    v.run_defocus_sweep = checked_sweep
    old_argv = sys.argv
    original_stat = Path.stat
    disabled_defocus_names = {name for name in stale_names if 'defocus' in name}

    def guarded_stat(path, *args, **kwargs):
        if mode == 'defocus-disabled' and path.name in disabled_defocus_names:
            raise AssertionError('Disabled defocus output was inspected: ' + path.name)
        return original_stat(path, *args, **kwargs)

    try:
        sys.argv = [str(source_file), '--output-dir', str(output_dir)]
        with patch.object(Path, 'stat', new=guarded_stat):
            v.main()
    finally:
        sys.argv = old_argv
        v.run_defocus_sweep = original_sweep
    assert source_sha == hashlib.sha256(source_file.read_bytes()).hexdigest()
    summary = (output_dir / 'validation_summary.txt').read_text(encoding='utf-8')
    manifest = summary.split('本次生成文件\n')[-1].splitlines()
    checks_section = summary.split('一、内部数值自检\n', 1)[1].split('\n二、', 1)[0]
    check_lines = [line for line in checks_section.splitlines() if line.startswith('- ')]
    assert check_lines and all(line.startswith('- PASS:') for line in check_lines), check_lines
    for name, sha in stale.items():
        assert hashlib.sha256((output_dir / name).read_bytes()).hexdigest() == sha, name
        assert name not in manifest, name
    for name in manifest:
        assert (output_dir / name).is_file() and (output_dir / name).stat().st_size > 0, name
    data_checks = []
    cross_mode_max_errors = {}
    if mode == 'defocus-disabled':
        assert calls == []
        assert manifest == ['validation_summary.txt'], manifest
        assert not any(name.startswith('defocus_') or name.startswith('figure_defocus_')
                       for name in manifest)
        assert '- defocus: NOT RUN' in summary
        data_checks += ['invalid_disabled_list_not_validated', 'disabled_sweep_not_called',
                        'disabled_defocus_outputs_not_inspected']
    else:
        assert len(calls) == 1, calls
        assert '- defocus: EXECUTED' in summary
        assert {'defocus_metrics.csv', 'defocus_representative_data.npz'} <= set(manifest)
        with (output_dir / 'defocus_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 5, len(rows)
        for key in ('reference_intensity', 'nominal_threshold_intensity',
                    'nominal_threshold_fraction'):
            assert len({row[key] for row in rows}) == 1, key
        assert all(float(row['on_intensity_scale']) == 1.0
                   and float(row['off_to_on_intensity_ratio']) == 0.0
                   and float(row['off_relative_phase_rad']) == 0.0 for row in rows)
        with np.load(output_dir / 'defocus_representative_data.npz', allow_pickle=False) as data:
            for key in data.files:
                assert data[key].dtype.kind != 'O', key
        data_checks += ['single_shared_reference', 'pure_defocus_response_independent',
                        'npz_allow_pickle_false']
        if mode == 'defocus-only':
            assert all(name == 'validation_summary.txt' or 'defocus' in name for name in manifest)
        else:
            assert active_flags == original_flags, 'combined must preserve every current default flag'
            assert {'leakage_metrics.csv', 'leakage_representative_data.npz'} <= set(manifest)
            if len(sys.argv) >= 5:
                previous_dir = Path(sys.argv[4]).resolve()
                with np.load(previous_dir / 'defocus_representative_data.npz', allow_pickle=False) as old, \
                     np.load(output_dir / 'defocus_representative_data.npz', allow_pickle=False) as current:
                    for key in ('x_um', 'y_um', 'defocus_values_um', 'raw_profiles',
                                'image_field_center_rows', 'baseline_raw_profile',
                                'baseline_image_field_center_row', 'reference_intensity',
                                'nominal_threshold_intensity', 'representative_image_field',
                                'representative_raw_intensity', 'representative_pupil'):
                        left, right = old[key], current[key]
                        assert left.shape == right.shape, key
                        error = float(np.max(np.abs(left - right)))
                        cross_mode_max_errors[key] = error
                        assert error <= 1e-12, (key, error)
                with (previous_dir / 'defocus_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
                    previous_rows = list(csv.DictReader(stream))
                assert previous_rows == rows, 'physical CSV rows changed with unrelated switches'
                data_checks.append('same_focus_independent_of_other_main_switches')
    report = {
        'mode': mode, 'status': 'PASS', 'source_sha256': source_sha,
        'test_case_kind': 'current_main_defaults' if mode == 'combined' else 'flag_only_override',
        'original_flags': original_flags, 'flags': active_flags,
        'self_checks': {'passed': len(check_lines), 'total': len(check_lines)},
        'defocus_branch_calls': len(calls), 'data_checks': data_checks,
        'cross_mode_max_absolute_errors': cross_mode_max_errors,
        'stale_files_unchanged_excluded': list(stale), 'generated_manifest': manifest,
    }
    (output_dir / 'switch_check.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[Switch gate] PASS:', mode)


def run_defocus_cases_mode() -> None:
    """固定工况的列表语义、顺序不变性与复漏光代数检查。"""
    from dataclasses import asdict, replace

    tolerance = 1.0e-12  # 运行之前固定，所有相同网格复场/强度/指标一致性共用。
    dmd = v.DMDConfig(num_mirrors_x=64, num_mirrors_y=32, dmd_mirror_pitch_um=7.56,
                      projection_magnification=1.5 / 7.56, samples_per_mirror=16,
                      active_side_ratio=0.95)
    optical = v.OpticalConfig2D(wavelength_um=0.405, numerical_aperture=0.065)
    dmd_before = asdict(dmd)
    keywords = dict(on_width_um=6.0, pitch_um=12.0, phase_offset_um=0.0,
                    on_intensity_scale=1.0, off_to_on_intensity_ratio=0.0,
                    off_relative_phase_rad=0.0, threshold_fraction=0.5,
                    evaluation_bounds_um=(-6.0, 6.0), search_bounds_um=(-6.0, 6.0),
                    dark_bounds_um=(4.5, 7.5), threshold_mode='target_cd')
    checks = []

    def check(name, error, limit=tolerance):
        error = float(error)
        passed = math.isfinite(error) and error <= limit
        checks.append({'name': name, 'value': error, 'limit': limit,
                       'status': 'PASS' if passed else 'FAIL'})
        assert passed, checks[-1]

    def compare(name, actual, expected):
        a, b = np.asarray(actual), np.asarray(expected)
        assert a.shape == b.shape, (name, a.shape, b.shape)
        assert np.array_equal(np.isnan(a), np.isnan(b)), name
        finite = np.isfinite(a) & np.isfinite(b)
        check(name, np.max(np.abs(a[finite] - b[finite])) if np.any(finite) else 0.0)

    original_render = v.render_dmd_field

    def execute_case(name, z_values, *, save_numeric=False, **changes):
        folder = output_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        snapshots = []

        def tracked_render(states, config):
            states_copy = np.array(states, copy=True)
            result = original_render(states, config)
            snapshots.append((states, states_copy, result.object_field,
                              result.object_field.copy()))
            return result

        with patch.object(v, 'render_dmd_field', side_effect=tracked_render):
            result = v.run_defocus_sweep(folder, False, dmd, optical,
                                        **dict(keywords, defocus_values_um=z_values, **changes))
        for index, (states, states_copy, field, field_copy) in enumerate(snapshots):
            check(f'{name}_states_{index}_unchanged', float(not np.array_equal(states, states_copy)), 0)
            compare(f'{name}_field_{index}_unchanged', field, field_copy)
        assert asdict(dmd) == dmd_before
        rows, data, plots, notes = result
        assert len(rows) == len(z_values)
        assert list(data['defocus_values_um']) == list(z_values)
        assert len({row['reference_intensity'] for row in rows}) == 1
        assert len({row['nominal_threshold_intensity'] for row in rows}) == 1
        compare(f'{name}_field_squared_is_raw_profile', data['raw_profiles'],
                np.abs(data['image_field_center_rows']) ** 2)
        for path in plots:
            assert path.is_file() and path.stat().st_size > 0, path
        if save_numeric:
            # 与main相同的CSV/NPZ序列化路径，保留无效指标的NaN及状态。
            with (folder / 'defocus_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            np.savez_compressed(folder / 'defocus_representative_data.npz', **data)
            with np.load(folder / 'defocus_representative_data.npz', allow_pickle=False) as loaded:
                for key in loaded.files:
                    check(f'{name}_{key}_pickle_free', float(loaded[key].dtype.kind == 'O'), 0)
        return rows, data

    base_rows, base = execute_case('base', (-50.0, -25.0, 0.0, 25.0, 50.0))
    base_copy = {key: np.array(value, copy=True) for key, value in base.items()
                 if isinstance(value, np.ndarray)}
    metrics = ('fixed_left_edge_um', 'fixed_right_edge_um', 'fixed_cd_um',
               'design_edge_nils', 'center_contrast', 'dark_max_intensity')
    for name, z_values in [('single', (25.0,)), ('no_zero', (-25.0, 25.0)),
                           ('reverse', (50.0, 25.0, 0.0, -25.0, -50.0)),
                           ('repeat', (-50.0, -25.0, 0.0, 25.0, 50.0))]:
        rows, data = execute_case(name, z_values)
        for key in ('reference_intensity', 'nominal_threshold_intensity',
                    'baseline_raw_profile', 'baseline_image_field_center_row'):
            compare(f'{name}_common_{key}', data[key], base[key])
        for i, z in enumerate(z_values):
            j = list(base['defocus_values_um']).index(z)
            for key in ('raw_profiles', 'image_field_center_rows'):
                compare(f'{name}_{z:g}_{key}', data[key][i], base[key][j])
            for key in metrics:
                compare(f'{name}_{z:g}_{key}', rows[i][key], base_rows[j][key])
            check(f'{name}_{z:g}_status', float(rows[i]['threshold_status'] !=
                                             base_rows[j]['threshold_status']), 0)

    rho, phi = 0.01, 0.7
    rows, data = execute_case('complex_leakage', (-25.0, 0.0, 25.0),
                             off_to_on_intensity_ratio=rho, off_relative_phase_rad=phi)
    for key in ('reference_intensity', 'nominal_threshold_intensity', 'baseline_raw_profile'):
        compare(f'complex_leakage_common_{key}', data[key], base[key])
    # 独立ON/OFF基场的复数叠加代数；瞳仍通过唯一生产传播入口使用。
    from projection_lens_2d import make_defocused_circular_pupil
    from coherent_imaging_2d import make_frequency_axes
    states = v.make_periodic_line_space_states(dmd, 12.0, 6.0)
    on = v.render_dmd_field(states, replace(dmd, on_amplitude=1 + 0j, off_amplitude=0 + 0j))
    off = v.render_dmd_field(states, replace(dmd, on_amplitude=0 + 0j, off_amplitude=1 + 0j))
    fx, fy = make_frequency_axes(len(on.x_um), len(on.y_um),
                                float(on.x_um[1] - on.x_um[0]),
                                float(on.y_um[1] - on.y_um[0]))
    pupil = make_defocused_circular_pupil(fx, fy, optical, 25.0)
    on_image = v.calculate_coherent_aerial_image_2d(on.object_field, on.x_um, on.y_um,
                                                   optical, pupil=pupil)
    off_image = v.calculate_coherent_aerial_image_2d(off.object_field, off.x_um, off.y_um,
                                                    optical, pupil=pupil)
    row_index = int(data['profile_row_index'])
    expected = (on_image.image_field[row_index] + math.sqrt(rho) * np.exp(1j * phi)
                * off_image.image_field[row_index])
    compare('complex_leakage_sum_fields_then_square', data['image_field_center_rows'][2], expected)
    compare('complex_leakage_raw_intensity', data['raw_profiles'][2], np.abs(expected) ** 2)
    incoherent = on_image.raw_intensity[row_index] + rho * off_image.raw_intensity[row_index]
    wrong_sum_difference = float(np.max(np.abs(data['raw_profiles'][2] - incoherent)))
    check('complex_leakage_interference_is_present', float(wrong_sum_difference <= 1e-3), 0)

    fraction_rows, fraction = execute_case('fixed_fraction', (-25.0, 0.0, 25.0),
                                          threshold_mode='fixed_fraction', save_numeric=True)
    compare('fixed_fraction_common_Iref', fraction['reference_intensity'], base['reference_intensity'])
    compare('fixed_fraction_threshold_is_half_Iref', fraction['nominal_threshold_intensity'],
            0.5 * fraction['reference_intensity'])
    check('fixed_fraction_rows_report_mode',
          float(not all(row['threshold_mode'] == 'fixed_fraction' for row in fraction_rows)), 0)
    invalid_rows, invalid = execute_case('zero_response_single', (25.0,),
                                        on_intensity_scale=0.0, save_numeric=True)
    check('zero_response_profile_is_zero', np.max(np.abs(invalid['raw_profiles'])), 0)
    check('zero_response_metrics_stay_nan', float(not all(
        math.isnan(invalid_rows[0][key]) for key in ('fixed_cd_um', 'design_edge_nils', 'center_contrast'))), 0)
    check('zero_response_retains_failure_reason',
          float(invalid_rows[0]['threshold_status'] in ('VALID', '')), 0)
    with (output_dir / 'zero_response_single' / 'defocus_metrics.csv').open(
            encoding='utf-8-sig', newline='') as stream:
        saved_invalid = list(csv.DictReader(stream))
    check('zero_response_saved_csv_nan_and_reason', float(not (
        len(saved_invalid) == 1 and math.isnan(float(saved_invalid[0]['fixed_cd_um']))
        and saved_invalid[0]['threshold_status'] == invalid_rows[0]['threshold_status'])), 0)
    for key, saved in base_copy.items():
        if saved.dtype.kind in 'fc':
            compare('previous_result_unchanged_' + key, base[key], saved)
        else:
            check('previous_result_unchanged_' + key, float(not np.array_equal(base[key], saved)), 0)
    # 开启时非法列表应早于DMD渲染被拒绝；关闭情形另由main模式guard验收。
    for name, z_values in [('empty', ()), ('bool', (True,)), ('complex', (1j,)),
                           ('nonfinite', (math.nan,))]:
        with patch.object(v, 'render_dmd_field', side_effect=AssertionError('Invalid list rendered DMD.')):
            try:
                v.run_defocus_sweep(output_dir, False, dmd, optical,
                                    **dict(keywords, defocus_values_um=z_values))
            except ValueError:
                check('invalid_list_rejected_' + name, 0, 0)
            else:
                check('invalid_list_rejected_' + name, 1, 0)
    assert source_sha == hashlib.sha256(source_file.read_bytes()).hexdigest()
    report = {'mode': mode, 'status': 'PASS', 'source_sha256': source_sha,
              'absolute_tolerance': tolerance, 'checks': checks,
              'wrong_incoherent_intensity_max_difference': wrong_sum_difference,
              'cases': ['base', 'single', 'no_zero', 'reverse', 'repeat', 'complex_leakage',
                        'fixed_fraction', 'zero_response_single']}
    (output_dir / 'switch_check.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[Switch gate] PASS:', mode, len(checks), 'checks')


if mode == 'defocus-cases':
    run_defocus_cases_mode()
    raise SystemExit(0)

if mode in {'defocus-only', 'combined', 'defocus-disabled'}:
    run_defocus_main_mode()
    raise SystemExit(0)

flags = dict.fromkeys(('run_textbook_validation','run_na_experiment','run_sampling_experiment',
    'run_original_cross','run_convergence','run_leakage_sweep','run_phase_sweep',
    'run_dose_check','run_cross_leakage','run_defocus_experiment'), False)
if mode == 'default':
    flags.update(run_leakage_sweep=True, run_phase_sweep=True, run_dose_check=True)
elif mode == 'legacy':
    for key in ('run_textbook_validation','run_na_experiment','run_sampling_experiment',
                'run_original_cross','run_convergence'):
        flags[key] = True
elif mode == 'phase-only':
    flags['run_phase_sweep'] = True
    flags['run_dose_check'] = False
elif mode == 'dose-only':
    flags['run_dose_check'] = True
elif mode == 'cross-only':
    flags['run_cross_leakage'] = True
parsed = ast.parse(source_file.read_text(encoding='utf-8'), filename=str(source_file))
main = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
found = []
for node in ast.walk(main):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
        if name in flags:
            node.value = ast.copy_location(ast.Constant(flags[name]), node.value)
            found.append(name)
        elif name in {'leakage_ratios', 'dose_scales'}:
            # 历史验收输入明示冻结；不修改产品main的当前默认列表。
            fixture = ((0.0, 1.0e-4, 1.0e-3, 1.0e-2) if name == 'leakage_ratios'
                       else (0.95, 1.0, 1.05))
            node.value = ast.copy_location(ast.Tuple(
                elts=[ast.Constant(value) for value in fixture], ctx=ast.Load()), node.value)
assert set(found) == set(flags), (found, flags)
exec(compile(ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[])),
             filename=str(source_file), mode='exec'), v.__dict__)

# 旧工况中离焦分支必须确实关闭，而非只不生成文件。
v.run_defocus_sweep = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError('Disabled defocus branch executed in legacy switch fixture.'))
legacy_calls = []
def forbidden_legacy(*args, **kwargs):
    legacy_calls.append('called')
    raise AssertionError('Disabled legacy experiment performed computation.')

stale = {}
if mode != 'legacy':
    for name in ('_make_base_image','run_textbook_cross_validation','run_na_sweep',
                 'run_sampling_ratio_experiment','run_cross_pattern_experiment','run_convergence_check',
                 'run_cross_leakage_comparison'):
        if not (mode == 'cross-only' and name == 'run_cross_leakage_comparison'):
            setattr(v, name, forbidden_legacy)
    # 模拟已有旧结果：应原样保留但不得计入本次运行清单。
    stale_names = ['figure_05_na_sweep.png','validation_metrics.csv']
    for flag, name in (('run_leakage_sweep', 'figure_leakage_ratio.png'),
                       ('run_phase_sweep', 'figure_leakage_phase.png'),
                       ('run_dose_check', 'figure_leakage_dose.png')):
        if not flags[flag]:
            stale_names.append(name)
    if mode == 'disabled':
        stale_names += ['leakage_metrics.csv','leakage_representative_data.npz']
    for name in stale_names:
        marker = ('STALE OUTPUT SENTINEL: '+name).encode()
        (output_dir / name).write_bytes(marker)
        stale[name] = hashlib.sha256(marker).hexdigest()
original_leakage = v.run_leakage_experiments
leakage_propagation_count = []
cache_input_checks = []
def checked_leakage(*args, **kwargs):
    if mode in {'disabled','legacy','cross-only'}:
        assert not kwargs['run_ratio_sweep'] and not kwargs['run_phase_sweep'] and not kwargs['run_dose_check']
        with patch.object(v, 'render_dmd_field', side_effect=AssertionError('Disabled leakage calculated field.')), \
             patch.object(v, 'calculate_coherent_aerial_image_2d', side_effect=AssertionError('Disabled leakage propagated field.')):
            result = original_leakage(*args, **kwargs)
        assert result == ([], {}, [], [])
        leakage_propagation_count.append(0)
        return result
    assert kwargs['run_ratio_sweep'] is (mode == 'default')
    assert kwargs['threshold_mode'] == 'target_cd', 'main must explicitly select target_cd.'
    if mode == 'phase-only':
        assert kwargs['run_phase_sweep'] and not kwargs['run_dose_check']
    elif mode == 'dose-only':
        assert kwargs['run_dose_check'] and not kwargs['run_phase_sweep']
        assert tuple(kwargs['dose_scales']) == (0.95, 1.0, 1.05)
        # bool True与整数1在字典键比较时相等，缓存命中前仍须验证输入。
        invalid_keywords = dict(kwargs, on_intensity_scale=True,
                                off_to_on_intensity_ratio=0.0, off_relative_phase_rad=0.0)
        with patch.object(v, 'plot_dose_comparison',
                          side_effect=AssertionError('Invalid response reached plotting.')):
            try:
                original_leakage(*args, **invalid_keywords)
            except ValueError as error:
                assert 'on_intensity_scale' in str(error), str(error)
            else:
                raise AssertionError('eta=True bypassed validation through baseline cache key.')
        cache_input_checks.append('eta_true_rho_zero_rejected_before_cache_reuse')
    else:
        assert mode == 'default'
        assert kwargs['run_phase_sweep'] and kwargs['run_dose_check']
    with patch.object(v, 'calculate_coherent_aerial_image_2d', wraps=v.calculate_coherent_aerial_image_2d) as propagation:
        result = original_leakage(*args, **kwargs)
        count = propagation.call_count
    leakage_propagation_count.append(count)
    expected_count = {'phase-only': 4, 'dose-only': 2, 'default': 6}[mode]
    assert count == expected_count, (
        f'{mode}: expected {expected_count} production propagations, got {count}'
    )
    return result
v.run_leakage_experiments = checked_leakage
sys.argv = [str(source_file), '--output-dir', str(output_dir)]
v.main()
assert source_sha == hashlib.sha256(source_file.read_bytes()).hexdigest(), 'Source changed during test.'
summary = (output_dir/'validation_summary.txt').read_text(encoding='utf-8')
manifest = summary.split('本次生成文件\n')[-1].splitlines()
check_section = summary.split('一、内部数值自检\n', 1)[1].split('\n二、', 1)[0]
check_lines = [line for line in check_section.splitlines() if line.startswith('- ')]
assert check_lines and all(line.startswith('- PASS:') for line in check_lines), check_lines
data_checks = []
for name, sha in stale.items():
    assert hashlib.sha256((output_dir/name).read_bytes()).hexdigest() == sha
    assert name not in manifest, name
if mode in {'default', 'phase-only', 'dose-only'}:
    with (output_dir/'leakage_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len({row['reference_intensity'] for row in rows}) == 1
    assert len({row['nominal_threshold_intensity'] for row in rows}) == 1
    assert len({row['nominal_threshold_fraction'] for row in rows}) == 1
    assert all(row['reference_group'] == 'periodic_lines' for row in rows)
    for row in rows:
        assert row['threshold_mode'] == 'target_cd'
        assert math.isclose(float(row['nominal_baseline_cd_um']), float(row['target_cd_um']),
                            rel_tol=0, abs_tol=1e-8)
        assert math.isclose(float(row['threshold_intensity']),
                            float(row['nominal_threshold_intensity']) / float(row['dose_scale']),
                            rel_tol=1e-12, abs_tol=1e-12)
        if row['experiment'] == 'dose':
            paired_baseline = next(candidate for candidate in rows
                                   if candidate['experiment'] == 'dose'
                                   and candidate['dose_state'] == 'baseline'
                                   and candidate['dose_scale'] == row['dose_scale'])
            same_dose_cd = float(paired_baseline['fixed_cd_um'])
            same_dose_status = paired_baseline['threshold_status']
        else:
            same_dose_cd = float(row['nominal_baseline_cd_um'])
            same_dose_status = row['nominal_baseline_status']
        assert row['same_dose_baseline_status'] == same_dose_status
        assert math.isclose(float(row['same_dose_baseline_cd_um']), same_dose_cd,
                            rel_tol=1e-12, abs_tol=1e-12)
        for key, expected in (
                ('delta_cd_to_nominal_baseline_um',
                 float(row['fixed_cd_um']) - float(row['nominal_baseline_cd_um'])),
                ('delta_cd_to_baseline_um',
                 float(row['fixed_cd_um']) - float(row['nominal_baseline_cd_um'])),
                ('delta_cd_to_same_dose_baseline_um', float(row['fixed_cd_um']) - same_dose_cd)):
            assert math.isclose(float(row[key]), expected, rel_tol=1e-12, abs_tol=1e-12), (
                row['case_name'], key, row[key], expected)
    data_checks.extend(['common_nominal_reference', 'effective_threshold_T0_over_s',
                        'nominal_and_same_dose_cd_deltas', 'same_dose_baseline_status'])
    expected_files = {'leakage_metrics.csv', 'leakage_representative_data.npz', 'validation_summary.txt'}
    if mode == 'default':
        expected_files.update(f'figure_leakage_{group}.png' for group in ('ratio', 'phase', 'dose'))
        assert '- leakage_ratio: EXECUTED' in summary
    else:
        assert '- leakage_ratio: NOT RUN' in summary
        expected_files.add('figure_leakage_phase.png' if mode == 'phase-only' else 'figure_leakage_dose.png')
    assert set(manifest) == expected_files, manifest
    if mode == 'phase-only':
        assert len(rows) == 3
        assert all(row['experiment'] == 'phase' for row in rows)
        assert '- leakage_phase: EXECUTED' in summary
        assert '- dose_check: NOT RUN' in summary
    else:
        if mode == 'dose-only':
            assert len(rows) == 6
            assert all(row['experiment'] == 'dose' for row in rows)
            assert '- leakage_phase: NOT RUN' in summary
        else:
            assert len(rows) == 13
            assert [sum(row['experiment'] == group for row in rows)
                    for group in ('ratio', 'phase', 'dose')] == [4, 3, 6]
            assert '- leakage_phase: EXECUTED' in summary
        assert '- dose_check: EXECUTED' in summary
        for state in ('baseline', 'representative'):
            state_rows = [row for row in rows if row['experiment'] == 'dose' and row['dose_state'] == state]
            assert len(state_rows) == 3
            assert {row['case_name'] for row in state_rows} == {
                f'dose_{state}_{index:02d}' for index in range(3)
            }
            assert {float(row['dose_scale']) for row in state_rows} == {0.95, 1.0, 1.05}
            for key in ('on_center_intensity', 'off_center_intensity', 'center_contrast',
                        'design_edge_nils', 'dark_max_intensity'):
                assert len({row[key] for row in state_rows}) == 1, (
                    state, key, [row[key] for row in state_rows])
            if mode == 'default':
                nominal = next(row for row in state_rows if float(row['dose_scale']) == 1.0)
                scan = next(row for row in rows if row['experiment'] == 'ratio'
                            and all(float(row[key]) == float(nominal[key]) for key in (
                                'on_intensity_scale', 'off_to_on_intensity_ratio', 'off_relative_phase_rad')))
                for key in ('fixed_left_edge_um', 'fixed_right_edge_um', 'fixed_cd_um',
                            'cd_error_to_design_um', 'design_edge_nils', 'dark_max_intensity'):
                    assert math.isclose(float(nominal[key]), float(scan[key]), rel_tol=1e-12, abs_tol=1e-12), (
                        state, key, nominal[key], scan[key]
                    )
        data_checks.append('dose_preserves_raw_centers_contrast_nils_darkmax')
        if mode == 'default':
            data_checks.append('dose_s1_matches_response_scan')
elif mode == 'disabled':
    assert manifest == ['validation_summary.txt'], manifest
    assert all(f'- {name}: NOT RUN' in summary for name in (
        'textbook_validation','na_sweep','sampling_ratio','original_cross','legacy_convergence',
        'leakage_ratio','leakage_phase','dose_check','leakage_cross'))
elif mode == 'cross-only':
    assert set(manifest) == {'figure_leakage_cross.png','leakage_representative_data.npz','validation_summary.txt'}, manifest
    assert '- leakage_cross: EXECUTED' in summary
    assert '- leakage_phase: NOT RUN' in summary
    assert '- leakage_ratio: NOT RUN' in summary
    assert '- dose_check: NOT RUN' in summary
else:
    assert len(manifest) == 9, manifest
    assert all((output_dir/f'figure_0{index}_{suffix}.png').exists() for index, suffix in (
        (1,'dmd_state_and_aperture'),(2,'spectrum_pupil_filtered_spectrum'),
        (3,'aerial_image_and_profiles'),(4,'1d_2d_cross_validation'),(5,'na_sweep'),
        (6,'sampling_ratio_comparison'),(7,'2d_cross_pattern')))
if 'leakage_representative_data.npz' in manifest:
    with np.load(output_dir/'leakage_representative_data.npz', allow_pickle=False) as data:
        field_keys = [key for key in data.files if 'image_field' in key]
        assert field_keys and all(np.iscomplexobj(data[key]) for key in field_keys), field_keys
        data_checks.append('npz_retains_complex_fields')
        if mode in {'default', 'dose-only'}:
            assert not any(re.fullmatch(r'dose_.+_\d+_raw_profile', key) for key in data.files)
            for state in ('baseline', 'representative'):
                profile_key = str(data[f'dose_{state}_profile_key'].item())
                assert profile_key in data.files, (state, profile_key)
                profile = data[profile_key]
                assert profile.shape == data['x_um'].shape and np.isrealobj(profile)
                assert np.all(np.isfinite(profile))
                if state == 'baseline':
                    assert profile_key == 'baseline_raw_profile'
                    field_profile = data['baseline_image_field_center_row']
                else:
                    center_row = int(np.argmin(np.abs(data['representative_y_um'])))
                    field_profile = data['representative_image_field'][center_row]
                assert np.allclose(profile, np.abs(field_profile)**2, rtol=1e-12, atol=1e-12)
            assert np.allclose(data['dose_effective_thresholds'],
                               data['nominal_threshold_intensity']/data['dose_scales'],
                               rtol=1e-12, atol=1e-12)
            data_checks.extend(['dose_npz_profiles_reused_without_per_s_copies',
                                'dose_npz_profile_keys_resolve_to_field_intensity'])
# 可选十字的交付目录只保留真实产物；测试用旧文件标记验收后清理。
if mode == 'cross-only':
    for name in stale:
        (output_dir/name).unlink()
report = {'mode':mode, 'test_case_kind':'explicit_historical_fixture', 'status':'PASS', 'source_sha256':source_sha, 'flags':flags,
          'self_checks': {'passed':len(check_lines), 'total':len(check_lines)},
          'data_checks': data_checks,
          'cache_input_checks': cache_input_checks,
          'legacy_disabled_guard_calls':len(legacy_calls),
          'leakage_propagations':leakage_propagation_count,
          'stale_files_unchanged_excluded':list(stale), 'generated_manifest':manifest}
(output_dir/'switch_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('[Switch gate] PASS:',mode)
