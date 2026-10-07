"""Capture fixed-input evidence from an explicitly selected original source tree.

Run before any edits: python baseline/capture_fixed_baseline.py --source ../original
This script intentionally imports only from that source directory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source))
    import dmd_model_2d as dmd
    import coherent_imaging_2d as imaging
    import run_dmd_2d_validation as runner
    for module in (dmd, imaging, runner):
        if Path(module.__file__).resolve().parent != source:
            raise RuntimeError(f"Imported unexpected source: {module.__file__}")
    config = dmd.DMDConfig(
        num_mirrors_x=64, num_mirrors_y=32, dmd_mirror_pitch_um=7.56,
        projection_magnification=1.5 / 7.56, samples_per_mirror=16,
        active_side_ratio=0.95,
    )
    optical = imaging.OpticalConfig2D(wavelength_um=0.405, numerical_aperture=0.065)
    states = dmd.make_periodic_line_space_states(config, 12.0, 6.0, phase_offset_um=0.0)
    arrays = {}
    cases = {}
    workpoint = None
    for name, rho, phi in (("no_leakage", 0.0, 0.0), ("complex_leakage", 0.01, np.pi/3)):
        response = dmd.with_state_response(
            config, on_intensity_scale=1.0, off_to_on_intensity_ratio=rho,
            off_relative_phase_rad=phi,
        )
        pattern = dmd.render_dmd_field(states, response)
        result = imaging.calculate_coherent_aerial_image_2d(
            pattern.object_field, pattern.x_um, pattern.y_um, optical,
        )
        row = int(np.argmin(abs(result.y_um)))
        profile = result.raw_intensity[row].copy()
        if workpoint is None:
            mask = (result.x_um >= -6.0) & (result.x_um <= 6.0)
            reference = float(np.max(profile[mask]))
            workpoint = runner.establish_nominal_workpoint(
                result.x_um, profile, on_width_um=6.0, center_um=0.0,
                reference_intensity=reference, search_bounds_um=(-6.0, 6.0),
                threshold_mode="target_cd", threshold_fraction=0.5,
            )
            arrays.update(x_um=result.x_um, y_um=result.y_um, mirror_states=states)
        metrics = runner.evaluate_leakage_profile(
            result.x_um, profile, on_width_um=6.0, pitch_um=12.0, center_um=0.0,
            reference_intensity=workpoint["reference_intensity"],
            threshold_intensity=workpoint["nominal_threshold_intensity"],
            search_bounds_um=(-6.0, 6.0), dark_bounds_um=(4.5, 7.5),
        )
        arrays.update({f"{name}_field": result.image_field,
                       f"{name}_intensity": result.raw_intensity,
                       f"{name}_profile": profile})
        cases[name] = dict(eta=1.0, rho=rho, phi_rad=phi, metrics=metrics,
                           input_energy=result.input_energy, output_energy=result.output_energy,
                           energy_transmission=result.energy_transmission,
                           profile_row=row, profile_y_um=float(result.y_um[row]))
    textbook_config = runner.TextbookExperimentConfig()
    _, textbook_image = runner._make_base_image(
        textbook_config.samples_per_mirror, textbook_config.numerical_aperture,
        textbook_config,
    )
    textbook_metrics = runner.calculate_periodic_profile_metrics(
        textbook_image.x_um, runner._extract_center_horizontal_profile(textbook_image),
        textbook_config.pitch_um, textbook_config.on_width_um,
        textbook_config.phase_offset_um,
    )
    arrays.update(textbook_field=textbook_image.image_field,
                  textbook_intensity=textbook_image.raw_intensity)
    np.savez_compressed(output / "fixed_na0065_baseline.npz", **arrays)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(source.iterdir()) if path.suffix in (".py", ".md")}
    report = dict(source_directory=str(source), source_sha256=hashes,
                  python=sys.version, numpy=np.__version__,
                  geometry=dict(num_mirrors_x=64, num_mirrors_y=32,
                                dmd_mirror_pitch_um=7.56, projected_mirror_pitch_um=1.5,
                                samples_per_mirror=16, active_side_ratio=0.95,
                                on_width_um=6.0, pitch_um=12.0, phase_offset_um=0.0),
                  optical=dict(wavelength_um=0.405, numerical_aperture=0.065),
                  workpoint=workpoint, cases=cases,
                  textbook=dict(config=asdict(textbook_config), metrics=asdict(textbook_metrics),
                                input_energy=textbook_image.input_energy,
                                output_energy=textbook_image.output_energy))
    (output / "fixed_na0065_baseline.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"baseline": str(output), "workpoint": workpoint,
                      "source_sha256": hashes}, indent=2))


if __name__ == "__main__":
    main()
