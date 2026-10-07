"""Independent numerical checks for the complex-pupil/physical-defocus increment.

No extra test framework is needed.  Run ``python test_projection_lens_2d.py``.
Expected finite-mode fields are direct plane-wave sums; they never use the
production pupil or its filtered spectrum.  All tolerances below are declared
before execution.  The baseline was captured from the unmodified source tree.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import sys
import traceback

import matplotlib
import numpy as np

from coherent_imaging_2d import (
    OpticalConfig2D, calculate_coherent_aerial_image_2d,
    make_frequency_axes, propagate_coherent_field_from_pupil,
)
from dmd_model_2d import (
    DMDConfig, make_field_coordinates, make_periodic_line_space_states,
    render_dmd_field, with_state_response,
)
from projection_lens_2d import make_defocused_circular_pupil
import run_dmd_2d_validation as runner


MACHINE_ATOL = 1e-12
CONVERGENCE_PROFILE_REL_MAX = 1e-3
CONVERGENCE_CD_ATOL_UM = 0.01
CONVERGENCE_NILS_REL = 0.01
TEXTBOOK_RAW_RMSE_MAX = 1e-3  # sampled rectangles vs continuous Fourier coefficients
OPTICAL = OpticalConfig2D(wavelength_um=0.405, numerical_aperture=0.065)


class Checks:
    def __init__(self) -> None:
        self.records: list[dict] = []
        self.details: dict = {}

    def error(self, name: str, value: float, tolerance: float, detail: str = "") -> None:
        self.records.append(dict(name=name, status="PASS" if np.isfinite(value)
                                 and value <= tolerance else "FAIL",
                                 error=float(value), tolerance=float(tolerance), detail=detail))

    def require(self, name: str, condition: bool, detail: str = "") -> None:
        self.error(name, 0.0 if condition else 1.0, 0.0, detail)

    def reject(self, name: str, function) -> None:
        try:
            function()
        except ValueError as error:
            self.require(name, True, str(error))
        else:
            self.require(name, False, "Expected ValueError was not raised.")

    def values(self, name: str, actual, expected) -> None:
        if isinstance(expected, dict):
            self.require(name + ".keys", set(actual) == set(expected))
            for key in expected:
                self.values(f"{name}.{key}", actual[key], expected[key])
        elif isinstance(expected, (bool, str)):
            self.require(name, actual == expected)
        elif expected is None:
            self.require(name, actual is None)
        elif np.isnan(expected):
            self.require(name, bool(np.isnan(actual)), "NaN retained in both paths.")
        else:
            self.error(name, abs(float(actual) - float(expected)), MACHINE_ATOL)


def geometry(samples: int = 16) -> DMDConfig:
    return DMDConfig(num_mirrors_x=64, num_mirrors_y=32,
                     dmd_mirror_pitch_um=7.56, projection_magnification=1.5/7.56,
                     samples_per_mirror=samples, active_side_ratio=0.95)


def dmd_image(config: DMDConfig, z: float | None = None):
    states = make_periodic_line_space_states(config, 12.0, 6.0, phase_offset_um=0.0)
    pattern = render_dmd_field(states, config)
    if z is None:
        pupil = None
    else:
        fx, fy = make_frequency_axes(config.field_num_samples_x, config.field_num_samples_y,
                                     config.sample_spacing_um, config.sample_spacing_um)
        pupil = make_defocused_circular_pupil(fx, fy, OPTICAL, z)
    return calculate_coherent_aerial_image_2d(
        pattern.object_field, pattern.x_um, pattern.y_um, OPTICAL, pupil=pupil)


def profile_metrics(result, workpoint: dict) -> dict:
    row = int(np.argmin(abs(result.y_um)))
    return runner.evaluate_leakage_profile(
        result.x_um, result.raw_intensity[row], on_width_um=6.0, pitch_um=12.0,
        center_um=0.0, reference_intensity=workpoint["reference_intensity"],
        threshold_intensity=workpoint["nominal_threshold_intensity"],
        search_bounds_um=(-6.0, 6.0), dark_bounds_um=(4.5, 7.5))


def check_legality(checks: Checks) -> None:
    for magnification in (0.424, 1.0, 1.3):
        config = replace(geometry(), num_mirrors_x=16, num_mirrors_y=8,
                         projection_magnification=magnification, samples_per_mirror=4)
        expected_pitch = 7.56 * magnification
        checks.error(f"magnification_{magnification}.pitch",
                     abs(config.projected_mirror_pitch_um - expected_pitch), MACHINE_ATOL)
        x, y = make_field_coordinates(config)
        expected_dx = expected_pitch / 4
        checks.error(f"magnification_{magnification}.coordinates",
                     max(np.max(abs(np.diff(x) - expected_dx)),
                         np.max(abs(np.diff(y) - expected_dx)), abs(x[0] + x[-1]),
                         abs(y[0] + y[-1])), MACHINE_ATOL)
        states = make_periodic_line_space_states(config, 8*expected_pitch, 4*expected_pitch)
        pattern = render_dmd_field(states, config)
        image = calculate_coherent_aerial_image_2d(pattern.object_field, x, y, OPTICAL)
        checks.require(f"magnification_{magnification}.render_propagate",
                       image.raw_intensity.shape == (32, 64) and np.all(np.isfinite(image.image_field)))
    for number, value in enumerate((0, -1, np.nan, np.inf, -np.inf, True, np.bool_(False), 1+0j)):
        checks.reject(f"invalid_magnification_{number}",
                      lambda value=value: replace(geometry(), projection_magnification=value))
    checks.reject("overflow_projected_pitch",
                  lambda: replace(geometry(), dmd_mirror_pitch_um=1e308, projection_magnification=1e308))
    checks.reject("underflow_projected_pitch",
                  lambda: replace(geometry(), dmd_mirror_pitch_um=1e-308, projection_magnification=1e-308))
    checks.reject("underflow_sample_spacing",
                  lambda: replace(geometry(), dmd_mirror_pitch_um=5e-324, projection_magnification=1.0))
    # These are configuration legality checks, not an assertion that every
    # arbitrary magnification also meets an experiment's integer-period rules.
    for magnification in (1.0, 1.3):
        projected_pitch = 7.56*magnification
        textbook = runner.TextbookExperimentConfig(projected_mirror_pitch_um=projected_pitch)
        cross = runner.CrossExperimentConfig(projected_mirror_pitch_um=projected_pitch)
        sampling = runner.SamplingExperimentConfig(n_cd_values=(6.0/projected_pitch,))
        for name, actual in (("textbook", textbook.projection_magnification),
                             ("cross", cross.projection_magnification),
                             ("sampling", sampling.projected_mirror_pitch_um(sampling.n_cd_values[0])/7.56)):
            checks.error(f"experiment_{name}_magnification_{magnification}",
                         abs(actual-magnification), MACHINE_ATOL)
    for name, cls in (("textbook", runner.TextbookExperimentConfig),
                      ("cross", runner.CrossExperimentConfig)):
        for label, options in (
            ("ratio_overflow", dict(projected_mirror_pitch_um=1e308, dmd_mirror_pitch_um=1e-308)),
            ("ratio_underflow", dict(projected_mirror_pitch_um=1e-308, dmd_mirror_pitch_um=1e308)),
            ("spacing_underflow", dict(projected_mirror_pitch_um=5e-324)),
        ):
            checks.reject(f"experiment_{name}_{label}", lambda cls=cls, options=options: cls(**options))
    checks.reject("experiment_textbook_period_overflow", lambda: runner.TextbookExperimentConfig(
        on_width_um=1e308, line_width_um=1e308))
    for label, options in (
        ("pitch_overflow", dict(target_on_width_um=1e308, n_cd_values=(1e-308,))),
        ("pitch_underflow", dict(target_on_width_um=1e-308, n_cd_values=(1e308,))),
        ("ratio_underflow", dict(target_on_width_um=1e-300, dmd_mirror_pitch_um=1e300, n_cd_values=(1.0,))),
        ("spacing_underflow", dict(target_on_width_um=5e-324, n_cd_values=(1.0,))),
        ("period_overflow", dict(target_on_width_um=1e308, target_line_width_um=1e308)),
    ):
        checks.reject(f"experiment_sampling_{label}", lambda options=options: runner.SamplingExperimentConfig(**options))
    field = np.ones((4, 6), dtype=complex)
    invalid_pupils = (np.ones((4, 5)), np.ones(24), np.ones((0, 0)),
                      np.full((4, 6), np.nan), np.full((4, 6), np.inf),
                      np.full((4, 6), 1+0.2j), np.full((4, 6), -1.01))
    for number, pupil in enumerate(invalid_pupils):
        checks.reject(f"invalid_pupil_{number}",
                      lambda pupil=pupil: propagate_coherent_field_from_pupil(field, pupil))
    fx, fy = make_frequency_axes(6, 4, 0.5, 0.5)
    for number, z in enumerate((True, np.bool_(False), 1+0j, 1+1j, np.nan, np.inf, "0")):
        checks.reject(f"invalid_defocus_{number}",
                      lambda z=z: make_defocused_circular_pupil(fx, fy, OPTICAL, z))
    checks.reject("air_na_above_one", lambda: make_defocused_circular_pupil(
        fx, fy, OpticalConfig2D(0.405, 1.01), 1.0))
    # Air grazing boundary is permitted; evanescent samples outside the pupil
    # must not enter sqrt even though this frequency grid extends far beyond 1/lambda.
    edge = np.array([-2.0, -1.0, 0.0, 1.0])
    with np.errstate(invalid="raise"):
        pupil = make_defocused_circular_pupil(edge, edge, OpticalConfig2D(1.0, 1.0), 0.3)
    checks.require("air_na_one_finite", np.all(np.isfinite(pupil)))
    checks.error("air_na_one_grazing_phase", abs(pupil[2, 3] - np.exp(-0.6j*np.pi)), MACHINE_ATOL)


def check_baseline(checks: Checks, baseline_dir: Path) -> None:
    metadata = json.loads((baseline_dir / "fixed_na0065_baseline.json").read_text(encoding="utf-8"))
    arrays = np.load(baseline_dir / "fixed_na0065_baseline.npz", allow_pickle=False)
    checks.details["original_source_sha256"] = metadata["source_sha256"]
    checks.details["baseline_workpoint"] = metadata["workpoint"]
    for name, case in metadata["cases"].items():
        config = with_state_response(geometry(), on_intensity_scale=case["eta"],
                                     off_to_on_intensity_ratio=case["rho"],
                                     off_relative_phase_rad=case["phi_rad"])
        for path, z in (("default", None), ("explicit_zero", 0.0)):
            result = dmd_image(config, z)
            checks.error(f"baseline.{name}.{path}.field",
                         np.max(abs(result.image_field-arrays[f"{name}_field"])), MACHINE_ATOL)
            checks.error(f"baseline.{name}.{path}.raw_intensity",
                         np.max(abs(result.raw_intensity-arrays[f"{name}_intensity"])), MACHINE_ATOL)
            for attr in ("input_energy", "output_energy", "energy_transmission"):
                checks.error(f"baseline.{name}.{path}.{attr}",
                             abs(getattr(result, attr)-case[attr])/max(abs(case[attr]), 1), MACHINE_ATOL,
                             "Relative error for discrete sum energies; no physical-power claim.")
            checks.values(f"baseline.{name}.{path}.metrics",
                          profile_metrics(result, metadata["workpoint"]), case["metrics"])
    textbook_config = runner.TextbookExperimentConfig(**metadata["textbook"]["config"])
    _, result = runner._make_base_image(textbook_config.samples_per_mirror,
                                      textbook_config.numerical_aperture, textbook_config)
    checks.error("textbook.original_field", np.max(abs(result.image_field-arrays["textbook_field"])), MACHINE_ATOL)
    checks.error("textbook.original_intensity", np.max(abs(result.raw_intensity-arrays["textbook_intensity"])), MACHINE_ATOL)
    metrics = runner.calculate_periodic_profile_metrics(
        result.x_um, runner._extract_center_horizontal_profile(result), textbook_config.pitch_um,
        textbook_config.on_width_um, textbook_config.phase_offset_um)
    checks.values("textbook.original_metrics", asdict(metrics), metadata["textbook"]["metrics"])
    cross = runner.run_textbook_cross_validation(Path(__file__).parent, result, textbook_config)
    checks.require("textbook.independent_1d_executed", cross.status == "COMPLETED", cross.reason)
    checks.error("textbook.continuous_1d_raw_rmse", cross.raw_rmse, TEXTBOOK_RAW_RMSE_MAX,
                 "Finite grid vs continuous rectangle coefficients: not a machine-precision identity.")
    arrays.close()


def check_phase_and_energy(checks: Checks) -> None:
    rng = np.random.default_rng(73019)
    field = rng.normal(size=(17, 24)) + 1j*rng.normal(size=(17, 24))
    fx, fy = make_frequency_axes(24, 17, 1.0, 1.0)
    zero = make_defocused_circular_pupil(fx, fy, OPTICAL, 0.0)
    _, _, reference = propagate_coherent_field_from_pupil(field, zero)
    phase = 0.73
    _, _, shifted = propagate_coherent_field_from_pupil(field, zero*np.exp(1j*phase))
    checks.error("constant_phase.field", np.max(abs(shifted-reference*np.exp(1j*phase))), MACHINE_ATOL)
    checks.error("constant_phase.intensity", np.max(abs(abs(shifted)**2-abs(reference)**2)), MACHINE_ATOL)
    _, _, negative = propagate_coherent_field_from_pupil(field, -zero.real)
    checks.error("negative_real_pupil", np.max(abs(negative+reference)), MACHINE_ATOL)
    reference_energy = float(np.sum(abs(reference)**2))
    input_energy = float(np.sum(abs(field)**2))
    energy_rows = []
    for z in (-50.0, -25.0, 0.0, 25.0, 50.0):
        pupil = make_defocused_circular_pupil(fx, fy, OPTICAL, z)
        _, _, output = propagate_coherent_field_from_pupil(field, pupil)
        energy = float(np.sum(abs(output)**2))
        checks.error(f"defocus_energy_{z}", abs(energy-reference_energy)/reference_energy, MACHINE_ATOL)
        checks.error(f"passive_energy_{z}", max(0, energy/input_energy-1), MACHINE_ATOL)
        checks.error(f"pupil_modulus_{z}", np.max(abs(abs(pupil)-abs(zero))), MACHINE_ATOL)
        checks.error(f"pupil_dc_{z}", abs(pupil[np.argmin(abs(fy)), np.argmin(abs(fx))]-1), MACHINE_ATOL)
        energy_rows.append(dict(defocus_um=z, output_discrete_energy=energy))
    checks.details["phase_only_energy"] = dict(input_discrete_energy=input_energy, cases=energy_rows)
    # A supplied pupil is the complete transfer function.  Identity pupil must
    # pass even frequencies outside nominal NA; no hidden second circle allowed.
    x = (np.arange(24)-11.5)
    y = (np.arange(17)-8.0)
    unity = np.ones(field.shape, complex)
    original_field, original_pupil = field.copy(), unity.copy()
    image = calculate_coherent_aerial_image_2d(field, x, y, OPTICAL, pupil=unity)
    checks.error("complete_pupil_no_hidden_circle", np.max(abs(image.image_field-field)), MACHINE_ATOL)
    checks.require("inputs_not_mutated", np.array_equal(field, original_field)
                   and np.array_equal(unity, original_pupil))
    checks.error("nominal_cutoff_metadata", abs(image.frequency_cutoff_cyc_per_um-.065/.405), MACHINE_ATOL)


def check_independent_modes(checks: Checks) -> None:
    mode_info = []
    for nx, ny in ((63, 48), (64, 47)):
        dx, dy = 1.0, 1.25
        x = (np.arange(nx)-(nx-1)/2)*dx
        y = (np.arange(ny)-(ny-1)/2)*dy
        modes = ((0, 0, 0.45+0.03j), (3, 2, 0.2+0.17j),
                 (-5, 1, -0.11+0.09j), (2, -4, 0.05-0.13j), (17, 0, 0.16+0.08j))
        field = np.zeros((ny, nx), complex)
        components = []
        for mx, my, coefficient in modes:
            frequency_x, frequency_y = mx/(nx*dx), my/(ny*dy)
            wave = coefficient*np.exp(2j*np.pi*(frequency_x*x[None, :]+frequency_y*y[:, None]))
            field += wave
            components.append((frequency_x, frequency_y, wave))
        fx, fy = make_frequency_axes(nx, ny, dx, dy)
        outputs = {}
        for z in (-50.0, 0.0, 50.0):
            expected = np.zeros_like(field)
            k = 2*np.pi/OPTICAL.wavelength_um
            for frequency_x, frequency_y, wave in components:
                transverse_k_squared = (2*np.pi*frequency_x)**2 + (2*np.pi*frequency_y)**2
                if transverse_k_squared <= (k*OPTICAL.numerical_aperture)**2:
                    kz = np.sqrt(k*k-transverse_k_squared)
                    expected += wave*np.exp(1j*(kz-k)*z)
            pupil = make_defocused_circular_pupil(fx, fy, OPTICAL, z)
            result = calculate_coherent_aerial_image_2d(field, x, y, OPTICAL, pupil=pupil)
            checks.error(f"independent_modes_{nx}x{ny}_{z}.field",
                         np.max(abs(result.image_field-expected)), MACHINE_ATOL,
                         "Direct finite plane-wave sum, independent kz-k, no phase/scale/shift fitting.")
            checks.error(f"independent_modes_{nx}x{ny}_{z}.intensity",
                         np.max(abs(result.raw_intensity-abs(expected)**2)), MACHINE_ATOL)
            outputs[z] = result.raw_intensity.copy()
        checks.require(f"complex_target_distinguishes_sign_{nx}x{ny}",
                       np.max(abs(outputs[50.0]-outputs[-50.0])) > 1e-3)
        mode_info.append(dict(nx=nx, ny=ny, dx_um=dx, dy_um=dy,
                              sign_intensity_max_difference=float(np.max(abs(outputs[50.0]-outputs[-50.0])))))
    checks.details["independent_modes"] = mode_info


def check_convergence(checks: Checks) -> None:
    coarse_reference = dmd_image(geometry(16), 0.0)
    row = int(np.argmin(abs(coarse_reference.y_um)))
    reference_profile = coarse_reference.raw_intensity[row].copy()
    mask = abs(coarse_reference.x_um) <= 6.0
    workpoint = runner.establish_nominal_workpoint(
        coarse_reference.x_um, reference_profile, on_width_um=6.0, center_um=0.0,
        reference_intensity=float(np.max(reference_profile[mask])), search_bounds_um=(-6.0, 6.0),
        threshold_mode="target_cd")
    del coarse_reference
    profiles = {}
    metrics = {}
    grids = []
    for samples in (16, 32):
        config = geometry(samples)
        result = dmd_image(config, 50.0)
        row = int(np.argmin(abs(result.y_um)))
        dx = config.sample_spacing_um
        nx, ny = config.field_num_samples_x, config.field_num_samples_y
        fc = OPTICAL.frequency_cutoff_cyc_per_um
        # Need two-sided circle and intensity Nyquist (2*fc), not only field fc.
        checks.require(f"grid_{samples}.circle_coverage",
                       result.frequency_x_cyc_per_um[0] <= -fc <= fc <= result.frequency_x_cyc_per_um[-1]
                       and result.frequency_y_cyc_per_um[0] <= -fc <= fc <= result.frequency_y_cyc_per_um[-1])
        checks.require(f"grid_{samples}.intensity_sampling", 1/(2*dx) > 2*fc)
        metrics[samples] = profile_metrics(result, workpoint)
        profiles[samples] = (result.x_um.copy(), result.raw_intensity[row].copy())
        grids.append(dict(samples_per_mirror=samples, nx=nx, ny=ny, dx_um=dx, dy_um=dx,
                          lx_um=nx*dx, ly_um=ny*dx, dfx_cyc_per_um=1/(nx*dx),
                          dfy_cyc_per_um=1/(ny*dx), cutoff_cyc_per_um=fc,
                          nyquist_cyc_per_um=1/(2*dx), intensity_max_bandwidth_cyc_per_um=2*fc,
                          samples_per_shortest_intensity_period=1/(2*fc*dx),
                          profile_row=row, profile_y_um=float(result.y_um[row])))
        del result
    coarse_x, coarse_profile = profiles[16]
    fine_x, fine_profile = profiles[32]
    roi = abs(coarse_x) <= 6.0
    fine_on_coarse = np.interp(coarse_x[roi], fine_x, fine_profile)
    profile_error = np.max(abs(coarse_profile[roi]-fine_on_coarse))/workpoint["reference_intensity"]
    cd_error = abs(metrics[16]["fixed_cd_um"]-metrics[32]["fixed_cd_um"])
    nils_error = abs(metrics[16]["design_edge_nils"]-metrics[32]["design_edge_nils"])/abs(metrics[32]["design_edge_nils"])
    checks.error("convergence.profile_relative_max", profile_error, CONVERGENCE_PROFILE_REL_MAX,
                 "Raw fine profile linearly interpolated to coarse sample centers in [-6,6] um; denominator is shared nominal I_ref.")
    checks.error("convergence.cd_um", cd_error, CONVERGENCE_CD_ATOL_UM)
    checks.error("convergence.nils_relative", nils_error, CONVERGENCE_NILS_REL)
    checks.require("convergence.status", metrics[16]["threshold_status"] == metrics[32]["threshold_status"] == "VALID")
    checks.details["convergence"] = dict(defocus_um=50.0, active_side_ratio=0.95,
                                       common_workpoint=workpoint, grids=grids,
                                       coarse_metrics=metrics[16], fine_metrics=metrics[32],
                                       note="More samples per mirror increases Nyquist and aperture quadrature accuracy; longer windows refine frequency spacing. Integer-period boundaries are retained. This is not isolated-pattern window convergence.")


def check_complex_leakage(checks: Checks) -> None:
    config = geometry()
    states = make_periodic_line_space_states(config, 12.0, 6.0)
    complex_config = with_state_response(config, on_intensity_scale=1.0,
                                        off_to_on_intensity_ratio=0.01,
                                        off_relative_phase_rad=np.pi/3)
    pattern = render_dmd_field(states, complex_config)
    on_only = render_dmd_field(states, replace(config, on_amplitude=1, off_amplitude=0))
    off_only = render_dmd_field(states, replace(config, on_amplitude=0, off_amplitude=1))
    fx, fy = make_frequency_axes(config.field_num_samples_x, config.field_num_samples_y,
                                 config.sample_spacing_um, config.sample_spacing_um)
    pupil = make_defocused_circular_pupil(fx, fy, OPTICAL, 25.0)
    actual = calculate_coherent_aerial_image_2d(pattern.object_field, pattern.x_um, pattern.y_um, OPTICAL, pupil=pupil)
    on = calculate_coherent_aerial_image_2d(on_only.object_field, pattern.x_um, pattern.y_um, OPTICAL, pupil=pupil)
    off = calculate_coherent_aerial_image_2d(off_only.object_field, pattern.x_um, pattern.y_um, OPTICAL, pupil=pupil)
    expected = on.image_field + (0.1*np.exp(1j*np.pi/3))*off.image_field
    checks.error("complex_leakage.field_superposition", np.max(abs(actual.image_field-expected)), MACHINE_ATOL)
    checks.error("complex_leakage.intensity_after_superposition", np.max(abs(actual.raw_intensity-abs(expected)**2)), MACHINE_ATOL)
    incoherent = on.raw_intensity + 0.01*off.raw_intensity
    checks.require("complex_leakage_interference_retained", np.max(abs(actual.raw_intensity-incoherent)) > 1e-3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path(__file__).parent/"baseline")
    parser.add_argument("--report", type=Path, default=Path(__file__).parent/"projection_validation_report.json")
    args = parser.parse_args()
    checks = Checks()
    groups = (("A_legality", lambda: check_legality(checks)),
              ("B_original_zero_textbook", lambda: check_baseline(checks, args.baseline_dir)),
              ("C_phase_energy", lambda: check_phase_and_energy(checks)),
              ("D_independent_modes", lambda: check_independent_modes(checks)),
              ("E_sampling_convergence", lambda: check_convergence(checks)),
              ("F_complex_leakage", lambda: check_complex_leakage(checks)))
    for name, function in groups:
        try:
            function()
        except Exception:
            checks.records.append(dict(name=name, status="FAIL", error=None,
                                       tolerance=None, detail=traceback.format_exc()))
        print(f"Executed {name}")
    counts = {status: sum(record["status"] == status for record in checks.records)
              for status in ("PASS", "FAIL", "SKIPPED")}
    report = dict(environment=dict(python=sys.version, platform=platform.platform(),
                                   numpy=np.__version__, matplotlib=matplotlib.__version__),
                  tolerances=dict(machine_atol=MACHINE_ATOL,
                                  convergence_profile_relative_max=CONVERGENCE_PROFILE_REL_MAX,
                                  convergence_cd_atol_um=CONVERGENCE_CD_ATOL_UM,
                                  convergence_nils_relative=CONVERGENCE_NILS_REL,
                                  textbook_continuous_raw_rmse=TEXTBOOK_RAW_RMSE_MAX),
                  counts=counts, checks=checks.records, details=checks.details)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(counts), args.report)
    for record in checks.records:
        if record["status"] != "PASS":
            print(json.dumps(record, ensure_ascii=False))
    if counts["FAIL"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
