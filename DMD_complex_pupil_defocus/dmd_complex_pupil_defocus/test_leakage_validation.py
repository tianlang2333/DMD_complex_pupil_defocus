"""ON/OFF 等效响应的少量独立验证；生产计算仍使用原二维成像链。

直接执行本文件会打印验证结果；``--output-dir`` 另存 JSON。
不引入测试框架。验证阈值均在运行前给定，失败也会完整返回结果。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from coherent_imaging_2d import OpticalConfig2D, calculate_coherent_aerial_image_2d
from dmd_model_2d import (
    DMDConfig,
    make_periodic_line_space_states,
    render_dmd_field,
    with_state_response,
)

if TYPE_CHECKING:
    from run_dmd_2d_validation import ValidationCheck


def _config(samples_per_mirror: int = 16, active_side_ratio: float = 0.95) -> DMDConfig:
    """4 个完整周期，y 方向保持较小以控制轻量自检内存。"""
    return DMDConfig(
        num_mirrors_x=32, num_mirrors_y=8,
        dmd_mirror_pitch_um=7.56, projection_magnification=1.5 / 7.56,
        samples_per_mirror=samples_per_mirror, active_side_ratio=active_side_ratio,
    )


def _response(config: DMDConfig, eta: float = 1.0, rho: float = 0.01,
              phi: float = 0.7) -> DMDConfig:
    return with_state_response(
        config, on_intensity_scale=eta, off_to_on_intensity_ratio=rho,
        off_relative_phase_rad=phi,
    )


def _image(config: DMDConfig, states: np.ndarray | None = None):
    if states is None:
        states = make_periodic_line_space_states(config, 12.0, 6.0)
    pattern = render_dmd_field(states, config)
    image = calculate_coherent_aerial_image_2d(
        pattern.object_field, pattern.x_um, pattern.y_um,
        OpticalConfig2D(wavelength_um=0.405, numerical_aperture=0.065),
    )
    return pattern, image


def _profile(image) -> np.ndarray:
    """偶数网格在 y=0 两侧均值；此周期案例在 y 方向像场均匀。"""
    middle = image.y_um.size // 2
    return np.mean(image.raw_intensity[middle - 1:middle + 1], axis=0)


def run_leakage_validations() -> list[ValidationCheck]:
    """返回真实执行结果，不因单组失败丢失后续证据。"""
    # 放在函数内导入，主入口调用本文件时不会产生循环导入。
    import run_dmd_2d_validation as runner

    checks = []

    def check(name, value, limit, detail):
        checks.append(runner.ValidationCheck(
            name=name, value=float(value), limit=float(limit), comparison="<=",
            passed=bool(np.isfinite(value) and value <= limit), detail=detail,
        ))

    def require(name, condition, detail):
        check(name, 0.0 if condition else 1.0, 0.0, detail)

    def parameters_and_algebra():
        config = _config()
        states = make_periodic_line_space_states(config, 12.0, 6.0)
        old_pattern, old_image = _image(config, states)
        baseline = _response(config, rho=0.0, phi=0.0)
        _, baseline_image = _image(baseline, states)
        check("leakage_zero_recovers_old_model",
              np.max(np.abs(old_image.raw_intensity - baseline_image.raw_intensity)),
              1e-13, "eta=1,rho=0 与原配置使用同一二维渲染/传播。")

        # 特意先放入非单位旧系数，验证替换而非与旧值重复相乘。
        previous = replace(config, on_amplitude=2 + 3j, off_amplitude=-4j)
        mapped = _response(previous, eta=0.36, rho=0.04, phi=0.5 * math.pi)
        amplitude_error = max(abs(mapped.on_amplitude - 0.6),
                              abs(mapped.off_amplitude - 0.12j))
        check("leakage_sqrt_ratio_and_replace", amplitude_error, 1e-14,
              "eta=.36,rho=.04,phi=pi/2 应得 Aon=.6,Aoff=.12j。")
        require("leakage_config_copy_preserves_geometry",
                mapped is not previous and previous.on_amplitude == 2 + 3j
                and mapped.samples_per_mirror == previous.samples_per_mirror
                and mapped.active_side_ratio == previous.active_side_ratio,
                "原冻结配置不变，返回配置副本。")

        invalid_values = {
            "on_intensity_scale": (-1.0, math.nan, math.inf, 1 + 0j, True, "1"),
            "off_to_on_intensity_ratio": (-0.01, 1.01, math.nan, math.inf, 1j, False),
            "off_relative_phase_rad": (math.nan, math.inf, -math.inf, 1j, True),
        }
        rejected, count = 0, 0
        for parameter, values in invalid_values.items():
            for value in values:
                kwargs = dict(on_intensity_scale=1.0, off_to_on_intensity_ratio=0.01,
                              off_relative_phase_rad=0.0)
                kwargs[parameter] = value
                count += 1
                try:
                    with_state_response(config, **kwargs)
                except ValueError:
                    rejected += 1
        require("leakage_invalid_parameters", rejected == count,
                f"有限实数/范围校验：正确拒绝 {rejected}/{count} 项。")

        response = _response(config)
        _, image = _image(response, states)
        _, scaled_image = _image(_response(config, eta=0.37), states)
        check("leakage_eta_intensity_scaling",
              np.max(np.abs(scaled_image.raw_intensity - 0.37 * image.raw_intensity)),
              3e-13, "相同 rho,phi 下 I(eta=.37)=.37 I(eta=1)。")
        _, full_on = _image(response, np.ones_like(states))
        _, full_off = _image(response, np.zeros_like(states))
        check("leakage_full_off_on_ratio",
              np.max(np.abs(full_off.raw_intensity - 0.01 * full_on.raw_intensity)),
              3e-13, "有限开口不变，全 OFF/ON 原始强度比为 rho=.01。")

        _, on_basis = _image(replace(config, on_amplitude=1, off_amplitude=0), states)
        _, off_basis = _image(replace(config, on_amplitude=0, off_amplitude=1), states)
        combined = response.on_amplitude * on_basis.image_field + response.off_amplitude * off_basis.image_field
        check("leakage_complex_field_superposition",
              np.max(np.abs(image.image_field - combined)), 3e-13,
              "总场一次传播等于 ON/OFF 基场传播后复数合成；不按强度相加。")
        incoherent_sum = (abs(response.on_amplitude) ** 2 * on_basis.raw_intensity
                          + abs(response.off_amplitude) ** 2 * off_basis.raw_intensity)
        difference = float(np.max(np.abs(image.raw_intensity - incoherent_sum)))
        require("leakage_nonzero_interference_term", difference > 0.01,
                f"与错误的分态强度相加最大差={difference:.9g}，预期>0.01。")

        _, periodic_phase = _image(_response(config, phi=0.7 + 2 * math.pi), states)
        check("leakage_phase_2pi_periodicity",
              np.max(np.abs(image.raw_intensity - periodic_phase.raw_intensity)),
              3e-13, "phi 和 phi+2pi 在固定标尺上比较。")
        _, conjugate_phase = _image(_response(config, phi=-0.7), states)
        check("leakage_phase_sign_symmetry_real_pupil",
              np.max(np.abs(image.raw_intensity - conjugate_phase.raw_intensity)),
              3e-13, "本轮实数基场/实偶圆瞳条件下 I(phi)=I(-phi)。")

        equal = _response(config, rho=1.0, phi=0.0)
        equal_pattern, equal_image = _image(equal, states)
        _, inverted_equal_image = _image(equal, ~states)
        check("leakage_equal_complex_states_remove_modulation",
              np.max(np.abs(equal_image.raw_intensity - inverted_equal_image.raw_intensity)),
              3e-13, "Aon=Aoff，翻转状态矩阵不改变场；开口仍保留。")
        require("leakage_equal_states_retain_aperture",
                float(np.ptp(np.abs(equal_pattern.object_field))) > 0.1,
                "相等双态不移除镜间隙及部分覆盖权重。")
        _, opposite_image = _image(_response(config, rho=1.0, phi=math.pi), states)
        opposite_difference = float(np.max(np.abs(opposite_image.raw_intensity - equal_image.raw_intensity)))
        require("leakage_equal_modulus_is_not_equal_response", opposite_difference > 0.1,
                f"同模反相仍有调制；与同相像强度最大差={opposite_difference:.9g}。")

        # .95 开口在低采样下含部分覆盖单元；验证系数乘在权重上。
        fractional = render_dmd_field(np.zeros_like(states), response)
        expected = np.tile(fractional.mirror_aperture_template, states.shape) * response.off_amplitude
        check("leakage_off_respects_fractional_aperture",
              np.max(np.abs(fractional.object_field - expected)), 1e-14,
              "部分覆盖单元继续保留原开口权重，不能整体填满 OFF 像元。")
        # .8/20 含明确的零间隙，避免只检查 .95/8 灰边界而遗漏真空隙。
        gap_config = _response(_config(20, 0.8))
        gap_pattern = render_dmd_field(np.zeros_like(states), gap_config)
        zero_mask = np.tile(gap_pattern.mirror_aperture_template == 0.0, states.shape)
        require("leakage_off_does_not_fill_gaps", np.any(zero_mask)
                and np.all(gap_pattern.object_field[zero_mask] == 0.0),
                f"active_side_ratio=.8,spm=20；{np.count_nonzero(zero_mask)} 个零间隙采样保持零。")

        _, zero_image = _image(_response(config, eta=0.0), states)
        zero_metrics = runner.evaluate_leakage_profile(
            zero_image.x_um, _profile(zero_image), on_width_um=6.0, pitch_um=12.0,
            center_um=0.0, reference_intensity=1.0, threshold_intensity=0.5,
            search_bounds_um=(-6.0, 6.0), dark_bounds_um=(4.5, 7.5),
        )
        require("leakage_eta_zero_and_invalid_metrics",
                np.all(zero_image.raw_intensity == 0.0)
                and math.isnan(zero_metrics["fixed_cd_um"])
                and math.isnan(zero_metrics["design_edge_nils"])
                and math.isnan(zero_metrics["center_contrast"])
                and zero_metrics["threshold_status"] != "VALID",
                f"全零场 CD/NILS 为 NaN；状态={zero_metrics['threshold_status']}。")

    def threshold_behavior():
        x = np.linspace(-8.0, 8.0, 321)
        triangle = np.maximum(1.0 - np.abs(x) / 4.0, 0.0)
        base = runner.measure_threshold_interval(
            x, triangle, 0.0, threshold_intensity=0.5, search_bounds_um=(-6.0, 6.0))
        scaled = runner.measure_threshold_interval(
            x, 0.8 * triangle, 0.0, threshold_intensity=0.5, search_bounds_um=(-6.0, 6.0))
        relative = runner.measure_threshold_interval(x, 0.8 * triangle, 0.0)
        check("leakage_fixed_threshold_not_renormalized",
              max(abs(base.width_um - 4.0), abs(scaled.width_um - 3.0),
                  abs(relative.width_um - 4.0)), 1e-12,
              f"解析三角形：基准 CD={base.width_um:g}；缩放后固定阈值 CD={scaled.width_um:g}；相对峰值 CD={relative.width_um:g}。")
        cases = [
            ("center_below", 0.4 * triangle, (-6.0, 6.0)),
            ("full_bright", np.ones_like(x), (-6.0, 6.0)),
            ("tight_search", triangle, (-1.0, 1.0)),
            ("one_edge_missing", triangle, (-6.0, 1.0)),
        ]
        for label, profile, bounds in cases:
            interval = runner.measure_threshold_interval(
                x, profile, 0.0, threshold_intensity=0.5, search_bounds_um=bounds)
            metrics = runner.evaluate_leakage_profile(
                x, profile, on_width_um=6.0, pitch_um=12.0, center_um=0.0,
                reference_intensity=1.0, threshold_intensity=0.5,
                search_bounds_um=bounds, dark_bounds_um=(4.5, 7.5),
            )
            require(f"leakage_threshold_invalid_{label}",
                    all(math.isnan(value) for value in (
                        interval.left_edge_um, interval.right_edge_um, interval.width_um))
                    and metrics["threshold_status"] != "VALID",
                    f"无双边缘不得给零 CD 或截断边界；状态={metrics['threshold_status']}。")

    def analytic_fourier():
        # 对居中矩形连续傅里叶系数，边缘对齐的中点采样使非零级次
        # 恰乘 1/sinc(j*dx/p)。此解析有限和不调用 FFT、不拟合位移/幅度。
        continuous_errors = []
        for spm in (16, 32):
            config = _response(_config(spm, 1.0), eta=0.8, rho=0.03, phi=0.7)
            pattern, image = _image(config)
            pitch, duty = 12.0, 0.5
            cutoff = 0.065 / 0.405
            maximum_order = math.floor(pitch * cutoff)
            orders = np.arange(-maximum_order, maximum_order + 1)
            coefficients = np.asarray([
                config.off_amplitude + (config.on_amplitude - config.off_amplitude) * duty
                if order == 0 else
                (config.on_amplitude - config.off_amplitude) * math.sin(math.pi * order * duty) / (math.pi * order)
                for order in orders], dtype=np.complex128)
            discrete_coefficients = coefficients.copy()
            nonzero = orders != 0
            discrete_coefficients[nonzero] /= np.sinc(orders[nonzero] * pattern.sample_spacing_um / pitch)
            exponentials = np.exp(2j * math.pi * orders[:, None] * pattern.x_um[None, :] / pitch)
            continuous_field = coefficients @ exponentials
            discrete_field = discrete_coefficients @ exponentials
            numeric_field = image.image_field[image.y_um.size // 2]
            intensity = _profile(image)
            continuous_intensity = np.abs(continuous_field) ** 2
            continuous_error = float(np.max(np.abs(intensity - continuous_intensity)))
            continuous_errors.append(continuous_error)
            field_error_bound = float(np.sum(np.abs(discrete_coefficients - coefficients)))
            field_amplitude_bound = float(np.sum(np.abs(coefficients)))
            intensity_bound = 2 * field_amplitude_bound * field_error_bound + field_error_bound ** 2
            check(f"leakage_analytic_exact_midpoint_field_spm{spm}",
                  np.max(np.abs(numeric_field - discrete_field)), 3e-12,
                  "独立有限 Fourier 和；中点修正已知，无位移拟合、无强度重缩放。")
            check(f"leakage_analytic_continuous_bound_spm{spm}",
                  continuous_error, intensity_bound + 3e-12,
                  f"绝对强度误差由 2*B*delta+delta^2 界定；dx={pattern.sample_spacing_um:g} um，order={maximum_order}。")
        ratio = continuous_errors[1] / continuous_errors[0]
        check("leakage_analytic_refinement_error_ratio", ratio, 0.30,
              f"中点采样 O(dx^2)：16→32 预期误差约 1/4；误差={continuous_errors[0]:.9g},{continuous_errors[1]:.9g}。")

    def finite_aperture_convergence():
        # 同一个高采样无漏光基准定标一次，粗细网格共用强度阈值。
        fine_config = _config(32, 0.95)
        _, baseline_image = _image(_response(fine_config, rho=0.0, phi=0.0))
        baseline_profile = _profile(baseline_image)
        evaluation = np.abs(baseline_image.x_um) <= 6.0
        reference = float(np.max(baseline_profile[evaluation]))
        threshold = 0.5 * reference
        results = []
        for spm in (16, 32):
            _, image = _image(_response(_config(spm, 0.95), rho=0.01, phi=0.7))
            profile = _profile(image)
            metrics = runner.evaluate_leakage_profile(
                image.x_um, profile, on_width_um=6.0, pitch_um=12.0, center_um=0.0,
                reference_intensity=reference, threshold_intensity=threshold,
                search_bounds_um=(-6.0, 6.0), dark_bounds_um=(4.5, 7.5),
            )
            results.append((image.x_um, profile, metrics))
        require("leakage_common_reference_preserved",
                all(m["reference_intensity"] == reference
                    and m["threshold_intensity"] == threshold for _, _, m in results),
                "粗细采样评价返回完全相同的输入参考强度和固定阈值。")
        coarse_x, coarse_profile, coarse = results[0]
        fine_x, fine_profile, fine = results[1]
        center = np.abs(coarse_x) <= 6.0
        difference = coarse_profile[center] - np.interp(coarse_x[center], fine_x, fine_profile)
        normalized_max_error = float(np.max(np.abs(difference)) / reference)
        check("leakage_finite_aperture_profile_16_to_32",
              normalized_max_error, 0.01,
              f"有限开口 .95,rho=.01,phi=.7；固定 Iref={reference:.9g},Ith={threshold:.9g}；预设截面变化≤1%。")
        cd_change = abs(coarse["fixed_cd_um"] - fine["fixed_cd_um"])
        check("leakage_finite_aperture_cd_16_to_32", cd_change, 0.01,
              f"固定阈值 CD：16={coarse['fixed_cd_um']:.9g},32={fine['fixed_cd_um']:.9g} um；预设差≤.01 um。")
        nils_change = abs(coarse["design_edge_nils"] - fine["design_edge_nils"]) / abs(fine["design_edge_nils"])
        check("leakage_finite_aperture_nils_16_to_32", nils_change, 0.01,
              f"设计边缘 NILS：16={coarse['design_edge_nils']:.9g},32={fine['design_edge_nils']:.9g}；预设相对差≤1%。")
        # 对称图形与实偶光瞳不应产生无故的中心漂移。
        center_shift = max(abs(0.5 * (m["fixed_left_edge_um"] + m["fixed_right_edge_um"]))
                           for _, _, m in results)
        check("leakage_symmetric_pattern_center", center_shift, 2e-12,
              "固定阈值左右边缘的中心位置；统一双态响应保持几何对称性。")

    def nominal_workpoint_and_dose_formula():
        # 独立分段线性三角形：I=max(1-|x|/4,0)。目标 CD=6 时
        # T0=.25；I>=T0/s 的解析宽度为 CD(s)=8-2/s，非拟合预期。
        x = np.linspace(-8.0, 8.0, 321)
        triangle = np.maximum(1.0 - np.abs(x) / 4.0, 0.0)
        inputs = dict(on_width_um=6.0, center_um=0.0, reference_intensity=1.0,
                      search_bounds_um=(-6.0, 6.0))
        target = runner.establish_nominal_workpoint(
            x, triangle, **inputs, threshold_mode="target_cd")
        legacy = runner.establish_nominal_workpoint(x, triangle, **inputs)
        check("nominal_triangle_target_and_legacy",
              max(abs(target["nominal_threshold_intensity"] - 0.25),
                  abs(target["nominal_threshold_fraction"] - 0.25),
                  abs(target["nominal_baseline_cd_um"] - 6.0),
                  abs(legacy["nominal_threshold_intensity"] - 0.5),
                  abs(legacy["nominal_baseline_cd_um"] - 4.0)), 1e-12,
              "独立解析三角形：target_cd 得 T0=.25,CD=6；默认 fixed_fraction 得 T0=.5,CD=4。")
        require("nominal_workpoint_metadata",
                target["threshold_mode"] == "target_cd"
                and target["target_cd_um"] == 6.0
                and target["reference_intensity"] == 1.0
                and target["design_left_intensity"] == 0.25
                and target["design_right_intensity"] == 0.25,
                "名义工作点保存模式、目标尺寸、共同强度参考及两侧设计边缘强度。")
        check("nominal_workpoint_declared_tolerances",
              max(abs(target["intensity_symmetry_tolerance"] - 1e-10) / 1e-10,
                  abs(target["cd_tolerance_um"] - 6e-9) / 6e-9), 1e-12,
              "按约定输出强度容差1e-10*Iref，CD容差1e-9*max(1,w) um。")

        dose_errors, equivalent_errors = [], []
        widths = []
        for scale in (0.95, 1.0, 1.05):
            threshold = runner.effective_threshold_for_dose(
                target["nominal_threshold_intensity"], scale)
            measured = runner.measure_threshold_interval(
                x, triangle, 0.0, threshold_intensity=threshold,
                search_bounds_um=(-6.0, 6.0))
            direct_dose = runner.measure_threshold_interval(
                x, scale * triangle, 0.0, threshold_intensity=0.25,
                search_bounds_um=(-6.0, 6.0))
            widths.append(measured.width_um)
            dose_errors.extend((abs(threshold - 0.25 / scale),
                                abs(measured.width_um - (8.0 - 2.0 / scale))))
            equivalent_errors.append(abs(measured.width_um - direct_dose.width_um))
        check("nominal_dose_independent_triangle_formula", max(dose_errors), 1e-12,
              f"s=.95,1,1.05；解析 CD(s)=8-2/s，实测 CD={widths} um。")
        check("nominal_dose_threshold_equivalence", max(equivalent_errors), 1e-12,
              "在相同线性插值坐标上 I>=T0/s 与 sI>=T0 的交点一致。")

        # target_cd 严格要求能用一个阈值表达目标中央连通区；不对异常
        # 图形取平均后假称已建立目标工作点。旧模式继续容许无效基准 CD。
        disconnected = np.interp(
            x, [-8, -4, -3, -2, -1, 0, 1, 2, 3, 4, 8],
            [0, 0, 0.5, 0, 0, 1, 0, 0, 0.5, 0, 0])
        invalid_profiles = (
            ("asymmetric", triangle * (1.0 + 0.01 * x)),
            ("disconnected", disconnected),
            ("all_bright", np.ones_like(x)),
            ("zero_design_edge", np.maximum(1.0 - np.abs(x) / 2.0, 0.0)),
        )
        for name, profile in invalid_profiles:
            rejected = False
            try:
                runner.establish_nominal_workpoint(
                    x, profile, **inputs, threshold_mode="target_cd")
            except ValueError:
                rejected = True
            require(f"nominal_target_rejects_{name}", rejected,
                    "target_cd 必须报错，不能静默使用错误阈值或相邻/分离区域边缘。")
        old_invalid = runner.establish_nominal_workpoint(x, np.ones_like(x), **inputs)
        require("nominal_legacy_keeps_invalid_baseline_behavior",
                math.isnan(old_invalid["nominal_baseline_cd_um"])
                and old_invalid["nominal_threshold_intensity"] == 0.5,
                "默认 fixed_fraction 模式保持无双边缘基准 CD=NaN 的原行为。")

        invalid_inputs = (
            {"threshold_mode": "unsupported"},
            {"on_width_um": 0.0}, {"reference_intensity": 0.0},
            {"reference_intensity": math.nan}, {"search_bounds_um": (-1.0, 1.0)},
        )
        rejected = 0
        for changes in invalid_inputs:
            options = {**inputs, "threshold_mode": "target_cd", **changes}
            try:
                runner.establish_nominal_workpoint(x, triangle, **options)
            except ValueError:
                rejected += 1
        require("nominal_workpoint_invalid_inputs", rejected == len(invalid_inputs),
                f"模式/尺寸/强度参考/目标搜索范围：拒绝{rejected}/{len(invalid_inputs)}项。")

        dose_invalid = (0.0, -0.1, math.nan, math.inf, -math.inf,
                        True, np.bool_(False), 1 + 0j, "1")
        rejected = 0
        for scale in dose_invalid:
            try:
                runner.effective_threshold_for_dose(0.25, scale)
            except ValueError:
                rejected += 1
        for nominal_threshold in (0.0, -0.1, math.nan, math.inf):
            try:
                runner.effective_threshold_for_dose(nominal_threshold, 1.0)
            except ValueError:
                rejected += 1
        require("nominal_dose_rejects_invalid_inputs", rejected == len(dose_invalid) + 4,
                f"s 必须为有限正实数（排除bool/复数/字符串），T0必须有限正数：拒绝{rejected}/{len(dose_invalid) + 4}项。")

    def nominal_default_dmd_and_dose_invariance():
        # 与默认演示一致的64×32镜、16采样/镜、.95开口；仅两次原二维传播。
        config = replace(_config(), num_mirrors_x=64, num_mirrors_y=32)
        _, baseline = _image(_response(config, rho=0.0, phi=0.0))
        x = baseline.x_um
        baseline_profile = _profile(baseline)
        reference = float(np.max(baseline_profile[np.abs(x) <= 6.0]))
        workpoint = runner.establish_nominal_workpoint(
            x, baseline_profile, on_width_um=6.0, center_um=0.0,
            reference_intensity=reference, search_bounds_um=(-6.0, 6.0),
            threshold_mode="target_cd")
        check("nominal_default_dmd_target_cd",
              abs(workpoint["nominal_baseline_cd_um"] - 6.0), 6e-9,
              f"默认DMD无漏光基准：CD={workpoint['nominal_baseline_cd_um']:.12g} um，"
              f"T0={workpoint['nominal_threshold_intensity']:.12g}，"
              f"T0/Iref={workpoint['nominal_threshold_fraction']:.12g}。")
        _, image = _image(_response(config, rho=0.01, phi=0.0))
        profile = _profile(image)
        original_profile = profile.copy()
        metrics = []
        for scale in (0.95, 1.0, 1.05):
            metrics.append(runner.evaluate_leakage_profile(
                x, profile, on_width_um=6.0, pitch_um=12.0, center_um=0.0,
                reference_intensity=reference,
                threshold_intensity=runner.effective_threshold_for_dose(
                    workpoint["nominal_threshold_intensity"], scale),
                search_bounds_um=(-6.0, 6.0), dark_bounds_um=(4.5, 7.5)))
        require("nominal_dose_preserves_raw_intensity", np.array_equal(profile, original_profile),
                "相同代表响应的原始强度截面不因dose_scale改变或被就地缩放。")
        check("nominal_dose_preserves_design_nils",
              max(abs(item["design_edge_nils"] - metrics[1]["design_edge_nils"])
                  for item in metrics), 1e-13,
              "dose_scale仅移动评价阈值，设计边缘NILS由同一原始强度计算。")
        widths = [item["fixed_cd_um"] for item in metrics]
        require("nominal_default_dose_changes_threshold_cd",
                all(math.isfinite(value) for value in widths) and widths[0] < widths[1] < widths[2],
                f"默认代表图形的中央单调边缘：s=.95,1,1.05，CD={widths} um。")

    for group in (parameters_and_algebra, threshold_behavior, analytic_fourier,
                  finite_aperture_convergence, nominal_workpoint_and_dose_formula,
                  nominal_default_dmd_and_dose_invariance):
        try:
            group()
        except Exception as error:
            require(f"leakage_{group.__name__}_execution", False,
                    f"{type(error).__name__}: {error}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args()
    checks = run_leakage_validations()
    for item in checks:
        print(f"{'PASS' if item.passed else 'FAIL'} {item.name}: "
              f"{item.value:.9g} <= {item.limit:.9g}; {item.detail}")
    if arguments.output_dir is not None:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        target = arguments.output_dir / "leakage_validation_checks.json"
        target.write_text(json.dumps([asdict(item) for item in checks], ensure_ascii=False,
                                     indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {target.resolve()}")
    raise SystemExit(0 if all(item.passed for item in checks) else 1)


if __name__ == "__main__":
    main()
