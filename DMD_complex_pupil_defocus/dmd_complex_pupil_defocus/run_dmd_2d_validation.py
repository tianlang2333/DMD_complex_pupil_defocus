from __future__ import annotations

"""运行二维DMD相干空中像的数值验证与示例实验。

本文件只负责组织实验、计算指标和保存结果。DMD几何建模与相干传播
分别由 ``dmd_model_2d.py`` 和 ``coherent_imaging_2d.py`` 提供。

同目录的一维教材程序通过其真实公开API接入交叉验证，统一长度单位、
图形中心、通过级次和强度标度。只有一维文件缺失时才记录为 ``SKIPPED``；
导入或API调用失败会直接报错，不会被当作正常跳过。
"""

import argparse
import csv
import importlib.util
import math
import json
import sys
from dataclasses import asdict, dataclass, replace
from numbers import Real
from pathlib import Path
from typing import Any, Final, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from coherent_imaging_2d import (
    CoherentImage2DResult,
    OpticalConfig2D,
    calculate_coherent_aerial_image_2d,
    centered_fft2,
    centered_ifft2,
    make_frequency_axes,
    propagate_coherent_field_from_pupil,
)
from projection_lens_2d import make_defocused_circular_pupil
from dmd_model_2d import (
    DMDConfig,
    DMDPatternResult,
    make_cross_states,
    make_ideal_cross_field,
    make_ideal_periodic_line_space_field,
    make_periodic_line_space_states,
    render_dmd_field,
    with_state_response,
)


BoolArray2D = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
FloatArray2D = NDArray[np.float64]
ComplexArray2D = NDArray[np.complex128]

TEXTBOOK_MODULE_FILENAME: Final[str] = "textbook_dense_lines_aerial_image.py"
DEFAULT_OUTPUT_DIRECTORY: Final[str] = "dmd_2d_validation_results"
THRESHOLD_FRACTION: Final[float] = 0.5


@dataclass(frozen=True)
class TextbookExperimentConfig:
    """教材周期线空基准、NA扫描与网格收敛的公共参数。"""

    on_width_um: float = 0.5
    line_width_um: float = 0.5
    wavelength_um: float = 0.248
    numerical_aperture: float = 0.85
    na_sweep_numerical_apertures: tuple[float, ...] = (0.20, 0.30, 0.85)
    dmd_mirror_pitch_um: float = 7.56
    projected_mirror_pitch_um: float = 0.125
    num_mirrors_x: int = 64
    num_mirrors_y: int = 64
    samples_per_mirror: int = 8
    convergence_samples_per_mirror: int = 16
    active_side_ratio: float = 1.0
    aperture_demo_active_side_ratio: float = 0.95
    phase_offset_um: float = 0.0
    profile_display_half_range_um: float = 1.5
    frequency_display_cutoff_multiple: float = 1.5

    def __post_init__(self) -> None:
        positive_values = (
            self.on_width_um,
            self.line_width_um,
            self.wavelength_um,
            self.numerical_aperture,
            self.dmd_mirror_pitch_um,
            self.projected_mirror_pitch_um,
            self.active_side_ratio,
            self.aperture_demo_active_side_ratio,
            self.profile_display_half_range_um,
            self.frequency_display_cutoff_multiple,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive_values):
            raise ValueError("教材实验配置的尺寸、NA和显示范围必须为有限正数。")
        if not math.isfinite(self.phase_offset_um):
            raise ValueError("phase_offset_um必须为有限实数。")
        if not self.na_sweep_numerical_apertures or not all(
            math.isfinite(value) and value > 0.0
            for value in self.na_sweep_numerical_apertures
        ):
            raise ValueError("na_sweep_numerical_apertures必须包含有限正数。")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (
                self.num_mirrors_x,
                self.num_mirrors_y,
                self.samples_per_mirror,
                self.convergence_samples_per_mirror,
            )
        ):
            raise ValueError("微镜数和每微镜采样数必须为正整数。")
        if not all(math.isfinite(value) and value > 0.0 for value in (
            self.projection_magnification, self.pitch_um,
            self.projected_mirror_pitch_um / self.samples_per_mirror,
        )):
            raise ValueError("教材实验的计算倍率、周期和采样间距必须为有限正数。")
        if self.active_side_ratio > 1.0 or self.aperture_demo_active_side_ratio > 1.0:
            raise ValueError("active_side_ratio必须位于(0, 1]。")

    @property
    def pitch_um(self) -> float:
        """返回等线宽线/空图形的周期。"""

        return self.on_width_um + self.line_width_um

    @property
    def projection_magnification(self) -> float:
        """返回曝光面尺寸与DMD面尺寸之比。"""

        return self.projected_mirror_pitch_um / self.dmd_mirror_pitch_um

    @property
    def n_cd(self) -> float:
        """返回透明ON线宽所包含的投影微镜数。"""

        return self.on_width_um / self.projected_mirror_pitch_um

    @property
    def maximum_geometric_order(self) -> int:
        """返回基准NA下的最高几何通过级次。"""

        return math.floor(
            self.pitch_um * self.numerical_aperture / self.wavelength_um
        )


@dataclass(frozen=True)
class SamplingExperimentConfig:
    """DMD地址采样比实验的公共参数。"""

    target_on_width_um: float = 6.0
    target_line_width_um: float = 6.0
    wavelength_um: float = 0.405
    numerical_aperture: float = 0.065
    dmd_mirror_pitch_um: float = 7.56
    n_cd_values: tuple[float, ...] = (3.5, 2.0, 1.0)
    phase_offset_fraction: float = 0.35
    num_mirrors_x: int = 64
    num_mirrors_y: int = 32
    samples_per_mirror: int = 8
    active_side_ratio: float = 0.95
    display_half_range_um: float = 12.0

    def __post_init__(self) -> None:
        positive_values = (
            self.target_on_width_um,
            self.target_line_width_um,
            self.wavelength_um,
            self.numerical_aperture,
            self.dmd_mirror_pitch_um,
            self.active_side_ratio,
            self.display_half_range_um,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive_values):
            raise ValueError("采样比实验配置的尺寸、NA和显示范围必须为有限正数。")
        if not math.isfinite(self.phase_offset_fraction):
            raise ValueError("phase_offset_fraction必须为有限实数。")
        if not self.n_cd_values or not all(
            math.isfinite(value) and value > 0.0 for value in self.n_cd_values
        ):
            raise ValueError("n_cd_values必须包含有限正数。")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (
                self.num_mirrors_x,
                self.num_mirrors_y,
                self.samples_per_mirror,
            )
        ):
            raise ValueError("微镜数和每微镜采样数必须为正整数。")
        if self.active_side_ratio > 1.0:
            raise ValueError("active_side_ratio必须位于(0, 1]。")
        derived_scales = [self.target_pitch_um]
        for n_cd in self.n_cd_values:
            projected_pitch = self.projected_mirror_pitch_um(n_cd)
            derived_scales.extend((projected_pitch,
                                   projected_pitch / self.dmd_mirror_pitch_um,
                                   projected_pitch / self.samples_per_mirror))
        if not all(math.isfinite(value) and value > 0.0 for value in derived_scales):
            raise ValueError("采样比实验的计算倍率、间距和周期必须为有限正数。")

    @property
    def target_pitch_um(self) -> float:
        """返回示例周期线/空图形的周期。"""

        return self.target_on_width_um + self.target_line_width_um

    def projected_mirror_pitch_um(self, n_cd: float) -> float:
        """返回指定 ``n_cd`` 对应的投影微镜间距。"""

        return self.target_on_width_um / n_cd

    def phase_offset_um(self, projected_mirror_pitch_um: float) -> float:
        """返回相对DMD网格的目标图形相位偏移。"""

        return self.phase_offset_fraction * projected_mirror_pitch_um


@dataclass(frozen=True)
class CrossExperimentConfig:
    """二维十字图形演示的公共参数。"""

    arm_width_um: float = 6.0
    arm_length_um: float = 30.0
    center_x_um: float = 0.0
    center_y_um: float = 0.0
    wavelength_um: float = 0.405
    numerical_aperture: float = 0.065
    dmd_mirror_pitch_um: float = 7.56
    projected_mirror_pitch_um: float = 3.0
    num_mirrors_x: int = 40
    num_mirrors_y: int = 40
    samples_per_mirror: int = 20
    active_side_ratio: float = 0.95
    arm_probe_offset_um: float = 9.0
    frequency_display_cutoff_multiple: float = 4.0

    def __post_init__(self) -> None:
        positive_values = (
            self.arm_width_um,
            self.arm_length_um,
            self.wavelength_um,
            self.numerical_aperture,
            self.dmd_mirror_pitch_um,
            self.projected_mirror_pitch_um,
            self.active_side_ratio,
            self.arm_probe_offset_um,
            self.frequency_display_cutoff_multiple,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive_values):
            raise ValueError("十字实验配置的尺寸、NA和显示范围必须为有限正数。")
        if not math.isfinite(self.center_x_um) or not math.isfinite(self.center_y_um):
            raise ValueError("十字中心坐标必须为有限实数。")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (
                self.num_mirrors_x,
                self.num_mirrors_y,
                self.samples_per_mirror,
            )
        ):
            raise ValueError("微镜数和每微镜采样数必须为正整数。")
        if self.arm_length_um < self.arm_width_um:
            raise ValueError("arm_length_um必须大于或等于arm_width_um。")
        if not 0.5 * self.arm_width_um < self.arm_probe_offset_um < 0.5 * self.arm_length_um:
            raise ValueError("臂宽探测位置必须在交叉区之外且在十字臂端之内。")
        if not all(math.isfinite(value) and value > 0.0 for value in (
            self.projection_magnification,
            self.projected_mirror_pitch_um / self.samples_per_mirror,
        )):
            raise ValueError("十字实验的计算倍率和采样间距必须为有限正数。")
        if self.active_side_ratio > 1.0:
            raise ValueError("active_side_ratio必须位于(0, 1]。")

    @property
    def projection_magnification(self) -> float:
        """返回曝光面尺寸与DMD面尺寸之比。"""

        return self.projected_mirror_pitch_um / self.dmd_mirror_pitch_um


TEXTBOOK_EXPERIMENT_CONFIG: Final = TextbookExperimentConfig()
SAMPLING_EXPERIMENT_CONFIG: Final = SamplingExperimentConfig()
CROSS_EXPERIMENT_CONFIG: Final = CrossExperimentConfig()


@dataclass(frozen=True)
class ValidationCheck:
    """保存一项内部数值自检的结果。"""

    name: str
    value: float
    limit: float
    comparison: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ThresholdInterval:
    """保存一维阈值图形左右交点和宽度。"""

    left_edge_um: float
    right_edge_um: float
    width_um: float


@dataclass(frozen=True)
class ProfileMetrics:
    """保存周期线空中央截面的图像质量指标。"""

    space_center_intensity: float
    line_center_intensity: float
    contrast: float
    threshold_interval: ThresholdInterval
    nils: float


@dataclass(frozen=True)
class TextbookCrossValidation:
    """保存真实一维教材程序与二维模型的比较结果。"""

    status: str
    reason: str
    x_um: FloatArray | None
    textbook_intensity: FloatArray | None
    dmd_2d_intensity: FloatArray | None
    residual: FloatArray | None
    raw_rmse: float
    peak_normalized_rmse: float
    textbook_metrics: ProfileMetrics | None
    dmd_2d_metrics: ProfileMetrics | None


@dataclass(frozen=True)
class ConvergenceResult:
    """保存8点与16点每微镜的收敛比较。"""

    metrics_8: ProfileMetrics
    metrics_16: ProfileMetrics
    normalized_profile_rmse: float
    space_center_relative_change: float
    line_center_relative_change: float
    contrast_relative_change: float
    nils_relative_change: float
    threshold_cd_relative_change: float
    maximum_metric_relative_change: float
    converged_within_two_percent: bool


@dataclass(frozen=True)
class NASweepCase:
    """保存一个NA扫描案例。"""

    numerical_aperture: float
    maximum_geometric_order: int
    result: CoherentImage2DResult
    profile: FloatArray
    metrics: ProfileMetrics


@dataclass(frozen=True)
class SamplingRatioCase:
    """保存一个DMD采样比案例。"""

    n_cd: float
    config: DMDConfig
    pattern: DMDPatternResult
    ideal_field: ComplexArray2D
    ideal_image: CoherentImage2DResult
    dmd_image: CoherentImage2DResult
    ideal_profile: FloatArray
    dmd_profile: FloatArray
    ideal_metrics: ProfileMetrics
    dmd_metrics: ProfileMetrics
    rasterized_interval: ThresholdInterval
    raster_left_epe_um: float
    raster_right_epe_um: float
    raster_cd_error_um: float
    threshold_left_epe_um: float
    threshold_right_epe_um: float
    threshold_cd_error_um: float
    ideal_threshold_left_epe_um: float
    ideal_threshold_right_epe_um: float
    ideal_threshold_cd_error_um: float
    dmd_minus_ideal_left_edge_um: float
    dmd_minus_ideal_right_edge_um: float
    dmd_minus_ideal_threshold_cd_um: float
    normalized_image_rmse: float


@dataclass(frozen=True)
class CrossPatternCase:
    """保存二维十字图形案例。"""

    config: DMDConfig
    pattern: DMDPatternResult
    ideal_field: ComplexArray2D
    image: CoherentImage2DResult
    threshold_mask: BoolArray2D
    central_horizontal_span_um: float
    central_vertical_span_um: float
    horizontal_arm_width_um: float
    vertical_arm_width_um: float
    threshold_area_um2: float
    centroid_x_um: float
    centroid_y_um: float
    centroid_shift_um: float
    central_span_difference_um: float
    arm_width_difference_um: float


METRIC_FIELDNAMES: Final[list[str]] = [
    "experiment",
    "case_name",
    "wavelength_um",
    "numerical_aperture",
    "projected_mirror_pitch_um",
    "samples_per_mirror",
    "active_side_ratio",
    "n_cd",
    "input_energy",
    "output_energy",
    "energy_transmission",
    "space_center_intensity",
    "line_center_intensity",
    "contrast",
    "threshold_cd_um",
    "nils",
    "left_epe_um",
    "right_epe_um",
    "rmse_to_reference",
    "rasterized_on_width_um",
    "raster_cd_error_um",
    "threshold_cd_error_um",
    "threshold_left_epe_um",
    "threshold_right_epe_um",
    "horizontal_cd_um",
    "vertical_cd_um",
    "central_horizontal_span_um",
    "central_vertical_span_um",
    "ideal_threshold_cd_um",
    "ideal_threshold_cd_error_um",
    "ideal_threshold_left_epe_um",
    "ideal_threshold_right_epe_um",
    "dmd_minus_ideal_threshold_cd_um",
    "dmd_minus_ideal_left_edge_um",
    "dmd_minus_ideal_right_edge_um",
    "threshold_area_um2",
    "centroid_x_um",
    "centroid_y_um",
    "notes",
]


def _validate_profile_inputs(x_um: FloatArray, intensity: FloatArray) -> None:
    """检查一维坐标和强度数组是否适合插值与求导。"""

    if x_um.ndim != 1 or intensity.ndim != 1:
        raise ValueError("x_um和intensity必须是一维数组。")
    if x_um.size != intensity.size or x_um.size < 3:
        raise ValueError("x_um和intensity必须等长且至少包含3个采样点。")
    if not np.all(np.isfinite(x_um)) or not np.all(np.isfinite(intensity)):
        raise ValueError("x_um和intensity不能包含NaN或无穷值。")
    if not np.all(np.diff(x_um) > 0.0):
        raise ValueError("x_um必须严格单调递增。")


def _local_polynomial_value_and_derivative(
    x_um: FloatArray,
    values: FloatArray,
    position_um: float,
    num_fit_samples: int = 7,
    polynomial_degree: int = 4,
) -> tuple[float, float]:
    """用局部多项式计算指定位置的函数值和一阶导数。

    拟合前将指定位置平移到局部坐标原点，因此升幂系数的第0项和第1项
    分别就是函数值与一阶导数。此函数统一用于亮/暗中心强度和NILS边缘。
    """

    _validate_profile_inputs(x_um, values)
    if not x_um[0] <= position_um <= x_um[-1]:
        return math.nan, math.nan
    if num_fit_samples < 2:
        raise ValueError("num_fit_samples必须至少为2。")
    actual_num_samples = min(num_fit_samples, x_um.size)
    fit_indices = np.sort(
        np.argsort(np.abs(x_um - position_um))[:actual_num_samples]
    )
    local_x_um = x_um[fit_indices] - position_um
    actual_degree = min(polynomial_degree, actual_num_samples - 1)
    coefficients = np.polynomial.polynomial.polyfit(
        local_x_um,
        values[fit_indices],
        actual_degree,
    )
    derivative = float(coefficients[1]) if coefficients.size >= 2 else 0.0
    return float(coefficients[0]), derivative


def _local_polynomial_value(
    x_um: FloatArray,
    values: FloatArray,
    position_um: float,
    num_fit_samples: int = 7,
    polynomial_degree: int = 4,
) -> float:
    """用局部多项式评估指定位置的函数值。"""

    value, _ = _local_polynomial_value_and_derivative(
        x_um,
        values,
        position_um,
        num_fit_samples,
        polynomial_degree,
    )
    return value


def _peak_normalize(values: FloatArray | FloatArray2D) -> FloatArray | FloatArray2D:
    """进行峰值归一化；全零数组返回同形状零数组。"""

    peak = float(np.max(values))
    if peak <= np.finfo(np.float64).eps:
        return np.zeros_like(values, dtype=np.float64)
    return np.asarray(values / peak, dtype=np.float64)


def _relative_difference(value: float, reference: float) -> float:
    """计算相对于参考值的绝对相对差；无法定义时返回NaN。"""

    if not math.isfinite(value) or not math.isfinite(reference):
        return math.nan
    denominator = abs(reference)
    if denominator <= np.finfo(np.float64).eps:
        return 0.0 if abs(value) <= np.finfo(np.float64).eps else math.inf
    return abs(value - reference) / denominator


def _finite_max(values: Sequence[float]) -> float:
    """返回有限数值的最大值；没有有限值时返回NaN。"""

    finite_values = [value for value in values if math.isfinite(value)]
    return max(finite_values) if finite_values else math.nan


def _normalized_rmse(reference: FloatArray | FloatArray2D, test: FloatArray | FloatArray2D) -> float:
    """计算两组峰值归一化数据的均方根误差。"""

    if reference.shape != test.shape:
        raise ValueError("计算RMSE的两组数组必须具有相同形状。")
    reference_normalized = _peak_normalize(reference)
    test_normalized = _peak_normalize(test)
    return float(np.sqrt(np.mean(np.square(test_normalized - reference_normalized))))


def _linear_threshold_crossing(
    x_left_um: float,
    value_left: float,
    x_right_um: float,
    value_right: float,
    threshold: float,
) -> float:
    """用相邻两个采样点线性插值阈值交点。"""

    value_difference = value_right - value_left
    if abs(value_difference) <= np.finfo(np.float64).eps:
        return math.nan
    fraction = (threshold - value_left) / value_difference
    return float(x_left_um + fraction * (x_right_um - x_left_um))


def measure_threshold_interval(
    x_um: FloatArray,
    intensity: FloatArray,
    center_um: float,
    threshold_fraction: float = THRESHOLD_FRACTION,
    *,
    threshold_intensity: float | None = None,
    search_bounds_um: tuple[float, float] | None = None,
) -> ThresholdInterval:
    """测量包含指定中心的阈值图形宽度。

    左右边缘均由相邻采样点线性插值得到。若中心不在阈值图形内，或任一
    方向找不到有效交点，则返回三个NaN。未传新参数时完整保留原有
    峰值归一化行为；threshold_intensity 表示原始强度标度上的固定阈值。
    """

    _validate_profile_inputs(x_um, intensity)
    if threshold_intensity is not None or search_bounds_um is not None:
        if threshold_intensity is None:
            if not 0.0 < threshold_fraction < 1.0:
                raise ValueError("threshold_fraction必须位于(0, 1)内。")
            threshold_intensity = float(np.max(intensity)) * threshold_fraction
        interval, _ = measure_threshold_interval_with_status(
            x_um, intensity, center_um,
            threshold_intensity=threshold_intensity,
            search_bounds_um=search_bounds_um,
        )
        return interval
    if not 0.0 < threshold_fraction < 1.0:
        raise ValueError("threshold_fraction必须位于(0, 1)内。")

    normalized = np.asarray(_peak_normalize(intensity), dtype=np.float64)
    center_index = int(np.argmin(np.abs(x_um - center_um)))
    if normalized[center_index] < threshold_fraction:
        return ThresholdInterval(math.nan, math.nan, math.nan)

    left_edge_um = math.nan
    for index in range(center_index - 1, -1, -1):
        if normalized[index] < threshold_fraction <= normalized[index + 1]:
            left_edge_um = _linear_threshold_crossing(
                float(x_um[index]),
                float(normalized[index]),
                float(x_um[index + 1]),
                float(normalized[index + 1]),
                threshold_fraction,
            )
            break

    right_edge_um = math.nan
    for index in range(center_index, x_um.size - 1):
        if normalized[index] >= threshold_fraction > normalized[index + 1]:
            right_edge_um = _linear_threshold_crossing(
                float(x_um[index]),
                float(normalized[index]),
                float(x_um[index + 1]),
                float(normalized[index + 1]),
                threshold_fraction,
            )
            break

    if not math.isfinite(left_edge_um) or not math.isfinite(right_edge_um):
        return ThresholdInterval(math.nan, math.nan, math.nan)
    return ThresholdInterval(
        left_edge_um,
        right_edge_um,
        right_edge_um - left_edge_um,
    )


def measure_threshold_interval_with_status(
    x_um: FloatArray,
    intensity: FloatArray,
    center_um: float,
    *,
    threshold_intensity: float,
    search_bounds_um: tuple[float, float] | None = None,
) -> tuple[ThresholdInterval, str]:
    """固定原始强度阈值下找中心所属区间，并给出失效原因。

    搜索边界和中心按同一分段线性曲线插值；只接受搜索范围内具有
    双侧阈值交点的目标区间，不借用邻周期交点，不把范围边界冒充边缘。
    """

    _validate_profile_inputs(x_um, intensity)
    if not math.isfinite(center_um):
        raise ValueError("center_um必须为有限实数。")
    if not math.isfinite(threshold_intensity) or threshold_intensity < 0.0:
        raise ValueError("threshold_intensity必须为有限非负数。")
    invalid = ThresholdInterval(math.nan, math.nan, math.nan)
    lower, upper = (
        (float(x_um[0]), float(x_um[-1]))
        if search_bounds_um is None else search_bounds_um
    )
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise ValueError("search_bounds_um必须为有限且严格递增的两个边界。")
    if not lower < center_um < upper:
        return invalid, "CENTER_OUTSIDE_SEARCH"
    if lower < x_um[0] or upper > x_um[-1]:
        return invalid, "SEARCH_OUTSIDE_DATA"
    if float(np.interp(center_um, x_um, intensity)) < threshold_intensity:
        return invalid, "CENTER_BELOW_THRESHOLD"

    inside = (x_um > lower) & (x_um < upper)
    search_x = np.unique(np.concatenate(([lower, center_um, upper], x_um[inside])))
    values = np.interp(search_x, x_um, intensity)
    center_index = int(np.searchsorted(search_x, center_um))
    if np.all(values >= threshold_intensity):
        return invalid, "ALL_ABOVE_THRESHOLD"

    edges = [math.nan, math.nan]
    for side, indices in enumerate((
        range(center_index - 1, -1, -1),
        range(center_index, search_x.size - 1),
    )):
        for index in indices:
            left_value, right_value = values[index:index + 2]
            brackets = (
                left_value < threshold_intensity <= right_value
                if side == 0 else
                left_value >= threshold_intensity > right_value
            )
            if brackets:
                # 同比例缩放仅防止小强度被旧插值函数的绝对eps误判；
                # 阈值仍来自该参考组，未按当前案例重新计算。
                scale = max(abs(left_value), abs(right_value), threshold_intensity)
                edges[side] = _linear_threshold_crossing(
                    float(search_x[index]), float(left_value / scale),
                    float(search_x[index + 1]), float(right_value / scale),
                    float(threshold_intensity / scale),
                )
                break
    if not math.isfinite(edges[0]):
        return invalid, "MISSING_LEFT_EDGE"
    if not math.isfinite(edges[1]):
        return invalid, "MISSING_RIGHT_EDGE"
    if edges[1] <= edges[0]:
        return invalid, "ZERO_WIDTH"
    return ThresholdInterval(edges[0], edges[1], edges[1] - edges[0]), "VALID"


def calculate_periodic_profile_metrics(
    x_um: FloatArray,
    intensity: FloatArray,
    pitch_um: float,
    on_width_um: float,
    on_center_um: float = 0.0,
    force_threshold_nan: bool = False,
) -> ProfileMetrics:
    """计算周期线空中央截面的中心对比度、阈值CD和数值NILS。

    NILS在设计右边缘附近选取7个采样点，以边缘为局部坐标原点进行四次
    多项式拟合，并由拟合系数计算边缘强度和一阶导数。局部坐标能避免
    多项式病态；该方法比先做低阶差分再插值具有更好的网格收敛性。
    """

    _validate_profile_inputs(x_um, intensity)
    if pitch_um <= 0.0 or on_width_um <= 0.0 or on_width_um >= pitch_um:
        raise ValueError("必须满足0 < on_width_um < pitch_um。")

    space_center_intensity = _local_polynomial_value(x_um, intensity, on_center_um)
    line_center_intensity = _local_polynomial_value(
        x_um,
        intensity,
        on_center_um + 0.5 * pitch_um,
    )
    denominator = space_center_intensity + line_center_intensity
    contrast = (
        (space_center_intensity - line_center_intensity) / denominator
        if math.isfinite(denominator) and abs(denominator) > np.finfo(np.float64).eps
        else math.nan
    )
    if math.isfinite(contrast) and abs(contrast) < 1.0e-12:
        contrast = 0.0

    if force_threshold_nan:
        threshold_interval = ThresholdInterval(math.nan, math.nan, math.nan)
    else:
        threshold_interval = measure_threshold_interval(x_um, intensity, on_center_um)

    design_edge_um = on_center_um + 0.5 * on_width_um
    edge_intensity, edge_derivative = _local_polynomial_value_and_derivative(
        x_um,
        intensity,
        design_edge_um,
        num_fit_samples=7,
        polynomial_degree=4,
    )
    if abs(edge_intensity) <= np.finfo(np.float64).eps:
        nils = math.nan
    else:
        nils = on_width_um * abs(edge_derivative / edge_intensity)
        if nils < 1.0e-12:
            nils = 0.0

    return ProfileMetrics(
        space_center_intensity=space_center_intensity,
        line_center_intensity=line_center_intensity,
        contrast=float(contrast),
        threshold_interval=threshold_interval,
        nils=float(nils),
    )


def _make_check(
    name: str,
    value: float,
    limit: float,
    comparison: str,
    detail: str,
) -> ValidationCheck:
    """按指定不等式构造一项内部自检。"""

    if comparison == "<":
        passed = value < limit
    elif comparison == "<=":
        passed = value <= limit
    else:
        raise ValueError("comparison只支持'<'或'<='。")
    return ValidationCheck(name, value, limit, comparison, bool(passed), detail)


def run_internal_validations() -> list[ValidationCheck]:
    """运行FFT、全通瞳孔、常数场、对称性和能量自检。"""

    checks: list[ValidationCheck] = []
    random_generator = np.random.default_rng(20260828)
    complex_test_field = np.asarray(
        random_generator.normal(size=(17, 19))
        + 1j * random_generator.normal(size=(17, 19)),
        dtype=np.complex128,
    )

    roundtrip_field = centered_ifft2(centered_fft2(complex_test_field))
    roundtrip_error = float(np.max(np.abs(roundtrip_field - complex_test_field)))
    checks.append(
        _make_check(
            "FFT往返误差",
            roundtrip_error,
            1.0e-12,
            "<",
            "max|IFFT(FFT(U))-U|",
        )
    )

    all_pass_pupil = np.ones(complex_test_field.shape, dtype=np.float64)
    _, _, all_pass_field = propagate_coherent_field_from_pupil(
        complex_test_field,
        all_pass_pupil,
    )
    all_pass_error = float(np.max(np.abs(all_pass_field - complex_test_field)))
    checks.append(
        _make_check(
            "全通瞳孔恢复输入",
            all_pass_error,
            1.0e-12,
            "<",
            "P(fx,fy)=1",
        )
    )

    constant_dmd_config = DMDConfig(
        num_mirrors_x=9,
        num_mirrors_y=8,
        dmd_mirror_pitch_um=TEXTBOOK_EXPERIMENT_CONFIG.dmd_mirror_pitch_um,
        projection_magnification=(
            TEXTBOOK_EXPERIMENT_CONFIG.projection_magnification
        ),
        samples_per_mirror=4,
        active_side_ratio=1.0,
        on_amplitude=1.0 + 0.0j,
        off_amplitude=0.0 + 0.0j,
    )
    constant_states = np.ones(
        (constant_dmd_config.num_mirrors_y, constant_dmd_config.num_mirrors_x),
        dtype=np.bool_,
    )
    constant_pattern = render_dmd_field(constant_states, constant_dmd_config)
    constant_field = constant_pattern.object_field
    _, _, constant_output = propagate_coherent_field_from_pupil(
        constant_field,
        np.ones(constant_field.shape, dtype=np.float64),
    )
    constant_input_error = float(np.max(np.abs(constant_field - (1.0 + 0.0j))))
    constant_propagation_error = float(np.max(np.abs(constant_output - constant_field)))
    constant_error = max(constant_input_error, constant_propagation_error)
    checks.append(
        _make_check(
            "常数场验证",
            constant_error,
            1.0e-12,
            "<",
            "真实DMD渲染：全部ON、active_side_ratio=1、全通瞳孔",
        )
    )

    num_symmetric_samples = 65
    symmetric_spacing_um = 0.08
    symmetric_axis_um = (
        np.arange(num_symmetric_samples, dtype=np.float64)
        - 0.5 * (num_symmetric_samples - 1)
    ) * symmetric_spacing_um
    symmetric_xx_um, symmetric_yy_um = np.meshgrid(
        symmetric_axis_um,
        symmetric_axis_um,
        indexing="xy",
    )
    symmetric_field = np.asarray(
        (
            (np.abs(symmetric_xx_um) <= 0.8)
            & (np.abs(symmetric_yy_um) <= 0.8)
        ),
        dtype=np.complex128,
    )
    symmetric_result = calculate_coherent_aerial_image_2d(
        symmetric_field,
        symmetric_axis_um,
        symmetric_axis_um,
        OpticalConfig2D(
            wavelength_um=SAMPLING_EXPERIMENT_CONFIG.wavelength_um,
            numerical_aperture=(
                TEXTBOOK_EXPERIMENT_CONFIG.na_sweep_numerical_apertures[0]
            ),
        ),
    )
    intensity_scale = max(float(np.max(symmetric_result.raw_intensity)), np.finfo(float).eps)
    left_right_error = float(
        np.max(
            np.abs(
                symmetric_result.raw_intensity
                - symmetric_result.raw_intensity[:, ::-1]
            )
        )
        / intensity_scale
    )
    up_down_error = float(
        np.max(
            np.abs(
                symmetric_result.raw_intensity
                - symmetric_result.raw_intensity[::-1, :]
            )
        )
        / intensity_scale
    )
    checks.append(
        _make_check(
            "左右对称性",
            left_right_error,
            1.0e-10,
            "<",
            "中心对称方孔与圆形瞳孔",
        )
    )
    checks.append(
        _make_check(
            "上下对称性",
            up_down_error,
            1.0e-10,
            "<",
            "中心对称方孔与圆形瞳孔",
        )
    )

    checks.append(
        _make_check(
            "能量不增加",
            symmetric_result.energy_transmission,
            1.0 + 1.0e-12,
            "<=",
            "Eout/Ein <= 1+1e-12",
        )
    )

    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(
            f"[Internal validation] {status}: {check.name}; "
            f"value={check.value:.6e}, criterion {check.comparison} {check.limit:.6e}"
        )
    return checks


def _base_dmd_config(
    samples_per_mirror: int | None = None,
    experiment_config: TextbookExperimentConfig = TEXTBOOK_EXPERIMENT_CONFIG,
) -> DMDConfig:
    """创建教材线空交叉验证使用的二维DMD配置。"""

    resolved_samples_per_mirror = (
        experiment_config.samples_per_mirror
        if samples_per_mirror is None
        else samples_per_mirror
    )
    return DMDConfig(
        num_mirrors_x=experiment_config.num_mirrors_x,
        num_mirrors_y=experiment_config.num_mirrors_y,
        dmd_mirror_pitch_um=experiment_config.dmd_mirror_pitch_um,
        projection_magnification=experiment_config.projection_magnification,
        samples_per_mirror=resolved_samples_per_mirror,
        active_side_ratio=experiment_config.active_side_ratio,
        on_amplitude=1.0 + 0.0j,
        off_amplitude=0.0 + 0.0j,
    )


def _make_base_pattern(
    samples_per_mirror: int | None = None,
    experiment_config: TextbookExperimentConfig = TEXTBOOK_EXPERIMENT_CONFIG,
) -> DMDPatternResult:
    """生成整数周期、沿y不变的50%占空比竖直线空图形。"""

    config = _base_dmd_config(samples_per_mirror, experiment_config)
    states = make_periodic_line_space_states(
        config,
        pitch_um=experiment_config.pitch_um,
        on_width_um=experiment_config.on_width_um,
        orientation="vertical",
        phase_offset_um=experiment_config.phase_offset_um,
    )
    return render_dmd_field(states, config)


def _make_finite_aperture_demo_pattern(
    experiment_config: TextbookExperimentConfig = TEXTBOOK_EXPERIMENT_CONFIG,
) -> tuple[DMDConfig, DMDPatternResult]:
    """为图1创建配置指定边长比的有限开口线空示例。"""

    config = DMDConfig(
        num_mirrors_x=experiment_config.num_mirrors_x,
        num_mirrors_y=experiment_config.num_mirrors_y,
        dmd_mirror_pitch_um=experiment_config.dmd_mirror_pitch_um,
        projection_magnification=experiment_config.projection_magnification,
        samples_per_mirror=experiment_config.samples_per_mirror,
        active_side_ratio=experiment_config.aperture_demo_active_side_ratio,
        on_amplitude=1.0 + 0.0j,
        off_amplitude=0.0 + 0.0j,
    )
    states = make_periodic_line_space_states(
        config,
        pitch_um=experiment_config.pitch_um,
        on_width_um=experiment_config.on_width_um,
        orientation="vertical",
        phase_offset_um=experiment_config.phase_offset_um,
    )
    return config, render_dmd_field(states, config)


def _make_base_image(
    samples_per_mirror: int | None = None,
    numerical_aperture: float | None = None,
    experiment_config: TextbookExperimentConfig = TEXTBOOK_EXPERIMENT_CONFIG,
) -> tuple[DMDPatternResult, CoherentImage2DResult]:
    """传播教材线空验证图形并返回DMD输入与空中像。"""

    resolved_numerical_aperture = (
        experiment_config.numerical_aperture
        if numerical_aperture is None
        else numerical_aperture
    )
    pattern = _make_base_pattern(samples_per_mirror, experiment_config)
    result = calculate_coherent_aerial_image_2d(
        pattern.object_field,
        pattern.x_um,
        pattern.y_um,
        OpticalConfig2D(
            wavelength_um=experiment_config.wavelength_um,
            numerical_aperture=resolved_numerical_aperture,
        ),
    )
    return pattern, result


def _extract_center_horizontal_profile(result: CoherentImage2DResult) -> FloatArray:
    """从二维强度中提取最接近y=0的横向截面。"""

    center_row = int(np.argmin(np.abs(result.y_um)))
    return np.asarray(result.raw_intensity[center_row, :], dtype=np.float64)


def _extract_center_vertical_profile(result: CoherentImage2DResult) -> FloatArray:
    """从二维强度中提取最接近x=0的纵向截面。"""

    center_column = int(np.argmin(np.abs(result.x_um)))
    return np.asarray(result.raw_intensity[:, center_column], dtype=np.float64)


def run_textbook_cross_validation(
    script_directory: Path,
    dmd_result: CoherentImage2DResult,
    experiment_config: TextbookExperimentConfig = TEXTBOOK_EXPERIMENT_CONFIG,
) -> TextbookCrossValidation:
    """调用真实一维傅里叶级次程序，与二维DMD中央截面比较。

    一维使用nm、二维使用um，故以(x - phase_offset_um)*1000调用一维API。
    一维refractive_index=1，与二维|U|²的原始强度定义一致。基准参数下
    DMD为无间隙、整数周期且边缘对齐的条带；剩余差异主要来自有限网格。
    曲线RMSE在中央一个周期的同一组坐标上计算，不平移拟合、不重缩放
    原始强度。COMPLETED只表示比较已执行，不代表通过预设误差门限。
    """

    module_path = script_directory / TEXTBOOK_MODULE_FILENAME
    if not module_path.is_file():
        return TextbookCrossValidation(
            status="SKIPPED",
            reason=f"缺少真实一维教材程序：{module_path.name}",
            x_um=None,
            textbook_intensity=None,
            dmd_2d_intensity=None,
            residual=None,
            raw_rmse=math.nan,
            peak_normalized_rmse=math.nan,
            textbook_metrics=None,
            dmd_2d_metrics=None,
        )

    # 精确加载指定目录的文件，避免误用搜索路径中另一个同名模块。
    # dataclass会查询sys.modules，因此执行模块前需要临时注册。
    print(f"[1D-to-2D cross-validation] Loading: {module_path.resolve()}")
    module_name = "_dmd_textbook_reference"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载一维教材程序：{module_path}")
    textbook = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = textbook
    try:
        spec.loader.exec_module(textbook)
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module

    textbook_config = textbook.SimulationConfig(
        space_width_nm=1000.0 * experiment_config.on_width_um,
        line_width_nm=1000.0 * experiment_config.line_width_um,
        wavelength_nm=1000.0 * experiment_config.wavelength_um,
        numerical_aperture=experiment_config.numerical_aperture,
        refractive_index=1.0,
    )
    textbook_config.validate()
    captured_orders = textbook.captured_orders_on_axis(
        wavelength_nm=textbook_config.wavelength_nm,
        numerical_aperture=textbook_config.numerical_aperture,
        pitch_nm=textbook_config.pitch_nm,
    )
    maximum_order = int(np.max(np.abs(captured_orders)))
    reference = textbook.calculate_coherent_aerial_image(
        x_nm=np.asarray(
            (dmd_result.x_um - experiment_config.phase_offset_um) * 1000.0,
            dtype=np.float64,
        ),
        max_diffraction_order=maximum_order,
        config=textbook_config,
    )
    textbook_profile = np.asarray(reference.intensity, dtype=np.float64)
    dmd_profile = _extract_center_horizontal_profile(dmd_result)
    _validate_profile_inputs(dmd_result.x_um, textbook_profile)

    # 一维指标使用原程序的密网格和解析导数API；阈值CD仍在与二维相同
    # 的采样坐标上按同一种线性插值方法计算。
    reference_metrics = textbook.calculate_image_metrics(
        textbook.calculate_coherent_aerial_image(
            x_nm=textbook.make_position_grid(textbook_config),
            max_diffraction_order=maximum_order,
            config=textbook_config,
        ),
        textbook_config,
    )
    textbook_metrics = ProfileMetrics(
        space_center_intensity=reference_metrics.space_center_intensity,
        line_center_intensity=reference_metrics.line_center_intensity,
        contrast=reference_metrics.center_based_contrast,
        threshold_interval=measure_threshold_interval(
            dmd_result.x_um, textbook_profile, experiment_config.phase_offset_um,
        ),
        nils=reference_metrics.nils_analytical,
    )
    dmd_metrics = calculate_periodic_profile_metrics(
        dmd_result.x_um,
        dmd_profile,
        pitch_um=experiment_config.pitch_um,
        on_width_um=experiment_config.on_width_um,
        on_center_um=experiment_config.phase_offset_um,
    )
    central_mask = (
        np.abs(dmd_result.x_um - experiment_config.phase_offset_um)
        <= 0.5 * experiment_config.pitch_um
    )
    if np.count_nonzero(central_mask) < 3:
        raise ValueError("一维—二维比较的中央周期内至少需要3个采样点。")
    textbook_period = textbook_profile[central_mask]
    dmd_period = dmd_profile[central_mask]
    residual = np.asarray(dmd_period - textbook_period, dtype=np.float64)
    return TextbookCrossValidation(
        status="COMPLETED",
        reason=(
            f"已调用{module_path.name}的真实API；nm/um已换算，"
            f"最高通过级次={maximum_order}，中央周期同坐标比较"
        ),
        x_um=dmd_result.x_um[central_mask].copy(),
        textbook_intensity=textbook_period,
        dmd_2d_intensity=dmd_period,
        residual=residual,
        raw_rmse=float(np.sqrt(np.mean(np.square(residual)))),
        peak_normalized_rmse=_normalized_rmse(textbook_period, dmd_period),
        textbook_metrics=textbook_metrics,
        dmd_2d_metrics=dmd_metrics,
    )


def run_convergence_check(
    result_8: CoherentImage2DResult,
    experiment_config: TextbookExperimentConfig = TEXTBOOK_EXPERIMENT_CONFIG,
) -> ConvergenceResult:
    """比较基准与高密度每微镜采样下的线空图指标。"""

    profile_8 = _extract_center_horizontal_profile(result_8)
    metrics_8 = calculate_periodic_profile_metrics(
        result_8.x_um,
        profile_8,
        pitch_um=experiment_config.pitch_um,
        on_width_um=experiment_config.on_width_um,
        on_center_um=experiment_config.phase_offset_um,
    )

    _, result_16 = _make_base_image(
        samples_per_mirror=experiment_config.convergence_samples_per_mirror,
        numerical_aperture=experiment_config.numerical_aperture,
        experiment_config=experiment_config,
    )
    profile_16 = _extract_center_horizontal_profile(result_16)
    metrics_16 = calculate_periodic_profile_metrics(
        result_16.x_um,
        profile_16,
        pitch_um=experiment_config.pitch_um,
        on_width_um=experiment_config.on_width_um,
        on_center_um=experiment_config.phase_offset_um,
    )

    comparison_left_um = (
        experiment_config.phase_offset_um - 0.5 * experiment_config.pitch_um
    )
    comparison_right_um = (
        experiment_config.phase_offset_um + 0.5 * experiment_config.pitch_um
    )
    comparison_mask = (
        (result_8.x_um >= comparison_left_um)
        & (result_8.x_um <= comparison_right_um)
    )
    comparison_x_um = result_8.x_um[comparison_mask]
    profile_8_period = profile_8[comparison_mask]
    profile_16_period = np.interp(comparison_x_um, result_16.x_um, profile_16)
    normalized_profile_rmse = _normalized_rmse(profile_16_period, profile_8_period)

    relative_changes = (
        _relative_difference(
            metrics_8.space_center_intensity,
            metrics_16.space_center_intensity,
        ),
        _relative_difference(
            metrics_8.line_center_intensity,
            metrics_16.line_center_intensity,
        ),
        _relative_difference(metrics_8.contrast, metrics_16.contrast),
        _relative_difference(metrics_8.nils, metrics_16.nils),
        _relative_difference(
            metrics_8.threshold_interval.width_um,
            metrics_16.threshold_interval.width_um,
        ),
    )
    maximum_change = _finite_max(relative_changes)
    return ConvergenceResult(
        metrics_8=metrics_8,
        metrics_16=metrics_16,
        normalized_profile_rmse=normalized_profile_rmse,
        space_center_relative_change=relative_changes[0],
        line_center_relative_change=relative_changes[1],
        contrast_relative_change=relative_changes[2],
        nils_relative_change=relative_changes[3],
        threshold_cd_relative_change=relative_changes[4],
        maximum_metric_relative_change=maximum_change,
        converged_within_two_percent=bool(
            math.isfinite(maximum_change) and maximum_change <= 0.02
        ),
    )


def run_na_sweep(
    pattern: DMDPatternResult,
    experiment_config: TextbookExperimentConfig = TEXTBOOK_EXPERIMENT_CONFIG,
) -> list[NASweepCase]:
    """在教材线空图形上扫描配置指定的NA。"""

    cases: list[NASweepCase] = []
    for numerical_aperture in experiment_config.na_sweep_numerical_apertures:
        maximum_order = math.floor(
            experiment_config.pitch_um
            * numerical_aperture
            / experiment_config.wavelength_um
        )
        result = calculate_coherent_aerial_image_2d(
            pattern.object_field,
            pattern.x_um,
            pattern.y_um,
            OpticalConfig2D(
                wavelength_um=experiment_config.wavelength_um,
                numerical_aperture=numerical_aperture,
            ),
        )
        profile = _extract_center_horizontal_profile(result)
        metrics = calculate_periodic_profile_metrics(
            result.x_um,
            profile,
            pitch_um=experiment_config.pitch_um,
            on_width_um=experiment_config.on_width_um,
            on_center_um=experiment_config.phase_offset_um,
            force_threshold_nan=maximum_order < 1,
        )
        cases.append(
            NASweepCase(
                numerical_aperture=numerical_aperture,
                maximum_geometric_order=maximum_order,
                result=result,
                profile=profile,
                metrics=metrics,
            )
        )
    return cases


def _mirror_center_axis(num_mirrors: int, projected_mirror_pitch_um: float) -> FloatArray:
    """生成DMD微镜中心坐标。"""

    return np.asarray(
        (
            np.arange(num_mirrors, dtype=np.float64)
            - 0.5 * (num_mirrors - 1)
        )
        * projected_mirror_pitch_um,
        dtype=np.float64,
    )


def _measure_rasterized_on_interval(
    mirror_states: BoolArray2D,
    config: DMDConfig,
    target_center_um: float,
) -> ThresholdInterval:
    """从微镜中心判定结果测量靠近目标中心的连续ON条带边界。"""

    center_row = config.num_mirrors_y // 2
    row_states = mirror_states[center_row, :]
    center_x_um = _mirror_center_axis(
        config.num_mirrors_x,
        config.projected_mirror_pitch_um,
    )
    on_indices = np.flatnonzero(row_states)
    if on_indices.size == 0:
        return ThresholdInterval(math.nan, math.nan, math.nan)
    seed_index = int(on_indices[np.argmin(np.abs(center_x_um[on_indices] - target_center_um))])

    left_index = seed_index
    while left_index > 0 and row_states[left_index - 1]:
        left_index -= 1
    right_index = seed_index
    while right_index < row_states.size - 1 and row_states[right_index + 1]:
        right_index += 1

    half_pitch_um = 0.5 * config.projected_mirror_pitch_um
    left_edge_um = float(center_x_um[left_index] - half_pitch_um)
    right_edge_um = float(center_x_um[right_index] + half_pitch_um)
    return ThresholdInterval(
        left_edge_um,
        right_edge_um,
        right_edge_um - left_edge_um,
    )


def run_sampling_ratio_experiment(
    experiment_config: SamplingExperimentConfig = SAMPLING_EXPERIMENT_CONFIG,
) -> list[SamplingRatioCase]:
    """比较配置指定的多组DMD投影采样比。"""

    cases: list[SamplingRatioCase] = []

    for n_cd in experiment_config.n_cd_values:
        projected_pitch_um = experiment_config.projected_mirror_pitch_um(n_cd)
        phase_offset_um = experiment_config.phase_offset_um(projected_pitch_um)
        config = DMDConfig(
            num_mirrors_x=experiment_config.num_mirrors_x,
            num_mirrors_y=experiment_config.num_mirrors_y,
            dmd_mirror_pitch_um=experiment_config.dmd_mirror_pitch_um,
            projection_magnification=(
                projected_pitch_um / experiment_config.dmd_mirror_pitch_um
            ),
            samples_per_mirror=experiment_config.samples_per_mirror,
            active_side_ratio=experiment_config.active_side_ratio,
            on_amplitude=1.0 + 0.0j,
            off_amplitude=0.0 + 0.0j,
        )
        states = make_periodic_line_space_states(
            config,
            pitch_um=experiment_config.target_pitch_um,
            on_width_um=experiment_config.target_on_width_um,
            orientation="vertical",
            phase_offset_um=phase_offset_um,
        )
        pattern = render_dmd_field(states, config)
        ideal_field = make_ideal_periodic_line_space_field(
            pattern.x_um,
            pattern.y_um,
            pitch_um=experiment_config.target_pitch_um,
            on_width_um=experiment_config.target_on_width_um,
            orientation="vertical",
            phase_offset_um=phase_offset_um,
        )
        optical_config = OpticalConfig2D(
            wavelength_um=experiment_config.wavelength_um,
            numerical_aperture=experiment_config.numerical_aperture,
        )
        ideal_image = calculate_coherent_aerial_image_2d(
            ideal_field,
            pattern.x_um,
            pattern.y_um,
            optical_config,
        )
        dmd_image = calculate_coherent_aerial_image_2d(
            pattern.object_field,
            pattern.x_um,
            pattern.y_um,
            optical_config,
        )
        ideal_profile = _extract_center_horizontal_profile(ideal_image)
        dmd_profile = _extract_center_horizontal_profile(dmd_image)
        ideal_metrics = calculate_periodic_profile_metrics(
            pattern.x_um,
            ideal_profile,
            pitch_um=experiment_config.target_pitch_um,
            on_width_um=experiment_config.target_on_width_um,
            on_center_um=phase_offset_um,
        )
        dmd_metrics = calculate_periodic_profile_metrics(
            pattern.x_um,
            dmd_profile,
            pitch_um=experiment_config.target_pitch_um,
            on_width_um=experiment_config.target_on_width_um,
            on_center_um=phase_offset_um,
        )

        rasterized_interval = _measure_rasterized_on_interval(
            states,
            config,
            phase_offset_um,
        )
        target_left_edge_um = (
            phase_offset_um - 0.5 * experiment_config.target_on_width_um
        )
        target_right_edge_um = (
            phase_offset_um + 0.5 * experiment_config.target_on_width_um
        )
        ideal_threshold_interval = ideal_metrics.threshold_interval
        threshold_interval = dmd_metrics.threshold_interval
        cases.append(
            SamplingRatioCase(
                n_cd=(
                    experiment_config.target_on_width_um
                    / config.projected_mirror_pitch_um
                ),
                config=config,
                pattern=pattern,
                ideal_field=np.asarray(ideal_field, dtype=np.complex128),
                ideal_image=ideal_image,
                dmd_image=dmd_image,
                ideal_profile=ideal_profile,
                dmd_profile=dmd_profile,
                ideal_metrics=ideal_metrics,
                dmd_metrics=dmd_metrics,
                rasterized_interval=rasterized_interval,
                raster_left_epe_um=rasterized_interval.left_edge_um - target_left_edge_um,
                raster_right_epe_um=rasterized_interval.right_edge_um - target_right_edge_um,
                raster_cd_error_um=(
                    rasterized_interval.width_um
                    - experiment_config.target_on_width_um
                ),
                threshold_left_epe_um=(
                    threshold_interval.left_edge_um - target_left_edge_um
                ),
                threshold_right_epe_um=(
                    threshold_interval.right_edge_um - target_right_edge_um
                ),
                threshold_cd_error_um=(
                    threshold_interval.width_um
                    - experiment_config.target_on_width_um
                ),
                ideal_threshold_left_epe_um=(
                    ideal_threshold_interval.left_edge_um - target_left_edge_um
                ),
                ideal_threshold_right_epe_um=(
                    ideal_threshold_interval.right_edge_um - target_right_edge_um
                ),
                ideal_threshold_cd_error_um=(
                    ideal_threshold_interval.width_um
                    - experiment_config.target_on_width_um
                ),
                dmd_minus_ideal_left_edge_um=(
                    threshold_interval.left_edge_um
                    - ideal_threshold_interval.left_edge_um
                ),
                dmd_minus_ideal_right_edge_um=(
                    threshold_interval.right_edge_um
                    - ideal_threshold_interval.right_edge_um
                ),
                dmd_minus_ideal_threshold_cd_um=(
                    threshold_interval.width_um - ideal_threshold_interval.width_um
                ),
                normalized_image_rmse=_normalized_rmse(
                    ideal_image.raw_intensity,
                    dmd_image.raw_intensity,
                ),
            )
        )
    return cases


def run_cross_pattern_experiment(
    experiment_config: CrossExperimentConfig = CROSS_EXPERIMENT_CONFIG,
) -> CrossPatternCase:
    """生成并传播二维十字图形，计算阈值尺寸、面积和质心。"""

    config = DMDConfig(
        num_mirrors_x=experiment_config.num_mirrors_x,
        num_mirrors_y=experiment_config.num_mirrors_y,
        dmd_mirror_pitch_um=experiment_config.dmd_mirror_pitch_um,
        projection_magnification=experiment_config.projection_magnification,
        samples_per_mirror=experiment_config.samples_per_mirror,
        active_side_ratio=experiment_config.active_side_ratio,
        on_amplitude=1.0 + 0.0j,
        off_amplitude=0.0 + 0.0j,
    )
    states = make_cross_states(
        config,
        arm_width_um=experiment_config.arm_width_um,
        arm_length_um=experiment_config.arm_length_um,
        center_x_um=experiment_config.center_x_um,
        center_y_um=experiment_config.center_y_um,
    )
    pattern = render_dmd_field(states, config)
    ideal_field = make_ideal_cross_field(
        pattern.x_um,
        pattern.y_um,
        arm_width_um=experiment_config.arm_width_um,
        arm_length_um=experiment_config.arm_length_um,
        center_x_um=experiment_config.center_x_um,
        center_y_um=experiment_config.center_y_um,
    )
    image = calculate_coherent_aerial_image_2d(
        pattern.object_field,
        pattern.x_um,
        pattern.y_um,
        OpticalConfig2D(
            wavelength_um=experiment_config.wavelength_um,
            numerical_aperture=experiment_config.numerical_aperture,
        ),
    )
    threshold_mask = np.asarray(
        image.normalized_intensity >= THRESHOLD_FRACTION,
        dtype=np.bool_,
    )
    horizontal_interval = measure_threshold_interval(
        image.x_um,
        _extract_center_horizontal_profile(image),
        center_um=experiment_config.center_x_um,
    )
    vertical_interval = measure_threshold_interval(
        image.y_um,
        _extract_center_vertical_profile(image),
        center_um=experiment_config.center_y_um,
    )
    horizontal_probe_x_um = (
        experiment_config.center_x_um + experiment_config.arm_probe_offset_um
    )
    vertical_probe_y_um = (
        experiment_config.center_y_um + experiment_config.arm_probe_offset_um
    )
    horizontal_arm_column = int(
        np.argmin(np.abs(image.x_um - horizontal_probe_x_um))
    )
    horizontal_arm_profile = np.asarray(
        image.raw_intensity[:, horizontal_arm_column],
        dtype=np.float64,
    )
    vertical_arm_row = int(
        np.argmin(np.abs(image.y_um - vertical_probe_y_um))
    )
    vertical_arm_profile = np.asarray(
        image.raw_intensity[vertical_arm_row, :],
        dtype=np.float64,
    )
    global_peak = float(np.max(image.raw_intensity))
    horizontal_local_peak = float(np.max(horizontal_arm_profile))
    vertical_local_peak = float(np.max(vertical_arm_profile))
    horizontal_threshold_fraction = (
        THRESHOLD_FRACTION * global_peak / horizontal_local_peak
    )
    vertical_threshold_fraction = (
        THRESHOLD_FRACTION * global_peak / vertical_local_peak
    )
    horizontal_arm_interval = measure_threshold_interval(
        image.y_um,
        horizontal_arm_profile,
        center_um=experiment_config.center_y_um,
        threshold_fraction=horizontal_threshold_fraction,
    )
    vertical_arm_interval = measure_threshold_interval(
        image.x_um,
        vertical_arm_profile,
        center_um=experiment_config.center_x_um,
        threshold_fraction=vertical_threshold_fraction,
    )

    sample_area_um2 = pattern.sample_spacing_um**2
    threshold_area_um2 = float(np.count_nonzero(threshold_mask) * sample_area_um2)
    if np.any(threshold_mask):
        xx_um, yy_um = np.meshgrid(image.x_um, image.y_um, indexing="xy")
        centroid_x_um = float(np.mean(xx_um[threshold_mask]))
        centroid_y_um = float(np.mean(yy_um[threshold_mask]))
        centroid_shift_um = math.hypot(
            centroid_x_um - experiment_config.center_x_um,
            centroid_y_um - experiment_config.center_y_um,
        )
    else:
        centroid_x_um = centroid_y_um = centroid_shift_um = math.nan

    return CrossPatternCase(
        config=config,
        pattern=pattern,
        ideal_field=np.asarray(ideal_field, dtype=np.complex128),
        image=image,
        threshold_mask=threshold_mask,
        central_horizontal_span_um=horizontal_interval.width_um,
        central_vertical_span_um=vertical_interval.width_um,
        horizontal_arm_width_um=horizontal_arm_interval.width_um,
        vertical_arm_width_um=vertical_arm_interval.width_um,
        threshold_area_um2=threshold_area_um2,
        centroid_x_um=centroid_x_um,
        centroid_y_um=centroid_y_um,
        centroid_shift_um=centroid_shift_um,
        central_span_difference_um=abs(
            horizontal_interval.width_um - vertical_interval.width_um
        ),
        arm_width_difference_um=abs(
            horizontal_arm_interval.width_um - vertical_arm_interval.width_um
        ),
    )


def _image_extent(x_um: FloatArray, y_um: FloatArray) -> tuple[float, float, float, float]:
    """根据采样中心坐标生成imshow使用的像素边界范围。"""

    dx_um = float(np.mean(np.diff(x_um)))
    dy_um = float(np.mean(np.diff(y_um)))
    return (
        float(x_um[0] - 0.5 * dx_um),
        float(x_um[-1] + 0.5 * dx_um),
        float(y_um[0] - 0.5 * dy_um),
        float(y_um[-1] + 0.5 * dy_um),
    )


def _mirror_extent(config: DMDConfig) -> tuple[float, float, float, float]:
    """生成微镜状态图使用的曝光面坐标范围。"""

    half_width_um = 0.5 * config.num_mirrors_x * config.projected_mirror_pitch_um
    half_height_um = 0.5 * config.num_mirrors_y * config.projected_mirror_pitch_um
    return (-half_width_um, half_width_um, -half_height_um, half_height_um)


def _save_figure(figure: plt.Figure, path: Path, show: bool) -> None:
    """统一紧凑布局，并用同目录临时文件原子保存图片。"""

    figure.tight_layout()
    temporary_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary_path, dpi=180, bbox_inches="tight")
    if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
        raise RuntimeError(f"图片保存失败或为空：{path.name}")
    temporary_path.replace(path)
    if not show:
        plt.close(figure)


def plot_dmd_state_and_aperture(
    output_path: Path,
    pattern: DMDPatternResult,
    config: DMDConfig,
    show: bool,
) -> None:
    """绘制图1：微镜状态、单微镜开口和展开场。"""

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].imshow(
        pattern.mirror_states,
        origin="lower",
        cmap="gray",
        interpolation="nearest",
        extent=_mirror_extent(config),
    )
    axes[0].set_title("Mirror ON/OFF states")
    axes[0].set_xlabel("x (um, image plane)")
    axes[0].set_ylabel("y (um, image plane)")

    axes[1].imshow(
        np.abs(pattern.mirror_aperture_template),
        origin="lower",
        cmap="gray",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title(
        "Single-mirror aperture\n"
        f"requested={config.active_side_ratio:.3f}, "
        f"actual={pattern.actual_active_side_ratio:.3f}"
    )
    axes[1].set_xlabel("sample column")
    axes[1].set_ylabel("sample row")

    axes[2].imshow(
        np.abs(pattern.object_field),
        origin="lower",
        cmap="gray",
        interpolation="nearest",
        extent=_image_extent(pattern.x_um, pattern.y_um),
        vmin=0.0,
        vmax=max(1.0, float(np.max(np.abs(pattern.object_field)))),
    )
    axes[2].set_title("Rendered DMD field magnitude")
    axes[2].set_xlabel("x (um)")
    axes[2].set_ylabel("y (um)")
    _save_figure(figure, output_path, show)


def plot_fourier_imaging_chain(
    output_path: Path,
    result: CoherentImage2DResult,
    show: bool,
    experiment_config: TextbookExperimentConfig = TEXTBOOK_EXPERIMENT_CONFIG,
) -> None:
    """绘制图2：输入场、频谱、瞳孔与滤波频谱。"""

    spatial_extent = _image_extent(result.x_um, result.y_um)
    frequency_extent = _image_extent(
        result.frequency_x_cyc_per_um,
        result.frequency_y_cyc_per_um,
    )
    frequency_display_limit = min(
        float(np.max(np.abs(result.frequency_x_cyc_per_um))),
        experiment_config.frequency_display_cutoff_multiple
        * result.frequency_cutoff_cyc_per_um,
    )
    panels = (
        (np.abs(result.object_field), "Input field magnitude", spatial_extent, "gray"),
        (
            np.log10(1.0 + np.abs(result.object_spectrum)),
            "log10(1 + spectrum magnitude)",
            frequency_extent,
            "magma",
        ),
        (np.abs(result.pupil), "Pupil modulus", frequency_extent, "gray"),
        (
            np.log10(1.0 + np.abs(result.filtered_spectrum)),
            "log10(1 + filtered spectrum)",
            frequency_extent,
            "magma",
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    for axis, (image, title, extent, color_map) in zip(axes.flat, panels):
        plot = axis.imshow(
            image,
            origin="lower",
            cmap=color_map,
            extent=extent,
            aspect="auto",
        )
        axis.set_title(title)
        figure.colorbar(plot, ax=axis, shrink=0.82)
    axes[0, 0].set_xlabel("x (um)")
    axes[0, 0].set_ylabel("y (um)")
    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.set_xlabel("fx (cycles/um)")
        axis.set_ylabel("fy (cycles/um)")
        axis.set_xlim(-frequency_display_limit, frequency_display_limit)
        axis.set_ylim(-frequency_display_limit, frequency_display_limit)
    _save_figure(figure, output_path, show)


def plot_aerial_image_and_profiles(
    output_path: Path,
    result: CoherentImage2DResult,
    metrics: ProfileMetrics,
    show: bool,
    experiment_config: TextbookExperimentConfig = TEXTBOOK_EXPERIMENT_CONFIG,
) -> None:
    """绘制图3：二维空中像、横向和纵向中央截面。"""

    horizontal_profile = _extract_center_horizontal_profile(result)
    vertical_profile = _extract_center_vertical_profile(result)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    image = axes[0].imshow(
        result.normalized_intensity,
        origin="lower",
        cmap="viridis",
        extent=_image_extent(result.x_um, result.y_um),
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    axes[0].set_title("Normalized 2D aerial image")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    figure.colorbar(image, ax=axes[0], shrink=0.82)

    horizontal_normalized = _peak_normalize(horizontal_profile)
    axes[1].plot(result.x_um, horizontal_normalized, color="tab:blue")
    axes[1].axhline(THRESHOLD_FRACTION, color="tab:red", linestyle="--", label="0.5 threshold")
    target_left_edge_um = (
        experiment_config.phase_offset_um - 0.5 * experiment_config.on_width_um
    )
    target_right_edge_um = (
        experiment_config.phase_offset_um + 0.5 * experiment_config.on_width_um
    )
    axes[1].axvline(
        target_left_edge_um,
        color="black",
        linestyle=":",
        label="target edges",
    )
    axes[1].axvline(target_right_edge_um, color="black", linestyle=":")
    axes[1].set_xlim(
        experiment_config.phase_offset_um
        - experiment_config.profile_display_half_range_um,
        experiment_config.phase_offset_um
        + experiment_config.profile_display_half_range_um,
    )
    axes[1].set_ylim(-0.05, 1.08)
    axes[1].set_title(
        f"Central x profile\nC={metrics.contrast:.4f}, NILS={metrics.nils:.4f}"
    )
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("normalized intensity")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    axes[2].plot(result.y_um, _peak_normalize(vertical_profile), color="tab:green")
    axes[2].axhline(THRESHOLD_FRACTION, color="tab:red", linestyle="--")
    axes[2].set_xlim(
        -experiment_config.profile_display_half_range_um,
        experiment_config.profile_display_half_range_um,
    )
    axes[2].set_ylim(-0.05, 1.08)
    axes[2].set_title("Central y profile")
    axes[2].set_xlabel("y (um)")
    axes[2].set_ylabel("normalized intensity")
    axes[2].grid(alpha=0.25)
    _save_figure(figure, output_path, show)


def plot_textbook_cross_validation(
    output_path: Path,
    cross_validation: TextbookCrossValidation,
    dmd_result: CoherentImage2DResult,
    show: bool,
    experiment_config: TextbookExperimentConfig = TEXTBOOK_EXPERIMENT_CONFIG,
) -> None:
    """绘制图4；一维文件缺失时在图中明确标记SKIPPED。"""

    figure, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=False)
    if cross_validation.status == "COMPLETED":
        assert cross_validation.x_um is not None
        assert cross_validation.textbook_intensity is not None
        assert cross_validation.dmd_2d_intensity is not None
        assert cross_validation.residual is not None
        axes[0].plot(
            cross_validation.x_um,
            cross_validation.textbook_intensity,
            label="1D textbook program",
            linewidth=2.0,
        )
        axes[0].plot(
            cross_validation.x_um,
            cross_validation.dmd_2d_intensity,
            "--",
            label="2D DMD central profile",
        )
        axes[0].set_title(
            "Real 1D-to-2D cross-validation\n"
            f"raw RMSE={cross_validation.raw_rmse:.3e}, "
            f"normalized RMSE={cross_validation.peak_normalized_rmse:.3e}"
        )
        axes[0].legend(loc="best")
        axes[1].plot(
            cross_validation.x_um,
            cross_validation.residual,
            color="tab:red",
        )
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_title("Residual: 2D - 1D")
        axes[1].set_xlabel("x (um)")
        axes[1].set_ylabel("raw intensity residual")
    else:
        dmd_profile = _extract_center_horizontal_profile(dmd_result)
        comparison_left_um = (
            experiment_config.phase_offset_um - 0.5 * experiment_config.pitch_um
        )
        comparison_right_um = (
            experiment_config.phase_offset_um + 0.5 * experiment_config.pitch_um
        )
        central_mask = (
            (dmd_result.x_um >= comparison_left_um)
            & (dmd_result.x_um <= comparison_right_um)
        )
        axes[0].plot(
            dmd_result.x_um[central_mask],
            dmd_profile[central_mask],
            label="2D DMD profile only",
        )
        axes[0].legend(loc="best")
        axes[0].set_title("1D-to-2D cross-validation: SKIPPED")
        axes[1].axis("off")
        axes[1].text(
            0.5,
            0.55,
            "SKIPPED\n\n"
            f"Required 1D source file is missing:\n"
            f"{TEXTBOOK_MODULE_FILENAME}",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            fontsize=12,
            wrap=True,
        )
        axes[1].text(
            0.5,
            0.20,
            "No analytical curve or metric was copied, guessed, or fabricated.",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            fontsize=10,
        )
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("raw intensity")
    axes[0].grid(alpha=0.25)
    _save_figure(figure, output_path, show)


def plot_na_sweep(
    output_path: Path,
    cases: Sequence[NASweepCase],
    show: bool,
    experiment_config: TextbookExperimentConfig = TEXTBOOK_EXPERIMENT_CONFIG,
) -> None:
    """绘制图5：逐个NA展示瞳孔、保留频谱、空中像、截面和指标。"""

    figure, axes = plt.subplots(len(cases), 5, figsize=(18, 10))
    shared_frequency_limit = experiment_config.frequency_display_cutoff_multiple * max(
        case.result.frequency_cutoff_cyc_per_um for case in cases
    )
    for row, case in enumerate(cases):
        result = case.result
        frequency_extent = _image_extent(
            result.frequency_x_cyc_per_um,
            result.frequency_y_cyc_per_um,
        )
        spatial_extent = _image_extent(result.x_um, result.y_um)
        axes[row, 0].imshow(
            np.abs(result.pupil),
            origin="lower",
            cmap="gray",
            extent=frequency_extent,
            aspect="auto",
        )
        axes[row, 0].set_title(
            f"NA={case.numerical_aperture:.2f}, jmax={case.maximum_geometric_order}\npupil"
        )
        axes[row, 1].imshow(
            np.log10(1.0 + np.abs(result.filtered_spectrum)),
            origin="lower",
            cmap="magma",
            extent=frequency_extent,
            aspect="auto",
        )
        axes[row, 1].set_title("retained spectrum (log)")
        axes[row, 2].imshow(
            result.normalized_intensity,
            origin="lower",
            cmap="viridis",
            extent=spatial_extent,
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        axes[row, 2].set_xlim(
            experiment_config.phase_offset_um
            - experiment_config.profile_display_half_range_um,
            experiment_config.phase_offset_um
            + experiment_config.profile_display_half_range_um,
        )
        axes[row, 2].set_ylim(
            -experiment_config.profile_display_half_range_um,
            experiment_config.profile_display_half_range_um,
        )
        axes[row, 2].set_title("normalized aerial image")

        central_mask = (
            result.x_um
            >= experiment_config.phase_offset_um
            - experiment_config.profile_display_half_range_um
        ) & (
            result.x_um
            <= experiment_config.phase_offset_um
            + experiment_config.profile_display_half_range_um
        )
        axes[row, 3].plot(
            result.x_um[central_mask],
            _peak_normalize(case.profile)[central_mask],
            color="tab:blue",
        )
        axes[row, 3].axhline(0.5, color="tab:red", linestyle="--")
        axes[row, 3].set_ylim(-0.05, 1.08)
        axes[row, 3].set_title("central profile")
        axes[row, 3].grid(alpha=0.25)

        axes[row, 4].axis("off")
        axes[row, 4].text(
            0.04,
            0.72,
            f"center contrast = {_format_number(case.metrics.contrast)}\n"
            f"NILS = {_format_number(case.metrics.nils)}\n"
            f"threshold CD = "
            f"{_format_number(case.metrics.threshold_interval.width_um)} um\n"
            f"discrete energy ratio = {case.result.energy_transmission:.6f}",
            transform=axes[row, 4].transAxes,
            va="top",
            fontsize=10,
        )
        for column in (0, 1):
            axes[row, column].set_xlabel("fx (cycles/um)")
            axes[row, column].set_ylabel("fy (cycles/um)")
            axes[row, column].set_xlim(
                -shared_frequency_limit,
                shared_frequency_limit,
            )
            axes[row, column].set_ylim(
                -shared_frequency_limit,
                shared_frequency_limit,
            )
        axes[row, 2].set_xlabel("x (um)")
        axes[row, 2].set_ylabel("y (um)")
        axes[row, 3].set_xlabel("x (um)")
        axes[row, 3].set_ylabel("normalized intensity")
    _save_figure(figure, output_path, show)


def _format_plot_number(value: float, precision: int = 3) -> str:
    """按显示精度舍入并去掉负零；原始计算值及CSV精度不变。"""

    if not math.isfinite(value):
        return f"{value}"
    rounded_value = round(value, precision)
    if rounded_value == 0.0:
        rounded_value = 0.0
    return f"{rounded_value:.{precision}f}"


def plot_sampling_ratio_comparison(
    output_path: Path,
    cases: Sequence[SamplingRatioCase],
    show: bool,
    experiment_config: SamplingExperimentConfig = SAMPLING_EXPERIMENT_CONFIG,
) -> None:
    """绘制图6：三种N_CD下的目标、DMD状态、空中像与阈值结果。"""

    figure, axes = plt.subplots(
        len(cases), 7, figsize=(27, 12), squeeze=False,
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1, 1.7]},
    )
    for row, case in enumerate(cases):
        spatial_extent = _image_extent(case.pattern.x_um, case.pattern.y_um)
        display_half_range_um = experiment_config.display_half_range_um
        axes[row, 0].imshow(
            np.abs(case.ideal_field),
            origin="lower",
            cmap="gray",
            extent=spatial_extent,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )
        axes[row, 0].set_title(f"N_CD={case.n_cd:.2f}: ideal target")
        axes[row, 1].imshow(
            case.pattern.mirror_states,
            origin="lower",
            cmap="gray",
            extent=_mirror_extent(case.config),
            interpolation="nearest",
            aspect="auto",
        )
        axes[row, 1].set_title("DMD states")
        axes[row, 2].imshow(
            np.abs(case.pattern.object_field),
            origin="lower",
            cmap="gray",
            extent=spatial_extent,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )
        axes[row, 2].set_title("Rendered DMD field")
        axes[row, 3].imshow(
            case.ideal_image.normalized_intensity,
            origin="lower",
            cmap="viridis",
            extent=spatial_extent,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )
        axes[row, 3].set_title("Ideal optical image")
        axes[row, 4].imshow(
            case.dmd_image.normalized_intensity,
            origin="lower",
            cmap="viridis",
            extent=spatial_extent,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )
        axes[row, 4].set_title("DMD aerial image")
        threshold_image = np.asarray(
            case.dmd_image.normalized_intensity >= THRESHOLD_FRACTION,
            dtype=np.float64,
        )
        axes[row, 5].imshow(
            threshold_image,
            origin="lower",
            cmap="gray",
            extent=spatial_extent,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )
        axes[row, 5].set_title("0.5 threshold result")

        central_mask = (
            (case.pattern.x_um >= -display_half_range_um)
            & (case.pattern.x_um <= display_half_range_um)
        )
        axes[row, 6].plot(
            case.pattern.x_um[central_mask],
            _peak_normalize(case.ideal_profile)[central_mask],
            label="ideal optical image",
        )
        axes[row, 6].plot(
            case.pattern.x_um[central_mask],
            _peak_normalize(case.dmd_profile)[central_mask],
            "--",
            label="DMD optical image",
        )
        axes[row, 6].axhline(0.5, color="black", linestyle=":")
        axes[row, 6].set_title(
            "CD (DMD/ideal) = "
            f"{_format_plot_number(case.dmd_metrics.threshold_interval.width_um)} / "
            f"{_format_plot_number(case.ideal_metrics.threshold_interval.width_um)} um\n"
            r"$\Delta$CD (DMD - ideal) = "
            f"{_format_plot_number(case.dmd_minus_ideal_threshold_cd_um)} um\n"
            "Image edge shift (L/R) = "
            f"{_format_plot_number(case.dmd_minus_ideal_left_edge_um)} / "
            f"{_format_plot_number(case.dmd_minus_ideal_right_edge_um)} um\n"
            "Raster EPE (L/R) = "
            f"{_format_plot_number(case.raster_left_epe_um)} / "
            f"{_format_plot_number(case.raster_right_epe_um)} um",
            fontsize=9,
        )
        axes[row, 6].set_ylim(-0.05, 1.08)
        axes[row, 6].grid(alpha=0.25)
        if row == 0:
            axes[row, 6].legend(fontsize=8, loc="best")
        for column in range(6):
            axes[row, column].set_xlim(-display_half_range_um, display_half_range_um)
            axes[row, column].set_ylim(-display_half_range_um, display_half_range_um)
        for axis in axes[row, :]:
            axis.set_xlabel("x (um)")
    _save_figure(figure, output_path, show)


def plot_cross_pattern(
    output_path: Path,
    case: CrossPatternCase,
    show: bool,
    experiment_config: CrossExperimentConfig = CROSS_EXPERIMENT_CONFIG,
) -> None:
    """绘制图7：十字目标、DMD、频谱链路、空中像和阈值图形。"""

    result = case.image
    spatial_extent = _image_extent(result.x_um, result.y_um)
    frequency_extent = _image_extent(
        result.frequency_x_cyc_per_um,
        result.frequency_y_cyc_per_um,
    )
    frequency_display_limit = min(
        float(np.max(np.abs(result.frequency_x_cyc_per_um))),
        experiment_config.frequency_display_cutoff_multiple
        * result.frequency_cutoff_cyc_per_um,
    )
    panels = (
        (np.abs(case.ideal_field), "Continuous ideal cross", spatial_extent, "gray"),
        (
            case.pattern.mirror_states,
            "DMD states",
            _mirror_extent(case.config),
            "gray",
        ),
        (np.abs(case.pattern.object_field), "Finite-aperture DMD field", spatial_extent, "gray"),
        (
            np.log10(1.0 + np.abs(result.object_spectrum)),
            "DMD spectrum (log)",
            frequency_extent,
            "magma",
        ),
        (np.abs(result.pupil), "Pupil modulus", frequency_extent, "gray"),
        (
            np.log10(1.0 + np.abs(result.filtered_spectrum)),
            "Filtered spectrum (log)",
            frequency_extent,
            "magma",
        ),
        (result.normalized_intensity, "Normalized aerial image", spatial_extent, "viridis"),
        (case.threshold_mask, "0.5 threshold pattern", spatial_extent, "gray"),
    )
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    for panel_index, (axis, (image, title, extent, color_map)) in enumerate(
        zip(axes.flat, panels)
    ):
        axis.imshow(
            image,
            origin="lower",
            cmap=color_map,
            extent=extent,
            aspect="auto",
            interpolation="nearest" if "states" in title.lower() else None,
        )
        axis.set_title(title)
        if panel_index in (3, 4, 5):
            axis.set_xlabel("fx (cycles/um)")
            axis.set_ylabel("fy (cycles/um)")
            axis.set_xlim(-frequency_display_limit, frequency_display_limit)
            axis.set_ylim(-frequency_display_limit, frequency_display_limit)
        else:
            axis.set_xlabel("x (um)")
            axis.set_ylabel("y (um)")
    figure.suptitle(
        "2D cross example: "
        f"central x/y spans=({case.central_horizontal_span_um:.3f}, "
        f"{case.central_vertical_span_um:.3f}) um; "
        f"transverse arm widths=({case.horizontal_arm_width_um:.3f}, "
        f"{case.vertical_arm_width_um:.3f}) um; "
        f"centroid shift={case.centroid_shift_um:.3e} um",
        y=1.01,
    )
    _save_figure(figure, output_path, show)


def _empty_metric_row(**updates: Any) -> dict[str, Any]:
    """创建带NaN默认值的CSV指标行。"""

    row: dict[str, Any] = {field: math.nan for field in METRIC_FIELDNAMES}
    row["notes"] = ""
    row.update(updates)
    return row


def build_metric_rows(
    base_pattern: DMDPatternResult | None,
    base_result: CoherentImage2DResult | None,
    base_metrics: ProfileMetrics | None,
    cross_validation: TextbookCrossValidation | None,
    convergence: ConvergenceResult | None,
    na_cases: Sequence[NASweepCase],
    sampling_cases: Sequence[SamplingRatioCase],
    cross_case: CrossPatternCase | None,
    textbook_config: TextbookExperimentConfig = TEXTBOOK_EXPERIMENT_CONFIG,
    sampling_config: SamplingExperimentConfig = SAMPLING_EXPERIMENT_CONFIG,
    cross_config: CrossExperimentConfig = CROSS_EXPERIMENT_CONFIG,
) -> list[dict[str, Any]]:
    """汇总全部实验的CSV数据。"""

    base_config = _base_dmd_config(
        textbook_config.samples_per_mirror,
        textbook_config,
    )
    rows: list[dict[str, Any]] = []
    if cross_validation is not None:
        if base_pattern is None or base_result is None or base_metrics is None:
            raise ValueError("教材比较结果需要对应的二维基准数据。")
        rows.append(
            _empty_metric_row(
                experiment="1d_2d_cross_validation",
                case_name=(
                    "2d_dmd_reference_spm"
                    f"{textbook_config.samples_per_mirror}"
                ),
                wavelength_um=textbook_config.wavelength_um,
                numerical_aperture=textbook_config.numerical_aperture,
                projected_mirror_pitch_um=base_config.projected_mirror_pitch_um,
                samples_per_mirror=textbook_config.samples_per_mirror,
                active_side_ratio=base_pattern.actual_active_side_ratio,
                n_cd=textbook_config.n_cd,
                input_energy=base_result.input_energy,
                output_energy=base_result.output_energy,
                energy_transmission=base_result.energy_transmission,
                space_center_intensity=base_metrics.space_center_intensity,
                line_center_intensity=base_metrics.line_center_intensity,
                contrast=base_metrics.contrast,
                threshold_cd_um=base_metrics.threshold_interval.width_um,
                nils=base_metrics.nils,
                rmse_to_reference=cross_validation.peak_normalized_rmse,
                notes=(
                    cross_validation.reason
                    if cross_validation.status == "SKIPPED"
                    else "真实一维教材程序比较已完成"
                ),
            )
        )
    if convergence is not None:
        rows.append(
            _empty_metric_row(
                experiment="grid_convergence",
                case_name=f"spm{textbook_config.convergence_samples_per_mirror}",
                wavelength_um=textbook_config.wavelength_um,
                numerical_aperture=textbook_config.numerical_aperture,
                projected_mirror_pitch_um=base_config.projected_mirror_pitch_um,
                samples_per_mirror=textbook_config.convergence_samples_per_mirror,
                active_side_ratio=textbook_config.active_side_ratio,
                n_cd=textbook_config.n_cd,
                space_center_intensity=convergence.metrics_16.space_center_intensity,
                line_center_intensity=convergence.metrics_16.line_center_intensity,
                contrast=convergence.metrics_16.contrast,
                threshold_cd_um=convergence.metrics_16.threshold_interval.width_um,
                nils=convergence.metrics_16.nils,
                rmse_to_reference=convergence.normalized_profile_rmse,
                notes=(
                    f"{textbook_config.samples_per_mirror}与"
                    f"{textbook_config.convergence_samples_per_mirror}采样/"
                    "微镜主要指标变化不超过2%"
                    if convergence.converged_within_two_percent
                    else f"{textbook_config.samples_per_mirror}采样/"
                    "微镜尚未满足2%收敛建议"
                ),
            )
        )

    if na_cases and base_pattern is None:
        raise ValueError("NA扫描结果需要对应的DMD基准图形。")
    for case in na_cases:
        rows.append(
            _empty_metric_row(
                experiment="na_sweep",
                case_name=f"NA_{case.numerical_aperture:.2f}",
                wavelength_um=textbook_config.wavelength_um,
                numerical_aperture=case.numerical_aperture,
                projected_mirror_pitch_um=base_config.projected_mirror_pitch_um,
                samples_per_mirror=textbook_config.samples_per_mirror,
                active_side_ratio=base_pattern.actual_active_side_ratio,
                n_cd=textbook_config.n_cd,
                input_energy=case.result.input_energy,
                output_energy=case.result.output_energy,
                energy_transmission=case.result.energy_transmission,
                space_center_intensity=case.metrics.space_center_intensity,
                line_center_intensity=case.metrics.line_center_intensity,
                contrast=case.metrics.contrast,
                threshold_cd_um=case.metrics.threshold_interval.width_um,
                nils=case.metrics.nils,
                notes=(
                    "仅零级通过，均匀像NILS为0，阈值CD不可定义"
                    if case.maximum_geometric_order < 1
                    else f"几何最高通过级次={case.maximum_geometric_order}"
                ),
            )
        )

    for case in sampling_cases:
        rows.append(
            _empty_metric_row(
                experiment="dmd_sampling_ratio_example",
                case_name=f"N_CD_{case.n_cd:.2f}",
                wavelength_um=sampling_config.wavelength_um,
                numerical_aperture=sampling_config.numerical_aperture,
                projected_mirror_pitch_um=case.config.projected_mirror_pitch_um,
                samples_per_mirror=case.config.samples_per_mirror,
                active_side_ratio=case.pattern.actual_active_side_ratio,
                n_cd=case.n_cd,
                input_energy=case.dmd_image.input_energy,
                output_energy=case.dmd_image.output_energy,
                energy_transmission=case.dmd_image.energy_transmission,
                space_center_intensity=case.dmd_metrics.space_center_intensity,
                line_center_intensity=case.dmd_metrics.line_center_intensity,
                contrast=case.dmd_metrics.contrast,
                threshold_cd_um=case.dmd_metrics.threshold_interval.width_um,
                nils=case.dmd_metrics.nils,
                left_epe_um=case.raster_left_epe_um,
                right_epe_um=case.raster_right_epe_um,
                rmse_to_reference=case.normalized_image_rmse,
                rasterized_on_width_um=case.rasterized_interval.width_um,
                raster_cd_error_um=case.raster_cd_error_um,
                threshold_cd_error_um=case.threshold_cd_error_um,
                threshold_left_epe_um=case.threshold_left_epe_um,
                threshold_right_epe_um=case.threshold_right_epe_um,
                ideal_threshold_cd_um=case.ideal_metrics.threshold_interval.width_um,
                ideal_threshold_cd_error_um=case.ideal_threshold_cd_error_um,
                ideal_threshold_left_epe_um=case.ideal_threshold_left_epe_um,
                ideal_threshold_right_epe_um=case.ideal_threshold_right_epe_um,
                dmd_minus_ideal_threshold_cd_um=case.dmd_minus_ideal_threshold_cd_um,
                dmd_minus_ideal_left_edge_um=case.dmd_minus_ideal_left_edge_um,
                dmd_minus_ideal_right_edge_um=case.dmd_minus_ideal_right_edge_um,
                notes="示例参数，不代表公司真实设备参数",
            )
        )

    if cross_case is not None:
        rows.append(
            _empty_metric_row(
                experiment="2d_cross_pattern_example",
                case_name=(
                    f"cross_{cross_config.arm_width_um:g}um_width_"
                    f"{cross_config.arm_length_um:g}um_length"
                ),
                wavelength_um=cross_config.wavelength_um,
                numerical_aperture=cross_config.numerical_aperture,
                projected_mirror_pitch_um=cross_case.config.projected_mirror_pitch_um,
                samples_per_mirror=cross_case.config.samples_per_mirror,
                active_side_ratio=cross_case.pattern.actual_active_side_ratio,
                input_energy=cross_case.image.input_energy,
                output_energy=cross_case.image.output_energy,
                energy_transmission=cross_case.image.energy_transmission,
                threshold_cd_um=cross_case.horizontal_arm_width_um,
                horizontal_cd_um=cross_case.horizontal_arm_width_um,
                vertical_cd_um=cross_case.vertical_arm_width_um,
                central_horizontal_span_um=cross_case.central_horizontal_span_um,
                central_vertical_span_um=cross_case.central_vertical_span_um,
                threshold_area_um2=cross_case.threshold_area_um2,
                centroid_x_um=cross_case.centroid_x_um,
                centroid_y_um=cross_case.centroid_y_um,
                notes="示例参数；孤立图形仅评价中央ROI",
            )
        )
    return rows


def write_metrics_csv(output_path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """以UTF-8编码保存指标CSV。"""

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _format_number(value: float, precision: int = 6) -> str:
    """将有限数值格式化；不可定义值统一写为NaN。"""

    return f"{value:.{precision}g}" if math.isfinite(value) else "NaN"


def write_validation_summary(
    output_path: Path,
    checks: Sequence[ValidationCheck],
    cross_validation: TextbookCrossValidation | None = None,
    convergence: ConvergenceResult | None = None,
    na_cases: Sequence[NASweepCase] = (),
    sampling_cases: Sequence[SamplingRatioCase] = (),
    cross_case: CrossPatternCase | None = None,
    textbook_config: TextbookExperimentConfig = TEXTBOOK_EXPERIMENT_CONFIG,
    sampling_config: SamplingExperimentConfig = SAMPLING_EXPERIMENT_CONFIG,
    cross_config: CrossExperimentConfig = CROSS_EXPERIMENT_CONFIG,
) -> None:
    """保存包含PASS/FAIL/SKIPPED状态和模型边界的文本摘要。"""

    maximum_order = textbook_config.maximum_geometric_order
    geometric_orders = tuple(range(-maximum_order, maximum_order + 1))
    nonzero_orders = tuple(
        order for order in geometric_orders if order == 0 or order % 2 != 0
    )
    geometric_order_text = ",".join(str(order) for order in geometric_orders)
    nonzero_order_text = ",".join(str(order) for order in nonzero_orders)

    lines = [
        "二维DMD相干空中像验证摘要",
        "=" * 42,
        "",
        "模型边界：有限开口二值DMD等效复振幅 + 理想圆瞳 + 完全相干成像。",
        "全部长度坐标均为曝光面等效坐标，FFT采用周期边界和正交归一化。",
        "energy_transmission定义为离散场Σ|Uout|²/Σ|Uin|²的频谱能量比例；"
        "它不是DMD衍射效率、实际光功率透过率或曝光效率。",
        "NILS的边缘强度与导数采用设计边缘附近7点四次局部多项式拟合；"
        "阈值左右交点采用相邻采样点线性插值；周期漏光工作点模式见本次配置。",
        "",
        "一、内部数值自检",
    ]
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(
            f"- {status}: {check.name}; value={check.value:.6e}; "
            f"criterion {check.comparison} {check.limit:.6e}; {check.detail}"
        )

    lines.extend(
        [
            "",
            "二、一维—二维交叉验证",
            f"- 基准：透明空间{textbook_config.on_width_um:g} um，"
            f"不透明线{textbook_config.line_width_um:g} um，"
            f"周期{textbook_config.pitch_um:g} um，"
            f"波长{textbook_config.wavelength_um:g} um，"
            f"NA={textbook_config.numerical_aperture:g}。",
            "- 几何最高通过级次："
            f"floor({textbook_config.pitch_um:g}*"
            f"{textbook_config.numerical_aperture:g}/"
            f"{textbook_config.wavelength_um:g})={maximum_order}。",
            f"- 几何通过级次：{geometric_order_text}。",
            f"- 50%占空比理论非零成像级次：{nonzero_order_text}。",
        ]
    )
    if cross_validation is None:
        lines.append("- NOT RUN: 未启用或未执行。")
    elif cross_validation.status == "SKIPPED":
        lines.append(f"- SKIPPED: {cross_validation.reason}")
        lines.append("- 一维RMSE、对比度差异和NILS差异均保留为NaN，未伪造指标。")
    else:
        assert cross_validation.textbook_metrics is not None
        assert cross_validation.dmd_2d_metrics is not None
        contrast_difference = _relative_difference(
            cross_validation.dmd_2d_metrics.contrast,
            cross_validation.textbook_metrics.contrast,
        )
        nils_difference = _relative_difference(
            cross_validation.dmd_2d_metrics.nils,
            cross_validation.textbook_metrics.nils,
        )
        lines.extend(
            [
                f"- COMPLETED: {cross_validation.reason}。",
                "- COMPLETED表示已执行比较，不等同于通过预设精度门限。",
                f"- 原始强度RMSE：{cross_validation.raw_rmse:.6e}",
                f"- 峰值归一化RMSE：{cross_validation.peak_normalized_rmse:.6e}",
                f"- 对比度相对差异：{contrast_difference:.3%}",
                f"- NILS相对差异：{nils_difference:.3%}",
                "- 一维NILS采用原程序解析导数；二维NILS采用局部多项式拟合。",
            ]
        )

    lines.extend(
        [
            "",
            f"三、{textbook_config.samples_per_mirror}与"
            f"{textbook_config.convergence_samples_per_mirror}采样/"
            "微镜收敛检查",
        ]
    )
    if convergence is None:
        lines.append("- NOT RUN")
    else:
        convergence_status = (
            "PASS" if convergence.converged_within_two_percent else "WARNING"
        )
        lines.extend(
            [
                f"- {convergence_status}: 最大主要指标相对变化="
                f"{convergence.maximum_metric_relative_change:.3%}",
                f"- 归一化中央周期曲线RMSE={convergence.normalized_profile_rmse:.6e}",
                f"- 中心亮区强度变化={convergence.space_center_relative_change:.3%}",
                f"- 中心暗区强度变化={convergence.line_center_relative_change:.3%}",
                f"- 对比度变化={convergence.contrast_relative_change:.3%}",
                f"- NILS变化={convergence.nils_relative_change:.3%}",
                f"- 阈值CD变化={convergence.threshold_cd_relative_change:.3%}",
            ]
        )

    lines.extend(["", "四、NA影响实验"])
    if not na_cases:
        lines.append("- NOT RUN")
    for case in na_cases:
        lines.append(
            f"- NA={case.numerical_aperture:.2f}, jmax={case.maximum_geometric_order}, "
            f"contrast={_format_number(case.metrics.contrast)}, "
            f"NILS={_format_number(case.metrics.nils)}, "
            f"threshold_CD_um={_format_number(case.metrics.threshold_interval.width_um)}"
        )
    unresolved_na_cases = [
        case for case in na_cases if case.maximum_geometric_order < 1
    ]
    if unresolved_na_cases:
        unresolved_na_text = ", ".join(
            f"{case.numerical_aperture:.2f}" for case in unresolved_na_cases
        )
        lines.append(
            f"- NA={unresolved_na_text}仅零级通过，图像接近均匀；"
            "数值NILS为0，"
            "但0.5阈值CD无法形成两个有效交点，故记为NaN。"
        )

    lines.extend(["", "五、DMD采样比实验（示例参数）"])
    if not sampling_cases:
        lines.append("- NOT RUN")
    for case in sampling_cases:
        lines.append(
            f"- N_CD={case.n_cd:.2f}, projected_pitch_um="
            f"{case.config.projected_mirror_pitch_um:.3f}, "
            f"raster_width_um={_format_number(case.rasterized_interval.width_um)}, "
            f"raster_EPE_L/R_um=({_format_number(case.raster_left_epe_um)}, "
            f"{_format_number(case.raster_right_epe_um)}), "
            f"ideal_threshold_CD_um="
            f"{_format_number(case.ideal_metrics.threshold_interval.width_um)}, "
            f"DMD_threshold_CD_um="
            f"{_format_number(case.dmd_metrics.threshold_interval.width_um)}, "
            f"ideal_threshold_CD_error_um="
            f"{_format_number(case.ideal_threshold_cd_error_um)}, "
            f"DMD_threshold_CD_error_um={_format_number(case.threshold_cd_error_um)}, "
            f"DMD_minus_ideal_CD_um="
            f"{_format_number(case.dmd_minus_ideal_threshold_cd_um)}, "
            f"DMD_minus_ideal_edge_L/R_um=("
            f"{_format_number(case.dmd_minus_ideal_left_edge_um)}, "
            f"{_format_number(case.dmd_minus_ideal_right_edge_um)}), "
            f"normalized_image_RMSE={case.normalized_image_rmse:.6e}"
        )
    if sampling_cases:
        n_cd_text = ", ".join(f"{case.n_cd:g}" for case in sampling_cases)
        lines.append(
            "- 指标定义：CD为各自峰值归一化后0.5阈值左右交点间距；"
            "DMD_minus_ideal_CD为两种空中像的CD之差，不是DMD线宽本身，"
            "也不是相对设计线宽的误差。"
        )
        lines.append(
            "- Image edge shift以理想空中像阈值边缘为参考；"
            "Raster EPE以连续设计图形边缘为参考。两者参考对象不同。"
            "图中按三位小数舍入并去掉负零，摘要和CSV保留原始差值。"
        )
        lines.append(
            "- 结论：在相同投影光学系统下，DMD网格仍限制可选边缘位置与"
            "图形尺寸；DMD地址采样与光学低通分辨率是两个不同问题。"
        )
        noninteger_values = [
            case.n_cd for case in sampling_cases
            if not math.isclose(case.n_cd, round(case.n_cd), rel_tol=0.0, abs_tol=1e-12)
        ]
        finite_raster_errors = [
            abs(case.raster_cd_error_um) for case in sampling_cases
            if math.isfinite(case.raster_cd_error_um)
        ]
        noninteger_text = (
            ", ".join(f"{value:g}" for value in noninteger_values)
            if noninteger_values else "无"
        )
        lines.append(
            f"- 本次实际计算N_CD={n_cd_text}；非整数采样比：{noninteger_text}。"
            f"有效案例中栅格化CD误差绝对值的最大值为"
            f"{_format_number(max(finite_raster_errors)) if finite_raster_errors else 'NaN'} um。"
            f"几何位置偏移为{sampling_config.phase_offset_fraction:g}像元；"
            "各案例的边缘偏移和尺寸量化以实际Raster EPE及CD误差为准。"
        )
        lines.append(
            f"- 上述{sampling_config.target_on_width_um:g} um/"
            f"{sampling_config.target_pitch_um:g} um参数仅作说明，"
            "不代表公司真实设备参数。"
        )

    lines.extend(["", "六、二维十字图形演示"])
    if cross_case is None:
        lines.append("- NOT RUN")
    else:
        lines.extend(
            [
                "- 中央横/纵截面跨越整个十字臂长，"
                f"不代表{cross_config.arm_width_um:g} um臂宽。",
                f"- 中央横向阈值跨度={_format_number(cross_case.central_horizontal_span_um)} um",
                f"- 中央纵向阈值跨度={_format_number(cross_case.central_vertical_span_um)} um",
                f"- 远离交叉区测得水平臂横向宽度="
                f"{_format_number(cross_case.horizontal_arm_width_um)} um",
                f"- 远离交叉区测得垂直臂横向宽度="
                f"{_format_number(cross_case.vertical_arm_width_um)} um",
                f"- 两方向臂宽绝对差={_format_number(cross_case.arm_width_difference_um)} um",
                f"- 0.5阈值面积={_format_number(cross_case.threshold_area_um2)} um^2",
                f"- 质心(x,y)=({_format_number(cross_case.centroid_x_um)}, "
                f"{_format_number(cross_case.centroid_y_um)}) um；"
                f"偏移={_format_number(cross_case.centroid_shift_um)} um",
                "- 图中可观察圆瞳低通作用下的角点圆化与线端变化。",
            ]
        )

    lines.extend(
        [
            "",
            "七、当前模型不能代表的过程",
            "- DLP9000X真实微镜倾转、闪耀条件及OFF漏光的物理来源和角分布；",
            "- 新增ON/OFF响应为选定投影通道内的等效复振幅，不预测器件衍射效率；",
            "- DMD窗口透过率、偏振、实际照明角谱和空间非均匀性；",
            "- 实际投影镜头MTF、离焦、像差和空间变化PSF；",
            "- 部分相干、扫描曝光、DMD刷新时序、运动误差；",
            "- 光刻胶化学、显影、绝对曝光剂量和最终工艺CD。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="运行二维DMD相干空中像验证实验。"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIRECTORY),
        help=f"结果目录，默认：{DEFAULT_OUTPUT_DIRECTORY}",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="保存结果后显示Matplotlib窗口。",
    )
    return parser.parse_args()


def effective_threshold_for_dose(nominal_threshold_intensity: float, dose_scale: float) -> float:
    """s=E/E0只改变评价阈值T0/s；不再把原始光强乘s，也不改变eta。"""
    for name, value in (("dose_scale", dose_scale),
                        ("nominal_threshold_intensity", nominal_threshold_intensity)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name}必须为有限正实数。")
    threshold = float(nominal_threshold_intensity) / float(dose_scale)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("T0/s超出有限正浮点强度范围。")
    return threshold


def establish_nominal_workpoint(
    x_um: FloatArray,
    intensity: FloatArray,
    *,
    on_width_um: float,
    center_um: float,
    reference_intensity: float,
    search_bounds_um: tuple[float, float],
    threshold_mode: str = "fixed_fraction",
    threshold_fraction: float = 0.5,
) -> dict[str, Any]:
    """仅给对称周期基准设名义工作点，不是材料/设备标定。

    设计边缘取同一分段线性曲线的值。强度对称容差为I_ref的1e-10，
    用于容纳对称FFT和插值的舍入误差，不容纳几何不对称。交点与目标
    位置允许1e-9*max(1,w/um) um误差；此为同网格反算，非连续解精度。
    可选关键字默认仍采用旧固定比例，主入口显式选择target_cd。
    """
    _validate_profile_inputs(x_um, intensity)
    if threshold_mode not in ("target_cd", "fixed_fraction"):
        raise ValueError("threshold_mode必须为target_cd或fixed_fraction。")
    if not math.isfinite(reference_intensity) or reference_intensity <= 0:
        raise ValueError("无效基准：reference_intensity必须有限且大于零。")
    if not math.isfinite(on_width_um) or on_width_um <= 0 or not math.isfinite(center_um):
        raise ValueError("目标设计宽度必须为有限正数，中心必须有限。")
    left, right = center_um - on_width_um / 2, center_um + on_width_um / 2
    intensity_tolerance = 1e-10 * reference_intensity
    cd_tolerance = 1e-9 * max(1.0, on_width_um)
    edge_values = (math.nan, math.nan)
    if threshold_mode == "target_cd":
        if (not x_um[0] <= left < right <= x_um[-1]
                or not search_bounds_um[0] < left < right < search_bounds_um[1]):
            raise ValueError("目标设计边缘越界：须位于数据内及固定搜索范围内部。")
        if np.any(intensity < 0) or np.max(intensity) <= 0:
            raise ValueError("无效基准：强度须非负且不为全零。")
        edge_values = tuple(float(v) for v in np.interp([left, right], x_um, intensity))
        if abs(edge_values[0] - edge_values[1]) > intensity_tolerance:
            raise ValueError(
                f"基准设计边缘强度不对称：left={edge_values[0]:.12g}, "
                f"right={edge_values[1]:.12g}, tolerance={intensity_tolerance:.3g}。")
        # 仅在对称检查通过后平均舍入差异；不优化非对称图形。
        threshold = (edge_values[0] + edge_values[1]) / 2
        if threshold <= 0:
            raise ValueError("目标边缘阈值非正，无法建立有效名义工作点。")
    else:
        if not 0 < threshold_fraction < 1:
            raise ValueError("threshold_fraction必须位于(0,1)。")
        threshold = threshold_fraction * reference_intensity
    interval, status = measure_threshold_interval_with_status(
        x_um, intensity, center_um, threshold_intensity=threshold,
        search_bounds_um=search_bounds_um,
    )
    if threshold_mode == "target_cd":
        if status != "VALID":
            raise ValueError(f"名义工作点没有有效目标区间：{status}。")
        if max(abs(interval.width_um - on_width_um), abs(interval.left_edge_um - left),
               abs(interval.right_edge_um - right)) > cd_tolerance:
            raise ValueError(
                "目标区间不连通或交点不在设计边缘："
                f"实际CD={interval.width_um:.12g} um，目标={on_width_um:g} um。")
    return {
        "threshold_mode": threshold_mode, "target_cd_um": on_width_um,
        "nominal_baseline_cd_um": interval.width_um,
        "nominal_baseline_status": status,
        "reference_intensity": reference_intensity,
        "nominal_threshold_intensity": threshold,
        "nominal_threshold_fraction": threshold / reference_intensity,
        "design_left_intensity": edge_values[0], "design_right_intensity": edge_values[1],
        "intensity_symmetry_tolerance": intensity_tolerance,
        "cd_tolerance_um": cd_tolerance,
    }


def evaluate_leakage_profile(
    x_um: FloatArray,
    intensity: FloatArray,
    *,
    on_width_um: float,
    pitch_um: float,
    center_um: float,
    reference_intensity: float,
    threshold_intensity: float,
    search_bounds_um: tuple[float, float],
    dark_bounds_um: tuple[float, float],
) -> dict[str, Any]:
    """复用原指标；固定阈值宽度与设计边缘NILS有不同的评价位置。"""
    if not np.isfinite(reference_intensity) or reference_intensity <= 0:
        raise ValueError("reference_intensity必须是正的有限基准强度。")
    metrics = calculate_periodic_profile_metrics(
        x_um, intensity, pitch_um, on_width_um, center_um,
        force_threshold_nan=True,
    )
    interval, status = measure_threshold_interval_with_status(
        x_um, intensity, center_um, threshold_intensity=threshold_intensity,
        search_bounds_um=search_bounds_um,
    )
    dark_left, dark_right = dark_bounds_um
    if not x_um[0] <= dark_left < dark_right <= x_um[-1]:
        raise ValueError("暗区范围必须位于计算坐标内。")
    dark_mask = (x_um >= dark_left) & (x_um <= dark_right)
    if not np.any(dark_mask):
        raise ValueError("指定暗区中没有采样点。")
    dark_max = float(np.max(intensity[dark_mask]))
    return {
        "reference_intensity": reference_intensity,
        "threshold_intensity": threshold_intensity,
        "fixed_left_edge_um": interval.left_edge_um,
        "fixed_right_edge_um": interval.right_edge_um,
        "fixed_cd_um": interval.width_um,
        "cd_error_to_design_um": interval.width_um - on_width_um,
        "left_error_to_design_um": interval.left_edge_um - (center_um - on_width_um / 2),
        "right_error_to_design_um": interval.right_edge_um - (center_um + on_width_um / 2),
        "design_edge_nils": metrics.nils,
        "nils_status": "VALID" if math.isfinite(metrics.nils) else "UNDEFINED_EDGE_INTENSITY",
        "on_center_intensity": metrics.space_center_intensity,
        "off_center_intensity": metrics.line_center_intensity,
        "center_contrast": metrics.contrast,
        "contrast_status": "VALID" if math.isfinite(metrics.contrast) else "UNDEFINED_ZERO_INTENSITY",
        "dark_max_intensity": dark_max,
        "dark_exceeds_threshold": bool(dark_max >= threshold_intensity),
        "dark_sample_left_um": float(x_um[dark_mask][0]),
        "dark_sample_right_um": float(x_um[dark_mask][-1]),
        "search_left_um": search_bounds_um[0],
        "search_right_um": search_bounds_um[1],
        "threshold_status": status,
    }


def plot_leakage_comparison(
    output_path: Path,
    rows: Sequence[dict[str, Any]],
    profiles: Sequence[FloatArray],
    x_um: FloatArray,
    baseline_profile: FloatArray,
    *,
    on_width_um: float,
    center_um: float,
    pitch_um: float,
    sweep_name: str,
    show: bool,
) -> None:
    """每类扫描只输出一张综合图；所有曲线共享同一个基准强度。"""
    figure = plt.figure(figsize=(11, 8))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.4, 1))
    profile_ax = figure.add_subplot(grid[0, :])
    edge_ax = figure.add_subplot(grid[1, 0])
    nils_ax = figure.add_subplot(grid[1, 1])
    ref = rows[0]["reference_intensity"]
    threshold = rows[0]["threshold_intensity"]
    view = np.abs(x_um - center_um) <= 0.8 * pitch_um
    profile_ax.plot(x_um[view], baseline_profile[view] / ref, "k--", lw=1.5,
                    label="Response baseline: eta=1, rho=0")
    labels = []
    for row, profile in zip(rows, profiles):
        label = (f"rho={row['off_to_on_intensity_ratio']:g}" if sweep_name == "ratio"
                 else f"phi/pi={row['off_relative_phase_rad'] / np.pi:g}")
        labels.append(label)
        profile_ax.plot(x_um[view], profile[view] / ref, lw=1.6, label=label)
    profile_ax.axhline(threshold / ref, color="0.25", ls=":", label="Nominal T0 (s=1)")
    for index, edge in enumerate((center_um - on_width_um / 2, center_um + on_width_um / 2)):
        profile_ax.axvline(edge, color="0.6", ls=":", lw=1,
                           label="Design edges" if index == 0 else None)
    representative = next((r for r in reversed(rows) if np.isfinite(r["relative_peak_cd_um"])), None)
    if representative is not None:
        profile_ax.text(
            0.99, 0.98,
            f"Representative {representative['case_name']}\n"
            f"Fixed-threshold CD = {representative['fixed_cd_um']:.4f} um\n"
            f"Own-peak CD = {representative['relative_peak_cd_um']:.4f} um",
            transform=profile_ax.transAxes, ha="right", va="top", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )
    profile_ax.set(xlabel="Exposure-plane x (um)", ylabel="Raw intensity / common I_ref")
    profile_ax.legend(loc="upper left", fontsize=8, ncol=2)
    positions = np.arange(len(rows))
    for key, label, marker in (
        ("delta_left_to_baseline_um", "Left edge", "o"),
        ("delta_right_to_baseline_um", "Right edge", "s"),
        ("delta_cd_to_baseline_um", "CD", "^"),
    ):
        edge_ax.plot(positions, [r[key] for r in rows], marker=marker, label=label)
    edge_ax.axhline(0, color="0.5", lw=0.8)
    edge_ax.set(ylabel="Change from response baseline (um)")
    edge_ax.legend(fontsize=8)
    nils_ax.plot(positions, [r["design_edge_nils"] for r in rows], "o-", color="C3")
    nils_ax.set(ylabel="NILS at design right edge")
    for axis in (edge_ax, nils_ax):
        axis.set_xticks(positions, labels, rotation=15)
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"DMD coherent leakage: {sweep_name} comparison\n"
        f"Mode: {rows[0].get('threshold_mode', 'fixed_fraction')}; s=1; "
        "demonstration parameters, not measured device data",
        fontsize=13,
    )
    _save_figure(figure, output_path, show)


def plot_dose_comparison(
    output_path: Path, rows: Sequence[dict[str, Any]], *, show: bool,
) -> None:
    """两种响应各一条CD曲线；原始强度不随相对曝光倍率重复生成。"""
    figure, axis = plt.subplots(figsize=(8, 5))
    for state, label in (("baseline", "Response baseline: eta=1, rho=0"),
                         ("representative", "Representative response")):
        selected = sorted((r for r in rows if r["dose_state"] == state),
                          key=lambda r: r["dose_scale"])
        axis.plot([r["dose_scale"] for r in selected],
                  [r["fixed_cd_um"] for r in selected], "o-", label=label)
    row = rows[0]
    axis.axhline(row["target_cd_um"], color="0.4", ls="--", label="Design target CD")
    axis.set(xlabel="Relative exposure s = E / E0", ylabel="Aerial-image threshold CD (um)")
    axis.grid(alpha=0.25)
    axis.legend()
    representative = next(r for r in rows if r["dose_state"] == "representative")
    figure.suptitle(
        f"CD vs relative exposure; mode: {row['threshold_mode']}\n"
        f"Fixed raw I; threshold T0/s; representative eta={representative['on_intensity_scale']:g}, "
        f"rho={representative['off_to_on_intensity_ratio']:g}, "
        f"phi/pi={representative['off_relative_phase_rad'] / np.pi:g}\n"
        "Demonstration parameters, not measured device or resist data", fontsize=11,
    )
    _save_figure(figure, output_path, show)


def run_leakage_experiments(
    output_directory: Path,
    show: bool,
    dmd_config: DMDConfig,
    optical_config: OpticalConfig2D,
    *,
    on_width_um: float,
    pitch_um: float,
    phase_offset_um: float,
    on_intensity_scale: float,
    off_to_on_intensity_ratio: float,
    off_relative_phase_rad: float,
    leakage_ratios: Sequence[float],
    leakage_phases_rad: Sequence[float],
    threshold_fraction: float,
    evaluation_bounds_um: tuple[float, float],
    search_bounds_um: tuple[float, float],
    dark_bounds_um: tuple[float, float],
    run_ratio_sweep: bool,
    run_phase_sweep: bool,
    threshold_mode: str = "fixed_fraction",
    run_dose_check: bool = False,
    dose_scales: Sequence[float] = (0.95, 1.0, 1.05),
) -> tuple[list[dict[str, Any]], dict[str, Any], list[Path], list[str]]:
    """建立一次工作点，按响应计算；曝光倍率只复用原始截面评价。"""
    if not (run_ratio_sweep or run_phase_sweep or run_dose_check):
        return [], {}, [], []
    if run_dose_check:
        if len(dose_scales) == 0:
            raise ValueError("已启用曝光量检查，但dose_scales为空。")
        for scale in dose_scales:
            if (isinstance(scale, (bool, np.bool_)) or not isinstance(scale, Real)
                    or not math.isfinite(scale) or scale <= 0):
                raise ValueError("dose_scale必须为有限正实数。")
    period_count = dmd_config.num_mirrors_x * dmd_config.projected_mirror_pitch_um / pitch_um
    mirrors_per_period = pitch_um / dmd_config.projected_mirror_pitch_um
    if not (np.isclose(period_count, round(period_count)) and
            np.isclose(mirrors_per_period, round(mirrors_per_period))):
        raise ValueError("本漏光周期实验需整数周期窗口及整数像元/周期，避免周期拼接误差。")
    specifications = []
    if run_ratio_sweep:
        if len(leakage_ratios) == 0:
            raise ValueError("已启用比例扫描，但列表为空。")
        specifications.append(("ratio", [(on_intensity_scale, rho, off_relative_phase_rad)
                                        for rho in leakage_ratios]))
    if run_phase_sweep:
        if len(leakage_phases_rad) == 0:
            raise ValueError("已启用相位对照，但列表为空。")
        specifications.append(("phase", [(on_intensity_scale, off_to_on_intensity_ratio, phi)
                                        for phi in leakage_phases_rad]))
    response_setting = (on_intensity_scale, off_to_on_intensity_ratio, off_relative_phase_rad)
    # 延续原扫描的代表场选择；只开曝光量时选集中配置的代表响应。
    representative_setting = (specifications[0][1][-1 if specifications[0][0] == "ratio" else 0]
                              if specifications else response_setting)
    states = make_periodic_line_space_states(
        dmd_config, pitch_um, on_width_um, phase_offset_um=phase_offset_um,
    )

    def calculate_case(eta: float, rho: float, phi: float) -> tuple[DMDConfig, CoherentImage2DResult]:
        config = with_state_response(
            dmd_config, on_intensity_scale=eta,
            off_to_on_intensity_ratio=rho, off_relative_phase_rad=phi,
        )
        pattern = render_dmd_field(states, config)
        result = calculate_coherent_aerial_image_2d(
            pattern.object_field, pattern.x_um, pattern.y_um, optical_config,
        )
        return config, result

    baseline_setting = (1.0, 0.0, 0.0)
    baseline_config, baseline = calculate_case(*baseline_setting)
    x_um = baseline.x_um.copy()
    baseline_profile = _extract_center_horizontal_profile(baseline).copy()
    eval_left, eval_right = evaluation_bounds_um
    if not x_um[0] <= eval_left < eval_right <= x_um[-1]:
        raise ValueError("评价范围必须完整位于计算坐标内。")
    evaluation_mask = (x_um >= eval_left) & (x_um <= eval_right)
    if np.count_nonzero(evaluation_mask) < 3:
        raise ValueError("基准评价范围至少包含三个采样点。")
    reference_intensity = float(np.max(baseline_profile[evaluation_mask]))
    workpoint = establish_nominal_workpoint(
        x_um, baseline_profile, on_width_um=on_width_um, center_um=phase_offset_um,
        reference_intensity=reference_intensity, search_bounds_um=search_bounds_um,
        threshold_mode=threshold_mode, threshold_fraction=threshold_fraction,
    )
    nominal_threshold = workpoint["nominal_threshold_intensity"]
    evaluation_keywords = dict(
        on_width_um=on_width_um, pitch_um=pitch_um, center_um=phase_offset_um,
        reference_intensity=reference_intensity,
        search_bounds_um=search_bounds_um, dark_bounds_um=dark_bounds_um,
    )

    def evaluate(profile: FloatArray, scale: float) -> dict[str, Any]:
        return evaluate_leakage_profile(
            x_um, profile, **evaluation_keywords,
            threshold_intensity=effective_threshold_for_dose(nominal_threshold, scale),
        )

    baseline_metrics = evaluate(baseline_profile, 1.0)
    data: dict[str, Any] = {
        "x_um": x_um, "baseline_raw_profile": baseline_profile,
        "baseline_image_field_center_row": baseline.image_field[np.argmin(abs(baseline.y_um)), :].copy(),
        **workpoint,
        # 兼容旧NPZ键：此处始终是名义T0，有效阈值逐行见CSV。
        "threshold_intensity": nominal_threshold,
    }
    if representative_setting == baseline_setting:
        data.update(representative_image_field=baseline.image_field.copy(),
                    representative_y_um=baseline.y_um.copy(), representative_case_name="baseline")
    del baseline
    # 仅缓存小截面与响应配置；不积存每个案例的二维FFT中间结果。
    cache = {baseline_setting: (baseline_config, baseline_profile, "baseline_raw_profile")}

    def profile_for(setting: tuple[float, float, float], case_name: str):
        # 命中缓存前仍校验响应，避免True与1等字典键相等而绕过原参数校验。
        with_state_response(dmd_config, on_intensity_scale=setting[0],
                            off_to_on_intensity_ratio=setting[1], off_relative_phase_rad=setting[2])
        if setting not in cache:
            config, result = calculate_case(*setting)
            profile = _extract_center_horizontal_profile(result).copy()
            key = f"{case_name}_raw_profile"
            cache[setting] = (config, profile, key)
            data[key] = profile
            if setting == representative_setting and "representative_image_field" not in data:
                data.update(representative_image_field=result.image_field.copy(),
                            representative_y_um=result.y_um.copy(), representative_case_name=case_name)
            del result
        return cache[setting]

    def make_row(setting, config, profile, *, experiment, case_name, scale=1.0,
                 dose_state="", same_dose_baseline=None, relative_peak=False):
        metrics = evaluate(profile, scale)
        same_baseline = baseline_metrics if same_dose_baseline is None else same_dose_baseline
        eta, rho, phi = setting
        row = {
            "reference_group": "periodic_lines", "experiment": experiment,
            "case_name": case_name, "dose_state": dose_state, "dose_scale": float(scale),
            "on_intensity_scale": eta, "off_to_on_intensity_ratio": rho,
            "off_relative_phase_rad": phi,
            "on_amplitude_real": config.on_amplitude.real,
            "on_amplitude_imag": config.on_amplitude.imag,
            "off_amplitude_real": config.off_amplitude.real,
            "off_amplitude_imag": config.off_amplitude.imag,
            "evaluation_sample_left_um": float(x_um[evaluation_mask][0]),
            "evaluation_sample_right_um": float(x_um[evaluation_mask][-1]),
            **workpoint, **metrics,
        }
        for metric, delta in (("fixed_left_edge_um", "delta_left_to_baseline_um"),
                              ("fixed_right_edge_um", "delta_right_to_baseline_um"),
                              ("fixed_cd_um", "delta_cd_to_baseline_um")):
            row[delta] = metrics[metric] - baseline_metrics[metric]
        # 旧delta列保留，明确其参考为baseline,s=1；同剂量增量另列。
        row["delta_cd_to_nominal_baseline_um"] = row["delta_cd_to_baseline_um"]
        row["same_dose_baseline_cd_um"] = same_baseline["fixed_cd_um"]
        row["same_dose_baseline_status"] = same_baseline["threshold_status"]
        row["delta_cd_to_same_dose_baseline_um"] = metrics["fixed_cd_um"] - same_baseline["fixed_cd_um"]
        row["relative_peak_cd_um"] = (
            measure_threshold_interval(x_um, profile, phase_offset_um,
                                       threshold_fraction).width_um
            if relative_peak else math.nan
        )
        return row

    rows: list[dict[str, Any]] = []
    outputs: list[Path] = []
    for group, settings in specifications:
        group_rows, group_profiles = [], []
        representative_index = (len(settings) - 1 if group == "ratio" else 0)
        for index, setting in enumerate(settings):
            name = f"{group}_{index:02d}"
            config, profile, _ = profile_for(setting, name)
            row = make_row(setting, config, profile, experiment=group, case_name=name,
                           relative_peak=index == representative_index)
            data[f"{name}_raw_profile"] = profile  # 保留原扫描NPZ键。
            group_rows.append(row)
            group_profiles.append(profile)
        rows.extend(group_rows)
        path = output_directory / f"figure_leakage_{group}.png"
        plot_leakage_comparison(
            path, group_rows, group_profiles, x_um, baseline_profile,
            on_width_um=on_width_um, center_um=phase_offset_um, pitch_um=pitch_um,
            sweep_name=group, show=show,
        )
        outputs.append(path)

    if run_dose_check:
        # 两种响应各计算一次；每个s只换阈值。开启其他扫描时复用已有截面。
        response_config, response_profile, response_key = profile_for(response_setting, "dose_representative")
        data.update(dose_scales=np.asarray(dose_scales, dtype=float),
                    dose_effective_thresholds=np.asarray([
                        effective_threshold_for_dose(nominal_threshold, s) for s in dose_scales]),
                    dose_baseline_profile_key="baseline_raw_profile",
                    dose_representative_profile_key=response_key,
                    dose_representative_response=np.asarray(response_setting, dtype=float))
        baseline_by_scale = {float(s): evaluate(baseline_profile, s) for s in dose_scales}
        dose_rows = []
        for state, setting, config, profile in (
            ("baseline", baseline_setting, baseline_config, baseline_profile),
            ("representative", response_setting, response_config, response_profile),
        ):
            for index, scale in enumerate(dose_scales):
                dose_rows.append(make_row(
                    setting, config, profile, experiment="dose", case_name=f"dose_{state}_{index:02d}",
                    scale=scale, dose_state=state, same_dose_baseline=baseline_by_scale[float(scale)],
                ))
        rows.extend(dose_rows)
        path = output_directory / "figure_leakage_dose.png"
        plot_dose_comparison(path, dose_rows, show=show)
        outputs.append(path)

    lines = [
        "八、ON/OFF等效响应与名义曝光工作点（演示参数，非实测设备参数）",
        f"- 模式={threshold_mode}；目标CD={on_width_um:.12g} um；"
        f"实际名义基准CD={workpoint['nominal_baseline_cd_um']:.12g} um。",
        f"- 共同基准eta=1,rho=0,s=1；I_ref={reference_intensity:.12g}；"
        f"T0={nominal_threshold:.12g}；T0/I_ref={workpoint['nominal_threshold_fraction']:.12g}。",
        f"- 设计边缘线性插值强度L/R=({workpoint['design_left_intensity']:.12g},"
        f"{workpoint['design_right_intensity']:.12g})；对称容差={workpoint['intensity_symmetry_tolerance']:.3g}；"
        f"CD/边缘位置容差={workpoint['cd_tolerance_um']:.3g} um（target_cd模式使用）。",
        f"- 基准评价Omega：中央横截面x在{evaluation_bounds_um} um，"
        f"实际采样范围[{x_um[evaluation_mask][0]:.6g},{x_um[evaluation_mask][-1]:.6g}] um。",
        f"- 基准设计右边缘NILS={baseline_metrics['design_edge_nils']:.9g}。",
        "- 名义工作点只设定一次；不是实验标定，也不证明设备能打印目标CD。",
        "- 原始I和I_ref不随s改变；材料等效剂量阈值固定，T(s)=T0/s，不同时乘s。",
        "- delta_cd_to_nominal_baseline及旧delta_*_to_baseline以baseline,s=1为参考；"
        "delta_cd_to_same_dose_baseline以baseline,s为参考；error_to_design以设计为参考。",
        "- 暗区使用当前有效阈值；只检查CSV的固定中部区间，不代表二维无缺陷。",
    ]
    for row in rows:
        lines.append(
            f"- {row['case_name']}: eta={row['on_intensity_scale']:g}, "
            f"rho={row['off_to_on_intensity_ratio']:g}, phi={row['off_relative_phase_rad']:.6g}, "
            f"s={row['dose_scale']:g}, T(s)={row['threshold_intensity']:.10g}; "
            f"CD={row['fixed_cd_um']:.10g} um; "
            f"delta_nominal={row['delta_cd_to_nominal_baseline_um']:.9g} um; "
            f"delta_same_dose={row['delta_cd_to_same_dose_baseline_um']:.9g} um; "
            f"same_dose_baseline_status={row['same_dose_baseline_status']}; "
            f"NILS={row['design_edge_nils']:.8g}; dark_max={row['dark_max_intensity']:.8g}; "
            f"dark_exceeds_threshold={row['dark_exceeds_threshold']}; {row['threshold_status']}"
        )
    lines.extend([
        "- 先叠加复场再求强度，交叉项可正可负；相位对照属于敏感性分析，"
        "不代表实际DMD可以独立控制漏光相位。",
        "- eta是当前通道内等效强度缩放，不是器件总效率，也不是s；"
        "rho不是混合图形暗亮中心比。",
        "- 双态单微镜空间响应近似成比例、均匀静态、完全相干；"
        "正常衍射暗区、OFF残余复场、其他路径杂散光必须区分。",
        "- CD仍为未标定的空中像阈值宽度；尺寸余量须结合NILS、暗区与曝光扰动判断。",
        "- 后续在固定材料与工艺下用CD—剂量实验校准，并用未参与校准的图形验证。"
        "本轮不实现绝对剂量、胶反应、显影、拟合或自动标定。",
    ])
    return rows, data, outputs, lines


def run_defocus_sweep(
    output_directory: Path,
    show: bool,
    dmd_config: DMDConfig,
    optical_config: OpticalConfig2D,
    *,
    on_width_um: float,
    pitch_um: float,
    phase_offset_um: float,
    on_intensity_scale: float,
    off_to_on_intensity_ratio: float,
    off_relative_phase_rad: float,
    defocus_values_um: Sequence[float],
    threshold_fraction: float,
    evaluation_bounds_um: tuple[float, float],
    search_bounds_um: tuple[float, float],
    dark_bounds_um: tuple[float, float],
    threshold_mode: str = "target_cd",
) -> tuple[list[dict[str, Any]], dict[str, Any], list[Path], list[str]]:
    """固定无漏光、零离焦名义参考，扫描观察面位置；始终使用同一传播API。

    输入响应在全部焦位固定，s=1。复场和强度取同一条实际采样行，
    不逐图归一化，也不逐焦位重新匹配CD。只保留全部一维截面和一个
    代表焦位的二维数据；CSV/NPZ由主入口保存。
    """
    if len(defocus_values_um) == 0:
        raise ValueError("已启用离焦实验，但defocus_values_um为空。")
    for value in defocus_values_um:
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(value)):
            raise ValueError("defocus_values_um必须仅包含有限实数，不能为布尔值或复数。")
    z_values = np.asarray(defocus_values_um, dtype=np.float64)
    test_config = with_state_response(
        dmd_config, on_intensity_scale=on_intensity_scale,
        off_to_on_intensity_ratio=off_to_on_intensity_ratio,
        off_relative_phase_rad=off_relative_phase_rad,
    )
    reference_config = with_state_response(
        dmd_config, on_intensity_scale=1.0,
        off_to_on_intensity_ratio=0.0, off_relative_phase_rad=0.0,
    )
    states = make_periodic_line_space_states(
        dmd_config, pitch_um, on_width_um, phase_offset_um=phase_offset_um,
    )
    period_count = dmd_config.num_mirrors_x * dmd_config.projected_mirror_pitch_um / pitch_um
    mirrors_per_period = pitch_um / dmd_config.projected_mirror_pitch_um
    if not (np.isclose(period_count, round(period_count), rtol=0.0, atol=1e-10) and
            np.isclose(mirrors_per_period, round(mirrors_per_period), rtol=0.0, atol=1e-10)):
        raise ValueError("本离焦周期实验需整数周期窗口及整数像元/周期（绝对容差1e-10）。")
    reference_pattern = render_dmd_field(states, reference_config)
    test_pattern = render_dmd_field(states, test_config)
    x_um, y_um = reference_pattern.x_um.copy(), reference_pattern.y_um.copy()
    nx, ny = x_um.size, y_um.size
    dx, dy = float(x_um[1] - x_um[0]), float(y_um[1] - y_um[0])
    fx, fy = make_frequency_axes(nx, ny, dx, dy)
    cutoff = optical_config.frequency_cutoff_cyc_per_um
    if not (fx[0] <= -cutoff <= cutoff <= fx[-1] and
            fy[0] <= -cutoff <= cutoff <= fy[-1]):
        raise ValueError("离焦实验网格未覆盖圆瞳通带。")
    if not (0.5 / dx > 2.0 * cutoff and 0.5 / dy > 2.0 * cutoff):
        raise ValueError("离焦实验不满足光强最高带宽2*NA/lambda的严格Nyquist条件。")
    design_edge = phase_offset_um + on_width_um / 2
    if nx < 7 or not x_um[0] <= design_edge <= x_um[-1]:
        raise ValueError("离焦实验设计边缘须在网格内且至少提供7个NILS拟合样点。")
    fit_indices = np.sort(np.argsort(np.abs(x_um - design_edge))[:7])
    grid = {
        "Nx": nx, "Ny": ny, "dx_um": dx, "dy_um": dy,
        "Lx_um": nx * dx, "Ly_um": ny * dy,
        "delta_fx_cyc_per_um": 1.0 / (nx * dx),
        "delta_fy_cyc_per_um": 1.0 / (ny * dy),
        "nominal_frequency_cutoff_cyc_per_um": cutoff,
        "intensity_bandlimit_cyc_per_um": 2.0 * cutoff,
        "nyquist_x_cyc_per_um": 0.5 / dx, "nyquist_y_cyc_per_um": 0.5 / dy,
        "intensity_shortest_period_samples_x": 1.0 / (2.0 * cutoff * dx),
        "intensity_shortest_period_samples_y": 1.0 / (2.0 * cutoff * dy),
        "nils_fit_samples": 7, "nils_polynomial_degree": 4,
        "nils_fit_sample_left_um": float(x_um[fit_indices[0]]),
        "nils_fit_sample_right_um": float(x_um[fit_indices[-1]]),
    }
    zero_pupil = make_defocused_circular_pupil(fx, fy, optical_config, 0.0)
    baseline = calculate_coherent_aerial_image_2d(
        reference_pattern.object_field, x_um, y_um, optical_config, pupil=zero_pupil,
    )
    row_index = int(np.argmin(np.abs(y_um)))
    baseline_profile = baseline.raw_intensity[row_index].copy()
    baseline_field_row = baseline.image_field[row_index].copy()
    eval_left, eval_right = evaluation_bounds_um
    if not x_um[0] <= eval_left < eval_right <= x_um[-1]:
        raise ValueError("离焦评价范围必须完整位于计算坐标内。")
    evaluation_mask = (x_um >= eval_left) & (x_um <= eval_right)
    if np.count_nonzero(evaluation_mask) < 3:
        raise ValueError("离焦基准评价范围至少包含三个采样点。")
    reference_intensity = float(np.max(baseline_profile[evaluation_mask]))
    workpoint = establish_nominal_workpoint(
        x_um, baseline_profile, on_width_um=on_width_um, center_um=phase_offset_um,
        reference_intensity=reference_intensity, search_bounds_um=search_bounds_um,
        threshold_mode=threshold_mode, threshold_fraction=threshold_fraction,
    )
    threshold = workpoint["nominal_threshold_intensity"]
    evaluation_keywords = dict(
        on_width_um=on_width_um, pitch_um=pitch_um, center_um=phase_offset_um,
        reference_intensity=reference_intensity, threshold_intensity=threshold,
        search_bounds_um=search_bounds_um, dark_bounds_um=dark_bounds_um,
    )
    baseline_metrics = evaluate_leakage_profile(x_um, baseline_profile, **evaluation_keywords)
    data: dict[str, Any] = {
        "x_um": x_um, "y_um": y_um, "defocus_values_um": z_values.copy(),
        "baseline_raw_profile": baseline_profile,
        "baseline_image_field_center_row": baseline_field_row,
        "baseline_input_energy": baseline.input_energy,
        "baseline_output_energy": baseline.output_energy,
        "profile_row_index": row_index, "profile_y_um": float(y_um[row_index]),
        "frequency_x_cyc_per_um": fx, "frequency_y_cyc_per_um": fy,
        "threshold_intensity": threshold, **workpoint, **grid,
        "reference_definition": "eta=1,rho=0,phi=0,defocus_um=0,s=1; same geometry and aperture",
    }
    del baseline, reference_pattern, zero_pupil
    # 最大|z|为代表；正负并列时选正值，使选择不随扫描顺序变化。
    representative_z = max(z_values, key=lambda z: (abs(z), z))
    rows: list[dict[str, Any]] = []
    profiles, field_rows = [], []
    for index, z_um in enumerate(z_values):
        pupil = make_defocused_circular_pupil(fx, fy, optical_config, float(z_um))
        result = calculate_coherent_aerial_image_2d(
            test_pattern.object_field, x_um, y_um, optical_config, pupil=pupil,
        )
        profile = result.raw_intensity[row_index].copy()
        profiles.append(profile)
        field_rows.append(result.image_field[row_index].copy())
        metrics = evaluate_leakage_profile(x_um, profile, **evaluation_keywords)
        row = {
            "case_name": f"defocus_{index:02d}", "defocus_um": float(z_um),
            "reference_definition": data["reference_definition"], "dose_scale": 1.0,
            "wavelength_um": optical_config.wavelength_um,
            "numerical_aperture": optical_config.numerical_aperture,
            "num_mirrors_x": dmd_config.num_mirrors_x,
            "num_mirrors_y": dmd_config.num_mirrors_y,
            "dmd_mirror_pitch_um": dmd_config.dmd_mirror_pitch_um,
            "projection_magnification": dmd_config.projection_magnification,
            "projected_mirror_pitch_um": dmd_config.projected_mirror_pitch_um,
            "samples_per_mirror": dmd_config.samples_per_mirror,
            "active_side_ratio": dmd_config.active_side_ratio,
            "on_width_um": on_width_um, "pitch_um": pitch_um,
            "phase_offset_um": phase_offset_um,
            "on_intensity_scale": float(on_intensity_scale),
            "off_to_on_intensity_ratio": float(off_to_on_intensity_ratio),
            "off_relative_phase_rad": float(off_relative_phase_rad),
            "on_amplitude_real": test_config.on_amplitude.real,
            "on_amplitude_imag": test_config.on_amplitude.imag,
            "off_amplitude_real": test_config.off_amplitude.real,
            "off_amplitude_imag": test_config.off_amplitude.imag,
            "profile_row_index": row_index, "profile_y_um": float(y_um[row_index]),
            "evaluation_left_um": eval_left, "evaluation_right_um": eval_right,
            "evaluation_sample_left_um": float(x_um[evaluation_mask][0]),
            "evaluation_sample_right_um": float(x_um[evaluation_mask][-1]),
            "dark_left_um": dark_bounds_um[0], "dark_right_um": dark_bounds_um[1],
            "input_energy": result.input_energy, "output_energy": result.output_energy,
            "energy_transmission": result.energy_transmission,
            **grid, **workpoint, **metrics,
            "center_shift_to_design_um": (
                (metrics["fixed_left_edge_um"] + metrics["fixed_right_edge_um"]) / 2
                - phase_offset_um
            ),
        }
        for metric, delta in (
            ("fixed_left_edge_um", "delta_left_to_nominal_baseline_um"),
            ("fixed_right_edge_um", "delta_right_to_nominal_baseline_um"),
            ("fixed_cd_um", "delta_cd_to_nominal_baseline_um"),
            ("design_edge_nils", "delta_nils_to_nominal_baseline"),
        ):
            row[delta] = metrics[metric] - baseline_metrics[metric]
        rows.append(row)
        if z_um == representative_z:
            data.update(
                representative_defocus_um=float(z_um),
                representative_image_field=result.image_field.copy(),
                representative_raw_intensity=result.raw_intensity.copy(),
                representative_pupil=result.pupil.copy(),
            )
        del result, pupil
    data.update(raw_profiles=np.asarray(profiles), image_field_center_rows=np.asarray(field_rows),
                threshold_statuses=np.asarray([row["threshold_status"] for row in rows]))

    figure = plt.figure(figsize=(11, 8))
    layout = figure.add_gridspec(2, 2, height_ratios=(1.5, 1))
    profile_axis = figure.add_subplot(layout[0, :])
    cd_axis, nils_axis = figure.add_subplot(layout[1, 0]), figure.add_subplot(layout[1, 1])
    view = np.abs(x_um - phase_offset_um) <= 0.8 * pitch_um
    profile_axis.plot(x_um[view], baseline_profile[view] / reference_intensity, "k--",
                      label="Nominal reference: no leakage, z=0")
    for row, profile in zip(rows, profiles):
        profile_axis.plot(x_um[view], profile[view] / reference_intensity,
                          label=f"z={row['defocus_um']:g} um ({row['threshold_status']})")
    profile_axis.axhline(threshold / reference_intensity, color="0.3", ls=":", label="Fixed T0 / I_ref")
    for edge in (phase_offset_um - on_width_um / 2, design_edge):
        profile_axis.axvline(edge, color="0.7", ls=":", lw=0.8)
    profile_axis.set(xlabel="Exposure-plane x (um)", ylabel="Raw intensity / common I_ref")
    profile_axis.legend(fontsize=8, ncol=2)
    sorted_rows = sorted(rows, key=lambda row: row["defocus_um"])
    for axis, key, ylabel in (
        (cd_axis, "fixed_cd_um", "Fixed-threshold CD (um)"),
        (nils_axis, "design_edge_nils", "NILS at design right edge"),
    ):
        axis.plot([row["defocus_um"] for row in sorted_rows],
                  [row[key] for row in sorted_rows], "o-")
        axis.set(xlabel="Observation-plane defocus (um)", ylabel=ylabel)
        axis.grid(alpha=0.25)
    cd_axis.axhline(on_width_um, color="0.5", ls="--", lw=0.8)
    figure.suptitle(
        f"Defocus with fixed reference; {threshold_mode}; s=1\n"
        f"eta={on_intensity_scale:g}, rho={off_to_on_intensity_ratio:g}, "
        f"phi/pi={off_relative_phase_rad / np.pi:g}; scalar air model, reference NA"
    )
    path = output_directory / "figure_defocus_comparison.png"
    _save_figure(figure, path, show)
    outputs = [path]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    pupil = data["representative_pupil"]
    extent = _image_extent(fx, fy)
    phase = np.ma.masked_where(np.abs(pupil) == 0, np.angle(pupil))
    for axis, values, title, cmap, vmin, vmax in (
        (axes[0], np.abs(pupil), "Pupil modulus", "gray", 0.0, 1.0),
        (axes[1], phase, "Pupil phase (rad); outside masked", "twilight", -np.pi, np.pi),
    ):
        artist = axis.imshow(values, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set(xlim=(-1.15 * cutoff, 1.15 * cutoff), ylim=(-1.15 * cutoff, 1.15 * cutoff),
                 xlabel="fx (cycles/um)", ylabel="fy (cycles/um)", title=title)
        figure.colorbar(artist, ax=axis, shrink=0.8)
    figure.suptitle(f"Representative defocus z={representative_z:g} um")
    path = output_directory / "figure_defocus_pupil.png"
    _save_figure(figure, path, show)
    outputs.append(path)
    notes = [
        "离焦实验：固定镜头，沿光传播方向的观察面位移为正；时间约定exp(-i omega t)。",
        "- 模型边界：空气n=1、单色、静态、标量、完全相干、局部空间不变；"
        "角谱相位去除公共exp(i k z)，未模拟移动镜组/光楔引起的倍率、像差或像移。",
        "- NA=0.065在默认LDI配置中仅为参考值，不是河南百合镜头实测或标定数据。",
        f"- 固定响应eta={on_intensity_scale:g},rho={off_to_on_intensity_ratio:g},"
        f"phi={off_relative_phase_rad:.12g} rad；所有焦位s=1。",
        f"- 名义参考eta=1,rho=0,phi=0,z=0,s=1；模式={threshold_mode}；"
        f"I_ref={reference_intensity:.12g},T0={threshold:.12g}；仅建立一次。",
        f"- 横截面实际row={row_index}, y={y_um[row_index]:.12g} um；复场和强度取同一行。",
        f"- 网格Nx={nx},Ny={ny},dx={dx:.12g},dy={dy:.12g} um；"
        f"Lx=Nx*dx={nx * dx:.12g},Ly=Ny*dy={ny * dy:.12g} um；"
        f"delta_fx={1 / (nx * dx):.12g},delta_fy={1 / (ny * dy):.12g},fc={cutoff:.12g} cycles/um。",
        "- 运行前检查完整圆瞳通带及光强2fc的Nyquist条件；NILS沿用设计右边缘7点四次拟合，"
        "实际拟合范围见CSV。提高每微镜采样数细化开口积分与截面导数；"
        "增大窗口改善频率间隔，不等同于前者。本实验是严格周期窗口，不证明孤立图形窗口收敛。",
        "- delta_*_to_nominal_baseline始终相对无漏光、零离焦、s=1；若打开固定漏光，"
        "变化包含漏光与离焦共同影响。能量为完整网格sum|U|^2，不是标定光功率。",
        "- 阈值失效保留NaN和原因，中心对比度保留符号；等线空三束图可能CD近似不变而NILS变化。",
    ]
    notes.extend(
        f"- z={row['defocus_um']:g} um: CD={row['fixed_cd_um']:.10g} um; "
        f"NILS={row['design_edge_nils']:.10g}; contrast={row['center_contrast']:.10g}; "
        f"dark_exceeds_threshold={row['dark_exceeds_threshold']}; {row['threshold_status']}"
        for row in rows
    )
    return rows, data, outputs, notes


def run_cross_leakage_comparison(
    output_directory: Path,
    show: bool,
    dmd_config: DMDConfig,
    optical_config: OpticalConfig2D,
    *,
    on_intensity_scale: float,
    off_to_on_intensity_ratio: float,
    off_relative_phase_rad: float,
    threshold_fraction: float = 0.5,
    arm_width_um: float = 6.0,
    arm_length_um: float = 30.0,
    roi_half_width_um: float = 18.0,
) -> tuple[list[ValidationCheck], dict[str, np.ndarray], list[Path], list[str]]:
    """可选二维十字对照，并用加倍视场检查 FFT 周期副本的影响。

    固定 ROI、采样间距、微镜间距和有限开口；ON/OFF 两种响应均沿用
    原二维相干传播。只保留中央 ROI，避免积存完整二维中间数组。
    """

    if not 0.0 < threshold_fraction < 1.0:
        raise ValueError("threshold_fraction 必须位于 (0, 1)。")
    if not np.isfinite(roi_half_width_um) or roi_half_width_um <= 0.5 * arm_length_um:
        raise ValueError("十字 ROI 必须为有限正数且覆盖完整十字及其边缘。")
    # 十字独立使用至少 64×64 微镜的窗口。取偶数保证加倍前后微镜及
    # 细采样网格相位一致；不改变调用方的原配置，也不改变图形尺寸。
    min_count = max(64, int(math.ceil(4.0 * roi_half_width_um / dmd_config.projected_mirror_pitch_um)))
    nx = max(dmd_config.num_mirrors_x, min_count)
    ny = max(dmd_config.num_mirrors_y, min_count)
    cross_config = replace(dmd_config, num_mirrors_x=nx + nx % 2, num_mirrors_y=ny + ny % 2)

    def simulate_roi(config: DMDConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states = make_cross_states(config, arm_width_um=arm_width_um, arm_length_um=arm_length_um)
        pattern = render_dmd_field(states, config)
        result = calculate_coherent_aerial_image_2d(
            pattern.object_field, pattern.x_um, pattern.y_um, optical_config
        )
        ix = np.flatnonzero(np.abs(result.x_um) <= roi_half_width_um)
        iy = np.flatnonzero(np.abs(result.y_um) <= roi_half_width_um)
        if ix.size < 4 or iy.size < 4:
            raise ValueError("十字 ROI 采样不足。")
        # copy 保证返回值不持有完整视场的大数组。
        return (
            result.x_um[ix].copy(), result.y_um[iy].copy(),
            result.raw_intensity[np.ix_(iy, ix)].copy(),
            result.image_field[np.ix_(iy, ix)].copy(),
        )

    baseline_config = with_state_response(
        cross_config, on_intensity_scale=1.0,
        off_to_on_intensity_ratio=0.0, off_relative_phase_rad=0.0,
    )
    case_config = with_state_response(
        cross_config, on_intensity_scale=on_intensity_scale,
        off_to_on_intensity_ratio=off_to_on_intensity_ratio,
        off_relative_phase_rad=off_relative_phase_rad,
    )
    x_um, y_um, baseline_intensity, baseline_field = simulate_roi(baseline_config)
    case_x_um, case_y_um, case_intensity, case_field = simulate_roi(case_config)
    if not (np.array_equal(x_um, case_x_um) and np.array_equal(y_um, case_y_um)):
        raise RuntimeError("十字基准与响应案例网格不一致。")
    reference_intensity = float(np.max(baseline_intensity))
    if reference_intensity <= 0.0:
        raise ValueError("十字基准强度为零，不能建立共同强度参考。")
    threshold_intensity = threshold_fraction * reference_intensity
    center_row = int(np.argmin(np.abs(y_um)))
    baseline_profile = baseline_intensity[center_row, :]
    case_profile = case_intensity[center_row, :]
    checks: list[ValidationCheck] = []
    notes = [
        "二维十字漏光对照：演示参数，非实测设备参数；两图使用共同基准和固定阈值。",
        f"十字宽/长={arm_width_um:g}/{arm_length_um:g} um，ROI=[{-roi_half_width_um:g}, {roi_half_width_um:g}] um²。",
        f"独立十字窗口={cross_config.num_mirrors_x}×{cross_config.num_mirrors_y} 微镜，"
        f"投影间距={cross_config.projected_mirror_pitch_um:g} um，samples_per_mirror={cross_config.samples_per_mirror}，"
        f"开口边长比={cross_config.active_side_ratio:g}；FOV 检查将两个方向的微镜数均加倍。",
        f"十字响应 eta={on_intensity_scale:g}, rho={off_to_on_intensity_ratio:g}, phi={off_relative_phase_rad:g} rad；"
        f"I_ref={reference_intensity:.12g}, I_th={threshold_intensity:.12g}。",
    ]
    fov_errors = []
    for label, config, small_intensity in (
        ("baseline", baseline_config, baseline_intensity),
        ("response", case_config, case_intensity),
    ):
        larger_config = replace(config, num_mirrors_x=2 * config.num_mirrors_x, num_mirrors_y=2 * config.num_mirrors_y)
        big_x, big_y, big_intensity, big_field = simulate_roi(larger_config)
        coordinates_match = (
            big_x.shape == x_um.shape and big_y.shape == y_um.shape
            and np.allclose(big_x, x_um, rtol=0.0, atol=1e-12)
            and np.allclose(big_y, y_um, rtol=0.0, atol=1e-12)
        )
        if not coordinates_match:
            raise RuntimeError("加倍视场未保留中央 ROI 的物理坐标，不能直接比较。")
        difference = big_intensity - small_intensity
        roi_error = float(np.sqrt(np.mean(difference**2)) / reference_intensity)
        profile_error = float(np.sqrt(np.mean(difference[center_row, :]**2)) / reference_intensity)
        checks.extend([
            _make_check(f"cross_leakage_fov_{label}_roi", roi_error, 0.02, "<=",
                        "中央固定 ROI 的强度 RMSE / 原视场共同 I_ref；两个方向视场加倍，采样间距不变。"),
            _make_check(f"cross_leakage_fov_{label}_profile", profile_error, 0.02, "<=",
                        "中央横截面 RMSE / 原视场共同 I_ref；无逐图归一化、平移拟合或重缩放。"),
        ])
        notes.append(f"十字 {label} 视场检查：ROI NRMSE={roi_error:.6g}，横截面 NRMSE={profile_error:.6g}；限值均为 0.02。")
        fov_errors.append((roi_error, profile_error))
        del big_intensity, big_field, difference

    figure = plt.figure(figsize=(11, 8))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 0.8))
    map_axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    vmax = max(float(np.max(baseline_intensity)), float(np.max(case_intensity))) / reference_intensity
    for axis, values, title in zip(
        map_axes, (baseline_intensity, case_intensity),
        ("Response baseline: eta=1, rho=0", f"eta={on_intensity_scale:g}, rho={off_to_on_intensity_ratio:g}, phi={off_relative_phase_rad:.3g} rad"),
    ):
        plot_values = values / reference_intensity
        im = axis.imshow(plot_values, extent=_image_extent(x_um, y_um), origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
        if float(np.min(values)) < threshold_intensity < float(np.max(values)):
            axis.contour(x_um, y_um, values, levels=[threshold_intensity], colors="cyan", linewidths=0.8)
        axis.set(title=title, xlabel="x (um)", ylabel="y (um)")
        figure.colorbar(im, ax=axis, label="I / common I_ref", shrink=0.82)
    profile_axis = figure.add_subplot(grid[1, :])
    profile_axis.plot(x_um, baseline_profile / reference_intensity, label="Baseline")
    profile_axis.plot(x_um, case_profile / reference_intensity, label="Nonideal response")
    profile_axis.axhline(threshold_fraction, color="black", linestyle="--", label="Fixed threshold")
    for edge in (-0.5 * arm_length_um, 0.5 * arm_length_um):
        profile_axis.axvline(edge, color="gray", linestyle=":")
    profile_axis.set(xlabel="x (um)", ylabel="I / common I_ref", title=f"Central horizontal profile, y={y_um[center_row]:.4g} um; dotted lines: design ends", xlim=(-roi_half_width_um, roi_half_width_um))
    profile_axis.legend(loc="upper right")
    profile_axis.grid(alpha=0.25)
    figure.suptitle("Cross pattern: coherent ON/OFF response (demonstration, not measured device data)")
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "figure_leakage_cross.png"
    _save_figure(figure, output_path, show)
    arrays = {
        "cross_x_um": x_um, "cross_y_um": y_um,
        "cross_baseline_intensity": baseline_intensity,
        "cross_response_intensity": case_intensity,
        "cross_baseline_image_field": baseline_field,
        "cross_response_image_field": case_field,
        "cross_reference_intensity": np.asarray(reference_intensity),
        "cross_threshold_intensity": np.asarray(threshold_intensity),
        "cross_fov_nrmse_baseline_response_roi_profile": np.asarray(fov_errors),
        "cross_window_num_mirrors_xy": np.asarray([cross_config.num_mirrors_x, cross_config.num_mirrors_y]),
    }
    return checks, arrays, [output_path], notes


def main() -> None:
    """集中设置参数与开关；关闭实验时连计算、绘图和文件检查一起关闭。"""
    arguments = parse_arguments()

    # ---------- 1. 运行开关（原实验参数保留，按需打开） ----------
    run_textbook_validation = False
    run_na_experiment = False
    run_sampling_experiment = False
    run_original_cross = False
    run_convergence = False
    run_leakage_sweep = True
    run_phase_sweep = False
    run_dose_check = True
    run_cross_leakage = True

    # ---------- 2. 几何与光学：长度均为曝光面um，NA为像方 ----------
    textbook_config = TextbookExperimentConfig()
    sampling_config = SamplingExperimentConfig()
    cross_config = CrossExperimentConfig()
    dmd_config = DMDConfig(
        num_mirrors_x=64, num_mirrors_y=32, dmd_mirror_pitch_um=7.56,
        projection_magnification=1.5 / 7.56, samples_per_mirror=16,
        active_side_ratio=0.95,
    )
    optical_config = OpticalConfig2D(wavelength_um=0.405, numerical_aperture=0.065)  # 参考NA，非实测镜头参数。
    on_width_um = 6.0
    pitch_um = 12.0
    phase_offset_um = 0.0  # 几何中心位置；与下方光学相位phi不同。
    # 可选十字采用更宽窗口；当前16点/微镜，加倍视场验证需注意内存。
    cross_dmd_config = replace(dmd_config, num_mirrors_y=64, samples_per_mirror=16)

    # ---------- 3. 等效响应：演示参数，非实测设备参数 ----------
    on_intensity_scale = 1.0
    off_to_on_intensity_ratio = 0.01
    off_relative_phase_rad = 0.0
    leakage_ratios = (0.0, 1.0e-4, 1.0e-2, 5.0e-2)
    leakage_phases_rad = (0.0, 0.5 * np.pi, np.pi)

    # ---------- 4. 评价设置：所有比较案例共享eta=1,rho=0基准 ----------
    threshold_mode = "target_cd"  # 或fixed_fraction；目标宽度直接使用on_width_um。
    threshold_fraction = 0.5
    dose_scales = (0.95, 1.0, 1.05, 1.2)  # s=E/E0，相对曝光量，独立于eta。
    evaluation_bounds_um = (phase_offset_um - pitch_um / 2, phase_offset_um + pitch_um / 2)
    search_bounds_um = evaluation_bounds_um
    # 只检查右侧暗区中部，避开设计边缘过渡；结果记录实际采样边界。
    dark_center_um = phase_offset_um + pitch_um / 2
    dark_half_width_um = (pitch_um - on_width_um) / 4
    dark_bounds_um = (dark_center_um - dark_half_width_um, dark_center_um + dark_half_width_um)

    # ---------- 5. 独立离焦实验：不继承旧漏光演示的响应 ----------
    run_defocus_experiment = True
    defocus_values_um = (-50.0, -25.0, 0.0, 25.0, 50.0)
    defocus_on_intensity_scale = 1.0
    defocus_off_to_on_intensity_ratio = 0.0
    defocus_off_relative_phase_rad = 0.0

    # ---------- 6. 运行和输出 ----------
    output_directory = arguments.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    switches = {
        "textbook_validation": run_textbook_validation, "na_sweep": run_na_experiment,
        "sampling_ratio": run_sampling_experiment, "original_cross": run_original_cross,
        "legacy_convergence": run_convergence, "leakage_ratio": run_leakage_sweep,
        "leakage_phase": run_phase_sweep, "leakage_cross": run_cross_leakage,
        "dose_check": run_dose_check, "defocus": run_defocus_experiment,
    }
    # 每次只追踪本次写出的文件，不扫描目录推断旧文件属于当前运行。
    output_files: list[Path] = []
    checks = run_internal_validations()
    from test_leakage_validation import run_leakage_validations
    checks.extend(run_leakage_validations())
    if not all(check.passed for check in checks):
        write_validation_summary(output_directory / "validation_summary.txt", checks,
                                 textbook_config=textbook_config,
                                 sampling_config=sampling_config, cross_config=cross_config)
        raise RuntimeError("基础或漏光验证失败；详见validation_summary.txt。")

    base_pattern = base_result = base_metrics = cross_validation = convergence = None
    na_cases, sampling_cases = [], []
    cross_case = None
    if run_textbook_validation or run_na_experiment or run_convergence:
        base_pattern, base_result = _make_base_image(
            textbook_config.samples_per_mirror, textbook_config.numerical_aperture, textbook_config,
        )
    if run_textbook_validation:
        base_metrics = calculate_periodic_profile_metrics(
            base_result.x_um, _extract_center_horizontal_profile(base_result),
            textbook_config.pitch_um, textbook_config.on_width_um, textbook_config.phase_offset_um,
        )
        cross_validation = run_textbook_cross_validation(
            Path(__file__).resolve().parent, base_result, textbook_config,
        )
        print(f"[1D-to-2D] {cross_validation.status}: {cross_validation.reason}")
        demo_config, demo_pattern = _make_finite_aperture_demo_pattern(textbook_config)
        outputs = [output_directory / name for name in (
            "figure_01_dmd_state_and_aperture.png", "figure_02_spectrum_pupil_filtered_spectrum.png",
            "figure_03_aerial_image_and_profiles.png", "figure_04_1d_2d_cross_validation.png",
        )]
        plot_dmd_state_and_aperture(outputs[0], demo_pattern, demo_config, arguments.show)
        plot_fourier_imaging_chain(outputs[1], base_result, arguments.show, textbook_config)
        plot_aerial_image_and_profiles(outputs[2], base_result, base_metrics, arguments.show, textbook_config)
        plot_textbook_cross_validation(outputs[3], cross_validation, base_result, arguments.show, textbook_config)
        output_files.extend(outputs)
    if run_convergence:
        convergence = run_convergence_check(base_result, textbook_config)
    if run_na_experiment:
        na_cases = run_na_sweep(base_pattern, textbook_config)
        path = output_directory / "figure_05_na_sweep.png"
        plot_na_sweep(path, na_cases, arguments.show, textbook_config)
        output_files.append(path)
    if run_sampling_experiment:
        sampling_cases = run_sampling_ratio_experiment(sampling_config)
        path = output_directory / "figure_06_sampling_ratio_comparison.png"
        plot_sampling_ratio_comparison(path, sampling_cases, arguments.show, sampling_config)
        output_files.append(path)
    if run_original_cross:
        cross_case = run_cross_pattern_experiment(cross_config)
        path = output_directory / "figure_07_2d_cross_pattern.png"
        plot_cross_pattern(path, cross_case, arguments.show, cross_config)
        output_files.append(path)
    if any((run_textbook_validation, run_na_experiment, run_sampling_experiment,
            run_original_cross, run_convergence)):
        legacy_rows = build_metric_rows(
            base_pattern, base_result, base_metrics, cross_validation, convergence,
            na_cases, sampling_cases, cross_case, textbook_config, sampling_config, cross_config,
        )
        path = output_directory / "validation_metrics.csv"
        write_metrics_csv(path, legacy_rows)
        output_files.append(path)

    rows, representative_data, plots, leakage_notes = run_leakage_experiments(
        output_directory, arguments.show, dmd_config, optical_config,
        on_width_um=on_width_um, pitch_um=pitch_um, phase_offset_um=phase_offset_um,
        on_intensity_scale=on_intensity_scale,
        off_to_on_intensity_ratio=off_to_on_intensity_ratio,
        off_relative_phase_rad=off_relative_phase_rad,
        leakage_ratios=leakage_ratios, leakage_phases_rad=leakage_phases_rad,
        threshold_fraction=threshold_fraction, evaluation_bounds_um=evaluation_bounds_um,
        search_bounds_um=search_bounds_um, dark_bounds_um=dark_bounds_um,
        run_ratio_sweep=run_leakage_sweep, run_phase_sweep=run_phase_sweep,
        threshold_mode=threshold_mode, run_dose_check=run_dose_check, dose_scales=dose_scales,
    )
    output_files.extend(plots)
    if rows:
        path = output_directory / "leakage_metrics.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        output_files.append(path)
    if run_cross_leakage:
        cross_checks, cross_data, cross_plots, cross_notes = run_cross_leakage_comparison(
            output_directory, arguments.show, cross_dmd_config, optical_config,
            on_intensity_scale=on_intensity_scale, off_to_on_intensity_ratio=off_to_on_intensity_ratio,
            off_relative_phase_rad=off_relative_phase_rad, threshold_fraction=threshold_fraction,
        )
        checks.extend(cross_checks)
        representative_data.update(cross_data)
        output_files.extend(cross_plots)
        leakage_notes.extend(cross_notes)
    if representative_data:
        path = output_directory / "leakage_representative_data.npz"
        np.savez_compressed(path, **representative_data)
        output_files.append(path)

    defocus_notes: list[str] = []
    if run_defocus_experiment:
        defocus_rows, defocus_data, defocus_plots, defocus_notes = run_defocus_sweep(
            output_directory, arguments.show, dmd_config, optical_config,
            on_width_um=on_width_um, pitch_um=pitch_um, phase_offset_um=phase_offset_um,
            on_intensity_scale=defocus_on_intensity_scale,
            off_to_on_intensity_ratio=defocus_off_to_on_intensity_ratio,
            off_relative_phase_rad=defocus_off_relative_phase_rad,
            defocus_values_um=defocus_values_um,
            threshold_fraction=threshold_fraction, evaluation_bounds_um=evaluation_bounds_um,
            search_bounds_um=search_bounds_um, dark_bounds_um=dark_bounds_um,
            threshold_mode=threshold_mode,
        )
        path = output_directory / "defocus_metrics.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(defocus_rows[0]))
            writer.writeheader()
            writer.writerows(defocus_rows)
        output_files.append(path)
        path = output_directory / "defocus_representative_data.npz"
        np.savez_compressed(path, **defocus_data)
        output_files.append(path)
        output_files.extend(defocus_plots)

    summary_path = output_directory / "validation_summary.txt"
    write_validation_summary(summary_path, checks, cross_validation, convergence,
                             na_cases, sampling_cases, cross_case,
                             textbook_config, sampling_config, cross_config)
    output_files.append(summary_path)
    configuration = {
        "switches": switches,
        "legacy_textbook": asdict(textbook_config), "legacy_sampling": asdict(sampling_config),
        "legacy_cross": asdict(cross_config), "leakage_dmd": asdict(dmd_config),
        "leakage_cross_dmd": asdict(cross_dmd_config), "optical": asdict(optical_config),
        "on_width_um": on_width_um, "pitch_um": pitch_um, "phase_offset_um": phase_offset_um,
        "eta": on_intensity_scale, "rho": off_to_on_intensity_ratio, "phi_rad": off_relative_phase_rad,
        "leakage_ratios": leakage_ratios, "leakage_phases_rad": leakage_phases_rad,
        "threshold_fraction": threshold_fraction, "evaluation_bounds_um": evaluation_bounds_um,
        "threshold_mode": threshold_mode, "dose_scales": dose_scales,
        "search_bounds_um": search_bounds_um, "dark_bounds_um": dark_bounds_um,
        "defocus_values_um": defocus_values_um,
        "defocus_eta": defocus_on_intensity_scale,
        "defocus_rho": defocus_off_to_on_intensity_ratio,
        "defocus_phi_rad": defocus_off_relative_phase_rad,
    }
    def json_value(value: Any) -> Any:
        if isinstance(value, complex):
            return {"real": value.real, "imag": value.imag}
        raise TypeError(type(value).__name__)
    with summary_path.open("a", encoding="utf-8") as stream:
        stream.write("\n本次实验开关（旧文件不计入本次结果）\n")
        for name, enabled in switches.items():
            stream.write(f"- {name}: {'EXECUTED' if enabled else 'NOT RUN'}\n")
        stream.write("\n" + "\n".join(leakage_notes + defocus_notes) + "\n\n完整本次配置\n")
        stream.write(json.dumps(configuration, default=json_value, ensure_ascii=False, indent=2))
        stream.write("\n\n本次生成文件\n" + "\n".join(path.name for path in output_files) + "\n")
    missing = [path.name for path in output_files if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("本次文件未生成：" + ", ".join(missing))
    if not all(check.passed for check in checks):
        raise RuntimeError("验证未通过；详见validation_summary.txt。")
    print(f"[Checks] {sum(check.passed for check in checks)}/{len(checks)} PASS")
    print(f"All requested outputs were written to: {output_directory}")
    if arguments.show:
        plt.show()


if __name__ == "__main__":
    main()
