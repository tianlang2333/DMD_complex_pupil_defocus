"""
教材验证模型：一维周期线/空图形的相干空中像
=================================================

本程序复现以下教材内容：
1. 周期性线/空图形的傅里叶衍射级次系数；
2. 物镜接收不同最高衍射级次 N=1,3,5,7,9,11 时的相干空中像；
3. 中心明区/暗区强度、图像对比度与 NILS；
4. λ=248 nm、NA=0.85、pitch=1000 nm 时物镜实际接收的衍射级次。

理论依据：
- Chris Mack, Fundamental Principles of Optical Lithography,
  Section 2.2.5, Eqs. (2.62)–(2.68), and Section 3.8.3.
- Kevin Berwick, Optical Lithography Modelling with MATLAB,
  Chapter 2: Aerial Image calculation - Dense Lines and Spaces.

模型边界：
- 一维、无限周期二值线/空图形；
- 正入射、单色、完全相干、标量成像；
- 理想无像差物镜，仅按衍射级次是否进入光瞳进行截断；
- 不包含部分相干、离焦、光刻胶、显影、DMD 微镜结构及平台扫描。

运行示例：
    python textbook_dense_lines_aerial_image.py
    python textbook_dense_lines_aerial_image.py --show
    python textbook_dense_lines_aerial_image.py --output-dir my_results

依赖：numpy、matplotlib
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SimulationConfig:
    """教材验证模型的统一参数配置。

    坐标约定
    --------
    - x = 0：透明空间（clear space）的中心；
    - x = ±space_width/2：空间的设计边缘；
    - x = ±pitch/2：相邻不透明线（opaque line）的中心。

    术语说明
    --------
    教材公式中的 w 是透明空间宽度，p 是周期：
        p = space_width + line_width
    """

    space_width_nm: float = 500.0
    line_width_nm: float = 500.0
    refractive_index: float = 1.0

    wavelength_nm: float = 248.0
    numerical_aperture: float = 0.85

    compared_max_orders: tuple[int, ...] = (1, 3, 5, 7, 9, 11)
    coefficient_max_order: int = 11
    sample_count: int = 4001

    @property
    def pitch_nm(self) -> float:
        """线/空图形周期，单位 nm。"""
        return self.space_width_nm + self.line_width_nm

    @property
    def duty_cycle(self) -> float:
        """透明空间占空比 w/p。"""
        return self.space_width_nm / self.pitch_nm

    @property
    def nominal_edge_nm(self) -> float:
        """右侧设计边缘位置 x=w/2，单位 nm。"""
        return 0.5 * self.space_width_nm

    def validate(self) -> None:
        """检查参数是否满足模型和数值计算要求。"""
        if self.space_width_nm <= 0.0 or self.line_width_nm <= 0.0:
            raise ValueError("space_width_nm 和 line_width_nm 必须为正数。")
        if self.refractive_index <= 0.0:
            raise ValueError("refractive_index 必须为正数。")
        if not (0.0 < self.numerical_aperture <= self.refractive_index):
            raise ValueError(
                "numerical_aperture 必须满足 0 < NA <= refractive_index。"
            )
        if self.wavelength_nm <= 0.0:
            raise ValueError("wavelength_nm 必须为正数。")
        if self.sample_count < 101 or self.sample_count % 2 == 0:
            raise ValueError("sample_count 应为不小于 101 的奇数。")
        if self.coefficient_max_order < 1:
            raise ValueError("coefficient_max_order 必须不小于 1。")
        if not self.compared_max_orders:
            raise ValueError("compared_max_orders 不能为空。")
        if any(order < 0 for order in self.compared_max_orders):
            raise ValueError("compared_max_orders 中的级次必须为非负整数。")


@dataclass(frozen=True)
class AerialImageResult:
    """单个最高衍射级次条件下的空中像结果。"""

    max_diffraction_order: int
    x_nm: FloatArray
    electric_field: FloatArray
    electric_field_derivative_per_nm: FloatArray
    intensity: FloatArray
    intensity_derivative_per_nm: FloatArray


@dataclass(frozen=True)
class ImageMetrics:
    """教材关注的线/空图形质量指标。"""

    max_diffraction_order: int
    space_center_intensity: float
    line_center_intensity: float
    nominal_edge_intensity: float
    center_based_contrast: float
    nils_analytical: float
    nils_numerical: float


def make_position_grid(config: SimulationConfig) -> FloatArray:
    """建立一个周期内的空间坐标网格 [-p/2, p/2]。"""
    half_pitch_nm = 0.5 * config.pitch_nm
    return np.linspace(
        -half_pitch_nm,
        half_pitch_nm,
        config.sample_count,
        dtype=np.float64,
    )


def fourier_order_coefficient(
    order: int,
    space_width_nm: float,
    pitch_nm: float,
) -> float:
    r"""计算周期透明空间的第 ``order`` 个电场傅里叶系数。

    对以 x=0 为中心、宽度为 w 的透明空间，周期为 p：

        a_0 = w / p

        a_j = sin(pi*j*w/p) / (pi*j),  j != 0

    对 50% 占空比，偶数级次理论上严格为零，奇数级次正负交替。
    """
    if pitch_nm <= 0.0:
        raise ValueError("pitch_nm 必须为正数。")
    if not (0.0 < space_width_nm < pitch_nm):
        raise ValueError("space_width_nm 必须满足 0 < w < p。")

    if order == 0:
        return space_width_nm / pitch_nm

    order_float = float(order)
    return float(
        np.sin(np.pi * order_float * space_width_nm / pitch_nm)
        / (np.pi * order_float)
    )


def calculate_fourier_coefficients(
    max_order: int,
    config: SimulationConfig,
) -> tuple[NDArray[np.int64], FloatArray]:
    """返回从 -max_order 到 +max_order 的衍射级次及其系数。"""
    if max_order < 0:
        raise ValueError("max_order 必须为非负整数。")

    orders = np.arange(-max_order, max_order + 1, dtype=np.int64)
    coefficients = np.array(
        [
            fourier_order_coefficient(
                int(order),
                config.space_width_nm,
                config.pitch_nm,
            )
            for order in orders
        ],
        dtype=np.float64,
    )
    return orders, coefficients


def calculate_coherent_aerial_image(
    x_nm: FloatArray,
    max_diffraction_order: int,
    config: SimulationConfig,
) -> AerialImageResult:
    r"""按教材式 (2.63) 计算对称级次截断后的相干空中像。

    对正入射、中心对称图形，物镜同时接收 +j 和 -j 级：

        E(x) = a_0 + 2 * sum[a_j cos(2*pi*j*x/p)]

    在折射率 n 的介质中：

        I(x) = n * |E(x)|^2

    注意
    ----
    ``max_diffraction_order=1`` 表示接收 0、+1、-1 级，属于三束成像，
    不是“只接收一束光”。
    """
    if max_diffraction_order < 0:
        raise ValueError("max_diffraction_order 必须为非负整数。")

    pitch_nm = config.pitch_nm
    a0 = fourier_order_coefficient(0, config.space_width_nm, pitch_nm)

    electric_field = np.full_like(x_nm, a0, dtype=np.float64)
    electric_field_derivative = np.zeros_like(x_nm, dtype=np.float64)

    for order in range(1, max_diffraction_order + 1):
        coefficient = fourier_order_coefficient(
            order,
            config.space_width_nm,
            pitch_nm,
        )
        phase = 2.0 * np.pi * order * x_nm / pitch_nm

        electric_field += 2.0 * coefficient * np.cos(phase)
        electric_field_derivative += (
            -4.0
            * np.pi
            * order
            * coefficient
            / pitch_nm
            * np.sin(phase)
        )

    intensity = config.refractive_index * np.square(electric_field)
    intensity_derivative = (
        2.0
        * config.refractive_index
        * electric_field
        * electric_field_derivative
    )

    return AerialImageResult(
        max_diffraction_order=max_diffraction_order,
        x_nm=x_nm,
        electric_field=electric_field,
        electric_field_derivative_per_nm=electric_field_derivative,
        intensity=intensity,
        intensity_derivative_per_nm=intensity_derivative,
    )


def textbook_three_beam_intensity(
    x_nm: FloatArray,
    config: SimulationConfig,
) -> FloatArray:
    r"""教材 N=1（0、±1 级）三束成像闭式结果，用于程序自检。

    对 50% 占空比：

        a_0 = 1/2,  a_1 = 1/pi

        E(x) = 1/2 + (2/pi) cos(2*pi*x/p)
        I(x) = n * E(x)^2
    """
    if not np.isclose(config.duty_cycle, 0.5, atol=1e-12):
        raise ValueError("该闭式验证只适用于等线宽/等空间宽度。")

    field = 0.5 + (2.0 / np.pi) * np.cos(
        2.0 * np.pi * x_nm / config.pitch_nm
    )
    return config.refractive_index * np.square(field)


def interpolate_value(
    x_nm: FloatArray,
    y: FloatArray,
    query_x_nm: float,
) -> float:
    """在一维规则结果上做线性插值。"""
    if query_x_nm < x_nm[0] or query_x_nm > x_nm[-1]:
        raise ValueError("query_x_nm 超出当前空间网格范围。")
    return float(np.interp(query_x_nm, x_nm, y))


def calculate_image_metrics(
    result: AerialImageResult,
    config: SimulationConfig,
) -> ImageMetrics:
    r"""计算中心对比度和设计边缘处的 NILS。

    对比度采用教材题目中的中心点近似：

        C = (I_max - I_min) / (I_max + I_min)

    其中比较 x=0 的空间中心和 x=p/2 的线中心。

    NILS 在设计边缘 x=w/2 处计算：

        NILS = w * |(1/I) dI/dx|

    同时输出解析导数和数值梯度两种结果，便于检查代码正确性。
    """
    x_nm = result.x_nm
    intensity = result.intensity

    space_center_intensity = interpolate_value(x_nm, intensity, 0.0)
    line_center_intensity = interpolate_value(
        x_nm,
        intensity,
        0.5 * config.pitch_nm,
    )

    center_max = max(space_center_intensity, line_center_intensity)
    center_min = min(space_center_intensity, line_center_intensity)
    denominator = center_max + center_min
    if denominator <= 0.0:
        raise ZeroDivisionError("中心点强度之和为零，无法计算图像对比度。")
    contrast = (center_max - center_min) / denominator

    edge_nm = config.nominal_edge_nm
    edge_intensity = interpolate_value(x_nm, intensity, edge_nm)
    if edge_intensity <= 0.0:
        raise ZeroDivisionError("设计边缘强度不为正，无法计算 NILS。")

    analytical_edge_slope = interpolate_value(
        x_nm,
        result.intensity_derivative_per_nm,
        edge_nm,
    )
    nils_analytical = config.space_width_nm * abs(
        analytical_edge_slope / edge_intensity
    )

    numerical_derivative = np.gradient(
        intensity,
        x_nm,
        edge_order=2,
    )
    numerical_edge_slope = interpolate_value(
        x_nm,
        numerical_derivative,
        edge_nm,
    )
    nils_numerical = config.space_width_nm * abs(
        numerical_edge_slope / edge_intensity
    )

    return ImageMetrics(
        max_diffraction_order=result.max_diffraction_order,
        space_center_intensity=space_center_intensity,
        line_center_intensity=line_center_intensity,
        nominal_edge_intensity=edge_intensity,
        center_based_contrast=contrast,
        nils_analytical=nils_analytical,
        nils_numerical=nils_numerical,
    )


def captured_orders_on_axis(
    wavelength_nm: float,
    numerical_aperture: float,
    pitch_nm: float,
) -> NDArray[np.int64]:
    r"""计算正入射相干照明时进入理想物镜的衍射级次。

    第 j 级空间频率为 |j|/p，物镜截止频率为 NA/lambda，因此：

        |j|/p <= NA/lambda

        |j| <= p*NA/lambda
    """
    if wavelength_nm <= 0.0 or pitch_nm <= 0.0:
        raise ValueError("wavelength_nm 和 pitch_nm 必须为正数。")
    if numerical_aperture <= 0.0:
        raise ValueError("numerical_aperture 必须为正数。")

    normalized_cutoff = pitch_nm * numerical_aperture / wavelength_nm
    max_order = int(np.floor(normalized_cutoff + 1e-12))
    return np.arange(-max_order, max_order + 1, dtype=np.int64)


def nonzero_captured_orders(
    captured_orders: NDArray[np.int64],
    config: SimulationConfig,
    tolerance: float = 1e-12,
) -> NDArray[np.int64]:
    """从物镜接收级次中筛出振幅非零的级次。"""
    return np.array(
        [
            int(order)
            for order in captured_orders
            if abs(
                fourier_order_coefficient(
                    int(order),
                    config.space_width_nm,
                    config.pitch_nm,
                )
            )
            > tolerance
        ],
        dtype=np.int64,
    )


def run_internal_validation(
    config: SimulationConfig,
    results: dict[int, AerialImageResult],
    metrics: dict[int, ImageMetrics],
) -> dict[str, float]:
    """执行若干可解释的内部一致性检查。"""
    validation_values: dict[str, float] = {}

    if 1 not in results:
        raise RuntimeError("教材验证至少需要 max_diffraction_order=1。")

    # 1) 通用傅里叶级次求和应与教材三束成像闭式解一致。
    three_beam_closed_form = textbook_three_beam_intensity(
        results[1].x_nm,
        config,
    )
    three_beam_max_abs_error = float(
        np.max(np.abs(results[1].intensity - three_beam_closed_form))
    )
    validation_values["three_beam_max_abs_error"] = three_beam_max_abs_error
    if three_beam_max_abs_error > 1e-12:
        raise RuntimeError(
            "N=1 通用模型与教材三束成像闭式解不一致："
            f"max error={three_beam_max_abs_error:.3e}"
        )

    # 2) 对 50% 占空比，偶数级次应为零。
    if np.isclose(config.duty_cycle, 0.5, atol=1e-12):
        even_coefficients = [
            abs(
                fourier_order_coefficient(
                    order,
                    config.space_width_nm,
                    config.pitch_nm,
                )
            )
            for order in range(2, config.coefficient_max_order + 1, 2)
        ]
        max_even_coefficient = max(even_coefficients, default=0.0)
        validation_values["max_even_order_coefficient"] = max_even_coefficient
        if max_even_coefficient > 1e-12:
            raise RuntimeError(
                "50% 占空比下偶数级次未接近零："
                f"max coefficient={max_even_coefficient:.3e}"
            )

    # 3) NILS 的解析导数与数值导数应一致。
    max_nils_relative_difference = 0.0
    for order, item in metrics.items():
        reference = max(abs(item.nils_analytical), 1e-15)
        relative_difference = abs(
            item.nils_numerical - item.nils_analytical
        ) / reference
        max_nils_relative_difference = max(
            max_nils_relative_difference,
            relative_difference,
        )
        if relative_difference > 1e-3:
            raise RuntimeError(
                f"N={order} 的解析 NILS 与数值 NILS 差异过大："
                f"relative difference={relative_difference:.3e}"
            )
    validation_values[
        "max_nils_relative_difference"
    ] = max_nils_relative_difference

    return validation_values


def save_coefficients_csv(
    output_dir: Path,
    config: SimulationConfig,
) -> Path:
    """保存 -N 到 +N 的衍射级次系数。"""
    orders, coefficients = calculate_fourier_coefficients(
        config.coefficient_max_order,
        config,
    )
    output_path = output_dir / "fourier_order_coefficients.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["diffraction_order", "electric_field_coefficient"])
        for order, coefficient in zip(orders, coefficients, strict=True):
            writer.writerow([int(order), f"{coefficient:.16g}"])
    return output_path


def save_metrics_csv(
    output_dir: Path,
    metrics: Iterable[ImageMetrics],
) -> Path:
    """保存各最高衍射级次对应的图像指标。"""
    output_path = output_dir / "aerial_image_metrics.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "max_diffraction_order",
                "space_center_intensity",
                "line_center_intensity",
                "nominal_edge_intensity",
                "center_based_contrast",
                "nils_analytical",
                "nils_numerical",
            ]
        )
        for item in metrics:
            writer.writerow(
                [
                    item.max_diffraction_order,
                    f"{item.space_center_intensity:.16g}",
                    f"{item.line_center_intensity:.16g}",
                    f"{item.nominal_edge_intensity:.16g}",
                    f"{item.center_based_contrast:.16g}",
                    f"{item.nils_analytical:.16g}",
                    f"{item.nils_numerical:.16g}",
                ]
            )
    return output_path


def plot_fourier_coefficients(
    output_dir: Path,
    config: SimulationConfig,
) -> Path:
    """绘制教材中的离散衍射级次幅值。"""
    orders, coefficients = calculate_fourier_coefficients(
        config.coefficient_max_order,
        config,
    )

    figure, axis = plt.subplots(figsize=(9, 5))
    markerline, stemlines, baseline = axis.stem(
        orders,
        coefficients,
        basefmt="k-",
    )
    plt.setp(markerline, markersize=5)
    plt.setp(stemlines, linewidth=1.2)
    plt.setp(baseline, linewidth=0.8)

    axis.set_xlabel("Diffraction order j")
    axis.set_ylabel(r"Electric-field coefficient $a_j$")
    axis.set_title("Fourier coefficients of equal lines and spaces")
    axis.grid(True, alpha=0.3)
    axis.set_xticks(orders)
    figure.tight_layout()

    output_path = output_dir / "figure_01_fourier_coefficients.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_aerial_images_vs_order(
    output_dir: Path,
    config: SimulationConfig,
    results: dict[int, AerialImageResult],
) -> Path:
    """复现实验手册 N=1,3,5,7,9,11 的空中像比较图。"""
    figure, axis = plt.subplots(figsize=(10, 6))

    for order in config.compared_max_orders:
        axis.plot(
            results[order].x_nm,
            results[order].intensity,
            linewidth=1.5,
            label=f"N={order}",
        )

    axis.axvline(
        -config.nominal_edge_nm,
        linestyle="--",
        linewidth=1.0,
        label="Nominal edges",
    )
    axis.axvline(
        config.nominal_edge_nm,
        linestyle="--",
        linewidth=1.0,
    )
    axis.set_xlabel("Horizontal position x (nm)")
    axis.set_ylabel("Relative intensity")
    axis.set_title("Coherent aerial image versus captured diffraction order")
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2)
    figure.tight_layout()

    output_path = output_dir / "figure_02_aerial_image_vs_order.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_contrast_vs_order(
    output_dir: Path,
    metrics: dict[int, ImageMetrics],
) -> Path:
    """绘制中心点定义下的图像对比度。"""
    orders = np.array(sorted(metrics), dtype=np.int64)
    contrast = np.array(
        [metrics[int(order)].center_based_contrast for order in orders],
        dtype=np.float64,
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(orders, contrast, marker="o")
    axis.set_xlabel("Maximum captured diffraction order N")
    axis.set_ylabel("Center-based image contrast")
    axis.set_title("Image contrast versus captured diffraction order")
    axis.grid(True, alpha=0.3)
    axis.set_xticks(orders)
    figure.tight_layout()

    output_path = output_dir / "figure_03_contrast_vs_order.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_nils_vs_order(
    output_dir: Path,
    metrics: dict[int, ImageMetrics],
) -> Path:
    """绘制设计边缘处的 NILS，并比较解析/数值导数。"""
    orders = np.array(sorted(metrics), dtype=np.int64)
    nils_analytical = np.array(
        [metrics[int(order)].nils_analytical for order in orders],
        dtype=np.float64,
    )
    nils_numerical = np.array(
        [metrics[int(order)].nils_numerical for order in orders],
        dtype=np.float64,
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(orders, nils_analytical, marker="o", label="Analytical derivative")
    axis.plot(
        orders,
        nils_numerical,
        marker="x",
        linestyle="--",
        label="Numerical gradient",
    )
    axis.set_xlabel("Maximum captured diffraction order N")
    axis.set_ylabel("NILS at nominal edge")
    axis.set_title("NILS versus captured diffraction order")
    axis.grid(True, alpha=0.3)
    axis.set_xticks(orders)
    axis.legend()
    figure.tight_layout()

    output_path = output_dir / "figure_04_nils_vs_order.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_three_beam_closed_form_validation(
    output_dir: Path,
    config: SimulationConfig,
    result_n1: AerialImageResult,
) -> Path:
    """比较通用级次求和与教材 N=1 闭式公式。"""
    closed_form = textbook_three_beam_intensity(result_n1.x_nm, config)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        result_n1.x_nm,
        result_n1.intensity,
        linewidth=2.0,
        label="General Fourier-order sum",
    )
    axis.plot(
        result_n1.x_nm,
        closed_form,
        linestyle="--",
        linewidth=1.5,
        label="Textbook three-beam closed form",
    )
    axis.set_xlabel("Horizontal position x (nm)")
    axis.set_ylabel("Relative intensity")
    axis.set_title("Validation of the N=1 three-beam image")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    output_path = output_dir / "figure_05_three_beam_validation.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_physical_pupil_case(
    output_dir: Path,
    config: SimulationConfig,
    result: AerialImageResult,
    captured_orders: NDArray[np.int64],
    effective_orders: NDArray[np.int64],
) -> Path:
    """绘制 λ、NA 决定的实际最高衍射级次空中像。"""
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(result.x_nm, result.intensity, linewidth=2.0)
    axis.axvline(-config.nominal_edge_nm, linestyle="--", linewidth=1.0)
    axis.axvline(config.nominal_edge_nm, linestyle="--", linewidth=1.0)
    axis.set_xlabel("Horizontal position x (nm)")
    axis.set_ylabel("Relative intensity")
    axis.set_title(
        "Physical pupil case: "
        f"wavelength={config.wavelength_nm:g} nm, "
        f"NA={config.numerical_aperture:g}, "
        f"captured orders={captured_orders.tolist()}, "
        f"nonzero orders={effective_orders.tolist()}"
    )
    axis.grid(True, alpha=0.3)
    figure.tight_layout()

    output_path = output_dir / "figure_06_physical_pupil_case.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def write_summary(
    output_dir: Path,
    config: SimulationConfig,
    metrics: dict[int, ImageMetrics],
    captured_orders: NDArray[np.int64],
    effective_orders: NDArray[np.int64],
    validation_values: dict[str, float],
) -> Path:
    """写出便于检查和汇报的文本摘要。"""
    output_path = output_dir / "summary.txt"

    coefficient_lines = []
    for order in range(0, config.coefficient_max_order + 1):
        coefficient = fourier_order_coefficient(
            order,
            config.space_width_nm,
            config.pitch_nm,
        )
        coefficient_lines.append(f"a_{order:02d} = {coefficient:+.12f}")

    metric_lines = []
    for order in sorted(metrics):
        item = metrics[order]
        metric_lines.append(
            "N={:2d}: I_space={:.12f}, I_line={:.12f}, "
            "I_edge={:.12f}, contrast={:.12f}, "
            "NILS(analytic)={:.12f}, NILS(numeric)={:.12f}".format(
                order,
                item.space_center_intensity,
                item.line_center_intensity,
                item.nominal_edge_intensity,
                item.center_based_contrast,
                item.nils_analytical,
                item.nils_numerical,
            )
        )

    n1 = metrics[1]
    summary = f"""Textbook validation: coherent dense lines/spaces
=================================================

Model parameters
----------------
Transparent space width w : {config.space_width_nm:.6f} nm
Opaque line width         : {config.line_width_nm:.6f} nm
Pitch p                   : {config.pitch_nm:.6f} nm
Duty cycle w/p            : {config.duty_cycle:.6f}
Refractive index n        : {config.refractive_index:.6f}
Wavelength                : {config.wavelength_nm:.6f} nm
Numerical aperture        : {config.numerical_aperture:.6f}

Fourier coefficients
--------------------
{chr(10).join(coefficient_lines)}

Image metrics
-------------
{chr(10).join(metric_lines)}

N=1 hand-check values
---------------------
a0 = 1/2
a1 = 1/pi
I(x=0, center of clear space) = {n1.space_center_intensity:.12f}
I(x=p/2, center of opaque line) = {n1.line_center_intensity:.12f}
Center-based image contrast     = {n1.center_based_contrast:.12f}
NILS at x=w/2                   = {n1.nils_analytical:.12f}

Physical pupil case
-------------------
Condition: |j|/p <= NA/lambda
Captured orders                : {captured_orders.tolist()}
Captured nonzero-amplitude orders: {effective_orders.tolist()}
Maximum captured order         : {int(np.max(np.abs(captured_orders)))}

Internal validation
-------------------
Three-beam max absolute error  : {validation_values['three_beam_max_abs_error']:.6e}
Maximum even-order coefficient : {validation_values.get('max_even_order_coefficient', float('nan')):.6e}
Maximum NILS relative mismatch : {validation_values['max_nils_relative_difference']:.6e}

Interpretation notes
--------------------
1. In this program, N is the maximum positive diffraction order retained.
   N=1 therefore means 0 and +/-1 orders are included: three-beam imaging.
2. For equal lines and spaces, all even Fourier coefficients are zero, so adding an
   even maximum order does not change the image until the next odd order is added.
3. The intensity is not peak-normalized; it follows I=n|E|^2 for unit incident field,
   matching the textbook plot where the N=1 peak is approximately 1.29.
4. NILS is evaluated at the nominal clear-space edge x=w/2 and reported as a
   positive magnitude by convention.
"""

    output_path.write_text(summary, encoding="utf-8")
    return output_path


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the coherent dense line/space aerial-image exercise "
            "from the optical lithography textbook and lab manual."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("textbook_validation_results"),
        help="Directory used to save figures, CSV files and summary text.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display all saved figures after calculation.",
    )
    return parser.parse_args()


def main() -> None:
    """程序主入口。"""
    args = parse_arguments()
    config = SimulationConfig()
    config.validate()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    x_nm = make_position_grid(config)

    results: dict[int, AerialImageResult] = {}
    metrics: dict[int, ImageMetrics] = {}

    for max_order in config.compared_max_orders:
        result = calculate_coherent_aerial_image(
            x_nm=x_nm,
            max_diffraction_order=max_order,
            config=config,
        )
        results[max_order] = result
        metrics[max_order] = calculate_image_metrics(result, config)

    captured_orders = captured_orders_on_axis(
        wavelength_nm=config.wavelength_nm,
        numerical_aperture=config.numerical_aperture,
        pitch_nm=config.pitch_nm,
    )
    effective_orders = nonzero_captured_orders(captured_orders, config)
    physical_max_order = int(np.max(np.abs(captured_orders)))

    physical_result = calculate_coherent_aerial_image(
        x_nm=x_nm,
        max_diffraction_order=physical_max_order,
        config=config,
    )

    validation_values = run_internal_validation(config, results, metrics)

    generated_files = [
        save_coefficients_csv(output_dir, config),
        save_metrics_csv(output_dir, (metrics[o] for o in sorted(metrics))),
        plot_fourier_coefficients(output_dir, config),
        plot_aerial_images_vs_order(output_dir, config, results),
        plot_contrast_vs_order(output_dir, metrics),
        plot_nils_vs_order(output_dir, metrics),
        plot_three_beam_closed_form_validation(output_dir, config, results[1]),
        plot_physical_pupil_case(
            output_dir,
            config,
            physical_result,
            captured_orders,
            effective_orders,
        ),
        write_summary(
            output_dir,
            config,
            metrics,
            captured_orders,
            effective_orders,
            validation_values,
        ),
    ]

    print("\n=== Textbook coherent aerial-image validation completed ===")
    print(f"Output directory: {output_dir.resolve()}")
    print(
        "Captured orders for the physical pupil case: "
        f"{captured_orders.tolist()}"
    )
    print(
        "Nonzero-amplitude captured orders: "
        f"{effective_orders.tolist()}"
    )
    print("\nN=1 hand-check:")
    print(
        f"  I(space center) = {metrics[1].space_center_intensity:.12f}"
    )
    print(f"  I(line center)  = {metrics[1].line_center_intensity:.12f}")
    print(f"  Contrast        = {metrics[1].center_based_contrast:.12f}")
    print(f"  NILS            = {metrics[1].nils_analytical:.12f}")
    print("\nGenerated files:")
    for path in generated_files:
        print(f"  - {path}")

    if args.show:
        for path in generated_files:
            if path.suffix.lower() == ".png":
                image = plt.imread(path)
                figure, axis = plt.subplots(figsize=(10, 6))
                axis.imshow(image)
                axis.axis("off")
                axis.set_title(path.name)
                figure.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
