"""空气中固定镜头、移动观察面的标量相干离焦复瞳。

长度单位 um，空间频率 cycles/um，数组采用 [row_y, column_x]。
时间约定 exp(-i omega t)，正离焦沿光传播方向。去除公共传播相位
exp(i k dz) 后，仅保留 exp[i (kz-k) dz]。本模块只构瞳，不传播场。
这不是高 NA 矢量模型，也不描述移动镜组/光楔导致的倍率或像差变化。
"""

from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import NDArray

from coherent_imaging_2d import OpticalConfig2D, make_circular_pupil


def _finite_real(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} 必须是有限实数，不接受布尔值或复数。")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} 必须是有限实数。") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} 必须是有限实数。")
    return result


def _frequency_axis(name: str, values: NDArray[np.generic]) -> NDArray[np.float64]:
    source = np.asarray(values)
    if np.iscomplexobj(source) or not np.issubdtype(source.dtype, np.number):
        raise ValueError(f"{name} 必须是有限实数频率轴。")
    axis = np.asarray(source, dtype=np.float64)
    if axis.ndim != 1 or axis.size == 0 or not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} 必须是非空、有限的一维频率轴。")
    if axis.size == 1:
        if axis[0] != 0.0:
            raise ValueError(f"{name} 的单点频率轴必须为直流点。")
        return axis
    step = float(axis[1] - axis[0])
    expected = (np.arange(axis.size, dtype=np.float64) - axis.size // 2) * step
    if not np.isfinite(step) or step <= 0.0 or not np.allclose(
        axis, expected, rtol=1.0e-10, atol=max(1.0e-14, step * 1.0e-12)
    ):
        raise ValueError(f"{name} 必须与 fftshift(fftfreq) 的居中等距顺序一致。")
    return axis


def make_defocused_circular_pupil(
    frequency_x_cyc_per_um: NDArray[np.generic],
    frequency_y_cyc_per_um: NDArray[np.generic],
    optical_config: OpticalConfig2D,
    defocus_um: float,
) -> NDArray[np.complex128]:
    """返回给定居中频率网格上的完整离焦圆瞳，形状为 (Ny, Nx)。

    空气 n=1，要求 0<NA<=1；lambda 是真空波长。圆瞳半径 NA/lambda。
    瞳内使用 exp{i(2*pi*dz/lambda)[sqrt(1-q)-1]}，
    q=lambda**2*(fx**2+fy**2)。瞳外直接为零，不计算平方根。
    采用 -q/(1+sqrt(1-q)) 稳定计算同一表达式，不是近轴替代。
    调用方应传入成像核心为同一空间采样生成的频率轴。
    """

    if not isinstance(optical_config, OpticalConfig2D):
        raise ValueError("optical_config 必须是 OpticalConfig2D 实例。")
    wavelength_um = _finite_real("wavelength_um", optical_config.wavelength_um)
    numerical_aperture = _finite_real(
        "numerical_aperture", optical_config.numerical_aperture
    )
    dz_um = _finite_real("defocus_um", defocus_um)
    if wavelength_um <= 0.0:
        raise ValueError("wavelength_um 必须为正数。")
    if not 0.0 < numerical_aperture <= 1.0:
        raise ValueError("空气离焦生成器要求 0 < numerical_aperture <= 1。")
    cutoff = numerical_aperture / wavelength_um
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("NA/lambda 必须为有限正数。")
    fx = _frequency_axis("frequency_x_cyc_per_um", frequency_x_cyc_per_um)
    fy = _frequency_axis("frequency_y_cyc_per_um", frequency_y_cyc_per_um)
    aperture = make_circular_pupil(fx, fy, cutoff)
    pupil = np.asarray(aperture, dtype=np.complex128)
    if dz_um == 0.0:
        return pupil

    inside = aperture != 0.0
    grid_x, grid_y = np.meshgrid(fx, fy, indexing="xy")
    # 先将频率化为无量纲量，避免单独求 lambda**2 引起无谓溢出。
    q = (wavelength_um * grid_x[inside]) ** 2 + (
        wavelength_um * grid_y[inside]
    ) ** 2
    # q 在空气传播瞳内应 <=1；仅允许边界的浮点舍入误差。
    root_roundoff_tolerance = 32.0 * np.finfo(np.float64).eps
    if not np.all(np.isfinite(q)) or np.any(q > 1.0 + root_roundoff_tolerance):
        raise ValueError("有效瞳内出现非传播频率；不支持倏逝波。")
    radicand = 1.0 - q
    radicand[(radicand < 0.0) & (radicand >= -root_roundoff_tolerance)] = 0.0
    relative_longitudinal_wavevector = -q / (1.0 + np.sqrt(radicand))
    with np.errstate(over="ignore", invalid="ignore"):
        phase = 2.0 * np.pi * (dz_um * relative_longitudinal_wavevector / wavelength_um)
    if not np.all(np.isfinite(phase)):
        raise ValueError("给定离焦与频率产生不可表示的相位。")
    pupil[inside] = np.exp(1j * phase)
    return pupil


__all__ = ["make_defocused_circular_pupil"]
