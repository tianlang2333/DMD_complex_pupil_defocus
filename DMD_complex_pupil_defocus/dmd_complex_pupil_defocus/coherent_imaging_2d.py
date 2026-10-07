from __future__ import annotations

"""使用理想圆瞳或外部完整复瞳的二维相干投影成像。

本模块只负责复振幅的二维傅里叶传播，不包含DMD图形生成、部分相干、
离焦/像差的瞳函数构造、扫描曝光或光刻胶过程。二维数组统一采用
``array[row_y, column_x]``，空间频率单位统一为 cycles/um。
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


BoolArray2D = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
FloatArray2D = NDArray[np.float64]
ComplexArray2D = NDArray[np.complex128]

# 模值容差引起的最坏强度增量约为 2e-13，低于能量相对容差 1e-12。
# 容差仅用于判定；绝不裁剪或重新归一化调用方的复瞳。
PASSIVE_PUPIL_MODULUS_TOLERANCE = 1.0e-13
ENERGY_RELATIVE_TOLERANCE = 1.0e-12
ENERGY_ABSOLUTE_TOLERANCE = 1.0e-14


@dataclass(frozen=True)
class OpticalConfig2D:
    """二维理想相干成像的光学参数。

    Attributes:
        wavelength_um: 真空波长，单位为 um。
        numerical_aperture: 像方数值孔径，无量纲。
    """

    wavelength_um: float
    numerical_aperture: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.wavelength_um) or self.wavelength_um <= 0.0:
            raise ValueError("wavelength_um 必须是有限正数。")
        if (
            not np.isfinite(self.numerical_aperture)
            or self.numerical_aperture <= 0.0
        ):
            raise ValueError("numerical_aperture 必须是有限正数。")

    @property
    def frequency_cutoff_cyc_per_um(self) -> float:
        """返回相干成像截止空间频率 NA/lambda，单位 cycles/um。"""

        return self.numerical_aperture / self.wavelength_um


@dataclass(frozen=True)
class CoherentImage2DResult:
    """二维相干传播的完整中间量和输出结果。

    能量使用离散样本强度之和定义。由于输入、输出使用同一空间采样，
    该定义不影响能量透过比例，并能直接利用正交归一化FFT检查Parseval
    能量关系。
    """

    x_um: FloatArray
    y_um: FloatArray
    frequency_x_cyc_per_um: FloatArray
    frequency_y_cyc_per_um: FloatArray
    object_field: ComplexArray2D
    object_spectrum: ComplexArray2D
    pupil: ComplexArray2D
    filtered_spectrum: ComplexArray2D
    image_field: ComplexArray2D
    raw_intensity: FloatArray2D
    normalized_intensity: FloatArray2D
    input_energy: float
    output_energy: float
    energy_transmission: float
    frequency_cutoff_cyc_per_um: float

    @property
    def input_field(self) -> ComplexArray2D:
        """``object_field``的语义别名。"""

        return self.object_field

    @property
    def input_spectrum(self) -> ComplexArray2D:
        """``object_spectrum``的语义别名。"""

        return self.object_spectrum

    @property
    def pupil_filtered_spectrum(self) -> ComplexArray2D:
        """``filtered_spectrum``的语义别名。"""

        return self.filtered_spectrum

    @property
    def image_complex_amplitude(self) -> ComplexArray2D:
        """``image_field``的语义别名。"""

        return self.image_field


def _as_valid_complex_field(
    field: NDArray[np.generic],
    *,
    field_name: str,
) -> ComplexArray2D:
    """将输入转换为有限的二维complex128复振幅数组。"""

    field_array = np.asarray(field, dtype=np.complex128)
    if field_array.ndim != 2:
        raise ValueError(f"{field_name} 必须是二维数组。")
    if field_array.size == 0:
        raise ValueError(f"{field_name} 不能为空数组。")
    if not np.all(np.isfinite(field_array)):
        raise ValueError(f"{field_name} 必须只包含有限数值。")
    return field_array


def _validate_coordinate_axis(
    coordinate_um: NDArray[np.generic],
    *,
    expected_size: int,
    coordinate_name: str,
) -> tuple[FloatArray, float]:
    """检查一维坐标轴，并返回float64坐标和采样间距。"""

    coordinate_array = np.asarray(coordinate_um, dtype=np.float64)
    if coordinate_array.ndim != 1:
        raise ValueError(f"{coordinate_name} 必须是一维数组。")
    if coordinate_array.size != expected_size:
        raise ValueError(
            f"{coordinate_name} 长度必须为 {expected_size}，"
            f"实际为 {coordinate_array.size}。"
        )
    if coordinate_array.size < 2:
        raise ValueError(f"{coordinate_name} 至少需要两个采样点。")
    if not np.all(np.isfinite(coordinate_array)):
        raise ValueError(f"{coordinate_name} 必须只包含有限数值。")

    coordinate_steps_um = np.diff(coordinate_array)
    if np.any(coordinate_steps_um <= 0.0):
        raise ValueError(f"{coordinate_name} 必须严格递增。")

    sample_spacing_um = float(coordinate_steps_um[0])
    absolute_tolerance_um = max(1.0e-14, abs(sample_spacing_um) * 1.0e-12)
    if not np.allclose(
        coordinate_steps_um,
        sample_spacing_um,
        rtol=1.0e-10,
        atol=absolute_tolerance_um,
    ):
        raise ValueError(f"{coordinate_name} 必须采用等间距采样。")

    return coordinate_array, sample_spacing_um


def centered_fft2(object_field: NDArray[np.generic]) -> ComplexArray2D:
    """计算居中的二维正交归一化FFT。

    输入和输出都按零坐标（零频率）位于数组中心的顺序排列。配套逆变换
    必须使用 :func:`centered_ifft2`，不得与其他移位约定混用。
    """

    object_field_array = _as_valid_complex_field(
        object_field,
        field_name="object_field",
    )
    return np.asarray(
        np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(object_field_array),
                norm="ortho",
            )
        ),
        dtype=np.complex128,
    )


def centered_ifft2(spectrum: NDArray[np.generic]) -> ComplexArray2D:
    """计算与 :func:`centered_fft2`严格配对的居中二维逆FFT。"""

    spectrum_array = _as_valid_complex_field(
        spectrum,
        field_name="spectrum",
    )
    return np.asarray(
        np.fft.fftshift(
            np.fft.ifft2(
                np.fft.ifftshift(spectrum_array),
                norm="ortho",
            )
        ),
        dtype=np.complex128,
    )


def make_frequency_axes(
    num_samples_x: int,
    num_samples_y: int,
    sample_spacing_x_um: float,
    sample_spacing_y_um: float,
) -> tuple[FloatArray, FloatArray]:
    """生成经过移位的x、y空间频率轴，单位 cycles/um。"""

    if isinstance(num_samples_x, bool) or not isinstance(
        num_samples_x,
        (int, np.integer),
    ):
        raise ValueError("num_samples_x 必须是整数。")
    if isinstance(num_samples_y, bool) or not isinstance(
        num_samples_y,
        (int, np.integer),
    ):
        raise ValueError("num_samples_y 必须是整数。")
    if num_samples_x < 2 or num_samples_y < 2:
        raise ValueError("x和y方向都至少需要两个采样点。")
    if (
        not np.isfinite(sample_spacing_x_um)
        or sample_spacing_x_um <= 0.0
    ):
        raise ValueError("sample_spacing_x_um 必须是有限正数。")
    if (
        not np.isfinite(sample_spacing_y_um)
        or sample_spacing_y_um <= 0.0
    ):
        raise ValueError("sample_spacing_y_um 必须是有限正数。")

    frequency_x_cyc_per_um = np.asarray(
        np.fft.fftshift(
            np.fft.fftfreq(
                int(num_samples_x),
                d=float(sample_spacing_x_um),
            )
        ),
        dtype=np.float64,
    )
    frequency_y_cyc_per_um = np.asarray(
        np.fft.fftshift(
            np.fft.fftfreq(
                int(num_samples_y),
                d=float(sample_spacing_y_um),
            )
        ),
        dtype=np.float64,
    )
    return frequency_x_cyc_per_um, frequency_y_cyc_per_um


def make_frequency_grid(
    x_um: NDArray[np.generic],
    y_um: NDArray[np.generic],
) -> tuple[FloatArray, FloatArray, FloatArray2D, FloatArray2D]:
    """根据等间距空间坐标生成频率轴和二维频率网格。

    返回顺序为 ``frequency_x, frequency_y, frequency_grid_x,
    frequency_grid_y``；二维网格形状为 ``(len(y_um), len(x_um))``。
    """

    x_array = np.asarray(x_um, dtype=np.float64)
    y_array = np.asarray(y_um, dtype=np.float64)
    x_array, sample_spacing_x_um = _validate_coordinate_axis(
        x_array,
        expected_size=x_array.size,
        coordinate_name="x_um",
    )
    y_array, sample_spacing_y_um = _validate_coordinate_axis(
        y_array,
        expected_size=y_array.size,
        coordinate_name="y_um",
    )

    frequency_x_cyc_per_um, frequency_y_cyc_per_um = make_frequency_axes(
        num_samples_x=x_array.size,
        num_samples_y=y_array.size,
        sample_spacing_x_um=sample_spacing_x_um,
        sample_spacing_y_um=sample_spacing_y_um,
    )
    frequency_grid_x_cyc_per_um, frequency_grid_y_cyc_per_um = np.meshgrid(
        frequency_x_cyc_per_um,
        frequency_y_cyc_per_um,
        indexing="xy",
    )
    return (
        frequency_x_cyc_per_um,
        frequency_y_cyc_per_um,
        np.asarray(frequency_grid_x_cyc_per_um, dtype=np.float64),
        np.asarray(frequency_grid_y_cyc_per_um, dtype=np.float64),
    )


def make_circular_pupil(
    frequency_x_cyc_per_um: NDArray[np.generic],
    frequency_y_cyc_per_um: NDArray[np.generic],
    frequency_cutoff_cyc_per_um: float,
) -> FloatArray2D:
    """生成截止半径为 ``frequency_cutoff_cyc_per_um`` 的理想圆瞳。

    前两个参数可以同时为一维频率轴，也可以同时为形状一致的二维频率
    网格。瞳内取1，瞳外取0，返回类型固定为float64。
    """

    if (
        not np.isfinite(frequency_cutoff_cyc_per_um)
        or frequency_cutoff_cyc_per_um <= 0.0
    ):
        raise ValueError("frequency_cutoff_cyc_per_um 必须是有限正数。")

    frequency_x_array = np.asarray(frequency_x_cyc_per_um, dtype=np.float64)
    frequency_y_array = np.asarray(frequency_y_cyc_per_um, dtype=np.float64)
    if not np.all(np.isfinite(frequency_x_array)) or not np.all(
        np.isfinite(frequency_y_array)
    ):
        raise ValueError("空间频率数组必须只包含有限数值。")

    if frequency_x_array.ndim == 1 and frequency_y_array.ndim == 1:
        frequency_grid_x, frequency_grid_y = np.meshgrid(
            frequency_x_array,
            frequency_y_array,
            indexing="xy",
        )
    elif frequency_x_array.ndim == 2 and frequency_y_array.ndim == 2:
        if frequency_x_array.shape != frequency_y_array.shape:
            raise ValueError("二维x、y频率网格的形状必须一致。")
        frequency_grid_x = frequency_x_array
        frequency_grid_y = frequency_y_array
    else:
        raise ValueError("x、y频率参数必须同时为一维轴或同时为二维网格。")

    radial_frequency_squared = (
        frequency_grid_x * frequency_grid_x
        + frequency_grid_y * frequency_grid_y
    )
    cutoff_frequency_squared = (
        frequency_cutoff_cyc_per_um * frequency_cutoff_cyc_per_um
    )
    return np.asarray(
        radial_frequency_squared <= cutoff_frequency_squared,
        dtype=np.float64,
    )


def propagate_coherent_field_from_pupil(
    object_field: NDArray[np.generic],
    pupil: NDArray[np.generic],
) -> tuple[ComplexArray2D, ComplexArray2D, ComplexArray2D]:
    """通过完整的被动复瞳函数传播二维复振幅。

    ``pupil`` 可以是实数或复数，模值须不超过 1（允许明确的舍入容差）。
    负实数代表 pi 相位，并非负透过率。函数不额外乘圆瞳或附加相位。

    Returns:
        ``(object_spectrum, filtered_spectrum, image_field)``。

    Raises:
        ValueError: 输入维度、数值范围或形状不合法。
        RuntimeError: 被动复瞳传播后出现超出舍入容差的能量增长。
    """

    object_field_array = _as_valid_complex_field(
        object_field,
        field_name="object_field",
    )
    pupil_array = _as_valid_complex_field(pupil, field_name="pupil")
    if pupil_array.shape != object_field_array.shape:
        raise ValueError(
            "pupil 与 object_field 的形状必须一致；"
            f"实际分别为 {pupil_array.shape} 和 {object_field_array.shape}。"
        )
    if np.any(np.abs(pupil_array) > 1.0 + PASSIVE_PUPIL_MODULUS_TOLERANCE):
        raise ValueError(
            "被动 pupil 必须满足 |P| <= 1；"
            f"模值舍入容差为 {PASSIVE_PUPIL_MODULUS_TOLERANCE:.1e}。"
        )

    object_spectrum = centered_fft2(object_field_array)
    filtered_spectrum = np.asarray(
        object_spectrum * pupil_array,
        dtype=np.complex128,
    )
    image_field = centered_ifft2(filtered_spectrum)

    input_energy = float(np.sum(np.abs(object_field_array) ** 2))
    output_energy = float(np.sum(np.abs(image_field) ** 2))
    if output_energy > (
        input_energy * (1.0 + ENERGY_RELATIVE_TOLERANCE)
        + ENERGY_ABSOLUTE_TOLERANCE
    ):
        raise RuntimeError(
            "被动复瞳传播出现异常能量增长："
            f"input_energy={input_energy:.16e}, "
            f"output_energy={output_energy:.16e}。"
        )

    return object_spectrum, filtered_spectrum, image_field


def calculate_coherent_aerial_image_2d(
    object_field: NDArray[np.generic],
    x_um: NDArray[np.generic],
    y_um: NDArray[np.generic],
    optical_config: OpticalConfig2D,
    *,
    pupil: NDArray[np.generic] | None = None,
) -> CoherentImage2DResult:
    """计算二维相干空中像；旧四位置参数调用仍使用理想圆瞳。

    关键字 ``pupil`` 若给定，表示完整最终有效复瞳，而非附加相位图；
    不再叠加圆瞳或离焦。调用方负责将其绑定到由当前 x/y 坐标生成的
    居中频率网格。返回的 ``frequency_cutoff_cyc_per_um`` 始终是
    optical_config 的名义 NA/lambda，不是对外部复瞳测得的实际截止。
    ``normalized_intensity`` 仅为兼容保留，固定参考比较应使用原始强度。
    """

    if not isinstance(optical_config, OpticalConfig2D):
        raise ValueError("optical_config 必须是 OpticalConfig2D 实例。")

    object_field_array = _as_valid_complex_field(
        object_field,
        field_name="object_field",
    )
    num_samples_y, num_samples_x = object_field_array.shape
    x_array, sample_spacing_x_um = _validate_coordinate_axis(
        x_um,
        expected_size=num_samples_x,
        coordinate_name="x_um",
    )
    y_array, sample_spacing_y_um = _validate_coordinate_axis(
        y_um,
        expected_size=num_samples_y,
        coordinate_name="y_um",
    )

    frequency_x_cyc_per_um, frequency_y_cyc_per_um = make_frequency_axes(
        num_samples_x=num_samples_x,
        num_samples_y=num_samples_y,
        sample_spacing_x_um=sample_spacing_x_um,
        sample_spacing_y_um=sample_spacing_y_um,
    )
    if pupil is None:
        pupil = make_circular_pupil(
            frequency_x_cyc_per_um=frequency_x_cyc_per_um,
            frequency_y_cyc_per_um=frequency_y_cyc_per_um,
            frequency_cutoff_cyc_per_um=(
                optical_config.frequency_cutoff_cyc_per_um
            ),
        )
    pupil_array = _as_valid_complex_field(pupil, field_name="pupil")
    object_spectrum, filtered_spectrum, image_field = (
        propagate_coherent_field_from_pupil(
            object_field=object_field_array,
            pupil=pupil_array,
        )
    )

    raw_intensity = np.asarray(np.abs(image_field) ** 2, dtype=np.float64)
    peak_intensity = float(np.max(raw_intensity))
    if peak_intensity > 0.0:
        normalized_intensity = np.asarray(
            raw_intensity / peak_intensity,
            dtype=np.float64,
        )
    else:
        normalized_intensity = np.zeros_like(raw_intensity, dtype=np.float64)

    input_energy = float(np.sum(np.abs(object_field_array) ** 2))
    output_energy = float(np.sum(raw_intensity))
    if input_energy > 0.0:
        energy_transmission = output_energy / input_energy
    else:
        energy_transmission = float("nan")

    return CoherentImage2DResult(
        x_um=x_array.copy(),
        y_um=y_array.copy(),
        frequency_x_cyc_per_um=frequency_x_cyc_per_um,
        frequency_y_cyc_per_um=frequency_y_cyc_per_um,
        object_field=object_field_array.copy(),
        object_spectrum=object_spectrum,
        pupil=pupil_array.copy(),
        filtered_spectrum=filtered_spectrum,
        image_field=image_field,
        raw_intensity=raw_intensity,
        normalized_intensity=normalized_intensity,
        input_energy=input_energy,
        output_energy=output_energy,
        energy_transmission=energy_transmission,
        frequency_cutoff_cyc_per_um=(
            optical_config.frequency_cutoff_cyc_per_um
        ),
    )


def run_internal_validations() -> dict[str, float]:
    """运行本模块的FFT往返、全通瞳孔、常数场和能量自检。

    任一检查不满足规定容差时直接抛出 ``AssertionError``，不会静默继续。
    返回值中的误差可由上层验证程序写入终端和摘要文件。
    """

    random_generator = np.random.default_rng(20260828)
    random_field = np.asarray(
        random_generator.standard_normal((31, 40))
        + 1j * random_generator.standard_normal((31, 40)),
        dtype=np.complex128,
    )
    roundtrip_field = centered_ifft2(centered_fft2(random_field))
    fft_roundtrip_max_error = float(
        np.max(np.abs(roundtrip_field - random_field))
    )
    if fft_roundtrip_max_error >= 1.0e-12:
        raise AssertionError(
            "FFT往返误差未通过："
            f"max_error={fft_roundtrip_max_error:.3e}。"
        )

    all_pass_pupil = np.ones(random_field.shape, dtype=np.float64)
    _, _, all_pass_field = propagate_coherent_field_from_pupil(
        object_field=random_field,
        pupil=all_pass_pupil,
    )
    all_pass_max_error = float(np.max(np.abs(all_pass_field - random_field)))
    if all_pass_max_error >= 1.0e-12:
        raise AssertionError(
            "全通瞳孔恢复输入未通过："
            f"max_error={all_pass_max_error:.3e}。"
        )

    constant_field = np.ones((32, 48), dtype=np.complex128)
    _, _, constant_output_field = propagate_coherent_field_from_pupil(
        object_field=constant_field,
        pupil=np.ones(constant_field.shape, dtype=np.float64),
    )
    constant_field_max_error = float(
        np.max(np.abs(constant_output_field - constant_field))
    )
    if constant_field_max_error >= 1.0e-12:
        raise AssertionError(
            "常数场全通瞳孔验证未通过："
            f"max_error={constant_field_max_error:.3e}。"
        )

    x_um = (np.arange(40, dtype=np.float64) - 19.5) * 0.08
    y_um = (np.arange(31, dtype=np.float64) - 15.0) * 0.10
    optical_result = calculate_coherent_aerial_image_2d(
        object_field=random_field,
        x_um=x_um,
        y_um=y_um,
        optical_config=OpticalConfig2D(
            wavelength_um=0.405,
            numerical_aperture=0.065,
        ),
    )
    energy_excess = optical_result.output_energy - optical_result.input_energy
    if energy_excess > optical_result.input_energy * 1.0e-12 + 1.0e-14:
        raise AssertionError(
            "能量单调性验证未通过："
            f"energy_excess={energy_excess:.3e}。"
        )

    return {
        "fft_roundtrip_max_error": fft_roundtrip_max_error,
        "all_pass_max_error": all_pass_max_error,
        "constant_field_max_error": constant_field_max_error,
        "input_energy": optical_result.input_energy,
        "output_energy": optical_result.output_energy,
        "energy_transmission": optical_result.energy_transmission,
        "energy_excess": energy_excess,
    }


__all__ = [
    "BoolArray2D",
    "FloatArray",
    "FloatArray2D",
    "ComplexArray2D",
    "OpticalConfig2D",
    "CoherentImage2DResult",
    "centered_fft2",
    "centered_ifft2",
    "make_frequency_axes",
    "make_frequency_grid",
    "make_circular_pupil",
    "propagate_coherent_field_from_pupil",
    "calculate_coherent_aerial_image_2d",
    "run_internal_validations",
]
