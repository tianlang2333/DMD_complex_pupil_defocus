"""二维 DMD 等效复振幅模型。

本模块只描述投影镜头已选中 ON 态反射方向后，DMD 在曝光面等效
坐标中的理想 ON/OFF 复振幅。它不包含微镜倾转、闪耀衍射、偏振、
窗口、照明角谱或其他严格电磁效应。

数组统一采用 ``array[row_y, column_x]``：第一维对应 y，第二维对应 x。
所有长度均使用微米（um），坐标位于各离散采样单元的中心。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import NDArray


BoolArray2D = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
FloatArray2D = NDArray[np.float64]
ComplexArray2D = NDArray[np.complex128]
Orientation = Literal["vertical", "horizontal"]


@dataclass(frozen=True)
class DMDConfig:
    """DMD 几何、采样和理想 ON/OFF 振幅配置。

    ``projection_magnification`` 定义为曝光面横向尺寸与 DMD 面横向
    尺寸之比的正倍率绝对值 M，允许任意有限 M>0（缩小、等倍、放大）。
    不隐含镜像/旋转，也不附加 1/M 振幅缩放。

    ``active_side_ratio`` 是有效开口边长与微镜像元间距之比，不是
    面积填充率。开口边界采用亚采样覆盖率表示；实际离散值会由
    :func:`render_dmd_field` 返回在 :class:`DMDPatternResult` 中。
    """

    num_mirrors_x: int
    num_mirrors_y: int
    dmd_mirror_pitch_um: float
    projection_magnification: float
    samples_per_mirror: int
    active_side_ratio: float = 0.92
    on_amplitude: complex = 1.0 + 0.0j
    off_amplitude: complex = 0.0 + 0.0j

    def __post_init__(self) -> None:
        _validate_positive_integer("num_mirrors_x", self.num_mirrors_x)
        _validate_positive_integer("num_mirrors_y", self.num_mirrors_y)
        _validate_positive_finite(
            "dmd_mirror_pitch_um", self.dmd_mirror_pitch_um
        )
        _validate_positive_finite(
            "projection_magnification", self.projection_magnification
        )
        _validate_positive_integer(
            "samples_per_mirror", self.samples_per_mirror
        )
        # 即使输入各自有限，乘积/商仍可能溢出或下溢到零。
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            _validate_positive_finite(
                "projected_mirror_pitch_um", self.projected_mirror_pitch_um
            )
            _validate_positive_finite("sample_spacing_um", self.sample_spacing_um)
        _validate_positive_finite("active_side_ratio", self.active_side_ratio)
        if self.active_side_ratio > 1.0:
            raise ValueError("active_side_ratio 必须位于区间 (0, 1]。")
        _validate_finite_complex("on_amplitude", self.on_amplitude)
        _validate_finite_complex("off_amplitude", self.off_amplitude)

    @property
    def projected_mirror_pitch_um(self) -> float:
        """返回曝光面上的投影微镜像元间距。"""

        return self.dmd_mirror_pitch_um * self.projection_magnification

    @property
    def sample_spacing_um(self) -> float:
        """返回有限开口复振幅网格的采样间距。"""

        return self.projected_mirror_pitch_um / self.samples_per_mirror

    @property
    def field_num_samples_x(self) -> int:
        """返回展开后复振幅网格在 x 方向的采样数。"""

        return self.num_mirrors_x * self.samples_per_mirror

    @property
    def field_num_samples_y(self) -> int:
        """返回展开后复振幅网格在 y 方向的采样数。"""

        return self.num_mirrors_y * self.samples_per_mirror


def with_state_response(
    config: DMDConfig,
    *,
    on_intensity_scale: float,
    off_to_on_intensity_ratio: float,
    off_relative_phase_rad: float,
) -> DMDConfig:
    """用 eta、rho、phi 替换双态等效复振幅，保留原几何与开口。

    eta 是模型参考下的 ON 强度缩放，rho 是 OFF/ON 强度比；二者
    先开平方再成为场振幅。phi 为光学相位（rad），不是几何偏移。
    返回新配置，不与原配置的振幅相乘，也不改变微镜间隙。
    """

    values = (
        ("on_intensity_scale", on_intensity_scale),
        ("off_to_on_intensity_ratio", off_to_on_intensity_ratio),
        ("off_relative_phase_rad", off_relative_phase_rad),
    )
    for name, value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} 必须是有限实数。")
        try:
            finite_value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} 必须是有限实数。") from error
        if not np.isfinite(finite_value):
            raise ValueError(f"{name} 必须是有限实数。")
    eta = float(on_intensity_scale)
    rho = float(off_to_on_intensity_ratio)
    phi = float(off_relative_phase_rad)
    if eta < 0.0:
        raise ValueError("on_intensity_scale 必须大于或等于 0。")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("off_to_on_intensity_ratio 必须位于 [0, 1]。")
    return replace(
        config,
        on_amplitude=complex(np.sqrt(eta)),
        off_amplitude=complex(np.sqrt(eta * rho) * np.exp(1j * phi)),
    )


@dataclass(frozen=True)
class DMDPatternResult:
    """DMD 状态经有限开口展开后的结果。

    ``mirror_states`` 的形状为 ``(num_mirrors_y, num_mirrors_x)``，
    ``object_field`` 的形状为 ``(field_num_samples_y,
    field_num_samples_x)``。``x_um`` 和 ``y_um`` 分别是一维 x、y
    采样中心坐标。
    """

    mirror_states: BoolArray2D
    object_field: ComplexArray2D
    x_um: FloatArray
    y_um: FloatArray
    sample_spacing_um: float
    projected_mirror_pitch_um: float
    mirror_aperture_template: FloatArray2D
    actual_active_side_ratio: float


def make_mirror_center_coordinates(
    config: DMDConfig,
) -> tuple[FloatArray, FloatArray]:
    """返回曝光面等效坐标中的微镜中心坐标 ``(x_um, y_um)``。

    整个 DMD 窗口的几何中心定义为坐标原点。若某方向包含偶数个
    微镜，则原点位于中央两个微镜中心之间，而不是某个微镜中心上。
    """

    projected_pitch_um = config.projected_mirror_pitch_um
    mirror_x_um = (
        np.arange(config.num_mirrors_x, dtype=np.float64)
        - 0.5 * (config.num_mirrors_x - 1)
    ) * projected_pitch_um
    mirror_y_um = (
        np.arange(config.num_mirrors_y, dtype=np.float64)
        - 0.5 * (config.num_mirrors_y - 1)
    ) * projected_pitch_um
    return mirror_x_um, mirror_y_um


def make_field_coordinates(
    config: DMDConfig,
) -> tuple[FloatArray, FloatArray]:
    """返回有限开口场的采样中心坐标 ``(x_um, y_um)``。"""

    sample_spacing_um = config.sample_spacing_um
    x_um = (
        np.arange(config.field_num_samples_x, dtype=np.float64)
        - 0.5 * (config.field_num_samples_x - 1)
    ) * sample_spacing_um
    y_um = (
        np.arange(config.field_num_samples_y, dtype=np.float64)
        - 0.5 * (config.field_num_samples_y - 1)
    ) * sample_spacing_um
    return x_um, y_um


def make_periodic_line_space_states(
    config: DMDConfig,
    pitch_um: float,
    on_width_um: float,
    orientation: Orientation = "vertical",
    phase_offset_um: float = 0.0,
) -> BoolArray2D:
    """按微镜中心位置生成二维周期线/空 ON/OFF 状态。

    Parameters
    ----------
    config:
        DMD 几何配置。
    pitch_um:
        周期线/空图形的周期。
    on_width_um:
        每个周期内 ON（透光或曝光）条带的宽度。
    orientation:
        ``"vertical"`` 表示条带沿 y 延伸、图形随 x 周期变化；
        ``"horizontal"`` 表示条带沿 x 延伸、图形随 y 周期变化。
    phase_offset_um:
        ON 条带中心相对坐标原点的偏移。该值可为任意有限实数，因而
        支持相对于 DMD 网格的亚像元相位偏移。

    Notes
    -----
    周期内采用左闭右开的 ON 区间，边缘恰好穿过微镜中心时也只有
    唯一、确定的归属。状态数组始终采用 ``[row_y, column_x]``。
    """

    pitch_um, on_width_um = _validate_line_space_parameters(
        pitch_um, on_width_um
    )
    orientation = _validate_orientation(orientation)
    phase_offset_um = _validate_finite_real(
        "phase_offset_um", phase_offset_um
    )

    mirror_x_um, mirror_y_um = make_mirror_center_coordinates(config)
    if orientation == "vertical":
        on_columns = _make_periodic_on_mask(
            mirror_x_um, pitch_um, on_width_um, phase_offset_um
        )
        mirror_states = np.broadcast_to(
            on_columns[np.newaxis, :],
            (config.num_mirrors_y, config.num_mirrors_x),
        )
    else:
        on_rows = _make_periodic_on_mask(
            mirror_y_um, pitch_um, on_width_um, phase_offset_um
        )
        mirror_states = np.broadcast_to(
            on_rows[:, np.newaxis],
            (config.num_mirrors_y, config.num_mirrors_x),
        )

    return np.array(mirror_states, dtype=np.bool_, copy=True)


def make_cross_states(
    config: DMDConfig,
    arm_width_um: float,
    arm_length_um: float,
    center_x_um: float = 0.0,
    center_y_um: float = 0.0,
) -> BoolArray2D:
    """按微镜中心位置生成由水平臂和竖直臂并集构成的十字图形。"""

    arm_width_um, arm_length_um = _validate_cross_parameters(
        arm_width_um, arm_length_um
    )
    center_x_um = _validate_finite_real("center_x_um", center_x_um)
    center_y_um = _validate_finite_real("center_y_um", center_y_um)

    mirror_x_um, mirror_y_um = make_mirror_center_coordinates(config)
    xx_um, yy_um = np.meshgrid(mirror_x_um, mirror_y_um, indexing="xy")
    relative_x_um = xx_um - center_x_um
    relative_y_um = yy_um - center_y_um

    horizontal_arm = (
        (np.abs(relative_x_um) <= 0.5 * arm_length_um)
        & (np.abs(relative_y_um) <= 0.5 * arm_width_um)
    )
    vertical_arm = (
        (np.abs(relative_x_um) <= 0.5 * arm_width_um)
        & (np.abs(relative_y_um) <= 0.5 * arm_length_um)
    )
    return np.asarray(horizontal_arm | vertical_arm, dtype=np.bool_)


def make_ideal_periodic_line_space_field(
    x_um: FloatArray,
    y_um: FloatArray,
    pitch_um: float,
    on_width_um: float,
    orientation: Orientation = "vertical",
    phase_offset_um: float = 0.0,
    on_amplitude: complex = 1.0 + 0.0j,
    off_amplitude: complex = 0.0 + 0.0j,
) -> ComplexArray2D:
    """在细网格上直接采样连续理想周期线/空复振幅。

    此函数不经过微镜中心量化，也不加入微镜有限开口，用于与 DMD
    栅格化目标作对照。返回数组形状为 ``(len(y_um), len(x_um))``。
    """

    x_um = _validate_coordinate_vector("x_um", x_um)
    y_um = _validate_coordinate_vector("y_um", y_um)
    pitch_um, on_width_um = _validate_line_space_parameters(
        pitch_um, on_width_um
    )
    orientation = _validate_orientation(orientation)
    phase_offset_um = _validate_finite_real(
        "phase_offset_um", phase_offset_um
    )
    on_amplitude = _validate_finite_complex("on_amplitude", on_amplitude)
    off_amplitude = _validate_finite_complex("off_amplitude", off_amplitude)

    if orientation == "vertical":
        on_columns = _make_periodic_on_mask(
            x_um, pitch_um, on_width_um, phase_offset_um
        )
        on_mask = np.broadcast_to(
            on_columns[np.newaxis, :], (y_um.size, x_um.size)
        )
    else:
        on_rows = _make_periodic_on_mask(
            y_um, pitch_um, on_width_um, phase_offset_um
        )
        on_mask = np.broadcast_to(
            on_rows[:, np.newaxis], (y_um.size, x_um.size)
        )

    return np.where(on_mask, on_amplitude, off_amplitude).astype(
        np.complex128, copy=False
    )


def make_ideal_cross_field(
    x_um: FloatArray,
    y_um: FloatArray,
    arm_width_um: float,
    arm_length_um: float,
    center_x_um: float = 0.0,
    center_y_um: float = 0.0,
    on_amplitude: complex = 1.0 + 0.0j,
    off_amplitude: complex = 0.0 + 0.0j,
) -> ComplexArray2D:
    """在细网格上直接采样连续理想十字复振幅。"""

    x_um = _validate_coordinate_vector("x_um", x_um)
    y_um = _validate_coordinate_vector("y_um", y_um)
    arm_width_um, arm_length_um = _validate_cross_parameters(
        arm_width_um, arm_length_um
    )
    center_x_um = _validate_finite_real("center_x_um", center_x_um)
    center_y_um = _validate_finite_real("center_y_um", center_y_um)
    on_amplitude = _validate_finite_complex("on_amplitude", on_amplitude)
    off_amplitude = _validate_finite_complex("off_amplitude", off_amplitude)

    xx_um, yy_um = np.meshgrid(x_um, y_um, indexing="xy")
    relative_x_um = xx_um - center_x_um
    relative_y_um = yy_um - center_y_um
    horizontal_arm = (
        (np.abs(relative_x_um) <= 0.5 * arm_length_um)
        & (np.abs(relative_y_um) <= 0.5 * arm_width_um)
    )
    vertical_arm = (
        (np.abs(relative_x_um) <= 0.5 * arm_width_um)
        & (np.abs(relative_y_um) <= 0.5 * arm_length_um)
    )
    on_mask = horizontal_arm | vertical_arm
    return np.where(on_mask, on_amplitude, off_amplitude).astype(
        np.complex128, copy=False
    )


def make_mirror_aperture_template(
    samples_per_mirror: int,
    active_side_ratio: float,
) -> tuple[FloatArray2D, float]:
    """生成严格居中的方形单微镜开口模板及实际离散边长比。

    模板元素表示一个数值采样单元被理想方形开口覆盖的面积比例，
    因此范围为 0 到 1，而不局限于二值。先精确计算每个一维采样区间
    与中心开口的重叠长度比例，再取二维外积。该亚采样覆盖率方法既
    保持严格左右、上下对称，也能在低采样数下表示窄间隙。例如，
    ``samples_per_mirror=8``、``active_side_ratio=0.95`` 时，首尾一维
    单元覆盖率为 0.8，而不会错误量化成无间隙的 1.0 开口。

    返回的实际边长比定义为一维覆盖率之和除以每微镜采样数。在浮点
    精度内，它应等于请求的 ``active_side_ratio``。
    """

    _validate_positive_integer("samples_per_mirror", samples_per_mirror)
    _validate_positive_finite("active_side_ratio", active_side_ratio)
    if active_side_ratio > 1.0:
        raise ValueError("active_side_ratio 必须位于区间 (0, 1]。")

    # 以下坐标以微镜像元间距为 1；每个数值单元是一个有限区间，
    # 而不是只在中心点判断 0/1，因此可以用部分覆盖率表达亚采样边界。
    sample_edges = np.linspace(
        -0.5,
        0.5,
        samples_per_mirror + 1,
        dtype=np.float64,
    )
    aperture_lower_edge = -0.5 * active_side_ratio
    aperture_upper_edge = 0.5 * active_side_ratio
    overlap_lower_edge = np.maximum(sample_edges[:-1], aperture_lower_edge)
    overlap_upper_edge = np.minimum(sample_edges[1:], aperture_upper_edge)
    sample_width = 1.0 / samples_per_mirror
    aperture_1d = np.clip(
        (overlap_upper_edge - overlap_lower_edge) / sample_width,
        0.0,
        1.0,
    ).astype(np.float64, copy=False)

    aperture_template = np.outer(aperture_1d, aperture_1d).astype(
        np.float64, copy=False
    )
    actual_active_side_ratio = float(np.sum(aperture_1d) / samples_per_mirror)
    return aperture_template, float(actual_active_side_ratio)


def render_dmd_field(
    mirror_states: BoolArray2D,
    config: DMDConfig,
) -> DMDPatternResult:
    """将微镜状态矩阵展开为带有限开口的二维 DMD 复振幅。

    每个微镜的复振幅系数为

    ``off_amplitude + state * (on_amplitude - off_amplitude)``。

    系数只作用于单微镜有效开口，像元间隙的振幅始终为零。展开使用
    ``np.kron``，避免对全部细网格采样点执行双重 Python 循环。
    """

    validated_states = _validate_mirror_states(mirror_states, config)
    aperture_template, actual_active_side_ratio = (
        make_mirror_aperture_template(
            config.samples_per_mirror, config.active_side_ratio
        )
    )

    mirror_amplitudes = (
        complex(config.off_amplitude)
        + validated_states.astype(np.float64)
        * (complex(config.on_amplitude) - complex(config.off_amplitude))
    ).astype(np.complex128, copy=False)
    object_field = np.kron(
        mirror_amplitudes,
        aperture_template.astype(np.complex128, copy=False),
    ).astype(np.complex128, copy=False)

    x_um, y_um = make_field_coordinates(config)
    expected_field_shape = (
        config.field_num_samples_y,
        config.field_num_samples_x,
    )
    if object_field.shape != expected_field_shape:
        raise RuntimeError(
            "DMD 有限开口展开结果形状错误："
            f"得到 {object_field.shape}，预期 {expected_field_shape}。"
        )

    return DMDPatternResult(
        mirror_states=validated_states.copy(),
        object_field=object_field,
        x_um=x_um,
        y_um=y_um,
        sample_spacing_um=config.sample_spacing_um,
        projected_mirror_pitch_um=config.projected_mirror_pitch_um,
        mirror_aperture_template=aperture_template,
        actual_active_side_ratio=actual_active_side_ratio,
    )


def _make_periodic_on_mask(
    coordinates_um: FloatArray,
    pitch_um: float,
    on_width_um: float,
    phase_offset_um: float,
) -> NDArray[np.bool_]:
    """返回以 ``phase_offset_um`` 为 ON 条带中心的周期一维掩模。"""

    wrapped_coordinate_um = np.remainder(
        coordinates_um - phase_offset_um + 0.5 * pitch_um,
        pitch_um,
    ) - 0.5 * pitch_um
    return np.asarray(
        (wrapped_coordinate_um >= -0.5 * on_width_um)
        & (wrapped_coordinate_um < 0.5 * on_width_um),
        dtype=np.bool_,
    )


def _validate_line_space_parameters(
    pitch_um: float,
    on_width_um: float,
) -> tuple[float, float]:
    pitch_um = _validate_positive_finite("pitch_um", pitch_um)
    on_width_um = _validate_positive_finite("on_width_um", on_width_um)
    if on_width_um > pitch_um:
        raise ValueError("on_width_um 必须小于或等于 pitch_um。")
    return pitch_um, on_width_um


def _validate_cross_parameters(
    arm_width_um: float,
    arm_length_um: float,
) -> tuple[float, float]:
    arm_width_um = _validate_positive_finite("arm_width_um", arm_width_um)
    arm_length_um = _validate_positive_finite("arm_length_um", arm_length_um)
    if arm_length_um < arm_width_um:
        raise ValueError("arm_length_um 必须大于或等于 arm_width_um。")
    return arm_width_um, arm_length_um


def _validate_orientation(orientation: str) -> Orientation:
    if not isinstance(orientation, str):
        raise ValueError("orientation 必须是 'vertical' 或 'horizontal'。")
    normalized_orientation = orientation.strip().lower()
    if normalized_orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation 必须是 'vertical' 或 'horizontal'。")
    return normalized_orientation  # type: ignore[return-value]


def _validate_coordinate_vector(name: str, values: FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} 必须是一维坐标数组。")
    if array.size == 0:
        raise ValueError(f"{name} 不能为空。")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须只包含有限实数。")
    if array.size > 1 and not np.all(np.diff(array) > 0.0):
        raise ValueError(f"{name} 必须严格递增。")
    return array


def _validate_mirror_states(
    mirror_states: BoolArray2D,
    config: DMDConfig,
) -> BoolArray2D:
    states = np.asarray(mirror_states)
    expected_shape = (config.num_mirrors_y, config.num_mirrors_x)
    if states.ndim != 2 or states.shape != expected_shape:
        raise ValueError(
            "mirror_states 必须采用 [row_y, column_x]，且形状为 "
            f"{expected_shape}；当前形状为 {states.shape}。"
        )

    if np.issubdtype(states.dtype, np.bool_):
        return states.astype(np.bool_, copy=False)

    if not (
        np.issubdtype(states.dtype, np.number)
        and np.all(np.isfinite(states))
        and np.all((states == 0) | (states == 1))
    ):
        raise ValueError("mirror_states 必须是布尔数组或只包含 0/1。")
    return states.astype(np.bool_, copy=False)


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} 必须是正整数。")
    if int(value) <= 0:
        raise ValueError(f"{name} 必须是正整数。")


def _validate_positive_finite(name: str, value: float) -> float:
    validated_value = _validate_finite_real(name, value)
    if validated_value <= 0.0:
        raise ValueError(f"{name} 必须是有限正数。")
    return validated_value


def _validate_finite_real(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} 必须是有限实数。")
    try:
        validated_value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} 必须是有限实数。") from error
    if not np.isfinite(validated_value):
        raise ValueError(f"{name} 必须是有限实数。")
    return validated_value


def _validate_finite_complex(name: str, value: complex) -> complex:
    try:
        validated_value = complex(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是有限复数。") from error
    if not (
        np.isfinite(validated_value.real) and np.isfinite(validated_value.imag)
    ):
        raise ValueError(f"{name} 必须是有限复数。")
    return validated_value


__all__ = [
    "BoolArray2D",
    "ComplexArray2D",
    "DMDConfig",
    "DMDPatternResult",
    "FloatArray",
    "FloatArray2D",
    "Orientation",
    "make_cross_states",
    "make_field_coordinates",
    "make_ideal_cross_field",
    "make_ideal_periodic_line_space_field",
    "make_mirror_aperture_template",
    "make_mirror_center_coordinates",
    "make_periodic_line_space_states",
    "render_dmd_field",
]
