# 二维 DMD 复瞳接口与物理离焦

本轮在原 ON/OFF 等效复振幅链路上增加完整复瞳入口和纯离焦实验。日常入口仍为 `run_dmd_2d_validation.py`；图形、有限开口、漏光响应、FFT 移位/正交归一化及原阈值算法保持原约定。参考 NA=0.065 是仿真参数，并非河南百合镜头实测值。

## 运行

Python 3.10+，依赖 NumPy、Matplotlib。解压进入本目录：

```bash
python -m pip install -r requirements.txt
python run_dmd_2d_validation.py --output-dir my_results
python run_dmd_2d_validation.py --output-dir my_results_show --show
```

PyCharm 可直接运行入口文件。自动验证使用 `MPLBACKEND=Agg`；`--show` 需要本机图形环境。默认结果目录为当前工作目录下 `dmd_2d_validation_results/`。每次可指定新目录；重复目录不会被清空，只有摘要“本次生成文件”属于本次运行。

## 参数和开关

常用参数仍集中在 `main()`，修改后运行即可。

| 设置 | 本轮实际默认值 |
|---|---|
| `run_defocus_experiment` | `True` |
| `defocus_values_um` | `(-50, -25, 0, 25, 50)` μm |
| 离焦实验响应 | eta=1、rho=0、phi=0，与旧漏光配置独立 |
| `run_leakage_sweep` | `True`；rho=(0, 1e-4, 1e-2, 5e-2) |
| `run_phase_sweep` | `False`；保留 phi=(0, π/2, π) |
| `run_dose_check` | `True`；s=(0.95, 1, 1.05, 1.2) |
| `run_cross_leakage` | `True` |
| 教材、NA、采样比、原十字、原收敛五个开关 | 均 `False` |
| 旧漏光代表响应 | eta=1、rho=0.01、phi=0，未改动 |
| `threshold_mode` | `"target_cd"`；也可选 `"fixed_fraction"` |

**只开离焦**：保持 `run_defocus_experiment=True`，将其余九个实验开关全部设为 `False`。基础自检仍执行；新增镜头验证文件不是主程序运行依赖。关闭离焦后，离焦列表不校验，离焦分支不渲染、不构瞳、不传播、不绘图或检查其文件。

参考几何：64×32 微镜，DMD 间距 7.56 μm，投影间距 1.5 μm，16 点/微镜，开口边长比 0.95，ON 宽 6 μm、周期 12 μm、中心偏移 0。LDI λ=0.405 μm、像方 NA=0.065（上传 main 原为 0.65，本轮按确认修正）；一维教材及二维教材交叉验证保留 λ=0.248 μm、NA=0.85。

`projection_magnification=M` 为曝光面尺寸/DMD 面尺寸的正倍率绝对值；有限 M>0 均可，含 1 和 1.3，拒绝负值、零、布尔值及非有限值。投影间距=pitch_DMD×M；采样间距=投影间距/每微镜采样数。不增加 1/M 振幅缩放。周期实验另外要求整数像元/周期及整数周期窗口；倍率合法不代表任何指定图形均满足该条件。

## 复瞳接口与物理约定

```python
from coherent_imaging_2d import make_frequency_axes, calculate_coherent_aerial_image_2d
from projection_lens_2d import make_defocused_circular_pupil

fx, fy = make_frequency_axes(len(pattern.x_um), len(pattern.y_um),
                            config.sample_spacing_um, config.sample_spacing_um)
pupil = make_defocused_circular_pupil(fx, fy, optical, defocus_um=25.0)
image = calculate_coherent_aerial_image_2d(
    pattern.object_field, pattern.x_um, pattern.y_um, optical, pupil=pupil)
```

上例 `pattern/config/optical` 为调用方已经建立的 DMD 渲染结果、DMDConfig 和 OpticalConfig2D。原四个位置参数不变；`pupil=None` 保留理想圆瞳路径。显式 pupil 是**完整有效复瞳**，不会再次乘圆瞳或离焦因子；调用方负责与本次居中频率网格一致。返回的 `frequency_cutoff_cyc_per_um` 始终为 optical 的名义 NA/λ，不是外部 pupil 的实测截止频率。

pupil 采用 complex128，必须二维、非空、形状匹配且有限；被动模型检查模值不超过 1（模值容差 1e-13；配套能量容差为相对 1e-12、绝对 1e-14），负实值表示 π 相位。非法输入报错，不裁剪、重归一化或丢弃虚部。绘图振幅用 |P|；arg(P) 在瞳外掩蔽。`normalized_intensity` 为兼容保留，离焦评价不用它。

空气 n=1，单色标量完全相干局部空间不变。时间因子 exp(-iωt)，Δz>0 沿光传播方向离开参考像面；固定镜头，只移动观察平面，去除公共 exp(ikΔz) 相位：

```text
fc = NA / λ
P(fx,fy,z) = A(fx,fy) exp{i (2πz/λ) [sqrt(1-λ²(fx²+fy²))-1]}
A = 1 inside fx²+fy² <= fc², otherwise 0
U_image = centered_ifft2(centered_fft2(U_DMD) * P)
I = abs(U_image)**2
```

生成器要求 0<NA≤1，Δz 为有限实数；只在瞳内计算根号和相位。以 `-q/(1+sqrt(1-q))` 稳定计算根号差，只有舍入量级的根号越界允许修正。Δz=0 恢复圆瞳；不含倏逝波、浸没、矢量效应或移动镜组/光楔的联动变化。数组为 `[row_y,column_x]`，长度 μm，频率 cycles/μm；FFT 均保留原 centered、`norm="ortho"` 约定。

## 固定参考与指标

每次离焦扫描先单独建立 eta=1、rho=0、phi=0、Δz=0、s=1 的名义参考，即使扫描列表没有零也一样。Iref 是指定评价区域内参考原始截面的采样峰值；T0 由原 `establish_nominal_workpoint` 一次建立。target_cd 使用对称设计边缘线性插值并验证目标区间；fixed_fraction 使用 fraction×Iref。不逐焦位归一化、重定阈值或拟合平移。

所有焦位共用 Iref、T0、目标和搜索/暗区范围。固定 s=1，旧漏光曝光量功能仍用 T0/s。NILS 仍是设计右边缘的七点四次局部多项式结果并取绝对值；中心对比度保留符号。截面选距 y=0 最近的一行，记录实际行号和 y；对应复场与强度取同一行。

无有效目标区间时 CD/边缘为 NaN，CSV 保留原因和该焦位。所有“相对名义基准”均相对无漏光零离焦 s=1；若修改成固定漏光，差值含漏光和离焦的共同影响。此为阈值空中像宽度，未标定为实际显影 CD。

对于对称等线空的三束成像，设计边缘附近可存在等焦交点，固定阈值 CD 可能变化很小而 NILS 已下降。不能以“离焦后 CD 必须明显改变”作为通过标准，也不要求全程单调。

## 输出

| 文件 | 内容 |
|---|---|
| `defocus_metrics.csv` | 各焦位参数、共同参考、CD/EPE、NILS、对比度、暗区和有效状态、网格与截面信息 |
| `defocus_representative_data.npz` | 全部原始截面及对应复场、固定参考、少量代表二维强度/复场/复瞳；`allow_pickle=False` |
| `figure_defocus_*.png` | 共同 Iref 的截面，CD/NILS 穿焦及代表瞳诊断 |
| `validation_summary.txt` | 自检、开关状态、完整配置、模型边界和实际产物清单 |
| 原 `leakage_*` 等文件 | 原字段含义保留，按旧开关生成 |

随包 `results/` 保存本轮实际默认运行的代表输出。原版九文件和哈希、显式 .065 基线在 `baseline/`；独立验收和报告在 `verification/`。原始复场和强度不作峰值归一化保存。

## 验证与复查

```bash
python test_projection_lens_2d.py --report check_lens/projection_validation_report.json
python test_leakage_validation.py --output-dir check_leakage
python verify_run_switches.py . check_only defocus-only
python verify_run_switches.py . check_combined combined
python verify_run_switches.py . check_disabled defocus-disabled
```

独立镜头测试预先规定机器精度与网格收敛容差，包含原版/默认路径/显式零离焦三路回归、被动复瞳、常量相位、二维有限平面波独立复场真值、正负离焦、奇偶混合非方网格、固定参考 16→32 点/微镜收敛。具体命令、数值误差、状态与缺项见 `verification/REPORT.md`。

完整网格能量定义为 Σ|U|²，仅用于同网格 Parseval/透过率检查，并非已标定光功率。不同网格比较需面积权重。提高每微镜采样数改变 dx 和 Nyquist；固定微镜数时 L=N×dx 不变、频率间隔不变。扩大物理窗口才减小 Δf；本轮周期边界检查不能代替孤立图形窗口收敛。

历史 `capture_original_baseline.py` 可用显式源目录和输出目录运行。`round2_regression.py` 依赖未随本次上传的 `round2_baseline/`，历史回归标记 SKIPPED；不新造旧证据。`verify_run_switches.py` 的历史模式使用明确固定的旧测试工况，当前默认与新离焦模式分别验收。

## 范围

已实现：正倍率、完整复瞳入口、空气物理离焦、固定参考穿焦与独立数值验证。预留：外部完整复瞳输入，可供后续波前数据构造。尚未实现：Zernike/OPD/Zemax 解析、实物标定、多视场、部分相干、RCWA、矢量传播、扫描、绝对剂量、胶/显影、OPC/ILT。当前结果不构成河南百合镜头验证。
