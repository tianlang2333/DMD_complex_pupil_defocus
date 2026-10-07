# 本轮实际验证报告

日期：2026-09-29。环境：Python 3.12.14、NumPy 2.3.5、Matplotlib 3.10.8，Linux，绘图后端 Agg。数值结果来自本轮真实运行；未使用设备验收精度充当算法容差。

## 修改与证据来源

职责和文件清单见根目录 `CHANGES.md`。主要改动为倍率检查、复瞳接口、独立离焦实验；新增两个生产/验证文件，未重写 FFT 或原指标体系。

修改前从实际九个上传源码文件建立快照；原始文件归档为 `baseline/original_sources.zip`，逐文件 SHA256 及来源记录在 `baseline/`。`fixed_na0065_baseline.npz/.json` 在未修改源码上显式传入 λ=.405、NA=.065 得到，包含无漏光 eta=1,rho=0,phi=0 与复漏光 eta=1,rho=.01,phi=π/3 的完整复场、强度、能量和关键指标。旧 main 的 .65 默认值未用于这项回归。

## 预设判据与结果

独立验证：**227 PASS，0 FAIL，0 SKIPPED**，逐条记录见 `projection_validation_report.json`。参数拒绝/状态断言按精确逻辑验收；机器精度比较采用绝对 1e-12，能量不变性采用相对 1e-12。被动瞳允许模值舍入 1e-13，最坏强度增量约 2e-13，配套能量检查为相对 1e-12 加绝对 1e-14；不裁剪输入。

| 验证 | 实际结果 | 预设判据 | 状态 |
|---|---:|---:|---|
| 原版无漏光/复漏光 → 新默认/显式零离焦，全复场与强度最大差 | 0 | ≤1e-12 | PASS |
| 上述两案例能量与关键指标回退 | 0 | ≤1e-12 | PASS |
| 常量瞳相位后的复场最大差 | 2.845e-16 | ≤1e-12 | PASS |
| 常量瞳相位后的原始强度最大差 | 6.661e-16 | ≤1e-12 | PASS |
| 改变离焦时完整网格能量相对差 | 2.567e-16 | ≤1e-12 | PASS |
| 独立二维有限和，复场最大差 | 2.027e-14 | ≤1e-12 | PASS |
| 独立二维有限和，强度最大差 | 2.132e-14 | ≤1e-12 | PASS |
| 复漏光的线性复场叠加差 | 2.763e-16 | ≤1e-12 | PASS |
| 16→32 点/微镜，原始截面差/Iref | 9.3847124e-5 | ≤1e-3 | PASS |
| 16→32 点/微镜，CD 绝对差 | 0.001217636 μm | ≤0.01 μm | PASS |
| 16→32 点/微镜，NILS 相对差 | 0.000163635 | ≤0.01 | PASS |

倍率 .424、1、1.3 均实际渲染传播；三类实验派生倍率也检查 M=1/1.3。零/负/布尔/复数/非有限以及派生尺度溢出、下溢均被拒绝。非法瞳、超单位模值、非法离焦和空气 NA>1 被拒绝；NA=1 边界有限，瞳外高频不参与平方根。负实瞳相位正确；外部完整全通 pupil 不会再次乘名义圆瞳。

独立有限和使用 63×48 与 64×47 混合奇偶非方网格、非零 fy、复系数与瞳外模式，分别覆盖零、正、负离焦。输入与期望输出由有限平面波直接求和，独立 kz 与 exp[i(kz-k)z] 构造真值；未调用生产构瞳、不读取 filtered_spectrum 作为真值，未拟合全局相位、幅度或平移。复目标的正负离焦强度差约 .304，能够区分符号。

## 收敛工况与适用性

active_side_ratio=.95，z=50 μm，64×32 微镜、投影间距1.5 μm。粗细网格共用**16点名义参考** Iref=1.0546765820801907、T0=.20382631104900517，未分别反算目标 CD。

| 项目 | 16点/微镜 | 32点/微镜 |
|---|---:|---:|
| Nx×Ny | 1024×512 | 2048×1024 |
| dx=dy / μm | .09375 | .046875 |
| Lx×Ly / μm | 96×48 | 96×48 |
| Δfx / cycles·μm⁻¹ | .010416667 | .010416667 |
| Δfy / cycles·μm⁻¹ | .020833333 | .020833333 |
| Nyquist / cycles·μm⁻¹ | 5.333333 | 10.666667 |
| 光强最短周期采样数 | 33.23077 | 66.46154 |
| 阈值 CD / μm | 6.000000000 | 5.998782364 |
| NILS | 7.248449922 | 7.249636215 |
| 状态 | VALID | VALID |

fc=.1604938272，强度最高带宽2fc=.3209876543 cycles/μm，两方向均满足。L=N×dx，不能用末采样中心减首采样中心。截面比较在相同物理评价区域，细网格插值到粗网格采样位置；同一真实周期模型在 y 方向的通带内无变化，实际采样行/y 坐标均记录在 JSON。提高每微镜采样数细化开口覆盖及截面，并提高 Nyquist；扩大物理窗口才改变 Δf。本轮只验证整数周期边界，不宣称孤立图形窗口收敛。能量为同网格离散 Σ|U|²，没有把不同网格未经面积加权的和当作物理能量比较。

## 原功能与历史材料

原基础自检6项、原漏光验证47项全部 PASS；修改前后47项输出 JSON 完全一致。`capture_original_baseline.py` 在新旧源码显式运行，教材/NA/采样比/十字的19个数组逐值相同，旧数值汇总也完全相同，见 `legacy_regression.json`。

教材条件保持 .248 μm/.85；二维采样矩形对一维连续系数的原始强度 RMSE=.000817780326，满足预设 .001，并非要求连续系数与离散采样机器精度相同。

原 `round2_regression.py` 所需 `round2_baseline/` 未随本次文件提供，其历史验收为 **SKIPPED**。保留该脚本，不伪造旧基线。本轮新建立的原版证据有独立命名和来源。

## 实际默认运行

`python run_dmd_2d_validation.py --output-dir results` 实际完成 **57/57** 内部检查（6基础+47漏光+4原十字检查）。本轮新增的227项镜头检查独立执行，不嵌入日常 main。默认保存10个实际产物：5张图、2个CSV、2个NPZ、1份摘要，均在 `results/`。

| Δz / μm | CD / μm | NILS | 中心对比度 | 状态 |
|---:|---:|---:|---:|---|
| -50 | 6.000000000 | 7.248449922 | 0.877697976 | VALID |
| -25 | 6.000000000 | 7.823819690 | 0.947368160 | VALID |
| 0 | 6.000000000 | 8.018769311 | 0.970974156 | VALID |
| 25 | 6.000000000 | 7.823819690 | 0.947368160 | VALID |
| 50 | 6.000000000 | 7.248449922 | 0.877697976 | VALID |

CD近似不变与三束等焦交点相符，NILS随本次范围内离焦下降；没有为制造CD变化而改变阈值，也不将此结论推广至所有图形。

## 调度验收

具体各模式记录见 `switch_checks.json`。案例组合共301项通过，最大数值误差4.441e-16；只开/默认组合的13项NPZ数据最大差0，CSV物理行完全一致。另实际验证fixed_fraction共享T0=.5Iref，以及eta=0单焦位的NaN、失效原因、图和CSV/NPZ正常保存。只开离焦、实际默认组合、关闭离焦，及单点、无零、逆序、重复、固定复漏光均通过真实生产计算验收。开关只在测试进程的 AST 中修改，不写产品源码；关闭分支使用非法列表和禁止调用哨兵检查。旧文件哈希保持不变且从本次manifest排除；只开/组合的同焦位原始数组与指标比较使用1e-12。

历史 default/phase-only/dose-only/disabled 模式用显式旧测试工况（rho=(0,1e-4,1e-3,.01)，s=(.95,1,1.05)），保留物理和缓存检查，不把硬编码当成新生产默认值。

## 复现命令

以下均从解压后的项目根目录运行；Linux无界面环境可先 `export MPLBACKEND=Agg`，Windows可直接使用本机Matplotlib后端。生成路径可自行改名。

```bash
python test_projection_lens_2d.py --report check_lens/projection_validation_report.json
python test_leakage_validation.py --output-dir check_leakage
python run_dmd_2d_validation.py --output-dir my_results
python verify_run_switches.py . check_only defocus-only
python verify_run_switches.py . check_combined combined check_only
python verify_run_switches.py . check_disabled defocus-disabled
python verify_run_switches.py . check_cases defocus-cases
python verify_run_switches.py . check_historical default
python verify_run_switches.py . check_phase phase-only
python verify_run_switches.py . check_dose dose-only
python verify_run_switches.py . check_all_disabled disabled
```

原版证据复现（输出到新目录，不覆盖随包证据）：

```bash
python -m zipfile -e baseline/original_sources.zip original_source_check
python baseline/capture_fixed_baseline.py --source original_source_check --output original_fixed_check
python capture_original_baseline.py original_source_check original_legacy_check
python capture_original_baseline.py . updated_legacy_check
```

本轮执行使用同样脚本与参数，scratch前缀与日志记录保留在JSON来源中；随包路径以以上相对命令为准。输出图已视觉检查，NPZ均以allow_pickle=False读取检查。

## 未验证/未实现

没有实测镜头波前/PSF/MTF数据，本轮不构成河南百合实物镜头验证。尚未实现Zernike、Zemax文件解析、实物标定、多视场、矢量/部分相干、扫描、绝对剂量及光刻胶/显影等。完整复瞳入口仅为这些后续工作的接口基础。未做本机交互式`--show`窗口验收；本轮已验证Agg保存路径。
