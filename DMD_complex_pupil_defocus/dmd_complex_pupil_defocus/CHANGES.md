# 本轮修改说明

- `dmd_model_2d.py`：允许有限实倍率 M>0，验证派生投影间距和采样间距；图形、坐标和 ON/OFF 渲染算法未改。
- `coherent_imaging_2d.py`：完整被动复瞳、complex128 返回类型、关键字 `pupil=None`；原 FFT/逆 FFT 与默认理想圆瞳路径保留。
- 新增 `projection_lens_2d.py`：按约定符号构造空气中精确标量角谱离焦圆瞳；稳定计算根号差，只在瞳内计算。
- `run_dmd_2d_validation.py`：LDI main NA 由 0.65 改为参考 0.065；教材 λ=0.248、NA=0.85 不变。修正重复倍率约束和复瞳显示；增加独立纯离焦参数区、固定参考实验、少量图与 CSV/NPZ 输出。旧开关及响应/扫描列表保留。
- 新增 `test_projection_lens_2d.py`：独立二维平面波真值、正负离焦复场、原版基线回退、相位/能量、参数与采样收敛检查。不成为主程序运行依赖。
- `verify_run_switches.py`：对旧硬编码断言显式设置历史验收工况；新增实际 main 的只开离焦、组合、关闭与案例顺序检查。未为通过测试改回旧生产默认值。
- `README.md` 与本轮验证报告：更新实际默认值、接口责任、离焦符号、物理边界、命令、实测数值和缺失历史证据说明。

未改动 `textbook_dense_lines_aerial_image.py`、`test_leakage_validation.py`、`capture_original_baseline.py`、`round2_regression.py`。原版源码归档和哈希在 `baseline/`；本轮基线均从未修改源码生成。

本轮仅建立外部完整复瞳输入接口。Zernike、Zemax 文件接入、镜头处方、实物标定及其它扩展仍未实现。
