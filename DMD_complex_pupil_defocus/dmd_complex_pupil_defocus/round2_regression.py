"""第二轮回归证据：旧调用默认 fixed_fraction、显式模式、target_cd 光学场不变。
代码包根目录运行：python verification/round2_regression.py --output-dir round2_recheck
只用于验收，复用真实生产函数，不实现另一套传播或测量算法。
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--source-dir', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--legacy-dir', type=Path, help='复用本轮已经实际执行的 legacy capture 目录。')
a = parser.parse_args()
source, out = a.source_dir.resolve(), a.output_dir.resolve()
before = Path(__file__).resolve().parent / 'round2_baseline'
out.mkdir(parents=True, exist_ok=True)
sys.dont_write_bytecode = True
os.environ.setdefault('MPLBACKEND', 'Agg')
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
commands = []
def execute(command, logfile):
    commands.append(command)
    with logfile.open('w', encoding='utf-8') as stream:
        subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

legacy = a.legacy_dir.resolve() if a.legacy_dir else out/'legacy'
if not a.legacy_dir:
    execute([sys.executable, str(source/'verification/capture_original_baseline.py'),
             str(source), str(legacy)], out/'legacy_capture.log')
report = {'legacy': {'numeric_count':0,'null_count':0,'bool_count':0}, 'commands':commands}
def same(x,y,path=''):
    if isinstance(x,dict):
        assert x.keys() == y.keys(), path
        for key in x: same(x[key],y[key],path+'/'+key)
    elif isinstance(x,list):
        assert len(x)==len(y), path
        for i,(left,right) in enumerate(zip(x,y)): same(left,right,path+'/'+str(i))
    else:
        assert x==y, (path,x,y)
        if isinstance(x,bool): report['legacy']['bool_count']+=1
        elif isinstance(x,(int,float)): report['legacy']['numeric_count']+=1
        elif x is None: report['legacy']['null_count']+=1
same(json.loads((before/'baseline_numeric.json').read_text()),
     json.loads((legacy/'baseline_numeric.json').read_text()))

def equal_arrays(original, updated, keys):
    results=[]
    for key in keys:
        left, right = np.asarray(original[key]), np.asarray(updated[key])
        assert left.shape == right.shape and left.dtype == right.dtype, (key,left.shape,right.shape)
        numeric = np.issubdtype(left.dtype,np.number)
        exact = np.array_equal(left,right,equal_nan=True) if numeric else np.array_equal(left,right)
        assert exact, f'NPZ mismatch: {key}'
        results.append({'key':key,'shape':list(left.shape),'dtype':str(left.dtype),'exact_equal':True,
                        'max_abs_difference':float(np.max(np.abs(left-right))) if numeric else None})
    return results
with np.load(before/'baseline_profiles.npz',allow_pickle=False) as old, np.load(legacy/'baseline_profiles.npz',allow_pickle=False) as new:
    assert old.files == new.files
    report['legacy']['arrays'] = equal_arrays(old,new,old.files)

sys.path.insert(0,str(source))
v = importlib.import_module('run_dmd_2d_validation')
dmd = v.DMDConfig(num_mirrors_x=64,num_mirrors_y=32,dmd_mirror_pitch_um=7.56,
                  projection_magnification=1.5/7.56,samples_per_mirror=16,active_side_ratio=0.95)
optical = v.OpticalConfig2D(wavelength_um=0.405,numerical_aperture=0.065)
# 完整保留旧公开调用的关键字集合：首次调用不传任何新增关键字。
kwargs = dict(on_width_um=6.,pitch_um=12.,phase_offset_um=0.,on_intensity_scale=1.,
    off_to_on_intensity_ratio=.01,off_relative_phase_rad=0.,
    leakage_ratios=(0.,1e-4,1e-3,1e-2),leakage_phases_rad=(0.,.5*np.pi,np.pi),
    threshold_fraction=.5,evaluation_bounds_um=(-6.,6.),search_bounds_um=(-6.,6.),
    dark_bounds_um=(4.5,7.5),run_ratio_sweep=True,run_phase_sweep=True)
with (before/'leakage_metrics.csv').open(encoding='utf-8-sig',newline='') as stream:
    reader=csv.DictReader(stream); old_rows=list(reader); old_columns=reader.fieldnames

def compare_rows(rows):
    by_name={row['case_name']:row for row in rows}
    counts={'row_count':len(rows),'old_numeric_cells':0,'old_text_cells':0,'old_columns':len(old_columns)}
    assert set(by_name)=={row['case_name'] for row in old_rows}
    for row in old_rows:
        current=by_name[row['case_name']]
        assert set(old_columns)<=current.keys()
        for key in old_columns:
            try: value=float(row[key])
            except ValueError:
                assert str(current[key])==row[key],(row['case_name'],key,row[key],current[key])
                counts['old_text_cells']+=1
            else:
                actual=float(current[key])
                assert value==actual or math.isnan(value) and math.isnan(actual),(row['case_name'],key,value,actual)
                counts['old_numeric_cells']+=1
    return counts

for label, additions in [('old_call_default',{}),('explicit_fixed_fraction',{'threshold_mode':'fixed_fraction'})]:
    folder=out/label; folder.mkdir(exist_ok=True)
    rows,data,plots,notes=v.run_leakage_experiments(folder,False,dmd,optical,**kwargs,**additions)
    comparison=compare_rows(rows)
    assert all(row['threshold_mode']=='fixed_fraction' for row in rows)
    with np.load(before/'leakage_representative_data.npz',allow_pickle=False) as old:
        comparison['arrays']=equal_arrays(old,data,old.files)
    comparison['baseline_cd_um']=rows[0]['fixed_cd_um']
    comparison['reference_intensity']=rows[0]['reference_intensity']
    comparison['threshold_intensity']=rows[0]['threshold_intensity']
    report[label]=comparison
    with (folder/'leakage_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(folder/'leakage_representative_data.npz',**data)
    (folder/'notes.txt').write_text('\n'.join(notes),encoding='utf-8')

# 实际主入口默认 target_cd；仅阈值评价/曝光倍率扩展，旧光学输出应逐值不变。
target=out/'target_main'
execute([sys.executable,str(source/'run_dmd_2d_validation.py'),'--output-dir',str(target)],out/'target_main.log')
with np.load(before/'leakage_representative_data.npz',allow_pickle=False) as old, np.load(target/'leakage_representative_data.npz',allow_pickle=False) as new:
    optical_keys=[key for key in old.files if key!='threshold_intensity']
    target_report={'unchanged_arrays':equal_arrays(old,new,optical_keys),
                   'old_threshold_intensity':float(old['threshold_intensity']),
                   'target_threshold_intensity':float(new['threshold_intensity']),
                   'representative_case_name':str(new['representative_case_name'])}
    assert str(new['representative_case_name'])=='ratio_03'
with (target/'leakage_metrics.csv').open(encoding='utf-8-sig',newline='') as stream:
    rows=list(csv.DictReader(stream))
assert all(row['threshold_mode']=='target_cd' for row in rows)
base_row=next(row for row in rows if row['case_name']=='ratio_00')
target_report.update(baseline_cd_um=float(base_row['fixed_cd_um']),total_rows=len(rows))
assert abs(target_report['baseline_cd_um']-6.)<=float(base_row['cd_tolerance_um'])
report['target_main']=target_report
report['current_source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob('*.py'))}
report['status']='PASS'
(out/'round2_regression_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print('PASS legacy:',report['legacy']['numeric_count'],'numeric,',len(report['legacy']['arrays']),'arrays')
for label in ('old_call_default','explicit_fixed_fraction'):
    r=report[label]
    print('PASS',label,':',r['row_count'],'rows,',r['old_numeric_cells'],'numeric cells,',len(r['arrays']),'NPZ keys; CD=',r['baseline_cd_um'])
print('PASS target_main: original optical outputs exact; target CD=',target_report['baseline_cd_um'],'threshold=',target_report['target_threshold_intensity'])
