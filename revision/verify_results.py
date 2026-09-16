"""Check saved predictions, protocol summaries and reported values without training."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon,rankdata
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
ARCHIVE=ROOT/'archived'

def metrics(d):
    return dict(R2=r2_score(d.y_true,d.y_pred),RMSE=np.sqrt(mean_squared_error(d.y_true,d.y_pred)),MAE=mean_absolute_error(d.y_true,d.y_pred))

def check_metrics(frame,row):
    assert not frame.idx.duplicated().any()
    for k,v in metrics(frame).items():
        # Stored predictions can originate as float32 and round during CSV export.
        assert np.isclose(v,row[k],atol=1e-7,rtol=1e-7),(k,v,row[k])

def verify():
    counts={'prediction_files':0,'selected_trials':0,'paired_comparisons':0}
    for kind in ['transform','budget']:
        folds=pd.read_csv(RESULTS/f'{kind}_summary_folds.csv')
        for row in folds.to_dict('records'):
            stem=f"{kind}_{row['dataset']}_{row['fold']}_{row['model']}_{row['transform']}_{row['score']}"
            frame=pd.read_csv(RESULTS/f"{stem}_{row['budget']}_predictions.csv")
            check_metrics(frame,row);counts['prediction_files']+=1
            trials=pd.read_csv(RESULTS/f'{stem}_trials.csv').iloc[:row['budget']]
            best=trials.loc[trials.value.idxmax()]
            assert int(best.number)==row['best_trial']
            assert np.isclose(best.value,row['inner_score'])
            counts['selected_trials']+=1
        keys=['dataset','model','transform','score','budget']
        recomputed=folds.groupby(keys)[['R2','RMSE','MAE']].mean().sort_index()
        saved=pd.read_csv(RESULTS/f'{kind}_summary.csv').set_index(keys).sort_index()
        assert np.allclose(recomputed,saved[['R2','RMSE','MAE']])
    foundation=pd.read_csv(RESULTS/'foundation_summary.csv')
    for row in foundation.to_dict('records'):
        if row['kind']=='new_random_cv':name=f"TabPFN_{row['dataset']}_{row['fold']}_predictions.csv"
        elif row['kind']=='split_conformal':name=f"TabPFN_{row['dataset']}_{row['fold']}_split_conformal.csv"
        else:name=f"Grouped_{row['dataset']}_{row['fold']}_{row['model']}_predictions.csv"
        d=pd.read_csv(RESULTS/name);check_metrics(d,row);counts['prediction_files']+=1
        if row['kind']=='split_conformal':
            coverage=((d.y_true>=d.lo)&(d.y_true<=d.hi)).mean()
            assert np.isclose(coverage,row['coverage'])
            assert np.isclose((d.hi-d.lo).mean(),row['width'])
    groups={}
    for tag in ['A','B','Korea']:
        for model in ['TabPFN','RF']:
            d=pd.concat([pd.read_csv(p) for p in sorted(RESULTS.glob(f'Grouped_{tag}_*_{model}_predictions.csv'))])
            assert len(d)==(63 if tag=='Korea' else 60) and d.idx.nunique()==len(d)
            groups[f'{tag}_{model}']=metrics(d)
    for name,value in {'A_TabPFN':.9216,'A_RF':.8783,'B_TabPFN':.6284,'B_RF':.6873,'Korea_TabPFN':.7425,'Korea_RF':.6185}.items():
        assert round(groups[name]['R2'],4)==value
    intervals={}
    for tag,expected in [('A',55),('Korea',61)]:
        d=pd.concat([pd.read_csv(p) for p in sorted(RESULTS.glob(f'TabPFN_{tag}_*_split_conformal.csv'))])
        assert d.idx.nunique()==len(d)
        covered=int(((d.y_true>=d.lo)&(d.y_true<=d.hi)).sum())
        assert covered==expected
        intervals[tag]={'covered':covered,'n':len(d),'mean_width':float((d.hi-d.lo).mean())}
    d=pd.read_csv(ARCHIVE/'UQ_XConf_samples_FSA.csv');d=d[np.isclose(d.alpha,.1)]
    assert len(d)==600 and d.idx.nunique()==60
    assert ((d.y_obs>=d.xc_lo)&(d.y_obs<=d.xc_hi)).sum()==553
    primary={}
    for tag,expected in [('A',.9190),('B',.6692)]:
        f=pd.read_csv(ARCHIVE/f'FS{tag}_all_folds.csv')
        selected=f[(f.model=='TabPFN')|(f['mode']=='bo')]
        agg=selected.groupby('model')[['R2','RMSE','MAE']].mean()
        assert round(agg.loc['TabPFN','R2'],4)==expected
        primary[tag]=agg
    gain=float((primary['A'].R2-primary['B'].R2).mean())
    assert round(gain,4)==.2582
    korea=pd.read_csv(ARCHIVE/'Korea_all_folds.csv').groupby('model')[['R2','RMSE','MAE']].mean()
    assert np.allclose(korea.loc['TabPFN'].to_numpy(),[.7748,.1947,.1630],atol=.00005)
    audit=json.loads((RESULTS/'paired_statistics_audit.json').read_text())
    for item in audit:
        dif=np.array(item['repeat_differences'])
        p=wilcoxon(dif,alternative='two-sided' if item['dataset']=='A_vs_B' else 'greater').pvalue
        assert np.isclose(p,item['p_raw']),item
        nz=dif[dif!=0];r=rankdata(np.abs(nz));effect=(r[nz>0].sum()-r[nz<0].sum())/r.sum()
        assert np.isclose(effect,item['rank_biserial'])
        counts['paired_comparisons']+=1
    # Verify all source rows against the published-table transcription supplied with this release.
    source=pd.read_csv(ROOT/'source_audit/Table_S11_60_catchments.csv')
    workbook=pd.read_excel(ROOT/'data/debris_flow_longmenshan.xlsx')
    numeric=['A_km2','H_m','L_km','D_km','J_permille','V_landslide_1e4m3','V0_1e4m3']
    assert np.allclose(source[numeric],workbook[numeric])
    manifest=ROOT/'SHA256.json'
    if manifest.exists():
        entries=json.loads(manifest.read_text(encoding='utf-8'))
        for name,expected in entries.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
        counts['checksum_files']=len(entries)
    return dict(status='passed',checks=counts,primary_mean_deposit_gain=gain,regional_metrics=groups,split_conformal=intervals,primary_residual_coverage={'covered':553,'instances':600,'catchments':60})

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path);args=ap.parse_args()
    report=verify();text=json.dumps(report,indent=2)
    if args.output:args.output.write_text(text+'\n',encoding='utf-8')
    print(text)
