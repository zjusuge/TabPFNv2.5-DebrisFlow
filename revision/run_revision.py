"""Revision sensitivities with model selection confined to outer training data."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='2'
os.environ['MPLBACKEND']='Agg'
os.environ['TABPFN_DISABLE_TELEMETRY']='1'
from pathlib import Path
import argparse, hashlib, json, time, warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold, LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'data'
OUT=ROOT/'rerun_results'
MODEL_PATH=None
SEED=20260915
optuna.logging.set_verbosity(optuna.logging.WARNING)

def datasets():
    d=pd.read_excel(BASE/'debris_flow_longmenshan.xlsx')
    assert d.No.tolist()==list(range(1,61))
    groups=np.repeat(['Gaochuan','Qingping','Yingxiu','Road 213','Longchi','Beichuan'],[21,8,13,5,10,3])
    d['Region']=groups
    d.to_csv(OUT/'longmenshan_with_source_region.csv',index=False)
    k=pd.read_excel(BASE/'debris_flow_korea.xlsx')
    k.to_csv(OUT/'korea_input.csv',index=False)
    cols=['A_km2','H_m','L_km','D_km','J_permille','V_landslide_1e4m3']
    result={}
    for tag,frame,features,target,group in [('A',d,cols,'V0_1e4m3',groups),('B',d,cols[:-1],'V0_1e4m3',groups),('Korea',k,['Aw_m2','Lc_m','Rw_m','CR_mm'],'V_m3',k.District.to_numpy())]:
        result[tag]=(frame[features].to_numpy(float),np.log10(frame[target].to_numpy(float)),np.asarray(group))
    return result

def metrics(y,p):
    return dict(R2=float(r2_score(y,p)),RMSE=float(np.sqrt(mean_squared_error(y,p))),MAE=float(mean_absolute_error(y,p)))

def params(trial,model):
    if model=='Ridge':return dict(alpha=trial.suggest_float('alpha',1e-3,1e3,log=True))
    if model=='Lasso':return dict(alpha=trial.suggest_float('alpha',1e-5,1e1,log=True),max_iter=5000)
    if model=='ElasticNet':return dict(alpha=trial.suggest_float('alpha',1e-5,1e1,log=True),l1_ratio=trial.suggest_float('l1_ratio',.05,.95),max_iter=5000)
    if model=='MLP':
        return dict(hidden_layer_sizes=tuple([trial.suggest_int('width',8,64)]*trial.suggest_int('layers',1,2)),alpha=trial.suggest_float('alpha',1e-5,.1,log=True),solver='lbfgs',max_iter=2000,random_state=SEED)
    if model=='RF':return dict(n_estimators=trial.suggest_int('n_estimators',50,500),max_depth=trial.suggest_int('max_depth',2,20),min_samples_split=trial.suggest_int('min_samples_split',2,10),min_samples_leaf=trial.suggest_int('min_samples_leaf',1,5),random_state=SEED,n_jobs=2)
    if model=='XGBoost':return dict(n_estimators=trial.suggest_int('n_estimators',50,500),max_depth=trial.suggest_int('max_depth',2,8),learning_rate=trial.suggest_float('learning_rate',.01,.3,log=True),subsample=trial.suggest_float('subsample',.5,1),colsample_bytree=trial.suggest_float('colsample_bytree',.5,1),reg_alpha=trial.suggest_float('reg_alpha',1e-3,10,log=True),reg_lambda=trial.suggest_float('reg_lambda',1e-3,10,log=True),random_state=SEED,n_jobs=2,verbosity=0)

CLASSES=dict(Ridge=Ridge,Lasso=Lasso,ElasticNet=ElasticNet,MLP=MLPRegressor,RF=RandomForestRegressor,XGBoost=XGBRegressor)

def prepare(xtr,xte,transform,scale):
    if transform=='log':
        assert (xtr>0).all() and (xte>0).all()
        xtr,xte=np.log10(xtr),np.log10(xte)
    if scale:
        sc=StandardScaler().fit(xtr);return sc.transform(xtr),sc.transform(xte)
    return xtr,xte

def optimise(x,y,model,transform,score,n_trials):
    configs={}; warning_count=0
    def objective(trial):
        nonlocal warning_count
        p=params(trial,model);configs[trial.number]=p;vals=[]
        for tr,va in KFold(3,shuffle=True,random_state=SEED).split(x):
            xt,xv=prepare(x[tr],x[va],transform,model not in ['RF','XGBoost'])
            m=CLASSES[model](**p)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');m.fit(xt,y[tr])
            warning_count+=len(caught)
            pred=m.predict(xv)
            vals.append(r2_score(y[va],pred) if score=='R2' else -np.sqrt(mean_squared_error(y[va],pred)))
        return float(np.mean(vals))
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective,n_trials=n_trials)
    return study,configs,warning_count

def save_predictions(path,tag,fold,indices,y,pred,extra=None):
    rows=pd.DataFrame(dict(dataset=tag,fold=fold,idx=indices,y_true=y,y_pred=pred))
    if extra:
        for key,value in extra.items():rows[key]=value
    rows.to_csv(path,index=False)

def sensitivity(kind,data):
    summary=[]
    for tag in ['A','B']:
        x,y,g=data[tag]
        for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=SEED).split(x)):
            jobs=[(m,t,'R2',30) for m in ['Ridge','Lasso','ElasticNet','MLP'] for t in ['raw','log']] if kind=='transform' else [(m,'raw',s,100) for m in ['RF','XGBoost'] for s in ['R2','RMSE']]
            for model,transform,score,budget in jobs:
                stem=f'{kind}_{tag}_{fold}_{model}_{transform}_{score}'
                status=OUT/(stem+'.json')
                if status.exists():summary.extend(json.loads(status.read_text())['results']);continue
                start=time.time();study,configs,nwarnings=optimise(x[tr],y[tr],model,transform,score,budget)
                study.trials_dataframe().to_csv(OUT/(stem+'_trials.csv'),index=False)
                results=[]
                for n in ([30,100] if budget==100 else [30]):
                    best=max(study.trials[:n],key=lambda t:t.value)
                    p=configs[best.number];xt,xv=prepare(x[tr],x[te],transform,model not in ['RF','XGBoost'])
                    m=CLASSES[model](**p)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always');m.fit(xt,y[tr])
                    pred=m.predict(xv);row=dict(kind=kind,dataset=tag,fold=fold,model=model,transform=transform,score=score,budget=n,best_trial=best.number,inner_score=best.value,warnings=nwarnings+len(caught),**metrics(y[te],pred));results.append(row)
                    save_predictions(OUT/(stem+f'_{n}_predictions.csv'),tag,fold,te,y[te],pred)
                status.write_text(json.dumps(dict(results=results,selected_params={str(n):configs[max(study.trials[:n],key=lambda t:t.value).number] for n in ([30,100] if budget==100 else [30])},seconds=time.time()-start),indent=2))
                summary.extend(results);pd.DataFrame(summary).to_csv(OUT/(kind+'_summary_folds.csv'),index=False)
                print(stem,'completed',round(time.time()-start,1),'s',flush=True)
    pd.DataFrame(summary).to_csv(OUT/(kind+'_summary_folds.csv'),index=False)

def foundation(data):
    import torch,tabpfn
    from tabpfn import TabPFNRegressor
    torch.set_num_threads(2)
    if MODEL_PATH is None:
        raise ValueError('Set --model-path or TABPFN_MODEL_PATH to the TabPFN-2.5 regression checkpoint.')
    weight=Path(MODEL_PATH).expanduser().resolve()
    if not weight.is_file():
        raise FileNotFoundError(weight)
    (OUT/'runtime.json').write_text(json.dumps(dict(tabpfn=tabpfn.__version__,weight=str(weight),weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),n_estimators=8,device='cpu',seed=SEED),indent=2))
    def make():return TabPFNRegressor(model_path=weight,n_estimators=8,random_state=SEED,device='cpu')
    rows=[]
    for tag,(x,y,groups) in data.items():
        for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=SEED).split(x)):
            stem=f'TabPFN_{tag}_{fold}';status=OUT/(stem+'.json')
            if status.exists():rows.extend(json.loads(status.read_text()));continue
            start=time.time();m=make();m.fit(x[tr],y[tr]);pred=m.predict(x[te]);result=[dict(dataset=tag,kind='new_random_cv',fold=fold,model='TabPFN',**metrics(y[te],pred))]
            save_predictions(OUT/(stem+'_predictions.csv'),tag,fold,te,y[te],pred)
            if tag in ['A','Korea']:
                proper,cal=train_test_split(tr,test_size=15,random_state=SEED+fold)
                cm=make();cm.fit(x[proper],y[proper]);res=np.abs(y[cal]-cm.predict(x[cal]));k=int(np.ceil((len(cal)+1)*.9));q=np.sort(res)[k-1]
                cp=cm.predict(x[te]);lo=cp-q;hi=cp+q
                result.append(dict(dataset=tag,kind='split_conformal',fold=fold,model='TabPFN',n_cal=len(cal),rank=k,coverage=float(np.mean((y[te]>=lo)&(y[te]<=hi))),width=float(2*q),**metrics(y[te],cp)))
                save_predictions(OUT/(stem+'_split_conformal.csv'),tag,fold,te,y[te],cp,dict(lo=lo,hi=hi,n_cal=len(cal)))
            status.write_text(json.dumps(result,indent=2));rows.extend(result);pd.DataFrame(rows).to_csv(OUT/'foundation_summary.csv',index=False);print(stem,'done',round(time.time()-start,1),'s',flush=True)
        for fold,(tr,te) in enumerate(LeaveOneGroupOut().split(x,y,groups)):
            stem=f'Grouped_{tag}_{fold}';status=OUT/(stem+'.json')
            if status.exists():rows.extend(json.loads(status.read_text()));continue
            start=time.time();result=[]
            for name in ['TabPFN','RF']:
                if name=='TabPFN':m=make()
                else:
                    study,configs,nw=optimise(x[tr],y[tr],'RF','raw','R2',30);m=RandomForestRegressor(**configs[study.best_trial.number]);study.trials_dataframe().to_csv(OUT/(stem+'_RF_trials.csv'),index=False)
                m.fit(x[tr],y[tr]);pred=m.predict(x[te]);result.append(dict(dataset=tag,kind='region_holdout',fold=fold,region=str(groups[te[0]]),model=name,n_test=len(te),**metrics(y[te],pred)))
                save_predictions(OUT/(stem+'_'+name+'_predictions.csv'),tag,fold,te,y[te],pred)
            status.write_text(json.dumps(result,indent=2));rows.extend(result);pd.DataFrame(rows).to_csv(OUT/'foundation_summary.csv',index=False);print(stem,'done',round(time.time()-start,1),'s',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'foundation_summary.csv',index=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--part',choices=['transform','budget','foundation'],required=True)
    ap.add_argument('--output-dir',type=Path,default=ROOT/'rerun_results')
    ap.add_argument('--model-path',default=os.environ.get('TABPFN_MODEL_PATH'))
    args=ap.parse_args()
    OUT=args.output_dir.expanduser().resolve()
    if OUT==ROOT/'results' or OUT.is_relative_to(ROOT/'results'):
        ap.error('Choose a rerun directory outside the archived results directory.')
    MODEL_PATH=args.model_path
    OUT.mkdir(parents=True,exist_ok=True)
    data=datasets()
    if args.part=='foundation':foundation(data)
    else:sensitivity(args.part,data)
