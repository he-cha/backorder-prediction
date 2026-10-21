"""Sample-only benchmark. Never uses the historical independent test for tuning.
Run from repo root: python scripts/improve_models.py --sample /path/sample-data.pkl
Only deserialize trusted local pickle/joblib files.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse, hashlib, json, platform, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]

def features(X):
    X=X.copy()
    for c in ['perf_6_month_avg','perf_12_month_avg']:
        X[c]=X[c].replace(-99,np.nan)
    X['stock_plus_transit']=X.national_inv+X.in_transit_qty
    for months in [3,6,9]:
        demand=X[f'forecast_{months}_month']
        X[f'coverage_{months}m']=X.stock_plus_transit/(demand+1)
        X[f'shortfall_{months}m']=demand-X.stock_plus_transit
    X['stock_vs_minimum']=X.national_inv-X.min_bank
    X['sales_acceleration']=X.sales_1_month-X.sales_3_month/3
    X['forecast_vs_recent_sales']=X.forecast_3_month-3*X.sales_1_month
    X['supplier_performance_change']=X.perf_6_month_avg-X.perf_12_month_avg
    return X.replace([np.inf,-np.inf],np.nan)

def threshold(y,p):
    precision,recall,t=precision_recall_curve(y,p)
    f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
    return float(t[np.argmax(f)])

def metrics(y,p,t):
    pred=p>=t
    return dict(threshold=t,precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred)),f1=float(f1_score(y,pred)),average_precision=float(average_precision_score(y,p)),roc_auc=float(roc_auc_score(y,p)),confusion_matrix=confusion_matrix(y,pred).tolist(),alerts=int(pred.sum()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sample',required=True);args=ap.parse_args()
    X,y=joblib.load(args.sample); X=X.reset_index(drop=True);y=y.reset_index(drop=True)
    assert len(X)==len(y) and set(y.unique())=={0,1}
    groups=pd.util.hash_pandas_object(X,index=False).to_numpy()
    train,remaining=next(GroupShuffleSplit(n_splits=1,test_size=.4,random_state=2027).split(X,y,groups))
    v,t=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=2028).split(X.iloc[remaining],y.iloc[remaining],groups[remaining]))
    val,test=remaining[v],remaining[t]
    for a,b in [(train,val),(train,test),(val,test)]:assert not set(groups[a])&set(groups[b])
    print('Split sizes:',len(train),len(val),len(test),flush=True)
    E=features(X)
    configs=[
        ('Reference RF + SelectKBest',False,make_pipeline(SimpleImputer(strategy='median'),SelectKBest(lambda a,b: mutual_info_classif(a,b,random_state=42),k=20),RandomForestClassifier(n_estimators=100,min_samples_leaf=5,class_weight='balanced',random_state=42,n_jobs=2))),
        ('Engineered Random Forest',True,make_pipeline(SimpleImputer(strategy='median',add_indicator=True),RandomForestClassifier(n_estimators=240,min_samples_leaf=2,max_features=.8,class_weight=None,random_state=42,n_jobs=2))),
        ('Histogram gradient boosting',False,make_pipeline(SimpleImputer(strategy='median',add_indicator=True),HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=15,learning_rate=.08,l2_regularization=5,early_stopping=False,random_state=42))),
        ('Engineered histogram boosting',True,make_pipeline(SimpleImputer(strategy='median',add_indicator=True),HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=15,learning_rate=.08,l2_regularization=5,early_stopping=False,random_state=42))),
    ]
    records=[];fitted=[]
    for name,eng,model in configs:
        frame=E if eng else X;start=time.time()
        model.fit(frame.iloc[train],y.iloc[train]);p=model.predict_proba(frame.iloc[val])[:,1]
        cut=threshold(y.iloc[val],p);m=metrics(y.iloc[val],p,cut)
        records.append(dict(name=name,engineered=eng,validation=m,seconds=round(time.time()-start,2)))
        fitted.append((model,frame))
        print(name,records[-1],flush=True)
    # Freeze model and threshold using validation AP; inspect test only after selection.
    winner=int(np.argmax([r['validation']['average_precision'] for r in records]))
    test_results={}
    predictions={}
    for idx,label in [(0,'reference'),(winner,'selected')]:
        model,frame=fitted[idx];p=model.predict_proba(frame.iloc[test])[:,1];cut=records[idx]['validation']['threshold']
        test_results[label]=dict(name=records[idx]['name'],tuned=metrics(y.iloc[test],p,cut),default=metrics(y.iloc[test],p,.5))
        predictions[label]=p
    report=dict(scope='Reused enriched sample only; not an independent production evaluation',selection='Highest validation average precision; threshold maximizes validation F1',seed=2027,grouping='Identical predictor rows kept in the same partition',source_sha256=hashlib.sha256(Path(args.sample).read_bytes()).hexdigest(),versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__),partitions={k:dict(rows=len(ix),positives=int(y.iloc[ix].sum()),prevalence=float(y.iloc[ix].mean())) for k,ix in [('train',train),('validation',val),('test',test)]},candidates=records,selected=records[winner]['name'],test=test_results)
    (ROOT/'reports/sample_benchmark.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for label,color in [('reference','#9c663e'),('selected','#137e82')]:
        pr,rc,_=precision_recall_curve(y.iloc[test],predictions[label]);axes[0].plot(rc,pr,label=label.capitalize(),color=color,lw=2)
    axes[0].axhline(y.iloc[test].mean(),color='gray',ls=':',label='Sample prevalence');axes[0].set(xlabel='Recall',ylabel='Precision',title='Precision–recall on reserved sample test');axes[0].legend()
    vals=[[test_results[k]['tuned'][m] for m in ['precision','recall','f1']] for k in ['reference','selected']]
    for j,(label,color) in enumerate([('Reference','#9c663e'),('Selected','#137e82')]):
        bars=axes[1].bar(np.arange(3)+(j-.5)*.34,vals[j],.34,label=label,color=color);axes[1].bar_label(bars,fmt='%.3f',padding=3)
    axes[1].set(xticks=np.arange(3),xticklabels=['Precision','Recall','F1'],ylim=(0,1.08),title='Validation-tuned thresholds');axes[1].legend(loc='lower right')
    fig.suptitle('Backorder prediction | Sample-only benchmark',fontsize=16,fontweight='bold')
    fig.savefig(ROOT/'reports/sample_benchmark.png',dpi=170)
    print(json.dumps(test_results,indent=2),flush=True)

if __name__=='__main__':main()
