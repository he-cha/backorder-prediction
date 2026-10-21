"""Export analysis charts. Run with --sample pointing to trusted sample-data.pkl.
Only aggregates are exported; no source rows or fitted models are published.
"""
import argparse
import hashlib
import json
from pathlib import Path
from improve_models import features, metrics
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/visualizations'
TEAL='#167d8d'; COPPER='#bd653f'; NAVY='#17334c'; GRAY='#657482'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':NAVY,'text.color':NAVY,'axes.titleweight':'bold','savefig.facecolor':'#ffffff','svg.fonttype':'none'})

def canvas(title, subtitle):
    fig,ax=plt.subplots(figsize=(11,6.5))
    fig.subplots_adjust(left=.12,right=.95,bottom=.23,top=.77)
    fig.text(.08,.94,title,fontsize=21,weight='bold')
    fig.text(.08,.87,subtitle,fontsize=11,color=GRAY)
    ax.set_axisbelow(True)
    return fig,ax

def save(fig,name,note):
    fig.text(.08,.065,note,fontsize=9,color=GRAY,va='bottom')
    for ext in ['png','svg']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=160)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sample',required=True);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    report=json.loads((ROOT/'reports/sample_benchmark.json').read_text())
    assert hashlib.sha256(Path(args.sample).read_bytes()).hexdigest()==report['source_sha256'], 'Use the benchmark source sample.'
    X,y=joblib.load(args.sample);X=X.reset_index(drop=True);y=y.reset_index(drop=True)
    groups=pd.util.hash_pandas_object(X,index=False).to_numpy()
    train,remaining=next(GroupShuffleSplit(n_splits=1,test_size=.4,random_state=2027).split(X,y,groups))
    v,t=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=2028).split(X.iloc[remaining],y.iloc[remaining],groups[remaining]))
    val,test=remaining[v],remaining[t]
    E=features(X)
    model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),RandomForestClassifier(n_estimators=240,min_samples_leaf=2,max_features=.8,class_weight=None,random_state=42,n_jobs=2))
    print('Reproducing selected model on frozen partitions...',flush=True)
    model.fit(E.iloc[train],y.iloc[train])
    cut=report['test']['selected']['tuned']['threshold']
    actual=metrics(y.iloc[test],model.predict_proba(E.iloc[test])[:,1],cut)
    for key in ['f1','average_precision','precision','recall']:
        assert np.isclose(actual[key],report['test']['selected']['tuned'][key],atol=1e-10),key
    assert actual['confusion_matrix']==report['test']['selected']['tuned']['confusion_matrix']
    p=model.predict_proba(E.iloc[val])[:,1]
    data={'scope':'Enriched sample; inherited preprocessing limitations apply. Historical evaluation is a different model and cohort.','source_sha256':report['source_sha256'],'validation_rows':len(val),'frozen_threshold':cut}
    # Inventory coverage is a descriptive association, never a population risk estimate.
    gap=E.forecast_3_month-E.stock_plus_transit
    bucket=pd.Series(np.select([gap<=0,(gap>0)&(E.stock_plus_transit>0)],[0,1],default=2))
    labels=['Stock covers forecast','Positive stock, below forecast','No positive stock, unmet forecast']
    rows=[]
    for i,label in enumerate(labels):
        mask=bucket==i
        rows.append({'group':label,'rows':int(mask.sum()),'backorders':int(y[mask].sum()),'share':float(y[mask].mean())})
    data['inventory_groups']=rows
    fig,ax=canvas('Where inventory falls short','Observed backorder share by stock + transit versus 3-month forecast | full enriched sample')
    fig.subplots_adjust(left=.25)
    bars=ax.barh(np.arange(3),[r['share'] for r in rows],color=[TEAL,COPPER,NAVY],height=.56)
    ax.set(yticks=np.arange(3),yticklabels=['Covers forecast','Stock below forecast','No positive stock*'],xlim=(0,1),xlabel='Backorder share within the sampled group')
    ax.invert_yaxis();ax.xaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='x',alpha=.15)
    for bar,row in zip(bars,rows):
        inside=row['share']>.4
        ax.text(bar.get_width()-.02 if inside else bar.get_width()+.02,bar.get_y()+bar.get_height()/2,f"{row['share']:.1%}  ·  n={row['rows']:,}",va='center',ha='right' if inside else 'left',color='white' if inside else NAVY,fontsize=10)
    save(fig,'01_inventory_coverage','*Forecast exceeds stock + transit, and stock + transit ≤ 0. Descriptive association; not causal.\nSample backorder prevalence is 25%; these percentages do not estimate real-world shortage risk.')
    # Threshold tradeoff on validation only: reserved sample test remains untouched for selection.
    thresholds=sorted(set(np.linspace(.05,.95,91).tolist()+[cut,.5]))
    sweep=[metrics(y.iloc[val],p,float(c)) for c in thresholds]
    data['validation_thresholds']=sweep
    fig,ax=canvas('Every alert threshold is an operating choice','Engineered Random Forest | validation partition only | 8,895 sampled records')
    for key,color in [('precision',TEAL),('recall',COPPER)]:ax.plot(thresholds,[r[key] for r in sweep],color=color,lw=2.6,label=key.capitalize())
    ax.plot(thresholds,[r['alerts']/len(val) for r in sweep],color=NAVY,lw=2,ls='--',label='Records flagged for review')
    ax.axvline(cut,color=GRAY,ls=':',label=f'Frozen F1 threshold ({cut:.3f})')
    ax.set(xlabel='Alert threshold (model score)',ylabel='Share',ylim=(0,1.04),xlim=(.05,.95));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.15);ax.legend(loc='lower left',fontsize=9)
    save(fig,'02_threshold_tradeoffs','Threshold was chosen by validation F1 in the existing benchmark, not by business cost.\nScores are not calibrated population probabilities. Curves reflect enriched-sample prevalence.')
    # Actual measured errors, matching cohorts and independently tuned thresholds.
    fig,ax=canvas('Fewer false alarms, with a measurable tradeoff','Same reserved sample test: 9,284 records | each threshold fixed on validation first')
    reference=report['test']['reference']['tuned'];selected=report['test']['selected']['tuned']
    for j,(label,m,color) in enumerate([('Reference RF',reference,COPPER),('Engineered RF',selected,TEAL)]):
        cm=m['confusion_matrix'];bars=ax.bar(np.arange(2)+(j-.5)*.34,[cm[0][1],cm[1][0]],.34,color=color,label=label);ax.bar_label(bars,padding=4)
    ax.set(xticks=[0,1],xticklabels=['False alarms','Missed backorders'],ylabel='Number of records',ylim=(0,760));ax.legend();ax.grid(axis='y',alpha=.15)
    save(fig,'03_error_tradeoff','False alarms: 637 → 574 (9.9% fewer). Missed backorders: 306 → 329 (+23).\nSample F1: 0.8079 → 0.8128. Average precision: 0.8799 → 0.8780; no broad superiority claim.')
    # Separate historical cohort, clear denominator and workload count.
    matrices=json.loads((ROOT/'reports/confusion_matrices.json').read_text())
    tn,fp,fn,tp=matrices['Random Forest — independent test']
    data['historical']={'rows':tn+fp+fn+tp,'true_positives':tp,'false_positives':fp,'false_negatives':fn,'true_negatives':tn}
    fig,ax=canvas('Catching shortages is only half the story','Historical Random Forest | independent test: 242,076 records, 1.11% backorders')
    ax.barh([0],[tp],color=TEAL,label='Actual backorders');ax.barh([0],[fp],left=[tp],color=COPPER,label='False alarms')
    ax.set(yticks=[],xlabel='Records flagged for review',xlim=(0,24000));ax.text(tp+fp+350,0,f'{tp+fp:,}\nalerts',va='center',weight='bold')
    ax.text(tp/2,0,f'{tp:,}',color='white',ha='center',va='center',fontsize=9);ax.text(tp+fp/2,0,f'{fp:,}',color='white',ha='center',va='center',weight='bold')
    ax.legend(loc='upper right');ax.text(.02,.12,f'Recall {tp/(tp+fn):.1%}  ·  Precision {tp/(tp+fp):.2%}  ·  {fn:,} missed backorders',transform=ax.transAxes,fontsize=13)
    save(fig,'04_historical_workload','Different model and cohort from the new sample benchmark; do not compare as a before/after improvement.\nAt historical prevalence, about 9 in 10 alerts were false alarms. No measured financial savings claimed.')
    fig,ax=canvas('Why sample scores need context','Backorder prevalence differs sharply across the available evaluation cohorts')
    rates=[11293/1687861,2688/242076,float(y.mean())]
    fig.subplots_adjust(left=.34)
    bars=ax.barh([0,1,2],rates,color=[GRAY,COPPER,TEAL],height=.55)
    ax.set(yticks=[0,1,2],yticklabels=['Original training data','Historical independent test','Enriched development sample'],xlabel='Backorder prevalence',xlim=(0,.32));ax.invert_yaxis();ax.xaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='x',alpha=.15)
    for b,r in zip(bars,rates):ax.text(r+.006,b.get_y()+b.get_height()/2,f'{r:.2%}',va='center')
    save(fig,'05_prevalence_context','Original training counts come from the historical notebook; the raw full dataset is unavailable here.\nEnriched sampling changes precision and review workload. New sample scores are not production estimates.')
    (OUT/'chart_data.json').write_text(json.dumps(data,indent=2))
    print('Exported 5 PNG + 5 editable SVG charts; model metrics reproduce benchmark.',flush=True)

if __name__=='__main__':main()
