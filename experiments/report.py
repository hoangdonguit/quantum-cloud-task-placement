"""Validation-selected baseline portfolio and paired crossed bootstrap over seeds/workloads."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from experiments.evaluate import eval_workloads,eval_seeds
from experiments.baseline_tuning import GRID,GreedyScoreTuned,holm,VAL_SEEDS
from src.heuristics import REGISTRY

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--bootstrap',type=int,default=2000);ap.add_argument('--n-val',type=int,default=30);a=ap.parse_args()
    root=Path(a.root);out=root/'tables';out.mkdir(exist_ok=True);rows=[];rng=np.random.default_rng(409)
    for case in sorted(root.iterdir()):
        if not (case/'case_config.json').exists():continue
        cfg=json.loads((case/'case_config.json').read_text());env=cfg['env'];val_seeds=list(range(5000,5000+a.n_val))
        candidates=dict(REGISTRY)
        for m,lam in GRID:candidates[f'GreedyScoreTuned_m{m}_l{lam}']=lambda m=m,lam=lam:GreedyScoreTuned(m,lam)
        val={n:np.mean([r['score'] for r in eval_workloads(env,'val',val_seeds,policy_cls=c)]) for n,c in candidates.items()}
        best=max(val,key=val.get);ref=eval_workloads(env,'test',eval_seeds(cfg),policy_cls=candidates[best]);h=np.array([x['score'] for x in ref])
        pd.DataFrame(ref).to_csv(case/'selected_baseline_test.csv',index=False)
        (case/'baseline_selection.json').write_text(json.dumps({'selected':best,'validation_scores':val},indent=2))
        for algo in cfg['algos']:
            paths=sorted(case.glob(f'{algo}_seed*/eval_per_workload.csv'))
            if not paths:continue
            mats=[]
            for p in paths:
                d=pd.read_csv(p).set_index('workload_seed').reindex(eval_seeds(cfg))
                if d.score.isna().any():raise ValueError(f'Missing workloads: {p}')
                mats.append(d.score.to_numpy())
            diff=np.stack(mats)-h[None,:];n,w=diff.shape
            boot=[diff[np.ix_(rng.integers(n,size=n),rng.integers(w,size=w))].mean() for _ in range(a.bootstrap)]
            # Seed t test is conditional on the fixed test workloads, not a population test over workloads.
            p=float(stats.ttest_1samp(diff.mean(axis=1),0).pvalue) if n>=2 else np.nan
            rows.append(dict(case=case.name,algo=algo,baseline=best,n_seeds=n,n_workloads=w,diff=diff.mean(),ci_lo=np.quantile(boot,.025),ci_hi=np.quantile(boot,.975),p_seed_conditional=p))
        figdir=root/'figures';figdir.mkdir(exist_ok=True)
        plt.figure(figsize=(7,4))
        for algo in cfg['algos']:
            curves=[pd.read_csv(p).assign(training_seed=p.parent.name) for p in sorted(case.glob(f'{algo}_seed*/train_log.csv'))]
            if curves:
                curves=pd.concat(curves);g=curves.groupby('timesteps').val_score.agg(['mean','std']).fillna(0)
                plt.plot(g.index,g['mean'],label=algo);plt.fill_between(g.index,g['mean']-g['std'],g['mean']+g['std'],alpha=.15)
        # Evaluate selected baseline on exactly the learning-curve validation workloads.
        curve_ref=np.mean([r['score'] for r in eval_workloads(env,'val',list(range(5000,5000+cfg['train']['n_val'])),policy_cls=candidates[best])])
        plt.axhline(curve_ref,color='black',linestyle='--',label='selected baseline / same validation workloads')
        plt.xlabel('actual environment steps');plt.ylabel('validation score');plt.legend(fontsize=7);plt.tight_layout();plt.savefig(figdir/f'learning_{case.name}.png',dpi=150);plt.close()
        print(case.name,'baseline:',best,flush=True)
    df=pd.DataFrame(rows);df['p_holm_seed_conditional']=holm(df.p_seed_conditional.fillna(1).to_numpy());df.to_csv(out/'comparison.csv',index=False);print(df.to_string(index=False))
if __name__=='__main__':main()
