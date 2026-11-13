"""Validation-retuned OOD at .4 task/s. No training, no historical result writes."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy import stats
from src.env import QCloudSim
from src.provenance import run_fingerprint
from experiments.evaluate import eval_workloads,load_agent
from experiments.baseline_tuning import holm
from post_common import candidates,digest,workload_hash,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--root',type=Path,default=Path('results/official'));p.add_argument('--smoke',action='store_true');p.add_argument('--bootstrap',type=int,default=5000);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    base=a.root/'ladder/base';cfg=json.loads((base/'case_config.json').read_text());env=dict(cfg['env'],arrival_rate=.4)
    assert cfg['env']['arrival_rate']==.2
    val_seeds=list(range(5000,5003 if a.smoke else 5100));test_seeds=list(range(30000,30003 if a.smoke else 30200));ts=list(range(1 if a.smoke else 10))
    portfolio=candidates();frozen=json.loads((base/'baseline_selection.json').read_text())['selected']
    protocol={'mode':'smoke' if a.smoke else 'full','train_env':cfg['env'],'test_env':env,'validation_seeds':val_seeds,'test_seeds':test_seeds,'training_seeds':ts,'family':'supplementary OOD04 six comparisons, separate from Holm63','script_sha256':digest(__file__),'helper_sha256':digest(Path(__file__).with_name('post_common.py')),'base_config_sha256':digest(base/'case_config.json'),'base_fingerprint':run_fingerprint(cfg),'reuse':'Recompute inference; no unverified saved OOD scores reused','workloads':{str(s):workload_hash(QCloudSim(env,'test'),s) for s in test_seeds}}
    write_json(a.out/'protocol.json',protocol)
    scores={name:np.mean([r['score'] for r in eval_workloads(env,'val',val_seeds,policy_cls=actor)]) for name,actor in portfolio.items()}
    selected=max(scores,key=scores.get)
    # Freeze selection on disk before any test evaluation.
    write_json(a.out/'baseline_selection.json',{'frozen_base':frozen,'retuned_validation':selected,'validation_scores':scores,'smoke_only':a.smoke})
    rows=[]
    for label,name in [('FrozenBase',frozen),('Retuned04',selected)]:
        rows.extend(dict(policy=label,train_seed=-1,**r) for r in eval_workloads(env,'test',test_seeds,policy_cls=portfolio[name]))
    models={}
    for algo in ['PPO','DQN','DDQN3']:
        for seed in ts:
            run=base/f'{algo}_seed{seed}';meta=json.loads((run/'run_meta.json').read_text())
            if meta['fingerprint']!=protocol['base_fingerprint']:raise ValueError(f'Fingerprint mismatch: {run}')
            weight=run/('model.zip' if algo=='PPO' else 'model.pt');models[str(weight)]=digest(weight)
            rows.extend(dict(policy=algo,train_seed=seed,**r) for r in eval_workloads(env,'test',test_seeds,act_fn=load_agent(algo,run,cfg)))
    write_json(a.out/'model_hashes.json',models);df=pd.DataFrame(rows);df.to_csv(a.out/'per_workload.csv',index=False)
    rng=np.random.default_rng(409);comparison=[]
    for algo in ['PPO','DQN','DDQN3']:
        mat=df[df.policy==algo].pivot(index='train_seed',columns='workload_seed',values='score').reindex(index=ts,columns=test_seeds).to_numpy()
        assert np.isfinite(mat).all()
        for ref in ['FrozenBase','Retuned04']:
            h=df[df.policy==ref].set_index('workload_seed').reindex(test_seeds).score.to_numpy();d=mat-h
            boot=[d[np.ix_(rng.integers(len(ts),size=len(ts)),rng.integers(len(h),size=len(h)))].mean() for _ in range(a.bootstrap)]
            comparison.append(dict(algo=algo,baseline=ref,diff=d.mean(),ci_lo=np.quantile(boot,.025),ci_hi=np.quantile(boot,.975),p_seed_conditional=float(stats.ttest_1samp(d.mean(1),0).pvalue) if len(ts)>1 else np.nan))
    c=pd.DataFrame(comparison);c['p_holm6']=holm(c.p_seed_conditional.fillna(1));c.to_csv(a.out/'comparison.csv',index=False);print(c.to_string(index=False));write_json(a.out/'COMPLETE.json',{'smoke_only':a.smoke,'rows':len(df)})
if __name__=='__main__':main()
