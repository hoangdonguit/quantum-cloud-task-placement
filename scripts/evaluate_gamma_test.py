"""Paired test evaluation of the locked tasks_t600 DQN gamma pilot checkpoints."""
import argparse,copy,hashlib,json,platform,sys
from pathlib import Path
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
 import numpy as np,pandas as pd
 from src.config import load_config
 from src.env import QCloudSim
 from src.provenance import run_fingerprint
 from experiments.evaluate import eval_seeds,eval_workloads
 from algorithms.dqn import DQNAgent
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 cfgpath=ROOT/'results_summary/cases/tasks_t600/case_config.json';cfg=load_config(cfgpath);pilot=ROOT/'results/gamma_t600_pilot'
 complete=json.loads((pilot/'COMPLETE.json').read_text());paired=pd.read_csv(pilot/'paired_validation.csv');summary=pd.read_csv(pilot/'validation_summary.csv')
 assert complete['runs']==9 and complete['smoke_only'] is False and complete['selected_gamma']==.99 and complete['test_evaluated'] is False
 assert len(paired)==9 and set(paired.gamma)=={.9,.95,.99} and len(summary)==3
 assert cfg['case']=='tasks_t600' and cfg['env']['n_tasks']==600
 fp=run_fingerprint(cfg);assert fp==json.loads((pilot/'protocol.json').read_text())['source_fingerprint']
 seeds=range(3);gammas=(.90,.99);ckpts={};configs={}
 for g in gammas:
  cc=copy.deepcopy(cfg);cc['algos']['DQN']['gamma']=g;configs[str(g)]=cc
  for seed in seeds:
   run=pilot/f'gamma_{g:.2f}'/f'DQN_seed{seed}';meta=json.loads((run/'pilot_meta.json').read_text());row=paired[(paired.gamma==g)&(paired.train_seed==seed)].iloc[0];weight=run/'model.pt'
   assert weight.is_file() and sha(weight)==row.checkpoint_sha256==meta['checkpoint_sha256']
   assert meta['config_fingerprint']==row.config_fingerprint and meta['actual_timesteps']==100000
   assert meta['n_validation_workloads']==20 and meta['test_evaluated'] is False
   run_cfg=json.loads((pilot/f'gamma_{g:.2f}'/'case_config.json').read_text());assert run_cfg==cc
   ckpts[f'{g:.2f}/seed{seed}']={'path':str(weight.relative_to(ROOT)),'sha256':sha(weight),'config_fingerprint':row.config_fingerprint}
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
 test_seeds=list(range(5000,5020)) if a.smoke else eval_seeds(cfg)
 if not a.smoke:assert len(test_seeds)==200
 else:test_seeds=test_seeds[:2]
 sim=QCloudSim(cfg['env'],'test');workloads={str(s):sim.make_workload(s) for s in test_seeds}
 hashes={str(s):hashlib.sha256(w[0].tobytes()+w[1].tobytes()).hexdigest() for s,w in workloads.items()}
 protocol={'protocol':'paired supplemental gamma test; primary contrast gamma .99 minus .90','case':'tasks_t600','config':cfg,'config_sha256':sha(cfgpath),'source_fingerprint':fp,'checkpoint_selection':'final checkpoints from validation pilot; gamma .99 locked by validation; seeds 0-2 participated in gamma selection','checkpoints':ckpts,'training_seeds':[0,1,2],'gammas':[.90,.99],'test_split':'test','test_workload_seeds':list(test_seeds),'workload_hashes':hashes,'n_workloads':len(test_seeds),'metrics':['score','mean_T','mean_wait','mean_exec','mean_frel','resched'],'paired_by':['train_seed','workload_seed'],'primary_difference':'gamma .99 minus gamma .90','family':'supplementary, outside Holm63','selected_baseline':'not included unless independently verified; gamma contrast is primary','smoke_only':a.smoke,'test_evaluated':not a.smoke,'code':{p:sha(ROOT/p) for p in ['scripts/evaluate_gamma_test.py','experiments/evaluate.py','algorithms/dqn.py','src/env.py']},'runtime':{'python':platform.python_version(),'packages':{p:version(p) for p in ['numpy','pandas','torch','gymnasium']}}}
 put(out/'protocol.json',protocol)
 rows=[]
 for g in gammas:
  for seed in seeds:
   agent=DQNAgent.load(pilot/f'gamma_{g:.2f}'/f'DQN_seed{seed}'/'model.pt')
   if agent.hp['gamma']!=g:raise ValueError('Checkpoint gamma mismatch')
   df=pd.DataFrame(eval_workloads(cfg['env'],'val' if a.smoke else 'test',test_seeds,act_fn=agent.predict));assert df.workload_seed.tolist()==list(test_seeds)
   df.insert(0,'gamma',g);df.insert(0,'train_seed',seed);rows.extend(df.to_dict('records'))
 outdf=pd.DataFrame(rows);metrics=['score','mean_T','mean_wait','mean_exec','mean_frel','resched']
 assert len(outdf)==len(gammas)*len(seeds)*len(test_seeds) and not outdf.duplicated(['gamma','train_seed','workload_seed']).any()
 outdf.to_csv(out/'per_workload.csv',index=False)
 piv=outdf.pivot(index=['train_seed','workload_seed'],columns='gamma',values=metrics)
 assert len(piv)==len(seeds)*len(test_seeds) and piv.notna().all().all()
 perseed=outdf.groupby(['gamma','train_seed'])[metrics].mean().reset_index();wide=perseed.pivot(index='train_seed',columns='gamma',values=metrics)
 diff=pd.DataFrame({'train_seed':wide.index})
 for m in metrics:diff[f'delta_{m}']=wide[(m,.99)].to_numpy()-wide[(m,.9)].to_numpy()
 diff.to_csv(out/'paired_by_seed.csv',index=False)
 rows=[]
 for m in metrics:
  for seed in seeds:
   rows.append({'train_seed':seed,'metric':m,'gamma_0.90':wide.loc[seed,(m,.9)],'gamma_0.99':wide.loc[seed,(m,.99)],'delta_0.99_minus_0.90':wide.loc[seed,(m,.99)]-wide.loc[seed,(m,.9)]})
 pd.DataFrame(rows).to_csv(out/'paired_metrics_by_seed.csv',index=False)
 overall=outdf.groupby('gamma')[metrics].mean();overall.loc['delta_0.99_minus_0.90']=overall.loc[.99]-overall.loc[.9];overall.to_csv(out/'summary.csv')
 # CI over workloads only, averaged across the three fixed training seeds; not uncertainty over training.
 ci=[];rng=np.random.default_rng(409);mat=outdf.pivot_table(index='workload_seed',columns=['gamma','train_seed'],values=metrics)
 for m in metrics:
  d=np.column_stack([mat[(m,.99,s)]-mat[(m,.9,s)] for s in seeds]).mean(axis=1)
  boot=np.array([d[rng.integers(len(d),size=len(d))].mean() for _ in range(5000)])
  ci.append({'metric':m,'mean_delta':d.mean(),'workload_bootstrap_ci95_low':np.quantile(boot,.025),'workload_bootstrap_ci95_high':np.quantile(boot,.975),'bootstrap_reps':5000,'scope':'resamples 200 workloads; conditions on these 3 selected training seeds'})
 pd.DataFrame(ci).to_csv(out/'workload_bootstrap_ci.csv',index=False)
 if a.smoke and list(test_seeds)!=[5000,5001]:raise AssertionError('Smoke must be validation only')
 if not a.smoke and list(test_seeds)!=list(range(30000,30200)):raise AssertionError('Unexpected official test workload IDs')
 put(out/'COMPLETE.json',{'smoke_only':a.smoke,'test_evaluated':not a.smoke,'n_checkpoints':6,'n_workloads':len(test_seeds),'rows':len(outdf),'all_gamma_seed_workload_pairs_complete':True,'primary_contrast':'.99-.90','protocol_sha256':sha(out/'protocol.json')})
 print('COMPLETE',out/'COMPLETE.json');print('PER_WORKLOAD',out/'per_workload.csv');print('BY_SEED',out/'paired_by_seed.csv')
if __name__=='__main__':main()
