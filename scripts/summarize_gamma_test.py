"""Validate and summarize the already completed paired gamma test evaluation."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from src.config import load_config
from src.env import QCloudSim
from src.provenance import run_fingerprint
from scripts.post_common import digest
METRICS=['score','mean_T','mean_wait','mean_exec','f_rel','resched']
SOURCE_COL={'score':'score','mean_T':'mean_T','mean_wait':'mean_wait','mean_exec':'mean_exec','f_rel':'mean_frel','resched':'resched'}
def require(df,cols,label):
 missing=[c for c in cols if c not in df.columns]
 if missing:raise ValueError(f'{label}: expected columns {missing}; actual header {list(df.columns)}')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();root=a.root;out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
 pr=json.loads((root/'protocol.json').read_text());done=json.loads((root/'COMPLETE.json').read_text());cfg=load_config('results_summary/cases/tasks_t600/case_config.json');pilot=json.loads(Path('results/gamma_t600_pilot/COMPLETE.json').read_text())
 assert done['test_evaluated'] and not done['smoke_only'] and done['rows']==1200 and done['n_workloads']==200 and done['n_checkpoints']==6 and pilot['selected_gamma']==.99
 assert pr['source_fingerprint']==run_fingerprint(cfg) and pr['test_evaluated'] is True
 d=pd.read_csv(root/'per_workload.csv');require(d,['gamma','train_seed','workload_seed']+[SOURCE_COL[m] for m in METRICS],'per_workload.csv')
 assert set(d.gamma)=={.9,.99} and set(d.train_seed)=={0,1,2} and set(d.workload_seed)==set(range(30000,30200))
 if d.duplicated(['gamma','train_seed','workload_seed']).any():raise ValueError('Duplicate gamma/seed/workload rows')
 for g in [.9,.99]:
  q=d[d.gamma==g];counts=q.groupby('train_seed').workload_seed.agg(['count','nunique']);assert counts.equals(pd.DataFrame({'count':[200]*3,'nunique':[200]*3},index=pd.Index([0,1,2],name='train_seed')))
 sim=QCloudSim(cfg['env'],'test');hashes={str(s):hashlib.sha256(x.tobytes()+t.tobytes()).hexdigest() for s in range(30000,30200) for x,t in [sim.make_workload(s)]};assert hashes==pr['workload_hashes']
 for g in [.9,.99]:
  for seed in range(3):
   f=Path('results/gamma_t600_pilot')/f'gamma_{g:.2f}'/f'DQN_seed{seed}'/'model.pt';assert digest(f)==pr['checkpoints'][f'{g:.2f}/seed{seed}']['sha256']
 # Explicit wide columns avoid pandas suffix and MultiIndex ambiguity.
 renamed=d.rename(columns={SOURCE_COL[m]:f'{m}_dqn' for m in METRICS})
 left=renamed[renamed.gamma==.99].drop(columns='gamma');right=renamed[renamed.gamma==.9].drop(columns='gamma')
 pair=left.merge(right,on=['train_seed','workload_seed'],validate='one_to_one',suffixes=('_099','_090'))
 assert len(pair)==600
 seedrows=[]
 for seed in range(3):
  sub=pair[pair.train_seed==seed];row={'train_seed':seed}
  for m in METRICS:row[f'{m}_090']=sub[f'{m}_dqn_090'].mean();row[f'{m}_099']=sub[f'{m}_dqn_099'].mean();row[f'delta_{m}']=row[f'{m}_099']-row[f'{m}_090']
  seedrows.append(row)
 pd.DataFrame(seedrows).to_csv(out/'paired_by_seed.csv',index=False)
 summary=pd.DataFrame([{'gamma':g,**{m:d.loc[d.gamma==g,SOURCE_COL[m]].mean() for m in METRICS}} for g in [.9,.99]])
 delta={'gamma':'delta_0.99_minus_0.90',**{m:summary.loc[1,m]-summary.loc[0,m] for m in METRICS}};summary=pd.concat([summary,pd.DataFrame([delta])],ignore_index=True);summary.to_csv(out/'gamma_summary.csv',index=False)
 rng=np.random.default_rng(409);ci=[]
 for m in METRICS:
  v=pair.groupby('workload_seed')[[f'{m}_dqn_099',f'{m}_dqn_090']].mean();dd=(v.iloc[:,0]-v.iloc[:,1]).to_numpy();boot=np.array([dd[rng.integers(200,size=200)].mean() for _ in range(5000)])
  ci.append({'metric':m,'mean_delta':dd.mean(),'ci95_low_workload_only':np.quantile(boot,.025),'ci95_high_workload_only':np.quantile(boot,.975),'scope':'resample 200 workloads; conditional on 3 training seeds used in gamma selection'})
 pd.DataFrame(ci).to_csv(out/'workload_bootstrap_ci.csv',index=False)
 basepath=Path('results/official/ladder/tasks_t600');baseline_verified=False
 basefile=basepath/'selected_baseline_test.csv'
 if basefile.exists():
  require(pd.read_csv(basefile),['workload_seed']+[SOURCE_COL[m] for m in METRICS],'selected_baseline_test.csv')
  bc=load_config(basepath/'case_config.json');b=pd.read_csv(basefile);selection=json.loads((basepath/'baseline_selection.json').read_text())['selected']
  assert bc==cfg and selection==json.loads((basepath/'baseline_selection.json').read_text())['selected']
  assert b.workload_seed.tolist()==list(range(30000,30200)) and set(b.workload_seed)==set(pr['test_workload_seeds'])
  b=b.rename(columns={SOURCE_COL[m]:f'{m}_baseline' for m in METRICS})
  for g in [.9,.99]:
   q=renamed[renamed.gamma==g].drop(columns='gamma');j=q.merge(b,on='workload_seed',validate='many_to_one');assert len(j)==600 and j.train_seed.nunique()==3
   for m in METRICS:j[f'{m}_delta_vs_baseline']=j[f'{m}_dqn']-j[f'{m}_baseline']
   j.to_csv(out/f'gamma_{g:.2f}_vs_selected_baseline_per_seed_workload.csv',index=False)
  rows=[]
  for name,x in [('SelectedBaseline',b),('DQN gamma 0.90',renamed[renamed.gamma==.9]),('DQN gamma 0.99',renamed[renamed.gamma==.99])]:
   rows.append({'policy':name,**{m:x[f'{m}_baseline' if name=='SelectedBaseline' else f'{m}_dqn'].mean() for m in METRICS}})
  desc=pd.DataFrame(rows)
  for g,name in [(.9,'DQN gamma 0.90'),(.99,'DQN gamma 0.99')]:
   for m in METRICS:desc.loc[desc.policy==name,f'delta_{m}_vs_baseline']=desc.loc[desc.policy==name,m].iloc[0]-desc.loc[desc.policy=='SelectedBaseline',m].iloc[0]
  desc.to_csv(out/'descriptive_vs_selected_baseline.csv',index=False);baseline_verified=True
 inputs={'evaluation_csv_sha256':digest(root/'per_workload.csv'),'protocol_sha256':digest(root/'protocol.json'),'pilot_complete_sha256':digest('results/gamma_t600_pilot/COMPLETE.json'),'config_sha256':digest('results_summary/cases/tasks_t600/case_config.json'),'baseline_csv_sha256':digest(basefile) if baseline_verified else None,'baseline_verified':baseline_verified,'baseline_selection':selection if baseline_verified else None,'analyzer_sha256':digest(__file__)}
 (out/'input_sources.json').write_text(json.dumps(inputs,indent=2));print(pd.DataFrame(seedrows).to_string(index=False));print(summary.to_string(index=False));print('BASELINE_VERIFIED',baseline_verified)
if __name__=='__main__':main()
