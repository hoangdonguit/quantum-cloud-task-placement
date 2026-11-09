import argparse,json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();root=Path(a.root);rows=[]
for case in sorted(root.iterdir()):
 if not (case/'case_config.json').exists():continue
 frames=[]
 for p in [*case.glob('*/eval_per_workload.csv'),*case.glob('heuristics_per_workload.csv')]:frames.append(pd.read_csv(p))
 if (case/'selected_baseline_test.csv').exists():
  frames.append(pd.read_csv(case/'selected_baseline_test.csv').assign(policy='SelectedBaseline',train_seed=-1,kind='heuristic'))
 df=pd.concat(frames,ignore_index=True)
 cfg=json.loads((case/'case_config.json').read_text())
 for policy,g in df.groupby('policy'):
  row={'case':case.name,'policy':policy,'n_training_seeds':g[g.train_seed>=0].train_seed.nunique(),'n_workloads':g.workload_seed.nunique()}
  row.update(width=cfg['env']['max_width'],tasks=cfg['env']['n_tasks'],steps=cfg['train']['timesteps'],rate=cfg['env']['arrival_rate'],nodes=len(cfg['env']['node_names']))
  for m in ['score','mean_T','p95_T','p99_T','mean_wait','mean_exec','mean_F','mean_frel','resched','time_over_scale','makespan']:
   if m not in g:continue
   row[m+'_mean']=g[m].mean();row[m+'_seed_sd']=g.groupby('train_seed')[m].mean().std();row[m+'_workload_sd']=g.groupby('workload_seed')[m].mean().std()
  rows.append(row)
out=root/'tables';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'all_metrics_mean_sd.csv',index=False)
print('Wrote',out/'all_metrics_mean_sd.csv')

summary=pd.DataFrame(rows);figdir=root/'figures';figdir.mkdir(exist_ok=True)
for axis in ['width','tasks','steps','rate','nodes']:
 sub=summary[(summary.case=='base')|summary.case.str.startswith(axis+'_')]
 for metric in ['score','mean_T','mean_F']:
  plt.figure(figsize=(7,4))
  for policy in ['PPO','DQN','DDQN3','SelectedBaseline','GreedyScore','GreedyFlow']:
   g=sub[sub.policy==policy].sort_values(axis)
   if g.empty:continue
   plt.plot(g[axis],g[metric+'_mean'],marker='o',label=policy)
   if policy in ['PPO','DQN','DDQN3']:
    plt.fill_between(g[axis],g[metric+'_mean']-g[metric+'_seed_sd'],g[metric+'_mean']+g[metric+'_seed_sd'],alpha=.12)
  plt.xlabel(axis);plt.ylabel(metric+' (DRL band: training-seed SD)');plt.legend(fontsize=7);plt.tight_layout();plt.savefig(figdir/f'ladder_{axis}_{metric}.png',dpi=150);plt.close()
