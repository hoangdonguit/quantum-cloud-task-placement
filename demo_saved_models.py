"""Replay saved models; root-level file leaves frozen training fingerprint unchanged."""
import argparse,hashlib,json,time,html,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
 import torch
 torch.set_num_threads(1)
 from src.env import QCloudSim
 from src.heuristics import REGISTRY
 from experiments.evaluate import load_agent
 from experiments.baseline_tuning import GreedyScoreTuned
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT/'results/official');ap.add_argument('--out',type=Path,default=None);ap.add_argument('--train-seed',type=int,default=0);ap.add_argument('--workload-seed',type=int,default=30000);ap.add_argument('--cases',nargs='+',default=['base','rate_r0.5','tasks_t600']);a=ap.parse_args()
 out=a.out or ROOT/'results'/('demo_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True,exist_ok=False)
 payload={'train_seed':a.train_seed,'workload_seed':a.workload_seed,'cases':[]};provenance={}
 for case in a.cases:
  cp=a.root/'ladder'/case;cfg=json.loads((cp/'case_config.json').read_text());name=json.loads((cp/'baseline_selection.json').read_text())['selected']
  if name.startswith('GreedyScoreTuned_m'):
   m,l=name.split('_m')[1].split('_l');baseline=GreedyScoreTuned(float(m),float(l))
  else:baseline=REGISTRY[name]()
  policies=[('SelectedBaseline',baseline),('GreedyFlow',REGISTRY['GreedyFlow']())]
  for algo in ['PPO','DQN','DDQN3']:
   run=cp/f'{algo}_seed{a.train_seed}';weight=run/('model.zip' if algo=='PPO' else 'model.pt');provenance[str(weight.relative_to(a.root))]=hashlib.sha256(weight.read_bytes()).hexdigest();policies.append((algo,load_agent(algo,run,cfg)))
  result={'name':case,'config':cfg['env'],'baseline':name,'policies':[]};sim=QCloudSim(cfg['env'],'test');workload=sim.make_workload(a.workload_seed)
  result['workload_sha256']=hashlib.sha256(np.asarray(workload[0]).tobytes()+np.asarray(workload[1]).tobytes()).hexdigest();provenance[str((cp/'case_config.json').relative_to(a.root))]=hashlib.sha256((cp/'case_config.json').read_bytes()).hexdigest()
  for label,actor in policies:
   obs=sim.reset(workload);learned=label in ['PPO','DQN','DDQN3']
   if not learned:actor.reset(sim,a.workload_seed)
   done=False;trace=[]
   while not done:
    v=sim.view();mask=sim.action_masks();idx=sim.i;nodes=[{'name':n['key'],'feasible':bool(mask[j]),'wait':float(v['wait'][j]),'exec':float(v['exec'][j]) if mask[j] else None,'frel':float(v['frel'][j]),'qlen':float(v['qlen'][j])} for j,n in enumerate(sim.nodes)]
    action=int(actor(obs,mask) if learned else actor(sim));assert mask[action]
    row={'task':idx,'width':int(sim.pool.width[v['row']]),'chosen':sim.nodes[action]['key'],'nodes':nodes};obs,reward,done,info=sim.step(action);row.update(reward=float(reward),T=float(info['T']));trace.append(row)
   metrics=sim.metrics()
   if learned:
    import pandas as pd
    ref=pd.read_csv(cp/f'{label}_seed{a.train_seed}'/'eval_per_workload.csv').set_index('workload_seed')
    if a.workload_seed in ref.index:
     for k in ['score','mean_T','mean_F','mean_frel']:assert np.isclose(metrics[k],ref.loc[a.workload_seed,k],rtol=1e-6,atol=1e-8),(case,label,k)
   result['policies'].append({'name':label,'metrics':metrics,'trace':trace});print(case,label,round(metrics['score'],5),flush=True)
  payload['cases'].append(result)
 (out/'replay.json').write_text(json.dumps(payload,allow_nan=False));(out/'provenance.json').write_text(json.dumps(provenance,indent=2))
 template=(ROOT/'docs/demo_viewer.html').read_text();data=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c');(out/'demo.html').write_text(template.replace('/*REPLAY_DATA*/ null',data))
 (out/'COMPLETE.json').write_text(json.dumps({'cases':a.cases,'train_seed':a.train_seed,'workload_seed':a.workload_seed}))
 print('OPEN:',(out/'demo.html').resolve(),flush=True)
if __name__=='__main__':main()
