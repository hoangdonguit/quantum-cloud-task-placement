"""Frozen official protocol. Tuning never evaluates test workloads."""
import argparse,json,sys,subprocess,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.config import load_config,deep_update,build_ladder
from src.provenance import run_fingerprint

def call(args,log=None):
 cmd=[sys.executable,*map(str,args)]
 if log:
  with open(log,'w') as f:subprocess.run(cmd,cwd=ROOT,check=True,stdout=f,stderr=subprocess.STDOUT)
 else:subprocess.run(cmd,cwd=ROOT,check=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=4);ap.add_argument('--check',action='store_true');ap.add_argument('--root',default='results/official');a=ap.parse_args()
 cfg=load_config(ROOT/'configs/official.json');cases=build_ladder(cfg);print(f'OFFICIAL: {len(cases)} cases x 3 algorithms x 10 training seeds = {len(cases)*30} training runs; 200 paired test workloads/case',flush=True)
 print('TUNING: 4 candidates x 3 seeds x 3 anchor cases x 3 algorithms =108 validation-only runs, 20000 steps each',flush=True)
 for c in cases:print(c['name'],c['cfg']['env']['max_width'],c['cfg']['env']['n_tasks'],c['cfg']['train']['timesteps'],flush=True)
 if a.check:
  call(['-m','unittest','discover','-s','tests','-v']);return
 out=ROOT/a.root;out.mkdir(parents=True,exist_ok=True)
 import fcntl
 lock=open(out/'RUNNING.lock','w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('Another official pipeline is already running')
 fp=run_fingerprint(cfg);marker=out/'protocol_fingerprint.txt'
 if marker.exists() and marker.read_text()!=fp:raise RuntimeError('Protocol/source/data/runtime changed: use a new result root')
 marker.write_text(fp)
 tuning=out/'tuning';tuning.mkdir(exist_ok=True)
 grids={
 'PPO':[{'learning_rate':lr,'clip_range':clip} for lr,clip in [(1e-4,.2),(1e-4,.3),(3e-4,.2),(3e-4,.3)]],
 'DQN':[{'lr':lr,'gamma':g} for lr,g in [(1e-4,.9),(5e-4,.9),(1e-4,.99),(5e-4,.99)]],
 'DDQN3':[{'lr':lr,'gamma':g} for lr,g in [(1e-4,.9),(5e-4,.9),(1e-4,.99),(5e-4,.99)]]}
 anchors=[{}, {'env':{'max_width':16,'arrival_rate':.05}}, {'env':{'n_tasks':120,'arrival_rate':.5}}]
 tasks=[]
 for algo,grid in grids.items():
  for k,hp in enumerate(grid):
   for j,anchor in enumerate(anchors):
    for seed in (100,101,102):
     d=tuning/f'{algo}_c{k}_a{j}_s{seed}';d.mkdir(exist_ok=True)
     c=deep_update(cfg,anchor);c=deep_update(c,{'train':{'timesteps':20000,'eval_every':20000,'n_val':50},'algos':{algo:hp}})
     cp=d/'config.json';cp.write_text(json.dumps(c,indent=2));tasks.append((algo,k,j,seed,d,cp,c))
 def tune(t):
  algo,k,j,seed,d,cp,c=t;meta=d/'validation_meta.json'
  if meta.exists():
   if json.loads(meta.read_text()).get('fingerprint')!=run_fingerprint(c):raise RuntimeError('Incompatible tuning run '+str(d))
  else:call(['algorithms/train.py','--case-config',cp,'--algo',algo,'--seed',seed,'--out-dir',d,'--validation-only'],d/'train.log')
  score=json.loads(meta.read_text())['val_score'];print('TUNE',algo,k,j,seed,score,flush=True);return algo,k,score
 with ThreadPoolExecutor(a.jobs) as ex:res=list(ex.map(tune,tasks))
 selected={};scores={}
 for algo,grid in grids.items():
  values=[sum(v for al,k2,v in res if al==algo and k2==k)/9 for k in range(len(grid))]
  k=max(range(len(values)),key=lambda k:values[k]);selected[algo]=grid[k];scores[algo]=values
 tuned=deep_update(cfg,{'algos':selected});cp=out/'selected_config.json';cp.write_text(json.dumps(tuned,indent=2))
 (out/'tuning_selection.json').write_text(json.dumps({'selected':selected,'mean_validation_scores':scores,'candidate_grid':grids,'anchors':anchors,'training_seeds':[100,101,102],'steps_per_run':20000,'validation_workloads':50,'test_used':False},indent=2))
 mainroot=out/'ladder'
 call(['experiments/run_ladder.py','--config',cp,'--out',mainroot,'--seeds',*range(10),'--jobs',a.jobs])
 call(['experiments/report.py','--root',mainroot,'--n-val',100,'--bootstrap',5000])
 call(['experiments/summarize_metrics.py','--root',mainroot])
 call(['experiments/explain.py','--root',mainroot,'--case','base','--repeats',3])
 call(['experiments/ood.py','--root',mainroot,'--case','base'])
 quantum=out/'quantum_validation'
 if not (quantum/'COMPLETE.json').exists():
  if quantum.exists():
   import datetime
   quantum.rename(out/('quantum_incomplete_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S_%f')))
  call(['experiments/validate_quantum.py','--out',quantum,'--shots',8192,'--n-seeds',5])
  (quantum/'COMPLETE.json').write_text(json.dumps({'n_configs':360,'shots':8192,'seeds':5}))
 (out/'COMPLETE.json').write_text(json.dumps({'training_runs':len(cases)*30,'tuning_runs':108,'train_seeds':list(range(10)),'test_workloads':200,'fingerprint':fp},indent=2))
 call(['tools/package_results.py',out]);print('OFFICIAL COMPLETE',out,flush=True)
if __name__=='__main__':main()
