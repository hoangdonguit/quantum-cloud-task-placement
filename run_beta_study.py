"""Optional, post-hoc beta sensitivity for DQN only. No feature ablation here."""
import argparse,copy,datetime,json,os,subprocess,sys,time
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=10);p.add_argument('--deadline',default='19:00');a=p.parse_args()
 if not 5<=a.seeds<=10:p.error('Use 5..10 seeds; five is preliminary, ten preferred')
 now=datetime.datetime.now(ZoneInfo('Asia/Ho_Chi_Minh'));hh,mm=map(int,a.deadline.split(':'));deadline=now.replace(hour=hh,minute=mm,second=0,microsecond=0)
 out=ROOT/'results'/('beta_study_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True,exist_ok=False)
 base=json.loads((ROOT/'results/official/ladder/base/case_config.json').read_text());tasks=[]
 for beta in [.5,2.]:
  case=out/f'beta_{beta:g}';case.mkdir();cfg=copy.deepcopy(base);cfg['env']['beta']=beta;cfg['algos']={'DQN':cfg['algos']['DQN']};cfg['train']['timesteps']=100000
  (case/'case_config.json').write_text(json.dumps(cfg,indent=2))
  for seed in range(a.seeds):tasks.append((case,seed))
 (out/'protocol.json').write_text(json.dumps({'hypothesis':'Sensitivity to beta, exploratory after main results','algorithm':'DQN','beta':[.5,2.],'baseline_beta1':'existing official/base; do not pool raw reward across beta','seeds':a.seeds,'steps':100000,'workloads':200,'deadline':deadline.isoformat()},indent=2))
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');duration=None
 for ix,(case,seed) in enumerate(tasks):
  left=(deadline-datetime.datetime.now(ZoneInfo('Asia/Ho_Chi_Minh'))).total_seconds();reserve=1800
  needed=(len(tasks)-ix)*(duration*1.5 if duration else 180)+reserve
  if left<needed:
   print('STOP: projected work plus 30-minute analysis reserve exceeds deadline. Keep main report; partial beta is not a completed study.',out,flush=True);return
  run=case/f'DQN_seed{seed}';run.mkdir();t=time.monotonic()
  with (run/'train.log').open('w') as f:
   subprocess.run([sys.executable,str(ROOT/'algorithms/train.py'),'--case-config',str(case/'case_config.json'),'--algo','DQN','--seed',str(seed),'--out-dir',str(run)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=max(1,left-reserve))
  elapsed=time.monotonic()-t;duration=max(duration or 0,elapsed);print('DONE',case.name,seed,'wall seconds',round(elapsed,1),'conservative remaining minutes',round((len(tasks)-ix-1)*duration*1.5/60,1),flush=True)
 # Main script tunes baseline per case on validation only, then evaluates paired test workloads.
 with (out/'analysis.log').open('w') as f:
  subprocess.run([sys.executable,str(ROOT/'experiments/report.py'),'--root',str(out),'--n-val','100','--bootstrap','5000'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)
 (out/'COMPLETE.json').write_text(json.dumps({'n_training_runs':len(tasks),'algorithm':'DQN','n_seeds':a.seeds,'exploratory':True}))
 print('BETA COMPLETE:',out,flush=True)
if __name__=='__main__':main()
