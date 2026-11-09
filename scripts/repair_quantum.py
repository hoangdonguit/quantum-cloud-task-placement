"""Standalone Aer recovery; leaves trained source, models and analysis unchanged."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):os.environ[k]='1'
os.environ['QISKIT_PARALLEL']='FALSE'
import argparse,csv,hashlib,itertools,json,math,subprocess,sys,time,zipfile
from pathlib import Path
from importlib.metadata import version
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED

def atomic(path,value):
 t=path.with_suffix(f'.{os.getpid()}.tmp');t.write_text(json.dumps(value,indent=2,default=str));t.replace(path)

def worker(spec,dest,signature):
 import numpy as np
 from qiskit import QuantumCircuit
 from qiskit.transpiler import generate_preset_pass_manager
 from qiskit.quantum_info import Statevector
 from qiskit_aer import AerSimulator
 from qiskit_ibm_runtime import fake_provider
 name,n,topology,level,seed=spec
 cls=getattr(fake_provider,'Fake'+name,None) or getattr(fake_provider,'Fake'+name+'V2');b=cls();target=b.target
 atomic(Path(dest).parent.parent/(name+'_snapshot.json'),b.properties().to_dict())
 qc=QuantumCircuit(n,n);qc.h(0)
 for i in range(1,n):qc.cx(i-1 if topology=='chain' else 0,i)
 qc.measure(range(n),range(n))
 print('START transpile',spec,flush=True);t=time.monotonic()
 isa=generate_preset_pass_manager(backend=b,optimization_level=level,seed_transpiler=seed).run(qc)
 transpile_s=time.monotonic()-t;ll=lm=0.;missing=0
 for inst in isa.data:
  gate=inst.operation.name
  if gate=='barrier':continue
  qubits=tuple(isa.find_bit(q).index for q in inst.qubits)
  if not target.instruction_supported(gate,qubits):raise RuntimeError((gate,qubits))
  prop=target[gate].get(qubits);err=None if prop is None else prop.error
  if err is None:
   if gate not in ('rz','delay'):missing+=1
   err=0.
  errs=[p.error for p in target[gate].values() if p is not None and p.error is not None]
  med=float(np.median(errs)) if errs else 0.
  ll+=math.log1p(-min(float(err),1-1e-15));lm+=math.log1p(-min(med,1-1e-15))
 print('START noisy simulation',spec,flush=True);t=time.monotonic()
 sim=AerSimulator.from_backend(b,method='density_matrix',max_parallel_threads=1,max_parallel_experiments=1,max_parallel_shots=1)
 counts=sim.run(isa,shots=8192,seed_simulator=seed).result().get_counts();simulation_s=time.monotonic()-t
 k=counts.get('0'*n,0)+counts.get('1'*n,0);p=k/8192;z=1.96;d=1+z*z/8192;mid=(p+z*z/(2*8192))/d;half=z*math.sqrt(p*(1-p)/8192+z*z/(4*8192**2))/d
 ideal=Statevector(qc.remove_final_measurements(inplace=False)).probabilities_dict();ideal_success=float(ideal.get('0'*n,0)+ideal.get('1'*n,0))
 row=dict(backend=name,n=n,topology=topology,level=level,seed=seed,shots=8192,method='density_matrix',depth=isa.depth(),gates_2q=isa.num_nonlocal_gates(),physical_layout=isa.layout.initial_index_layout(filter_ancillas=True),proxy_layout=math.exp(ll),proxy_median=math.exp(lm),ideal_success=ideal_success,noisy_success=p,success_ci_lo=mid-half,success_ci_hi=mid+half,missing_error_ops=missing,transpile_seconds=transpile_s,simulation_seconds=simulation_s,signature=signature)
 atomic(Path(dest),row);print('DONE',spec,'success',p,'seconds',round(transpile_s+simulation_s,2),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--worker');ap.add_argument('--dest');ap.add_argument('--signature');ap.add_argument('--probe',action='store_true');ap.add_argument('--jobs',type=int,default=2);ap.add_argument('--timeout',type=int,default=120);a=ap.parse_args()
 if a.worker:worker(json.loads(a.worker),a.dest,a.signature);return
 root=Path.cwd();out=root/'results/official/quantum_validation_repaired';out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir(exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
 packages={p:version(p) for p in ('qiskit','qiskit-aer','qiskit-ibm-runtime','numpy')}
 signature=hashlib.sha256(Path(__file__).read_bytes()+json.dumps(packages,sort_keys=True).encode()).hexdigest()
 atomic(out/'repair_protocol.json',dict(method='density_matrix',shots=8192,seeds=list(range(409,414)),packages=packages,signature=signature,original_training_fingerprint=(root/'results/official/protocol_fingerprint.txt').read_text(),note='Standalone repair. Classical training/statistical results unchanged. Small native circuits simulated with full mixed-state evolution; no-operation-error proxies differ from Z success.'))
 def run(spec):
  key='_'.join(map(str,spec));dest=out/'checkpoints'/(key+'.json');log=out/'logs'/(key+'.log')
  if dest.exists():
   if json.loads(dest.read_text()).get('signature')!=signature:raise RuntimeError('Checkpoint provenance changed: '+str(dest))
   print('SKIP',key,flush=True);return
  print('RUN',key,flush=True)
  try:
   with log.open('w') as f:subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',json.dumps(spec),'--dest',str(dest),'--signature',signature],stdout=f,stderr=subprocess.STDOUT,timeout=a.timeout,check=True)
  except Exception as e:
   print('FAILED',key,str(e),'READ',log,flush=True);raise
  row=json.loads(dest.read_text());print('OK',key,'success',row['noisy_success'],'seconds',round(row['simulation_seconds']+row['transpile_seconds'],2),flush=True)
 if a.probe:
  for spec in [('Perth',4,'chain',0,409),('Perth',7,'star',0,409)]:run(spec)
  print('PROBE PASSED. Full validation can now run.',flush=True);return
 specs=list(itertools.product(['Perth','Lagos','Nairobi'],range(2,8),['chain','star'],[0,2],range(409,414)))
 with ThreadPoolExecutor(a.jobs) as ex:
  remaining=iter(specs);futures={ex.submit(run,s) for s in itertools.islice(remaining,a.jobs)}
  while futures:
   done,futures=wait(futures,return_when=FIRST_COMPLETED)
   for f in done:
    f.result()
    s=next(remaining,None)
    if s is not None:futures.add(ex.submit(run,s))
 import pandas as pd
 from scipy.stats import spearmanr
 rows=[json.loads((out/'checkpoints'/('_'.join(map(str,s))+'.json')).read_text()) for s in specs];df=pd.DataFrame(rows);df.to_csv(out/'quantum_validation.csv',index=False)
 summary=df.groupby(['backend','n','topology','level'])[['proxy_layout','proxy_median','noisy_success']].mean();summary.to_csv(out/'quantum_summary.csv')
 ranks=[]
 for key,g in summary.reset_index().groupby(['n','topology','level']):
  for col in ['proxy_layout','proxy_median']:ranks.append(dict(n=key[0],topology=key[1],level=key[2],proxy=col,spearman=spearmanr(g[col],g.noisy_success).statistic,n_backends=len(g)))
 pd.DataFrame(ranks).to_csv(out/'backend_rank_correlation.csv',index=False)
 atomic(out/'COMPLETE.json',dict(n_configs=len(rows),shots=8192,seeds=5,signature=signature))
 (out/'repair_source.py').write_bytes(Path(__file__).read_bytes())
 # Verify completed classical evaluation artifacts before marking overall completion.
 runs=list((root/'results/official/ladder').glob('*/*/run_meta.json'))
 if len(runs)!=630:raise RuntimeError('Expected 630 classical runs; quantum results saved but overall completion withheld')
 for meta in runs:
  with (meta.parent/'eval_per_workload.csv').open() as f:
   if sum(1 for _ in csv.DictReader(f))!=200:raise RuntimeError('Incomplete workload evaluation: '+str(meta))
 atomic(root/'results/official/COMPLETE.json',dict(training_runs=630,quantum_configs=360,quantum_validation_directory='quantum_validation_repaired',repair_signature=signature))
 zp=root/'results'/('official_return_'+time.strftime('%Y%m%d_%H%M%S')+'.zip')
 with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED) as z:
  for p in sorted((root/'results/official').rglob('*')):
   if p.is_file() and p.suffix not in ('.pt','.zip','.pyc'):z.write(p,p.relative_to(root/'results/official'))
 print('SEND THIS FILE:',zp,flush=True)
if __name__=='__main__':main()
