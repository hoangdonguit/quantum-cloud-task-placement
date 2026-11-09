"""Lab01 extension: native-gate proxies vs Bell/GHZ success; no hardware execution."""
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider

def wilson(k,n):
 z=1.96;p=k/n;d=1+z*z/n;m=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return m-h,m+h

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--shots',type=int,default=None);ap.add_argument('--n-seeds',type=int,default=None);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 names=['Perth','Lagos'] if a.smoke else ['Perth','Lagos','Nairobi'];sizes=[2,3] if a.smoke else list(range(2,8));seeds=[409,410] if a.smoke else [409,410,411];shots=a.shots or (512 if a.smoke else 4096);seeds=list(range(409,409+a.n_seeds)) if a.n_seeds else seeds;rows=[]
 for name in names:
  cls=getattr(fake_provider,'Fake'+name,None) or getattr(fake_provider,'Fake'+name+'V2')
  backend=cls();target=backend.target
  props=backend.properties().to_dict()
  (out/(name+'_snapshot.json')).write_text(json.dumps(props,indent=2,default=str))
  for n in sizes:
   for topology in ['chain','star']:
    qc=QuantumCircuit(n,n);qc.h(0)
    for i in range(1,n):qc.cx(i-1 if topology=='chain' else 0,i)
    qc.measure(range(n),range(n))
    for level in [0,2]:
     for seed in seeds:
      isa=generate_preset_pass_manager(backend=backend,optimization_level=level,seed_transpiler=seed).run(qc)
      log_layout=0.;log_median=0.;missing=0
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
       log_layout+=math.log1p(-min(float(err),1-1e-15));log_median+=math.log1p(-min(med,1-1e-15))
      sim=AerSimulator.from_backend(backend,method='matrix_product_state')
      counts=sim.run(isa,shots=shots,seed_simulator=seed).result().get_counts()
      k=counts.get('0'*n,0)+counts.get('1'*n,0);lo,hi=wilson(k,shots)
      ideal=AerSimulator(method='matrix_product_state').run(qc,shots=shots,seed_simulator=seed).result().get_counts()
      rows.append(dict(backend=name,physical_layout=json.dumps(isa.layout.initial_index_layout(filter_ancillas=True)),n=n,topology=topology,level=level,seed=seed,shots=shots,depth=isa.depth(),gates_2q=isa.num_nonlocal_gates(),proxy_layout=math.exp(log_layout),proxy_median=math.exp(log_median),theory_success=1.,ideal_success=(ideal.get('0'*n,0)+ideal.get('1'*n,0))/shots,noisy_success=k/shots,success_ci_lo=lo,success_ci_hi=hi,missing_error_ops=missing))
      print(name,n,topology,level,seed,'success',round(k/shots,4),flush=True)
 df=pd.DataFrame(rows);df.to_csv(out/'quantum_validation.csv',index=False)
 summary=df.groupby(['backend','n','topology','level'])[['proxy_layout','proxy_median','noisy_success']].mean();summary.to_csv(out/'quantum_summary.csv')
 ranks=[]
 for key,g in summary.reset_index().groupby(['n','topology','level']):
  for col in ['proxy_layout','proxy_median']:
   ranks.append(dict(n=key[0],topology=key[1],level=key[2],proxy=col,spearman=spearmanr(g[col],g.noisy_success).statistic,n_backends=len(g)))
 pd.DataFrame(ranks).to_csv(out/'backend_rank_correlation.csv',index=False)
 (out/'interpretation.txt').write_text('Proxy no-operation-error products and Bell/GHZ Z-basis success are different quantities. Wilson intervals cover shot uncertainty only. Small backend counts make rank correlation exploratory. Matching Z counts does not certify entanglement.\n')
if __name__=='__main__':main()
