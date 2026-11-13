"""Small separate Aer validation: same ISA for median/layout proxies, Z success and state fidelity."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.quantum_info import DensityMatrix,state_fidelity
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider
from post_common import digest,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    from importlib.metadata import version
    packages={k:version(k) for k in ['qiskit','qiskit-aer','qiskit-ibm-runtime','numpy']}
    rows=[];specs=[('Perth',2,'chain',0,409)] if a.smoke else [(b,n,t,l,s) for b in ['Perth','Lagos','Nairobi'] for n in [2,3,4] for t in ['chain','star'] for l in [0,2] for s in [409,410]]
    write_json(a.out/'protocol.json',{'smoke_only':a.smoke,'specs':specs,'shots':8192,'packages':packages,'script_sha256':digest(__file__),'repair_source_sha256':digest('scripts/repair_quantum.py'),'fidelity':'Squared Uhlmann fidelity on full physical density matrices before readout; ideal and noisy evolve the same unmeasured transpiled ISA','Z_success':'Measured ISA, including readout noise; separate from premeasurement fidelity','scheduler_reward_changed':False})
    for name,n,topology,level,seed in specs:
        cls=getattr(fake_provider,'Fake'+name,None) or getattr(fake_provider,'Fake'+name+'V2');backend=cls();target=backend.target
        qc=QuantumCircuit(n,n);qc.h(0)
        for i in range(1,n):qc.cx(i-1 if topology=='chain' else 0,i)
        qc.measure(range(n),range(n));isa=generate_preset_pass_manager(backend=backend,optimization_level=level,seed_transpiler=seed).run(qc)
        ll=lm=0.;missing=0
        for inst in isa.data:
            gate=inst.operation.name
            if gate=='barrier':continue
            qubits=tuple(isa.find_bit(q).index for q in inst.qubits)
            assert target.instruction_supported(gate,qubits)
            prop=target[gate].get(qubits);err=None if prop is None else prop.error
            if err is None:
                if gate not in ['rz','delay']:missing+=1
                err=0.
            errors=[v.error for v in target[gate].values() if v is not None and v.error is not None]
            ll+=math.log1p(-min(float(err),1-1e-15));lm+=math.log1p(-min(float(np.median(errors)) if errors else 0.,1-1e-15))
        noisy=AerSimulator.from_backend(backend,method='density_matrix',max_parallel_threads=1,max_parallel_experiments=1,max_parallel_shots=1)
        counts=noisy.run(isa,shots=8192,seed_simulator=seed).result().get_counts();success=(counts.get('0'*n,0)+counts.get('1'*n,0))/8192
        state=isa.remove_final_measurements(inplace=False);state.save_density_matrix()
        rho=noisy.run(state,shots=1,seed_simulator=seed).result().data(0)['density_matrix']
        ideal=AerSimulator(method='density_matrix',max_parallel_threads=1).run(state,shots=1).result().data(0)['density_matrix']
        fid=state_fidelity(DensityMatrix(rho),DensityMatrix(ideal))
        ideal_counts=AerSimulator(method='density_matrix',max_parallel_threads=1).run(isa,shots=1024,seed_simulator=seed).result().get_counts()
        ideal_success=(ideal_counts.get('0'*n,0)+ideal_counts.get('1'*n,0))/1024
        assert ideal_success==1. and 0<=fid<=1.+1e-10
        row=dict(backend=name,n=n,topology=topology,level=level,seed=seed,proxy_layout=math.exp(ll),proxy_median=math.exp(lm),noisy_Z_success=success,state_fidelity=fid,ideal_Z_success=ideal_success,missing_error_ops=missing,layout=json.dumps(isa.layout.final_index_layout(filter_ancillas=True)))
        rows.append(row);pd.DataFrame(rows).to_csv(a.out/'per_circuit.csv',index=False);print(name,n,topology,level,seed,round(success,5),round(fid,5),flush=True)
    write_json(a.out/'COMPLETE.json',{'smoke_only':a.smoke,'configs':len(rows)})
if __name__=='__main__':main()
