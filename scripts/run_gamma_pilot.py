"""Validation-only DQN gamma pilot at tasks_t600; never resumes or evaluates test."""
import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.provenance import run_fingerprint

GAMMAS=(.90,.95,.99)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,value):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true',help='3 runs: seed0, 1200 steps, two validation workloads')
    args=parser.parse_args()
    os.chdir(ROOT)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    import numpy as np
    import pandas as pd
    from algorithms.dqn import DQNAgent
    from experiments.evaluate import eval_workloads
    from src.env import QCloudSim

    source=ROOT/'results_summary/cases/tasks_t600/case_config.json'
    cfg=json.loads(source.read_text())
    assert cfg['case']=='tasks_t600' and cfg['env']['n_tasks']==600
    assert cfg['train']==dict(timesteps=100000,eval_every=10000,n_val=20)
    assert cfg['algos']['DQN']['kind']=='dqn' and cfg['algos']['DQN']['gamma']==.90
    assert cfg['env']['mask_infeasible'] and cfg['env']['reward_mode']=='log1p'
    official=ROOT/'results/official/ladder/tasks_t600/case_config.json'
    if official.exists() and json.loads(official.read_text())!=cfg:
        raise RuntimeError('Official tasks_t600 config differs from the curated source; inspect before running')
    source_fp=run_fingerprint(cfg)
    out=args.out.resolve()
    # Fail before training if an output already exists, including any partial run.
    out.mkdir(parents=True,exist_ok=False)
    seeds=[0] if args.smoke else [0,1,2]
    steps=1200 if args.smoke else 100000
    val_seeds=list(range(5000,5002 if args.smoke else 5020))
    runtime={'python':platform.python_version(),'executable':sys.executable,'platform':platform.platform(),
        'packages':{p:version(p) for p in ('numpy','pandas','torch','gymnasium','stable-baselines3','sb3-contrib')},
        'threads':{k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}}
    workload_sim=QCloudSim(cfg['env'],'val')
    workload_hashes={}
    for seed in val_seeds:
        idx,arrival=workload_sim.make_workload(seed)
        workload_hashes[str(seed)]=hashlib.sha256(idx.tobytes()+arrival.tobytes()).hexdigest()
    protocol={'case':'tasks_t600','algorithm':'DQN','smoke_only':args.smoke,'gammas':list(GAMMAS),
        'training_seeds':seeds,'steps_per_run':steps,'validation_split':'val','validation_seeds':val_seeds,
        'validation_workload_sha256':workload_hashes,'test_evaluated':False,
        'selection':'Highest mean final-checkpoint validation score across paired training seeds; exact tie -> smaller gamma',
        'source_config':str(source.relative_to(ROOT)),'source_config_sha256':sha(source),
        'source_fingerprint':source_fp,'script_sha256':sha(__file__),
        'training_runner':'algorithms/train.py --validation-only','training_runner_sha256':sha(ROOT/'algorithms/train.py'),
        'runtime':runtime,'full_reference_gamma_retrained':True,'resume_supported':False,
        'started_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Exploratory validation-only pilot, outside Holm63; smoke is pipeline verification only'}
    protocol['pilot_fingerprint']=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    write_json(out/'protocol.json',protocol)
    rows=[]
    for gamma in GAMMAS:
        label=f'{gamma:.2f}'
        gc=copy.deepcopy(cfg);gc['algos']['DQN']['gamma']=gamma
        if args.smoke:gc['train'].update(timesteps=1200,eval_every=1200,n_val=2)
        group=out/f'gamma_{label}';group.mkdir()
        config_path=group/'case_config.json';write_json(config_path,gc)
        fp=run_fingerprint(gc)
        (group/'config_fingerprint.txt').write_text(fp+'\n')
        for seed in seeds:
            if sha(source)!=protocol['source_config_sha256'] or sha(__file__)!=protocol['script_sha256'] or run_fingerprint(cfg)!=source_fp:
                raise RuntimeError('Source/runtime changed during pilot; preserve partial output and inspect')
            run=group/f'DQN_seed{seed}';run.mkdir()
            command=[sys.executable,'-u',str(ROOT/'algorithms/train.py'),'--validation-only','--case-config',str(config_path),
                     '--algo','DQN','--seed',str(seed),'--out-dir',str(run)]
            write_json(run/'invocation.json',{'command':command,'pilot_fingerprint':protocol['pilot_fingerprint']})
            started=time.monotonic()
            print(f'START gamma={label} seed={seed} steps={steps} LOG={run / "train.log"}',flush=True)
            with (run/'train.log').open('x') as log:
                subprocess.run(command,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True)
            meta=json.loads((run/'validation_meta.json').read_text())
            if meta!={**meta,'algo':'DQN','seed':seed,'actual_timesteps':steps,'fingerprint':fp}:
                raise RuntimeError(f'Invalid validation metadata: {run}')
            curve=pd.read_csv(run/'train_log.csv')
            expected=[1200] if args.smoke else list(range(10000,100001,10000))
            if curve.timesteps.tolist()!=expected:raise RuntimeError(f'Invalid validation curve: {run}')
            final=curve.iloc[-1]
            if not np.isfinite(final[['val_score','val_mean_T','val_frel','val_resched']].to_numpy(dtype=float)).all():
                raise RuntimeError(f'Non-finite final validation: {run}')
            if (run/'eval_per_workload.csv').exists() or (run/'run_meta.json').exists():
                raise RuntimeError('Test artifact found in validation-only run')
            # Reload and evaluate only the same validation workloads; also retain per-workload metrics.
            agent=DQNAgent.load(run/'model.pt')
            hp=dict(gc['algos']['DQN']);hp.pop('kind')
            if agent.hp!=hp:raise RuntimeError(f'Checkpoint hyperparameters mismatch: {run}')
            validation=pd.DataFrame(eval_workloads(gc['env'],'val',val_seeds,act_fn=agent.predict))
            assert validation.workload_seed.tolist()==val_seeds
            for column,metric in [('val_score','score'),('val_mean_T','mean_T'),('val_frel','mean_frel'),('val_resched','resched')]:
                if not np.isclose(final[column],validation[metric].mean(),rtol=1e-10,atol=1e-10):
                    raise RuntimeError(f'Reloaded checkpoint validation mismatch: {run} {metric}')
            if not np.isclose(meta['val_score'],final.val_score,rtol=1e-10,atol=1e-10):raise RuntimeError('Metadata score mismatch')
            if run_fingerprint(gc)!=fp:raise RuntimeError('Protocol/source/runtime changed within run')
            validation.to_csv(run/'validation_per_workload.csv',index=False)
            row={'gamma':gamma,'train_seed':seed,'validation_score':float(final.val_score),
                 'mean_T':float(final.val_mean_T),'f_rel':float(final.val_frel),'resched':float(final.val_resched),
                 'actual_timesteps':steps,'n_validation_workloads':len(val_seeds),'config_fingerprint':fp,
                 'checkpoint_sha256':sha(run/'model.pt')}
            rows.append(row)
            write_json(run/'pilot_meta.json',{**row,'pilot_fingerprint':protocol['pilot_fingerprint'],
                'smoke_only':args.smoke,'reload_validated':True,'test_evaluated':False,
                'wall_seconds_including_validation_reload':time.monotonic()-started})
            print(f'DONE gamma={label} seed={seed} validation_score={row["validation_score"]:.8f}',flush=True)
    table=pd.DataFrame(rows)
    ref=table[table.gamma==.90].set_index('train_seed')
    for metric in ['validation_score','mean_T','f_rel','resched']:
        table[f'delta_{metric}_vs_gamma090']=table[metric]-table.train_seed.map(ref[metric])
    if len(table)!=len(GAMMAS)*len(seeds) or table.duplicated(['gamma','train_seed']).any():raise RuntimeError('Missing or duplicate runs')
    for gamma in GAMMAS:
        if sorted(table[table.gamma==gamma].train_seed)!=seeds:raise RuntimeError('Unpaired training seeds')
    numeric=table.select_dtypes(include='number')
    if not np.isfinite(numeric.to_numpy()).all():raise RuntimeError('Incomplete paired validation table')
    table.to_csv(out/'paired_validation.csv',index=False)
    summary=table.groupby('gamma')[['validation_score','mean_T','f_rel','resched','delta_validation_score_vs_gamma090']].mean().reset_index()
    summary=summary.sort_values(['validation_score','gamma'],ascending=[False,True],kind='stable')
    summary.to_csv(out/'validation_summary.csv',index=False)
    selected=float(summary.iloc[0].gamma)
    if list(out.rglob('eval_per_workload.csv')):raise RuntimeError('Unexpected test output in pilot')
    write_json(out/'selection.json',{'selected_gamma':selected,'smoke_only':args.smoke,'eligible_for_test_proposal':not args.smoke,
        'criterion':protocol['selection'],'test_evaluated':False,'note':'Smoke selection is not a scientific recommendation' if args.smoke else 'Review paired seed differences before deciding whether to confirm on test'})
    write_json(out/'COMPLETE.json',{'runs':len(rows),'smoke_only':args.smoke,'paired_table_rows':len(table),
        'selected_gamma':selected,'pilot_fingerprint':protocol['pilot_fingerprint'],'test_evaluated':False,
        'paired_validation_sha256':sha(out/'paired_validation.csv')})
    print(table[['gamma','train_seed','validation_score','delta_validation_score_vs_gamma090']].to_string(index=False),flush=True)
    print(f'COMPLETE: {out / "COMPLETE.json"}\nPAIRED: {out / "paired_validation.csv"}',flush=True)

if __name__=='__main__':main()
