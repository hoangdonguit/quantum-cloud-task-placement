"""Post-training analyses. Keep at repository root: do not change training fingerprints.
OOD inference is exploratory; it does not replace the official comparisons.
"""
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

def holm(p):
    p=np.asarray(p);order=np.argsort(p);out=np.empty(len(p))
    out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out

def baseline(case):
    from src.heuristics import REGISTRY
    from experiments.baseline_tuning import GreedyScoreTuned
    name=json.loads((case/'baseline_selection.json').read_text())['selected']
    if name.startswith('GreedyScoreTuned_m'):
        m,l=name.split('_m')[1].split('_l');return name,GreedyScoreTuned(float(m),float(l))
    return name,REGISTRY[name]()

def ood(args,out):
    source=args.root/'ladder/tables/ood_per_workload_base.csv';df=pd.read_csv(source)
    rng=np.random.default_rng(409);rows=[]
    for scen in sorted(df.scenario.unique()):
        sub=df[df.scenario==scen];ref=sub[sub.policy=='FrozenValidationBaseline']
        if ref.workload_seed.duplicated().any() or len(ref)!=200:raise ValueError('Invalid frozen baseline workload set')
        ref=ref.set_index('workload_seed').score.sort_index()
        for a in ['PPO','DQN','DDQN3']:
            mat=sub[sub.policy==a].pivot(index='train_seed',columns='workload_seed',values='score').reindex(columns=ref.index)
            if mat.shape!=(10,200) or mat.isna().any().any():raise ValueError(f'Incomplete OOD: {scen}/{a}')
            delta=mat.to_numpy()-ref.to_numpy()[None,:];n,w=delta.shape
            boot=np.array([delta[np.ix_(rng.integers(n,size=n),rng.integers(w,size=w))].mean() for _ in range(args.bootstrap)])
            means=delta.mean(axis=1);p=float(stats.ttest_1samp(means,0).pvalue)
            if np.isnan(p):p=1.0 if np.allclose(means,0) else 0.0
            rows.append(dict(scenario=scen,algo=a,n_seeds=n,n_workloads=w,diff=delta.mean(),ci_lo=np.quantile(boot,.025),ci_hi=np.quantile(boot,.975),p_seed_conditional=p))
    result=pd.DataFrame(rows);result['p_holm_seed_conditional']=holm(result.p_seed_conditional)
    result.to_csv(out/'ood_inference.csv',index=False)
    print(result.round(6).to_string(index=False),flush=True)
    print('Exploratory family: all 24 OOD/ID comparisons. CI unadjusted; p conditional on fixed 200 workloads.')

def load_core():
    import torch
    torch.set_num_threads(1)
    from src.env import QCloudSim
    from experiments.evaluate import load_agent
    from src.heuristics import REGISTRY
    return QCloudSim,load_agent,REGISTRY

def agents(args,case,cfg,loader):
    result=[]
    for a in ['PPO','DQN','DDQN3']:
        for seed in range(args.n_seeds):
            run=case/f'{a}_seed{seed}';weight=run/('model.zip' if a=='PPO' else 'model.pt')
            if not weight.is_file():raise FileNotFoundError(f'Missing model: {weight}; restore model backup, do not retrain automatically')
            result.append((a,seed,loader(a,run,cfg)))
    return result

def diagnostics(args,out):
    Sim,loader,registry=load_core();results=[];nodes=[];traces=[]
    for label in args.cases:
        case=args.root/'ladder'/label;cfg=json.loads((case/'case_config.json').read_text());name,ref=baseline(case)
        actors=agents(args,case,cfg,loader)+[('SelectedBaseline',-1,ref),('GreedyFlow',-1,registry['GreedyFlow']())]
        for policy,seed,actor in actors:
            sim=Sim(cfg['env'],'test');is_drl=seed>=0
            historical_path=case/f'{policy}_seed{seed}'/'eval_per_workload.csv'
            historical=pd.read_csv(historical_path).set_index('workload_seed') if is_drl and historical_path.exists() else None
            for ws in range(30000,30000+args.workloads):
                obs=sim.reset(sim.make_workload(ws));ref.reset(sim,ws)
                if not is_drl:actor.reset(sim,ws)
                done=False;decision=0;acc={}
                while not done:
                    v=sim.view();ref_action=ref(sim);mask=sim.action_masks()
                    action=int(actor(obs,mask) if is_drl else actor(sim))
                    if not mask[action]:raise ValueError('Infeasible action from feasible-policy diagnostic')
                    key=sim.nodes[action]['key'];z=acc.setdefault(key,dict(n_tasks=0,exec_seconds=0.,selected_wait_sum=0.,width_sum=0.,frel_sum=0.,disagree=0))
                    z['n_tasks']+=1;z['exec_seconds']+=float(v['exec'][action]);z['selected_wait_sum']+=float(v['wait'][action]);z['width_sum']+=int(sim.pool.width[v['row']]);z['frel_sum']+=float(v['frel'][action]);z['disagree']+=int(action!=ref_action)
                    before=dict(node=key,baseline_node=sim.nodes[ref_action]['key'],width=int(sim.pool.width[v['row']]),wait=float(v['wait'][action]),exec=float(v['exec'][action]),frel=float(v['frel'][action]))
                    obs,reward,done,info=sim.step(action)
                    if ws<30000+args.trace_workloads and (seed in (-1,0)):
                        traces.append(dict(case=label,policy=policy,train_seed=seed,workload_seed=ws,task=decision,reward=reward,T=info.get('T'),**before))
                    decision+=1
                measured=sim.metrics()
                if historical is not None:
                    for metric in ['score','mean_T','mean_F','mean_frel']:
                        if not np.isclose(measured[metric],historical.loc[ws,metric],rtol=1e-6,atol=1e-8):
                            raise RuntimeError(f'Replay differs from official: {label}/{policy}/{seed}/{ws}/{metric}. Check code, data, dependencies; do not mix results.')
                results.append(dict(case=label,policy=policy,train_seed=seed,workload_seed=ws,**measured))
                for node,z in acc.items():nodes.append(dict(case=label,policy=policy,train_seed=seed,workload_seed=ws,node=node,**z))
            pd.DataFrame(results).to_csv(out/'diagnostics_per_workload.csv',index=False)
            pd.DataFrame(nodes).to_csv(out/'node_allocation.csv',index=False)
            pd.DataFrame(traces).to_csv(out/'decision_traces.csv',index=False)
            print('DIAGNOSTICS',label,policy,seed,flush=True)
    g=pd.DataFrame(nodes).groupby(['case','policy','train_seed','node'],as_index=False)[['n_tasks','exec_seconds','selected_wait_sum','width_sum','frel_sum','disagree']].sum()
    g['selection_share']=g.n_tasks/g.groupby(['case','policy','train_seed']).n_tasks.transform('sum')
    for col,new in [('selected_wait_sum','mean_selected_wait'),('width_sum','mean_width'),('frel_sum','mean_frel'),('disagree','disagreement_rate')]:g[new]=g[col]/g.n_tasks
    g.to_csv(out/'node_allocation_summary.csv',index=False)
    # Aggregate at seed level before averaging; heuristic train_seed=-1 is not a training replication.
    metrics=['score','mean_T','p95_T','mean_wait','mean_exec','mean_F','mean_frel']
    run=pd.DataFrame(results).groupby(['case','policy','train_seed'])[metrics].mean()
    run.groupby(['case','policy']).agg(['mean','std']).to_csv(out/'diagnostics_summary.csv')

def latency(args,out):
    Sim,loader,registry=load_core();rows=[];build=[]
    for label in args.cases:
        case=args.root/'ladder'/label;cfg=json.loads((case/'case_config.json').read_text());name,ref=baseline(case)
        acts=agents(args,case,cfg,loader);sim=Sim(cfg['env'],'val');drivers=[('SelectedBaseline',ref),('GreedyFlow',registry['GreedyFlow']())]
        ws=7000;obs=sim.reset(sim.make_workload(ws))
        for _,s,a in acts:
            for _ in range(25):a(obs,sim.action_masks())
        for _,p in drivers:p.reset(sim,ws)
        rng=np.random.default_rng(409);count=0
        while count<args.timing_states:
            obs=sim.obs();mask=sim.action_masks()
            tasks=[(a,s,fn,True) for a,s,fn in acts]+[(a,-1,fn,False) for a,fn in drivers]
            # Randomize policy measurement order at each state to reduce drift/order effects.
            for ix in rng.permutation(len(tasks)):
                policy,seed,fn,is_drl=tasks[ix]
                for rep in range(args.timing_repeats):
                    t=time.perf_counter_ns();fn(obs,mask) if is_drl else fn(sim);elapsed=time.perf_counter_ns()-t
                    rows.append(dict(case=label,policy=policy,train_seed=seed,state=count,repeat=rep,decision_us=elapsed/1000))
            t=time.perf_counter_ns();sim.obs();build.append(dict(case=label,state=count,observation_us=(time.perf_counter_ns()-t)/1000))
            obs,_,done,_=sim.step(ref(sim));count+=1
            if done:
                ws+=1;obs=sim.reset(sim.make_workload(ws))
                for _,p in drivers:p.reset(sim,ws)
        print('LATENCY',label,'states',count,flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'decision_latency_raw.csv',index=False)
    runs=df.groupby(['case','policy','train_seed']).decision_us.agg(mean_us='mean',median_us='median',p95_us=lambda x:x.quantile(.95)).reset_index()
    runs.to_csv(out/'decision_latency_per_seed.csv',index=False)
    runs.groupby(['case','policy'])[['mean_us','median_us','p95_us']].agg(['mean','std']).to_csv(out/'decision_latency_summary.csv')
    pd.DataFrame(build).to_csv(out/'observation_latency.csv',index=False)
    print('Latency = warmed batch-1 policy call on identical baseline-trajectory states; excludes model loading, state building and scheduling update. CPU torch threads=1. Not end-to-end cloud latency.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT/'results/official');ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--mode',choices=['all','ood','diagnostics','latency'],default='all');ap.add_argument('--cases',nargs='+',default=['base','rate_r0.5','tasks_t600','interaction_wide_busy'])
    ap.add_argument('--n-seeds',type=int,default=10);ap.add_argument('--workloads',type=int,default=200);ap.add_argument('--trace-workloads',type=int,default=2);ap.add_argument('--bootstrap',type=int,default=5000);ap.add_argument('--timing-states',type=int,default=100);ap.add_argument('--timing-repeats',type=int,default=3);args=ap.parse_args()
    if not 1<=args.n_seeds<=10 or not 1<=args.workloads<=200 or args.bootstrap<100:ap.error('Invalid size')
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'protocol.json').exists():raise RuntimeError('Output exists: choose a fresh --out path to avoid mixing runs')
    def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    protocol={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()};protocol.update(script_sha256=digest(Path(__file__)),python=sys.version,platform=platform.platform(),cpu=platform.processor(),threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']})
    protocol['inputs']={str(p.relative_to(args.root)):digest(p) for p in [args.root/'protocol_fingerprint.txt',args.root/'ladder/tables/ood_per_workload_base.csv'] if p.exists()}
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    if args.mode in ('all','diagnostics','latency'):
        models=[]
        for label in args.cases:
            case=args.root/'ladder'/label
            for name in ['case_config.json','baseline_selection.json']:
                if not (case/name).is_file():raise FileNotFoundError(case/name)
                models.append(dict(path=str((case/name).relative_to(args.root)),sha256=digest(case/name)))
            for algo in ['PPO','DQN','DDQN3']:
                for seed in range(args.n_seeds):
                    weight=case/f'{algo}_seed{seed}'/('model.zip' if algo=='PPO' else 'model.pt')
                    if not weight.is_file():raise FileNotFoundError(f'Missing checkpoint: {weight}; restore backup, no automatic training')
                    models.append(dict(path=str(weight.relative_to(args.root)),sha256=digest(weight)))
        (args.out/'model_inputs.json').write_text(json.dumps(models,indent=2))
    if args.mode in ('all','ood'):ood(args,args.out)
    if args.mode in ('all','diagnostics'):diagnostics(args,args.out)
    if args.mode in ('all','latency'):latency(args,args.out)
    (args.out/'COMPLETE.json').write_text(json.dumps({'mode':args.mode,'n_seeds':args.n_seeds,'workloads':args.workloads}))
    print('COMPLETE:',args.out,flush=True)
if __name__=='__main__':main()
