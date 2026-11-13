"""Validate and summarize completed supplementary runs without modifying raw artifacts."""
import argparse,hashlib,json,sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.env import QCloudSim
from src.provenance import run_fingerprint
from experiments.baseline_tuning import holm
from experiments.evaluate import eval_workloads
from scripts.feature_retraining import load_variant
from scripts.post_common import digest,workload_hash,write_json

METRICS=['score','mean_T','mean_wait','mean_exec','mean_F','mean_frel','resched']

def paired(diff,rng,reps):
    n,w=diff.shape
    b=np.array([diff[np.ix_(rng.integers(n,size=n),rng.integers(w,size=w))].mean() for _ in range(reps)])
    return dict(diff=float(diff.mean()),ci_lo=float(np.quantile(b,.025)),ci_hi=float(np.quantile(b,.975)),p_seed_conditional=float(stats.ttest_1samp(diff.mean(1),0).pvalue))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path('results_summary/cases/base/case_config.json').read_text());core=run_fingerprint(cfg);inputs={};checks={};rng=np.random.default_rng(409)
    for part in ['ood_smoke','ablation_smoke','quantum_smoke','ood04','ablation','quantum']:
        inputs[str(a.root/part/'COMPLETE.json')]=digest(a.root/part/'COMPLETE.json')
    ap=json.loads((a.root/'ablation/protocol.json').read_text());op=json.loads((a.root/'ood04/protocol.json').read_text());qp=json.loads((a.root/'quantum/protocol.json').read_text())
    assert not ap['smoke_only'] and op['mode']=='full' and not json.loads((a.root/'quantum/COMPLETE.json').read_text())['smoke_only']
    for proto,script in [(ap,'feature_retraining.py'),(op,'ood_rate04.py'),(qp,'quantum_state_check.py')]:assert proto['script_sha256']==digest(Path('scripts')/script)
    assert ap['helper_sha256']==op['helper_sha256']==digest('scripts/post_common.py')
    assert ap['historical_core_fingerprint']==op['base_fingerprint']==core
    assert op['base_config_sha256']==digest('results/official/ladder/base/case_config.json')
    assert qp['repair_source_sha256']==digest('scripts/repair_quantum.py')
    assert ap['training_seeds']==op['training_seeds']==list(range(10));assert ap['test_seeds']==op['test_seeds']==list(range(30000,30200));assert ap['steps']==100000
    assert ap['validation_seeds']==list(range(5000,5020)) and op['validation_seeds']==list(range(5000,5100))
    test_seeds=ap['test_seeds'];allrows=[];reloads=[];original_diffs=[]
    for variant,groups in ap['variants'].items():
        for seed in ap['training_seeds']:
            run=a.root/'ablation'/variant/f'DQN_seed{seed}';rc=json.loads((run/'config.json').read_text());meta=json.loads((run/'run_meta.json').read_text())
            assert rc==dict(cfg,drop_groups=groups,training_seed=seed,actual_steps=100000)
            fp=hashlib.sha256((json.dumps(rc,sort_keys=True)+ap['script_sha256']+ap['helper_sha256']+core).encode()).hexdigest()
            assert meta['fingerprint']==fp and not meta['smoke_only'] and meta['model_sha256']==digest(run/'model.pt')
            curve=pd.read_csv(run/'train_log.csv');assert curve.timesteps.tolist()==list(range(10000,100001,10000))
            d=pd.read_csv(run/'eval_per_workload.csv');assert d.workload_seed.tolist()==test_seeds and np.isfinite(d[METRICS]).all().all()
            inputs[str(run/'eval_per_workload.csv')]=digest(run/'eval_per_workload.csv');inputs[str(run/'model.pt')]=meta['model_sha256']
            if seed==0:
                fresh=pd.DataFrame(eval_workloads(cfg['env'],'test',test_seeds[:3],act_fn=load_variant(run)))
                error=float(np.max(np.abs(fresh[METRICS].to_numpy()-d[METRICS].iloc[:3].to_numpy())));assert error<1e-9;reloads.append(dict(variant=variant,seed=seed,max_error=error))
            if variant=='full':
                old=pd.read_csv(Path('results/official/ladder/base')/f'DQN_seed{seed}/eval_per_workload.csv')
                assert old.workload_seed.tolist()==test_seeds
                original_diffs.append(float(np.max(np.abs(old[METRICS].to_numpy()-d[METRICS].to_numpy()))))
            allrows.extend(d.assign(variant=variant,train_seed=seed).to_dict('records'))
    assert len(allrows)==6000
    df=pd.DataFrame(allrows);df.to_csv(a.out/'ablation_per_workload.csv',index=False)
    per_seed=df.groupby(['variant','train_seed'])[METRICS].mean();per_seed.to_csv(a.out/'ablation_per_seed.csv');summary=per_seed.groupby('variant').agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(a.out/'ablation_summary.csv')
    full=df[df.variant=='full'].pivot(index='train_seed',columns='workload_seed',values='score').reindex(index=range(10),columns=test_seeds).to_numpy();comparisons=[]
    for v in ['no_queue','no_estimates']:
        mat=df[df.variant==v].pivot(index='train_seed',columns='workload_seed',values='score').reindex(index=range(10),columns=test_seeds).to_numpy()
        comparisons.append(dict(variant=v,reference='full',**paired(mat-full,rng,5000)))
    c=pd.DataFrame(comparisons);c['p_holm2']=holm(c.p_seed_conditional);c.to_csv(a.out/'ablation_comparison.csv',index=False)
    # Recompute the OOD comparison using exactly the original deterministic bootstrap order.
    od=pd.read_csv(a.root/'ood04/per_workload.csv');assert len(od)==6400 and not od.duplicated(['policy','train_seed','workload_seed']).any()
    env=dict(cfg['env'],arrival_rate=.4);assert op['train_env']==cfg['env'] and op['test_env']==env
    sim=QCloudSim(env,'test');assert op['workloads']=={str(s):workload_hash(sim,s) for s in test_seeds}
    hashes=json.loads((a.root/'ood04/model_hashes.json').read_text());assert len(hashes)==30
    for path,value in hashes.items():assert digest(path)==value
    selection=json.loads((a.root/'ood04/baseline_selection.json').read_text());assert selection['retuned_validation']==max(selection['validation_scores'],key=selection['validation_scores'].get)
    assert selection['frozen_base']==json.loads(Path('results/official/ladder/base/baseline_selection.json').read_text())['selected']
    assert (a.root/'ood04/baseline_selection.json').stat().st_mtime_ns<(a.root/'ood04/per_workload.csv').stat().st_mtime_ns
    oc=[];rng=np.random.default_rng(409)
    for algo in ['PPO','DQN','DDQN3']:
        block=od[od.policy==algo];assert sorted(block.train_seed.unique())==list(range(10))
        mat=block.pivot(index='train_seed',columns='workload_seed',values='score').reindex(index=range(10),columns=test_seeds).to_numpy();assert np.isfinite(mat).all()
        for ref in ['FrozenBase','Retuned04']:
            h=od[od.policy==ref].set_index('workload_seed').reindex(test_seeds).score.to_numpy();assert np.isfinite(h).all()
            oc.append(dict(algo=algo,baseline=ref,**paired(mat-h,rng,5000)))
    oc=pd.DataFrame(oc);oc['p_holm6']=holm(oc.p_seed_conditional);old=pd.read_csv(a.root/'ood04/comparison.csv');np.testing.assert_allclose(oc.iloc[:,2:].to_numpy(float),old.iloc[:,2:].to_numpy(float),rtol=1e-10,atol=1e-12)
    oc.to_csv(a.out/'ood04_comparison.csv',index=False);od.groupby(['policy','train_seed'])[METRICS].mean().groupby('policy').mean().to_csv(a.out/'ood04_summary.csv')
    # Verify all saved DRL OOD scores against the historical rate_x2 rows as an extra provenance check.
    history=pd.read_csv('results/official/ladder/tables/ood_per_workload_base.csv');h=history[(history.scenario=='rate_x2') & history.policy.isin(['PPO','DQN','DDQN3'])]
    merged=od[od.policy.isin(['PPO','DQN','DDQN3'])].merge(h,on=['policy','train_seed','workload_seed'],validate='one_to_one',suffixes=('_new','_old'));assert len(merged)==6000
    err=max(float((merged[k+'_new']-merged[k+'_old']).abs().max()) for k in METRICS);assert err<1e-9
    q=pd.read_csv(a.root/'quantum/per_circuit.csv');keys=['backend','n','topology','level','seed'];assert len(q)==72 and not q.duplicated(keys).any()
    assert set(map(tuple,q[keys].itertuples(index=False,name=None)))==set(map(tuple,qp['specs']))
    assert (q.ideal_Z_success==1).all() and (q.missing_error_ops==0).all()
    assert ((q[['proxy_layout','proxy_median','noisy_Z_success','state_fidelity']]>=0)&(q[['proxy_layout','proxy_median','noisy_Z_success','state_fidelity']]<=1+1e-10)).all().all()
    for backend in q.backend.unique():
        bell=q[(q.backend==backend)&(q.n==2)]
        x=bell[bell.topology=='chain'].sort_values(['level','seed']);y=bell[bell.topology=='star'].sort_values(['level','seed'])
        np.testing.assert_array_equal(x[['proxy_layout','proxy_median','noisy_Z_success','state_fidelity']].to_numpy(),y[['proxy_layout','proxy_median','noisy_Z_success','state_fidelity']].to_numpy())
    unique=q[~((q.n==2)&(q.topology=='star'))].copy();assert len(unique)==60
    qrows=[]
    for name,frame in [('all72',q),('bell_deduplicated60',unique)]:
        for backend,g in [('ALL',frame),*list(frame.groupby('backend'))]:
            for target in ['noisy_Z_success','state_fidelity']:
                for proxy in ['proxy_layout','proxy_median']:
                    qrows.append(dict(sample=name,backend=backend,target=target,proxy=proxy,n=len(g),mae=float((g[proxy]-g[target]).abs().mean()),mean_bias=float((g[proxy]-g[target]).mean())))
    pd.DataFrame(qrows).to_csv(a.out/'quantum_proxy_errors.csv',index=False)
    unique.groupby('backend')[['proxy_layout','proxy_median','noisy_Z_success','state_fidelity']].mean().to_csv(a.out/'quantum_means_unique.csv')
    q.to_csv(a.out/'quantum_per_circuit.csv',index=False)
    for part in ['ood04','ablation','quantum']:
        for f in ['protocol.json','COMPLETE.json']:
            dest=a.out/part;dest.mkdir(exist_ok=True);shutil.copy2(a.root/part/f,dest/f)
    shutil.copy2(a.root/'ood04/baseline_selection.json',a.out/'ood04/baseline_selection.json');shutil.copy2(a.root/'ood04/model_hashes.json',a.out/'ood04/model_hashes.json')
    od.to_csv(a.out/'ood04_per_workload.csv',index=False)
    for name in ['ood_smoke','ablation_smoke','quantum_smoke','ood04','ablation','quantum']:
        f=a.root/(name+'.log');inputs[str(f)]=digest(f)
    for f in [a.root/'ood04/per_workload.csv',a.root/'quantum/per_circuit.csv']:inputs[str(f)]=digest(f)
    checks={'complete_markers':6,'ablation_runs':30,'ablation_rows':len(df),'ablation_reload_checks':reloads,'full_retraining_vs_historical_max_errors':original_diffs,'ood_rows':len(od),'ood_historical_drl_rows_matched':6000,'ood_historical_max_metric_error':err,'ood_bootstrap5000_reproduced':True,'quantum_configs':len(q),'quantum_after_bell_deduplication':len(unique),'core_fingerprint':core}
    write_json(a.out/'verification.json',checks);write_json(a.out/'input_hashes.json',inputs)
    # Standalone figures for report integration.
    fig,ax=plt.subplots(figsize=(6,3.5));xx=np.arange(len(c));ax.errorbar(xx,c['diff'],yerr=[c['diff']-c.ci_lo,c.ci_hi-c['diff']],fmt='o',capsize=4);ax.axhline(0,color='gray',linestyle='--');ax.set_xticks(xx,c.variant);ax.set_ylabel('Score difference vs full (95% crossed bootstrap)');fig.tight_layout();fig.savefig(a.out/'ablation_score_difference.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,target,title in zip(axes,['noisy_Z_success','state_fidelity'],['Z success (with readout)','State fidelity (before readout)']):
        for proxy in ['proxy_layout','proxy_median']:ax.scatter(unique[target],unique[proxy],s=14,alpha=.6,label=proxy)
        ax.plot([0,1],[0,1],color='gray',linestyle='--');ax.set(xlabel=title,ylabel='Proxy',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(a.out/'quantum_proxy_targets.png',dpi=180);plt.close(fig)
    write_json(a.out/'COMPLETE.json',dict(validated=True,bootstrap=5000,ablation_family=2,ood_family=6,analyzer_sha256=digest(__file__)))
    print('ABLATION SUMMARY\n',summary.to_string());print('ABLATION COMPARISON\n',c.to_string(index=False));print('OOD\n',oc.to_string(index=False));print('QUANTUM\n',pd.DataFrame(qrows).query("sample=='bell_deduplicated60'").to_string(index=False));print('VERIFIED',a.out)
if __name__=='__main__':main()
