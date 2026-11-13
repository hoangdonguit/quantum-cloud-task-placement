"""DQN base paired retraining with identical observation transform at every stage."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from algorithms.dqn import DQNAgent
from src.env import QCloudGymEnv
from src.provenance import run_fingerprint
from experiments.evaluate import eval_workloads
from post_common import transform,digest,write_json

class FeatureEnv(QCloudGymEnv):
    def __init__(self,cfg,groups):super().__init__(cfg);self.groups=groups
    def reset(self,**kwargs):
        o,i=super().reset(**kwargs);return transform(o,self.groups,self.sim.M),i
    def step(self,a):
        o,r,t,u,i=super().step(a);return transform(o,self.groups,self.sim.M),r,t,u,i

def load_variant(run):
    """Only supported loader: reads the persisted transform alongside weights."""
    run=Path(run);cfg=json.loads((run/'config.json').read_text());agent=DQNAgent.load(run/'model.pt');groups=cfg['drop_groups'];n=len(cfg['env']['node_names'])
    return lambda o,m:agent.predict(transform(o,groups,n),m)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--node-static',action='store_true');p.add_argument('--seeds',type=int,default=10);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path('results_summary/cases/base/case_config.json').read_text());steps=1100 if a.smoke else cfg['train']['timesteps'];val=list(range(5000,5002 if a.smoke else 5020));test=list(range(30000,30002 if a.smoke else 30200));nseeds=1 if a.smoke else a.seeds
    variants={'full':[],'no_queue':['queue'],'no_estimates':['estimates']}
    if a.node_static:variants['no_node_static']=['node_static']
    write_json(a.out/'protocol.json',{'smoke_only':a.smoke,'steps':steps,'validation_seeds':val,'test_seeds':test,'training_seeds':list(range(nseeds)),'variants':variants,'full_reuse':False,'checkpoint_selection':'final checkpoint, same as main','script_sha256':digest(__file__),'helper_sha256':digest(Path(__file__).with_name('post_common.py')),'historical_core_fingerprint':run_fingerprint(cfg),'baseline':'Reuse frozen base selection only; no retuning against test','resume':False})
    for name,groups in variants.items():
        for seed in range(nseeds):
            out=a.out/name/f'DQN_seed{seed}';out.mkdir(parents=True)
            spec=dict(cfg['algos']['DQN']);spec.pop('kind');env=FeatureEnv(cfg['env'],groups);ag=DQNAgent(env.sim.obs_dim,env.sim.M,spec,seed);curve=[]
            config=dict(cfg,drop_groups=groups,training_seed=seed,actual_steps=steps)
            write_json(out/'config.json',config)
            fingerprint=__import__('hashlib').sha256((json.dumps(config,sort_keys=True)+digest(__file__)+digest(Path(__file__).with_name('post_common.py'))+run_fingerprint(cfg)).encode()).hexdigest()
            def validation(actor):
                transformed=lambda o,m:actor(transform(o,groups,env.sim.M),m)
                return {'val_score':np.mean([r['score'] for r in eval_workloads(cfg['env'],'val',val,act_fn=transformed)])}
            start=time.monotonic();ag.learn(env,steps,eval_fn=validation,eval_every=steps if a.smoke else cfg['train']['eval_every'],log=lambda t,r:curve.append(dict(timesteps=t,**r)));ag.save(out/'model.pt')
            actor=load_variant(out);original=lambda o,m:ag.predict(transform(o,groups,env.sim.M),m);obs,_=env.reset(seed=409)
            assert actor(obs,env.action_masks())==original(obs,env.action_masks())
            pd.DataFrame(curve).to_csv(out/'train_log.csv',index=False);pd.DataFrame(eval_workloads(cfg['env'],'test',test,act_fn=actor)).to_csv(out/'eval_per_workload.csv',index=False)
            write_json(out/'run_meta.json',{'fingerprint':fingerprint,'train_seconds':time.monotonic()-start,'smoke_only':a.smoke,'model_sha256':digest(out/'model.pt')});print(name,seed,'DONE',flush=True)
    write_json(a.out/'COMPLETE.json',{'smoke_only':a.smoke,'runs':len(variants)*nseeds})
if __name__=='__main__':main()
