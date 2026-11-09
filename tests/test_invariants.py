import copy,json,unittest
from pathlib import Path
import numpy as np
from src.env import QCloudSim,time_cost,QCloudGymEnv
from src.data import get_pool
from src.heuristics import REGISTRY,lexmin
ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/base.json').read_text())['env']
class Invariants(unittest.TestCase):
 def test_split(self):
  keys=lambda p:set(zip(p.algo,p.width))
  a,b,c=[keys(get_pool(s,'circuit')) for s in ('train','val','test')]
  self.assertTrue(a and b and c);self.assertFalse(a&b or a&c or b&c)
 def test_determinism(self):
  s=QCloudSim(CFG,'test');a=s.make_workload(409);b=s.make_workload(409)
  for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
 def test_lexicographic(self):
  self.assertEqual(lexmin(np.array([True,True]),np.array([0,1]),np.array([1e9,0])),0)
 def test_reward_monotonic(self):
  x=time_cost(np.array([119,120,121,1200]),120,'log1p');self.assertTrue(np.all(np.diff(x)>0))
 def test_clock_and_reward(self):
  for name in REGISTRY:
   s=QCloudSim(CFG,'test');s.reset(s.make_workload(1000));p=REGISTRY[name]();p.reset(s,1000);done=False;clock=s.cur_arr;reward=0
   while not done:
    obs,r,done,_=s.step(p(s));reward+=r
    self.assertGreaterEqual(s.cur_arr,clock);clock=s.cur_arr;self.assertTrue(np.isfinite(obs).all())
   m=s.metrics();self.assertAlmostEqual(reward/s.n,m['score']);self.assertAlmostEqual(m['mean_T'],m['mean_wait']+m['mean_exec'])
 def test_no_mutation(self):
  s=QCloudSim(CFG,'test');s.reset(s.make_workload(409));a=s.next_free.copy();s.view();s.obs();s.action_masks();np.testing.assert_array_equal(a,s.next_free)
 def test_feasible_policies(self):
  for name in REGISTRY:
   if name=='DRLQ_GreedyWait':continue
   s=QCloudSim(CFG,'test');s.reset(s.make_workload(409));p=REGISTRY[name]();p.reset(s,409);done=False
   while not done:
    a=p(s);self.assertTrue(s.action_masks()[a]);_,_,done,_=s.step(a)
 def test_observation_space(self):
  e=QCloudGymEnv(CFG);o,_=e.reset(seed=409);self.assertTrue(e.observation_space.contains(o))
if __name__=='__main__':unittest.main()
