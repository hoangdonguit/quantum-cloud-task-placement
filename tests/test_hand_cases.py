"""Analytical cases using existing simulator and exact Bellman targets."""
import copy,json,unittest
from pathlib import Path
from collections import deque
from unittest.mock import patch
import numpy as np
import torch
from src.env import QCloudSim
from algorithms.dqn import DQNAgent,Replay
ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'results_summary/cases/base/case_config.json').read_text())

class HandCases(unittest.TestCase):
    def sim(self):
        s=QCloudSim(CFG['env'],'test');row=int(s.allowed[0]);s.exec=s.exec.copy();s.feas=s.feas.copy();s.logF=s.logF.copy();s.exec[row]=2.;s.feas[row]=True;s.logF[row]=0.;return s,row
    def test_fifo_and_reward(self):
        s,row=self.sim();s.reset((np.array([row,row,row]),np.array([0.,1.,5.])))
        rewards=[]
        for _ in range(3):rewards.append(s.step(0)[1])
        np.testing.assert_allclose(np.array(s.rec)[:,:3],[[2,0,2],[3,1,2],[2,0,2]])
        expected=1-np.log1p(np.array([2,3,2])/120)/np.log(2)
        np.testing.assert_allclose(rewards,expected);self.assertAlmostEqual(s.metrics()['score'],expected.mean())
        # Requested two-task reference, with an independent closed-form answer.
        cfg=copy.deepcopy(CFG['env']);cfg.update(beta=1.,t_max=1.,reward_mode='log1p')
        s=QCloudSim(cfg,'test');row=int(s.allowed[0]);s.exec=s.exec.copy();s.logF=s.logF.copy()
        s.exec[row,0]=2.;s.logF[row,s.feas[row]]=0.
        self.assertTrue(s.feas[row,0])
        s.reset((np.array([row,row]),np.array([0.,1.])))
        _,r0,done,_=s.step(0);self.assertFalse(done)
        self.assertEqual(s.next_free[0],2.);self.assertEqual(s.fin[0],[2.])
        _,r1,done,_=s.step(0);self.assertTrue(done)
        np.testing.assert_allclose(s.fin[0],[2.,4.])
        np.testing.assert_allclose(np.array(s.rec)[:,:3],[[2.,0.,2.],[3.,1.,2.]])
        self.assertEqual(s.metrics()['mean_wait'],.5);self.assertEqual(s.metrics()['mean_exec'],2.)
        self.assertAlmostEqual(s.metrics()['score'],-np.log2(3.)/2)
        self.assertAlmostEqual((r0+r1)/2,-np.log2(3.)/2)

    def test_reschedule_delay_penalty_and_fallback(self):
        s,row=self.sim();s.feas[row,1]=False;s.exec[row,1]=np.inf;s.reset((np.array([row]),np.array([0.])))
        rewards=[]
        for k in range(5):
            _,r,done,info=s.step(1);rewards.append(r);self.assertFalse(done);self.assertFalse(info['placed']);self.assertAlmostEqual(r,-1-.1*k)
        _,r,done,_=s.step(1);rewards.append(r);self.assertTrue(done);self.assertEqual(s.rec[0][7],s.biggest)
        self.assertAlmostEqual(s.rec[0][0],7);self.assertAlmostEqual(s.rec[0][1],5);self.assertAlmostEqual(sum(rewards),s.metrics()['score'])
        # Exactly one retry: retain task at first failure, then use valid fallback.
        cfg=copy.deepcopy(CFG['env']);cfg.update(node_names=['washington','perth'],max_resched=1,
            resched_delay=1.,fail_penalty=1.,beta=1.,t_max=1.,reward_mode='log1p')
        s=QCloudSim(cfg,'test');row=int(s.allowed[0]);s.feas=s.feas.copy();s.exec=s.exec.copy();s.logF=s.logF.copy()
        s.feas[row]=[True,False];s.exec[row]=[2.,np.inf];s.logF[row]=[0.,-np.inf]
        self.assertEqual(s.biggest,0);s.reset((np.array([row]),np.array([0.])))
        _,r0,done,info=s.step(1)
        self.assertEqual(r0,-1.);self.assertFalse(done);self.assertFalse(info['placed'])
        self.assertEqual(s.i,0);self.assertEqual(s.cur_arr,1.);self.assertEqual(s.kappa,1)
        self.assertEqual(s.arrival[0],0.);self.assertEqual(len(s.rec),0)
        np.testing.assert_array_equal(s.next_free,[0.,0.])
        _,r1,done,info=s.step(1)
        self.assertTrue(done);self.assertTrue(info['placed']);self.assertEqual(s.rec[0][7],0)
        self.assertEqual(info['T'],3.);self.assertEqual(s.fin[0],[3.]);self.assertEqual(s.metrics()['resched'],1.)
        self.assertAlmostEqual(r1,-1.);self.assertAlmostEqual(r0+r1,-2.)
        self.assertAlmostEqual(s.metrics()['score'],-2.)

    def test_nstep_terminal_tail(self):
        hp=dict(CFG['algos']['DDQN3']);hp.pop('kind');a=DQNAgent(1,2,hp,0);buf=Replay(10,1,2)
        q=deque((np.array([i]),0,r,np.array([i+1]),np.ones(2,bool),i==2) for i,r in enumerate([1,2,3]))
        while q:a._emit(q,buf,.9);q.popleft()
        np.testing.assert_allclose(buf.r[:3],[1+.9*2+.81*3,2+.9*3,3]);np.testing.assert_allclose(buf.g[:3],[.729,.81,.9]);np.testing.assert_array_equal(buf.d[:3],1)
        # Exact three-step reference and the suffixes at a terminal boundary.
        buf=Replay(10,1,2)
        terminal_state=np.array([30.],np.float32);terminal_mask=np.array([False,True])
        q=deque((np.array([i],np.float32),i%2,r,
                 terminal_state if i==2 else np.array([i+1],np.float32),
                 terminal_mask if i==2 else np.ones(2,bool),i==2)
                for i,r in enumerate([2.,4.,8.]))
        while q:a._emit(q,buf,.5);q.popleft()
        self.assertEqual(buf.n,3)
        np.testing.assert_allclose(buf.r[:3],[6.,8.,8.])
        np.testing.assert_allclose(buf.g[:3],[.125,.25,.5])
        np.testing.assert_array_equal(buf.d[:3],[1.,1.,1.])
        np.testing.assert_array_equal(buf.s2[:3],np.tile(terminal_state,(3,1)))
        np.testing.assert_array_equal(buf.m2[:3],np.tile(terminal_mask,(3,1)))
        np.testing.assert_array_equal(buf.s[:3],[[0.],[1.],[2.]])
        np.testing.assert_array_equal(buf.a[:3],[0,1,0])

    def test_masked_vanilla_double_and_terminal_targets(self):
        # Online argmax differs from target argmax; infeasible action has largest Q.
        for double,expected in [(False,3+.9*8),(True,3+.9*2)]:
            hp=dict(CFG['algos']['DQN']);hp.pop('kind');hp.update(batch_size=1,double=double)
            a=DQNAgent(1,3,hp,0)
            with torch.no_grad():
                for net,values in [(a.net,[9,1,100]),(a.tgt,[2,8,999])]:
                    for p in net.parameters():p.zero_()
                    net.adv.bias.copy_(torch.tensor(values,dtype=torch.float32))
            for terminal,target in [(False,expected),(True,3.)]:
                b=Replay(2,1,3);b.add([0],0,3,[0],[True,True,False],terminal,.9)
                real=torch.nn.functional.smooth_l1_loss
                with patch('algorithms.dqn.F.smooth_l1_loss',wraps=real) as loss:
                    a._update(b);self.assertAlmostEqual(loss.call_args.args[1].item(),target,places=5)
        # Drive the real _update target path with exact deterministic Q values.
        for double,nonterminal_target in [(False,5.),(True,3.)]:
            for terminal in [False,True]:
                with self.subTest(reference=True,double=double,terminal=terminal):
                    hp=dict(CFG['algos']['DQN']);hp.pop('kind');hp.update(batch_size=1,double=double)
                    agent=DQNAgent(1,3,hp,0)
                    with torch.no_grad():
                        for net,values in [(agent.net,[9.,3.,5.]),(agent.tgt,[99.,8.,4.])]:
                            for param in net.parameters():param.zero_()
                            net.adv.bias.copy_(torch.tensor(values))
                    replay=Replay(1,1,3);replay.add([0.],1,1.,[1.],[False,True,True],terminal,.5)
                    real=torch.nn.functional.smooth_l1_loss
                    with patch('algorithms.dqn.F.smooth_l1_loss',wraps=real) as loss:
                        agent._update(replay)
                    self.assertEqual(loss.call_count,1)
                    self.assertEqual(loss.call_args.args[1].item(),1. if terminal else nonterminal_target)

class FeatureCases(unittest.TestCase):
    def test_group_transform(self):
        from src.env import obs_groups
        from scripts.post_common import transform
        x=np.arange(55,dtype=np.float32)+1
        for group,size in [('queue',10),('estimates',10),('node_static',25)]:
            y=transform(x,[group],5);indices=obs_groups(5)[group]
            self.assertEqual(len(indices),size);self.assertTrue((y[indices]==0).all())
            keep=np.ones(55,bool);keep[indices]=False
            np.testing.assert_array_equal(y[keep],x[keep]);np.testing.assert_array_equal(x,np.arange(55)+1)
if __name__=='__main__':unittest.main()
