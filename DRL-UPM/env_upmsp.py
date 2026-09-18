import gymnasium as gym
from gymnasium import spaces
import numpy as np

class UPMSPEnv(gym.Env):

    def __init__(self,data,pad_n=None,pad_m=None):
        self.n=data["n"]
        self.m=data["m"]
        self.p=data["processing"]
        self.s=data["setup"]

        # pad all instances to one shared shape so a single network can serve them
        self.pad_n=pad_n or self.n
        self.pad_m=pad_m or self.m

        # rough makespan lower bound: every job needs at least its fastest
        # processing time, spread over m machines. Used to normalize rewards
        # and machine loads across instances of very different sizes.
        self.scale=float(self.p.min(axis=0).sum()/self.m)+1e-6

        # static job features (filled once, finished flag updated in step)
        self.job_feat=np.zeros((3,self.pad_n),dtype=np.float32)
        self.job_feat[1,:self.n]=self.p.min(axis=0)/self.scale
        self.job_feat[2,:self.n]=self.p.mean(axis=0)/self.scale

        self.action_space=spaces.Discrete(self.pad_n*self.pad_m)
        self.observation_space=spaces.Box(
            -np.inf,np.inf,
            shape=(3*self.pad_n+3*self.pad_m+self.pad_n*self.pad_m,),
            dtype=np.float32
        )

    def reset(self,seed=None):
        self.finished=np.zeros(self.pad_n,dtype=bool)
        self.finished[self.n:]=True
        self.machine_time=np.zeros(self.pad_m,dtype=np.float32)
        self.machine_jobs=[[] for _ in range(self.m)]
        self.last_job=[-1]*self.m
        self.job_feat[0,:self.n]=0
        return self.state(),{}

    def state(self):
        counts=np.zeros(self.pad_m,dtype=np.float32)
        counts[:self.m]=[len(x) for x in self.machine_jobs]
        valid=np.zeros(self.pad_m,dtype=np.float32)
        valid[:self.m]=1

        # incremental cost of assigning each job to each machine, including
        # the setup from that machine's last job. Job-major, same layout as
        # the flattened action space. This is the key decision feature.
        cost=np.zeros((self.pad_n,self.pad_m),dtype=np.float32)
        for mm in range(self.m):
            base=self.p[mm]
            if self.last_job[mm]>=0:
                base=base+self.s[mm,self.last_job[mm],:]
            cost[:self.n,mm]=base

        return np.concatenate([
            self.job_feat[0],
            self.job_feat[1],
            self.job_feat[2],
            self.machine_time/self.scale,
            counts/self.n,
            valid,
            (cost/self.scale).reshape(-1)
        ]).astype(np.float32)

    def action_mask(self):
        mask=np.zeros((self.pad_n,self.pad_m),dtype=bool)
        mask[:self.n,:self.m]=~self.finished[:self.n,None]
        return mask.reshape(-1)

    def step(self,action):
        job=action//self.pad_m
        machine=action%self.pad_m

        if self.finished[job] or machine>=self.m:
            return self.state(),-20.0,False,False,{}

        old=self.machine_time.max()

        cost=self.p[machine,job]

        if self.last_job[machine]>=0:
            cost+=self.s[machine,self.last_job[machine],job]

        self.machine_time[machine]+=cost
        self.machine_jobs[machine].append(job)
        self.last_job[machine]=job
        self.finished[job]=True
        self.job_feat[0,job]=1

        new=self.machine_time.max()

        reward=-(new-old)

        done=bool(self.finished[:self.n].all())

        if done:
            reward-=new

        return self.state(),reward,done,False,{
            "makespan":new,
            "solution":self.machine_jobs
        }
