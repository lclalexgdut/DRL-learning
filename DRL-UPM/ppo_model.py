import torch.nn as nn
import torch

class PPO(nn.Module):
    # factorized autoregressive policy: pick a job, then pick a machine
    # conditioned on that job. The machine head reads the per-machine cost
    # slice (processing + setup) of the chosen job directly from the state,
    # so pair preferences no longer squeeze through a 256-dim bottleneck.
    def __init__(self,state_dim,action_dim,pad_n,pad_m):
        super().__init__()
        self.pad_n=pad_n
        self.pad_m=pad_m
        self.body=nn.Sequential(
            nn.Linear(state_dim,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU()
        )
        self.job_head=nn.Linear(256,pad_n)
        self.mach_head=nn.Sequential(
            nn.Linear(256+2*pad_m,128),
            nn.ReLU(),
            nn.Linear(128,pad_m)
        )
        self.critic=nn.Linear(256,1)

def _machine_logits(model,s,mask,h,j):
    pn,pm=model.pad_n,model.pad_m
    idx=3*pn+3*pm+j[:,None]*pm+torch.arange(pm,device=s.device)[None,:]
    feat=torch.cat([h,s[:,3*pn:3*pn+pm],s.gather(1,idx)],dim=1)
    mlogits=model.mach_head(feat)
    mrow=mask.view(-1,pn,pm)[torch.arange(s.shape[0],device=s.device),j]
    return mlogits.masked_fill(~mrow,float("-inf"))

def greedy_action(model,s,mask):
    with torch.no_grad():
        h=model.body(s)
        jlogits=model.job_head(h).masked_fill(
            ~mask.view(-1,model.pad_n,model.pad_m).any(dim=2),float("-inf"))
        j=jlogits.argmax(dim=-1)
        return j*model.pad_m+_machine_logits(model,s,mask,h,j).argmax(dim=-1)

def sample_action(model,s,mask):
    with torch.no_grad():
        h=model.body(s)
        jlogits=model.job_head(h).masked_fill(
            ~mask.view(-1,model.pad_n,model.pad_m).any(dim=2),float("-inf"))
        j=torch.distributions.Categorical(logits=jlogits).sample()
        m=torch.distributions.Categorical(
            logits=_machine_logits(model,s,mask,h,j)).sample()
        return j*model.pad_m+m
