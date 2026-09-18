import glob,time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from reader import read_instance
from env_upmsp import UPMSPEnv
from ppo_model import PPO

# ---------------- hyperparameters ----------------
ITERS=2000       # PPO iterations
ROLLOUT=256      # lockstep env steps collected per iteration (x15 instances)
EPOCHS=4         # optimization passes over each rollout
MINIBATCH=1024
LR=3e-4
GAMMA=1.0        # episodic task: total return is exactly -2*makespan
LAMBDA=0.95      # GAE lambda
CLIP=0.2
ENT_COEF=0.01
VAL_COEF=0.5
MAX_GRAD=0.5
EVAL_EVERY=50
SEED=0
# -------------------------------------------------

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

files=sorted(glob.glob("instances/*.txt"))
dataset=[read_instance(x) for x in files]

# one shared network: states/actions of every instance are padded to the
# largest one, invalid (job,machine) pairs are masked out before sampling
pad_n=max(d["n"] for d in dataset)
pad_m=max(d["m"] for d in dataset)
envs=[UPMSPEnv(d,pad_n,pad_m) for d in dataset]
K=len(envs)
A=pad_n*pad_m
obs_dim=envs[0].observation_space.shape[0]

model=PPO(obs_dim,A,pad_n,pad_m).to(device)
opt=torch.optim.Adam(model.parameters(),lr=LR,eps=1e-5)

COST_BASE=3*pad_n+3*pad_m   # offset of the flattened cost block in the state
MT_BASE=3*pad_n             # offset of the machine-load block

def policy_value(s,mask,job=None):
    # autoregressive factorization: pi(a) = pi_job(j) * pi_machine(m | j)
    h=model.body(s)
    jmask=mask.view(-1,pad_n,pad_m).any(dim=2)
    jlogits=model.job_head(h).masked_fill(~jmask,float("-inf"))
    jdist=torch.distributions.Categorical(logits=jlogits)
    if job is None:
        job=jdist.sample()
    B=s.shape[0]
    idx=COST_BASE+job[:,None]*pad_m+torch.arange(pad_m,device=s.device)[None,:]
    feat=torch.cat([h,s[:,MT_BASE:MT_BASE+pad_m],s.gather(1,idx)],dim=1)
    mlogits=model.mach_head(feat)
    mrow=mask.view(-1,pad_n,pad_m)[torch.arange(B,device=s.device),job]
    mlogits=mlogits.masked_fill(~mrow,float("-inf"))
    mdist=torch.distributions.Categorical(logits=mlogits)
    return jdist,mdist,job,model.critic(h).squeeze(-1)

def greedy_action(s,mask):
    with torch.no_grad():
        h=model.body(s)
        jlogits=model.job_head(h).masked_fill(
            ~mask.view(-1,pad_n,pad_m).any(dim=2),float("-inf"))
        job=jlogits.argmax(dim=-1)
        idx=COST_BASE+job[:,None]*pad_m+torch.arange(pad_m,device=s.device)[None,:]
        feat=torch.cat([h,s[:,MT_BASE:MT_BASE+pad_m],s.gather(1,idx)],dim=1)
        mlogits=model.mach_head(feat)
        mrow=mask.view(-1,pad_n,pad_m)[torch.arange(s.shape[0],device=s.device),job]
        mlogits=mlogits.masked_fill(~mrow,float("-inf"))
        return job*pad_m+mlogits.argmax(dim=-1)

def batch_obs():
    return torch.tensor(np.stack([e.state() for e in envs]),dtype=torch.float32,device=device)

def batch_mask():
    return torch.tensor(np.stack([e.action_mask() for e in envs]),device=device)

def evaluate():
    # deterministic greedy (argmax) episode on every instance, in lockstep
    for e in envs: e.reset()
    ratios=np.zeros(K)
    left=np.ones(K,dtype=bool)
    while left.any():
        action=greedy_action(batch_obs(),batch_mask()).cpu().numpy()
        for i,e in enumerate(envs):
            if not left[i]:
                continue
            _,_,done,_,info=e.step(int(action[i]))
            if done:
                ratios[i]=info["makespan"]/e.scale
                left[i]=False
                e.reset()
    return ratios

obs_buf=np.zeros((ROLLOUT,K,obs_dim),dtype=np.float32)
mask_buf=np.zeros((ROLLOUT,K,A),dtype=bool)
act_buf=np.zeros((ROLLOUT,K),dtype=np.int64)
logp_buf=np.zeros((ROLLOUT,K),dtype=np.float32)
val_buf=np.zeros((ROLLOUT,K),dtype=np.float32)
rew_buf=np.zeros((ROLLOUT,K),dtype=np.float32)
done_buf=np.zeros((ROLLOUT,K),dtype=np.float32)

makespans=[deque(maxlen=50) for _ in range(K)]

best=evaluate().mean()
saved=False
print(f"device={device} instances={K} obs_dim={obs_dim} actions={A}",flush=True)
print(f"init greedy eval mean Cmax/LB: {best:.3f}",flush=True)

log_t=time.time()
for it in range(1,ITERS+1):

    for g in opt.param_groups:
        g["lr"]=LR*(1.0-(it-1)/ITERS)
    ent_coef=ENT_COEF*(0.1+0.9*(1.0-(it-1)/ITERS))

    # -------- collect one rollout, all instances in lockstep --------
    for t in range(ROLLOUT):
        obs_np=np.stack([e.state() for e in envs])
        mask_np=np.stack([e.action_mask() for e in envs])
        s=torch.tensor(obs_np,dtype=torch.float32,device=device)
        mask=torch.tensor(mask_np,device=device)

        with torch.no_grad():
            jdist,mdist,job,value=policy_value(s,mask)
            mach=mdist.sample()
            action=job*pad_m+mach
            logp=jdist.log_prob(job)+mdist.log_prob(mach)
        a_np=action.cpu().numpy()
        lv=torch.stack([logp,value],dim=-1).cpu().numpy()

        obs_buf[t]=obs_np
        mask_buf[t]=mask_np
        act_buf[t]=a_np
        logp_buf[t]=lv[:,0]
        val_buf[t]=lv[:,1]

        for i,e in enumerate(envs):
            _,r,done,_,info=e.step(int(a_np[i]))
            rew_buf[t,i]=r/e.scale
            done_buf[t,i]=float(done)
            if done:
                makespans[i].append(info["makespan"]/e.scale)
                e.reset()

    # -------- GAE --------
    with torch.no_grad():
        _,_,_,next_val=policy_value(batch_obs(),batch_mask())
        next_val=next_val.cpu().numpy()

    adv=np.zeros((ROLLOUT,K),dtype=np.float32)
    gae=np.zeros(K,dtype=np.float32)
    for t in reversed(range(ROLLOUT)):
        nextv=val_buf[t+1] if t<ROLLOUT-1 else next_val
        nonterm=1.0-done_buf[t]
        delta=rew_buf[t]+GAMMA*nextv*nonterm-val_buf[t]
        gae=delta+GAMMA*LAMBDA*nonterm*gae
        adv[t]=gae
    ret=adv+val_buf

    # -------- clipped PPO update --------
    b_obs=torch.tensor(obs_buf.reshape(-1,obs_dim),device=device)
    b_mask=torch.tensor(mask_buf.reshape(-1,A),device=device)
    b_act=torch.tensor(act_buf.reshape(-1),device=device)
    b_logp=torch.tensor(logp_buf.reshape(-1),device=device)
    b_val=torch.tensor(val_buf.reshape(-1),device=device)
    b_ret=torch.tensor(ret.reshape(-1),dtype=torch.float32,device=device)
    b_adv=torch.tensor(adv.reshape(-1),dtype=torch.float32,device=device)
    b_adv=(b_adv-b_adv.mean())/(b_adv.std()+1e-8)

    n=b_obs.shape[0]
    idx=np.arange(n)
    pg_l=v_l=0.0
    updates=0
    for _ in range(EPOCHS):
        np.random.shuffle(idx)
        for lo in range(0,n,MINIBATCH):
            mb=torch.tensor(idx[lo:lo+MINIBATCH],device=device)
            job=b_act[mb]//pad_m
            mach=b_act[mb]%pad_m
            jdist,mdist,_,value=policy_value(b_obs[mb],b_mask[mb],job=job)
            logp=jdist.log_prob(job)+mdist.log_prob(mach)
            ratio=(logp-b_logp[mb]).exp()

            adv_mb=b_adv[mb]
            pg_loss=torch.max(-adv_mb*ratio,
                              -adv_mb*torch.clamp(ratio,1-CLIP,1+CLIP)).mean()

            v_clip=b_val[mb]+torch.clamp(value-b_val[mb],-CLIP,CLIP)
            v_loss=0.5*torch.max((value-b_ret[mb])**2,
                                 (v_clip-b_ret[mb])**2).mean()

            # exact job entropy + MC estimate of E_j~pi[H(m|j)]
            jsamp=jdist.sample().detach()
            _,mdist2,_,_=policy_value(b_obs[mb],b_mask[mb],job=jsamp)
            ent=(jdist.entropy()+mdist2.entropy()).mean()

            loss=pg_loss+VAL_COEF*v_loss-ent_coef*ent

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),MAX_GRAD)
            opt.step()

            pg_l+=pg_loss.item(); v_l+=v_loss.item(); updates+=1

    if it%20==0:
        recent=[np.mean(m) for m in makespans if m]
        rate=20*ROLLOUT*K/(time.time()-log_t)
        print(f"iter {it:4d} | pi {pg_l/updates:+.4f} v {v_l/updates:.4f} "
              f"| rollout Cmax/LB {np.mean(recent):.3f} "
              f"| {rate:.0f} env-steps/s",flush=True)
        log_t=time.time()

    if it%EVAL_EVERY==0 or it==ITERS:
        ratios=evaluate()
        score=ratios.mean()
        mark=""
        if score<best:
            best=score
            mark="  <- best, saved"
            saved=True
            torch.save({"model":model.state_dict(),"arch":"factored",
                        "obs_dim":obs_dim,"action_dim":A,
                        "pad_n":pad_n,"pad_m":pad_m},
                       "weights/ppo_upmsp.pth")
        print(f"iter {it:4d} | greedy eval mean Cmax/LB {score:.3f}{mark} | "
              +" ".join(f"{x:.2f}" for x in ratios),flush=True)

if not saved:
    torch.save({"model":model.state_dict(),"arch":"factored",
                "obs_dim":obs_dim,"action_dim":A,
                "pad_n":pad_n,"pad_m":pad_m},
               "weights/ppo_upmsp.pth")
print(f"training finished, best mean Cmax/LB {best:.3f}",flush=True)
