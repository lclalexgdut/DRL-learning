import glob
import numpy as np
import torch
import torch.nn as nn
from reader import read_instance
from env_upmsp import UPMSPEnv
from ppo_model import PPO,_machine_logits

device="cuda" if torch.cuda.is_available() else "cpu"
ckpt=torch.load("weights/ppo_upmsp.pth",map_location=device)
model=PPO(ckpt["obs_dim"],ckpt["action_dim"],ckpt["pad_n"],ckpt["pad_m"]).to(device)
model.load_state_dict(ckpt["model"]); model.eval()

files=sorted(glob.glob("instances/*.txt"))
dataset=[read_instance(f) for f in files]
envs=[UPMSPEnv(d,ckpt["pad_n"],ckpt["pad_m"]) for d in dataset]
K=len(envs)
pn,pm=ckpt["pad_n"],ckpt["pad_m"]

def batch_obs():
    return torch.tensor(np.stack([e.state() for e in envs]),
                        dtype=torch.float32,device=device)
def batch_mask():
    return torch.tensor(np.stack([e.action_mask() for e in envs]),device=device)

# ---- 1. collect states from the current policy ----
T=1500
obs_buf=np.zeros((T*K,3*pn+3*pm+pn*pm),dtype=np.float32)
from ppo_model import sample_action
for e in envs: e.reset()
for t in range(T):
    with torch.no_grad():
        a=sample_action(model,batch_obs(),batch_mask()).cpu().numpy()
    for i,e in enumerate(envs):
        obs_buf[t*K+i]=e.state()
        _,_,done,_,_=e.step(int(a[i]))
        if done: e.reset()
print(f"collected {len(obs_buf)} states")

# ---- 2. myopic-best (job, machine) targets from the normalized state ----
targets=np.zeros(len(obs_buf),dtype=np.int64)
with torch.no_grad():
    for lo in range(0,len(obs_buf),2048):
        o=torch.tensor(obs_buf[lo:lo+2048],device=device)
        finished=o[:,:pn]>0.5
        mt=o[:,3*pn:3*pn+pm]
        valid_m=o[:,3*pn+2*pm:3*pn+3*pm]>0.5
        cost=o[:,3*pn+3*pm:].view(-1,pn,pm)
        finish=mt[:,None,:]+cost
        cur=mt.max(dim=1).values
        delta=(finish.clamp(min=cur[:,None,None])-cur[:,None,None])
        invalid=finished[:,:,None]|~valid_m[:,None,:]
        delta=delta.masked_fill(invalid,float("inf"))
        score=delta*1e6+torch.where(torch.isinf(delta),0.0,finish)
        flat=score.view(-1,pn*pm).argmin(dim=1)
        targets[lo:lo+2048]=flat.cpu().numpy()
tj=torch.tensor(targets//pm,device=device)
print(f"target job distribution: {pn} slots, entropy done")

# ---- 3. probe arm A: frozen backbone, train job head only ----
with torch.no_grad():
    H=[]
    for lo in range(0,len(obs_buf),4096):
        o=torch.tensor(obs_buf[lo:lo+4096],device=device)
        H.append(model.body(o))
    H=torch.cat(H)
rng=np.random.default_rng(0)
perm=rng.permutation(len(H))
ntr=int(0.9*len(H)); tr,va=perm[:ntr],perm[ntr:]

def train_head(params,feats_fn,epochs,lr,batch=None):
    opt=torch.optim.Adam(params,lr=lr)
    best=0.0; best_state=None
    for ep in range(epochs):
        if batch is None:
            idx=slice(None)
            logits=feats_fn(H[tr])
            loss=nn.functional.cross_entropy(logits,tj[tr])
            opt.zero_grad(); loss.backward(); opt.step()
        else:
            sel=rng.permutation(ntr)[:batch]
            logits=feats_fn(H[tr][sel])
            loss=nn.functional.cross_entropy(logits,tj[tr][sel])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep%20==19 or ep==epochs-1:
            with torch.no_grad():
                acc=(feats_fn(H[va]).argmax(1)==tj[va]).float().mean().item()
            if acc>best: best=acc; best_state={k:v.clone() for k,v in
                         [(n,p) for n,p in params_by_name.items()]}
    return best,best_state

# arm A: linear probe on frozen features
probe=nn.Linear(256,pn).to(device)
params_by_name={"probe":probe.weight,"probe_b":probe.bias}
accA,_=train_head([probe.weight,probe.bias],lambda h:probe(h),epochs=400,lr=1e-3)
print(f"arm A (frozen body, linear probe)   val top-1 agreement: {accA*100:.1f}%")

# arm B: end-to-end, body + job head
model2=PPO(ckpt["obs_dim"],ckpt["action_dim"],pn,pm).to(device)
model2.load_state_dict(ckpt["model"])
for p in model2.mach_head.parameters(): p.requires_grad=False
for p in model2.critic.parameters(): p.requires_grad=False
params_by_name={n:p for n,p in model2.named_parameters() if p.requires_grad}
def feats_fn_full(idx):
    o=torch.tensor(obs_buf[idx],device=device)
    return model2.job_head(model2.body(o))
# train with minibatches over raw states
opt=torch.optim.Adam([p for p in model2.parameters() if p.requires_grad],lr=1e-4)
bestB=0.0
for ep in range(30):
    sel=rng.permutation(ntr)[:4096]
    logits=feats_fn_full(tr[sel])
    loss=nn.functional.cross_entropy(logits,tj[tr][sel])
    opt.zero_grad(); loss.backward(); opt.step()
    if ep%5==4 or ep==29 or ep==0:
        with torch.no_grad():
            accs=[]
            for lo in range(0,len(va),4096):
                vi=va[lo:lo+4096]
                accs.append((feats_fn_full(vi).argmax(1)==tj[vi]).float())
            accB=torch.cat(accs).mean().item()
        print(f"  arm B epoch {ep}: loss {loss.item():.4f} val acc {accB*100:.1f}%")
        bestB=max(bestB,accB)
print(f"arm B (end-to-end supervised)       val top-1 agreement: {bestB*100:.1f}%")

# ---- 4. hybrid rollout: probe job head + PPO machine head ----
def rollout_with_jobhead(head_fn,tag):
    for e in envs: e.reset()
    ratios=np.zeros(K); left=np.ones(K,dtype=bool)
    while left.any():
        s=batch_obs(); mask=batch_mask()
        with torch.no_grad():
            h=model.body(s)
            jl=head_fn(h,s).masked_fill(
                ~mask.view(-1,pn,pm).any(dim=2),float("-inf"))
            j=jl.argmax(dim=-1)
            a=(j*pm+_machine_logits(model,s,mask,h,j).argmax(dim=-1)).cpu().numpy()
        for i,e in enumerate(envs):
            if not left[i]: continue
            _,_,done,_,info=e.step(int(a[i]))
            if done:
                ratios[i]=info["makespan"]/e.scale; left[i]=False; e.reset()
    print(f"{tag:<28} mean ratio {ratios.mean():6.2f}   "
          + " ".join(f"{x:.2f}" for x in ratios))
    return ratios

def probe_head(h,s):
    return probe(h)
rollout_with_jobhead(probe_head,"hybrid A (probe job head)")

def probe_head_full(h,s):
    o=s
    return model2.job_head(model2.body(o))
rollout_with_jobhead(probe_head_full,"hybrid B (supervised job head)")

with torch.no_grad():
    cur=np.zeros(K); left=np.ones(K,dtype=bool)
    for e in envs: e.reset()
    while left.any():
        s=batch_obs(); mask=batch_mask()
        h=model.body(s)
        jl=model.job_head(h).masked_fill(
            ~mask.view(-1,pn,pm).any(dim=2),float("-inf"))
        j=jl.argmax(dim=-1)
        a=(j*pm+_machine_logits(model,s,mask,h,j).argmax(dim=-1)).cpu().numpy()
        for i,e in enumerate(envs):
            if not left[i]: continue
            _,_,done,_,info=e.step(int(a[i]))
            if done:
                cur[i]=info["makespan"]/e.scale; left[i]=False; e.reset()
    print(f"{'current PPO policy':<28} mean ratio {cur.mean():6.2f}   "
          + " ".join(f"{x:.2f}" for x in cur))
