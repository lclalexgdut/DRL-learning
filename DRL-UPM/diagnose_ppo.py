import glob,csv
import numpy as np
import torch
from reader import read_instance
from env_upmsp import UPMSPEnv
from ppo_model import PPO,greedy_action,sample_action

device="cuda" if torch.cuda.is_available() else "cpu"
ckpt=torch.load("weights/ppo_upmsp.pth",map_location=device)
model=PPO(ckpt["obs_dim"],ckpt["action_dim"],ckpt["pad_n"],ckpt["pad_m"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

files=sorted(glob.glob("instances/*.txt"))
dataset=[read_instance(f) for f in files]
envs=[UPMSPEnv(d,ckpt["pad_n"],ckpt["pad_m"]) for d in dataset]
K=len(envs)

def batch_obs():
    return torch.tensor(np.stack([e.state() for e in envs]),
                        dtype=torch.float32,device=device)
def batch_mask():
    return torch.tensor(np.stack([e.action_mask() for e in envs]),device=device)

# ---- 1. greedy rollout: agreement with per-step best action, step regret ----
stats=[]
for f,d,env in zip(files,dataset,envs):
    env.reset()
    agree=steps=0; regret=0.0
    while True:
        mask=env.action_mask()
        s=torch.tensor(env.state(),dtype=torch.float32,device=device).unsqueeze(0)
        m=torch.tensor(mask,device=device).unsqueeze(0)
        a=int(greedy_action(model,s,m).item())
        cost=env.p.copy()
        for mm in range(env.m):
            if env.last_job[mm]>=0:
                cost[mm]=cost[mm]+env.s[mm,env.last_job[mm],:]
        finish=env.machine_time[:env.m,None]+cost
        cur=env.machine_time.max()
        valid=~env.finished[:env.n][None,:]
        delta=np.where(valid,np.maximum(finish,cur)-cur,np.inf)
        min_delta=delta.min()
        dm,dj=np.unravel_index(
            np.argmin(delta*1e6+np.where(np.isinf(delta),0,finish)),delta.shape)
        if a==int(dj*env.pad_m+dm): agree+=1
        regret+=delta[a%env.pad_m,a//env.pad_m]-min_delta
        steps+=1
        _,_,done,_,info=env.step(a)
        if done: break
    stats.append(dict(name=f.split("\\")[-1].split("/")[-1],n=d["n"],m=d["m"],
                      actions=d["n"]*d["m"],ratio=info["makespan"]/env.scale,
                      agree=agree/steps,regret=regret/steps/env.scale))

# ---- 2. same inference budget as GA: sample 32 schedules, keep best ----
NS=32
for e in envs: e.reset()
best=[np.inf]*K; total=[0.0]*K; cnt=[0]*K
while any(c<NS for c in cnt):
    with torch.no_grad():
        a=sample_action(model,batch_obs(),batch_mask()).cpu().numpy()
    for i,e in enumerate(envs):
        if cnt[i]>=NS: continue
        _,_,done,_,info=e.step(int(a[i]))
        if done:
            r=info["makespan"]/e.scale
            cnt[i]+=1; best[i]=min(best[i],r); total[i]+=r
            e.reset()

print(f"{'instance':<24}{'actions':>8}{'greedy':>8}{'agree%':>8}"
      f"{'regret/step':>12}{'best-of-32':>12}{'mean-32':>9}")
out=[]
for st,env,b,t in zip(stats,envs,best,total):
    print(f"{st['name']:<24}{st['actions']:>8}{st['ratio']:>8.2f}"
          f"{st['agree']*100:>8.1f}{st['regret']:>12.4f}{b:>12.2f}{t/NS:>9.2f}")
    out.append([st["name"],st["actions"],round(st["ratio"],3),
                round(st["agree"],4),round(st["regret"],4),
                round(b,3),round(t/NS,3)])
print(f"\nMEAN  greedy {np.mean([s['ratio'] for s in stats]):.2f}   "
      f"agree {np.mean([s['agree'] for s in stats])*100:.1f}%   "
      f"best-of-32 {np.mean(best):.2f}   mean-32 {np.mean([t/NS for t in total]):.2f}")

with open("results/ppo_diagnosis.csv","w",encoding="utf-8-sig",newline="") as f:
    w=csv.writer(f)
    w.writerow(["instance","action_space","greedy_ratio","agree_with_stepwise_best",
                "step_regret_scaled","best_of_32_ratio","mean_of_32_ratio"])
    w.writerows(out)
print("saved results/ppo_diagnosis.csv")
