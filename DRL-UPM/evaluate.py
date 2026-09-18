import glob,os,csv
import numpy as np
import torch
from reader import read_instance
from env_upmsp import UPMSPEnv
from ppo_model import PPO,greedy_action
import solvers

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt=torch.load("weights/ppo_upmsp.pth",map_location=device)
model=PPO(ckpt["obs_dim"],ckpt["action_dim"],ckpt["pad_n"],ckpt["pad_m"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

files=sorted(glob.glob("instances/*.txt"))
dataset=[read_instance(f) for f in files]
envs=[UPMSPEnv(d,ckpt["pad_n"],ckpt["pad_m"]) for d in dataset]

GA_SEEDS=(0,1,2)   # independent GA restarts, averaged in the table

def ppo_greedy(env):
    env.reset()
    while True:
        s=torch.tensor(env.state(),dtype=torch.float32,device=device).unsqueeze(0)
        mask=torch.tensor(env.action_mask(),device=device).unsqueeze(0)
        a=int(greedy_action(model,s,mask).item())
        _,_,done,_,info=env.step(a)
        if done:
            return info["makespan"]/env.scale,info["makespan"]

def random_policy(env,episodes=5):
    rng=np.random.default_rng(0)
    ratios=[]
    for _ in range(episodes):
        env.reset()
        while True:
            valid=np.flatnonzero(env.action_mask())
            _,_,done,_,info=env.step(int(rng.choice(valid)))
            if done:
                ratios.append(info["makespan"]/env.scale)
                break
    return float(np.mean(ratios))

def myopic(env):
    # greedy heuristic: always place the job that finishes earliest
    env.reset()
    while True:
        cost=env.p.copy()
        for mm in range(env.m):
            if env.last_job[mm]>=0:
                cost[mm]=cost[mm]+env.s[mm,env.last_job[mm],:]
        finish=env.machine_time[:env.m,None]+cost
        cur=env.machine_time.max()
        score=np.where(~env.finished[:env.n][None,:],np.maximum(finish,cur),np.inf)
        m_idx,j_idx=np.unravel_index(np.argmin(score),score.shape)
        _,_,done,_,info=env.step(int(j_idx*env.pad_m+m_idx))
        if done:
            return info["makespan"]/env.scale

print(f"{'instance':<24}{'n':>4}{'m':>4}{'LB':>8}{'rand':>8}{'SPT':>8}"
      f"{'myopic':>8}{'PPO':>8}{'GA':>8}")
os.makedirs("results",exist_ok=True)
rows=[]
tot=np.zeros(5)
for f,d,env in zip(files,dataset,envs):
    r=random_policy(env)
    c_spt,_=solvers.spt(d)
    my=myopic(env)
    pp,cmax_ppo=ppo_greedy(env)
    ga_c=[solvers.run_ga(d,seed=sd)[0] for sd in GA_SEEDS]
    c_ga=float(np.mean(ga_c))
    ratios=np.array([r,c_spt/env.scale,my,pp,c_ga/env.scale])
    tot+=ratios
    print(f"{os.path.basename(f):<24}{env.n:>4}{env.m:>4}{env.scale:>8.1f}"
          f"{ratios[0]:>8.1f}{ratios[1]:>8.1f}{ratios[2]:>8.1f}"
          f"{ratios[3]:>8.1f}{ratios[4]:>8.1f}")
    rows.append([os.path.basename(f),env.n,env.m,round(env.scale,2),
                 round(r*env.scale,1),round(c_spt,1),round(my*env.scale,1),
                 round(cmax_ppo,1),round(c_ga,1),
                 round(ratios[0],3),round(ratios[1],3),round(ratios[2],3),
                 round(ratios[3],3),round(ratios[4],3)])
k=len(files)
mean=tot/k
print(f"{'MEAN RATIO':<40}{mean[0]:>16.3f}{mean[1]:>8.3f}"
      f"{mean[2]:>8.3f}{mean[3]:>8.3f}{mean[4]:>8.3f}")

header=["instance","n","m","LB","Cmax_random","Cmax_SPT","Cmax_myopic","Cmax_PPO",
        "Cmax_GA","ratio_random","ratio_SPT","ratio_myopic","ratio_PPO","ratio_GA"]
with open("results/evaluation_results.csv","w",encoding="utf-8-sig",newline="") as fcsv:
    w=csv.writer(fcsv)
    w.writerow(header)
    w.writerows(rows)
    w.writerow(["MEAN","","","","","","","","",
                round(mean[0],3),round(mean[1],3),round(mean[2],3),
                round(mean[3],3),round(mean[4],3)])

with open("results/evaluation_results.md","w",encoding="utf-8") as fmd:
    fmd.write("# UP-MSP 求解结果对比\n\n")
    fmd.write("Cmax=完工时间（越小越好）；ratio=Cmax/LB，LB 为仅计加工时间的下界，"
              "换型范围大的实例（S_1-124/S_1-49）下界偏松、比值天然偏高。\n\n")
    fmd.write("SPT 与 GA 共用同一换型感知解码器（按优先级顺序逐作业分配到增量最优机器）："
              "SPT 为静态最短加工时间优先排序；GA 为排列进化（种群 80、150 代、"
              "锦标赛选择 + OX 交叉 + 逆转变异 + 精英保留，3 次独立运行取均值）。"
              "随机策略为 5 次随机排程均值；PPO 为训练后贪心 argmax。\n\n")
    fmd.write("| " + " | ".join(header) + " |\n")
    fmd.write("|" + "---|"*len(header) + "\n")
    for row in rows:
        fmd.write("| " + " | ".join(str(x) for x in row) + " |\n")
    fmd.write(f"| **MEAN** |  |  |  |  |  |  |  |  | "
              f"**{mean[0]:.3f}** | **{mean[1]:.3f}** | **{mean[2]:.3f}** | "
              f"**{mean[3]:.3f}** | **{mean[4]:.3f}** |\n")
print("saved results/evaluation_results.csv and results/evaluation_results.md")
