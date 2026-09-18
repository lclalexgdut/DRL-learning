import glob,os,re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from reader import read_instance
from env_upmsp import UPMSPEnv
from ppo_model import PPO,greedy_action

plt.rcParams.update({
    "font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
    "font.family":"sans-serif",
    "axes.unicode_minus":False,
    "hatch.color":"white",
})
os.makedirs("figures",exist_ok=True)

C_DYN  ="#5B0DAD"   # PPO 主曲线（深紫）
C_NODYN="#5BBCCA"   # rollout 曲线（柔和青绿）
C_REF  ="#3D78C2"   # 参考线（独立蓝）
C_CMP  =["#C8C8C8","#707070","#C0392B"]   # 随机/贪心/PPO
C_BEST ="#8B0000"

# ---------- 1. parse train_log.txt ----------
rollout_it,rollout_ratio,speed=[],[],[]
eval_it,eval_ratio,best_mark=[],[],[]
init_eval=None
for line in open("train_log.txt",encoding="utf-8"):
    m=re.match(r"init greedy eval mean Cmax/LB: ([\d.]+)",line)
    if m: init_eval=float(m.group(1))
    m=re.match(r"iter\s+(\d+) \| pi [+-][\d.]+ v [\d.]+ \| rollout Cmax/LB ([\d.]+) \| (\d+) env-steps/s",line)
    if m:
        rollout_it.append(int(m.group(1)))
        rollout_ratio.append(float(m.group(2)))
        speed.append(int(m.group(3)))
    m=re.match(r"iter\s+(\d+) \| greedy eval mean Cmax/LB ([\d.]+)(.*)",line)
    if m:
        eval_it.append(int(m.group(1)))
        eval_ratio.append(float(m.group(2)))
        best_mark.append("best" in m.group(3))

rollout_it=np.array(rollout_it); rollout_ratio=np.array(rollout_ratio)
eval_it=np.array(eval_it); eval_ratio=np.array(eval_ratio)
best_mark=np.array(best_mark)
print(f"parsed: {len(rollout_it)} rollout points, {len(eval_it)} eval points, init eval {init_eval}")

# ---------- 2. reward trend figure ----------
# total return of one episode = -2 * makespan/scale (rewards normalized by instance LB)
fig,axes=plt.subplots(2,1,figsize=(9,6.8),sharex=True,constrained_layout=True)
ax=axes[0]
ax.plot(rollout_it,-2*rollout_ratio,color=C_NODYN,lw=1.6,label="采样策略（训练中）")
ax.plot(eval_it,-2*eval_ratio,color=C_DYN,lw=1.9,label="贪心评估（argmax）")
ax.scatter(eval_it[best_mark],-2*eval_ratio[best_mark],s=14,color=C_DYN,zorder=5)
myo_mean=None   # filled after baselines computed below if available
ax.set_ylabel("平均回合回报（按下界归一化）",fontsize=12,fontweight="bold")
ax.set_title("PPO 训练趋势（reward = $-2\\times$ Cmax/LB，越高越好）",fontsize=12.5)
for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(1.0)
ax.tick_params(direction="out",length=4,width=0.8)
ax.legend(loc="lower right",frameon=True,edgecolor="#999999",fontsize=9.5)

ax=axes[1]
if init_eval is not None:
    ax.scatter([0],[init_eval],s=26,color=C_DYN,zorder=5,label="初始随机策略")
ax.plot(eval_it,eval_ratio,color=C_DYN,lw=1.9)
ax.scatter(eval_it[best_mark],eval_ratio[best_mark],s=14,color=C_BEST,zorder=5,label="历史最优（已保存）")
ax.set_ylabel("贪心评估 Cmax / 下界",fontsize=12,fontweight="bold")
ax.set_xlabel("PPO 迭代轮次",fontsize=12,fontweight="bold")
for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(1.0)
ax.tick_params(direction="out",length=4,width=0.8)
ax.legend(loc="upper right",frameon=True,edgecolor="#999999",fontsize=9.5)
fig.savefig("figures/fig1_training_trend.png",dpi=300)
plt.close(fig)
print("saved figures/fig1_training_trend.png")

# ---------- 3. baselines + per-instance comparison ----------
device=torch.device("cpu")
ckpt=torch.load("weights/ppo_upmsp.pth",map_location=device)
model=PPO(ckpt["obs_dim"],ckpt["action_dim"],ckpt["pad_n"],ckpt["pad_m"]).to(device)
model.load_state_dict(ckpt["model"]); model.eval()

files=sorted(glob.glob("instances/*.txt"))
envs=[UPMSPEnv(read_instance(f),ckpt["pad_n"],ckpt["pad_m"]) for f in files]

def ppo_greedy(env):
    env.reset()
    while True:
        s=torch.tensor(env.state(),dtype=torch.float32,device=device).unsqueeze(0)
        mask=torch.tensor(env.action_mask(),device=device).unsqueeze(0)
        a=int(greedy_action(model,s,mask).item())
        _,_,done,_,info=env.step(a)
        if done:
            return info["makespan"]/env.scale,info

def random_policy(env,episodes=5):
    rng=np.random.default_rng(0); ratios=[]
    for _ in range(episodes):
        env.reset()
        while True:
            valid=np.flatnonzero(env.action_mask())
            _,_,done,_,info=env.step(int(rng.choice(valid)))
            if done: ratios.append(info["makespan"]/env.scale); break
    return float(np.mean(ratios))

def myopic(env):
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
        if done: return info["makespan"]/env.scale

R_rand,R_my,R_ppo=[],[],[]
for env in envs:
    R_rand.append(random_policy(env))
    R_my.append(myopic(env))
    r,_=ppo_greedy(env)
    R_ppo.append(r)
R_rand,R_my,R_ppo=map(np.array,(R_rand,R_my,R_ppo))
print(f"means: random {R_rand.mean():.3f}  myopic {R_my.mean():.3f}  ppo {R_ppo.mean():.3f}")

def short(name):
    b=name.replace("I_","").replace(".txt","").split("_")
    return f"{b[0]}×{b[1]}\nS:{b[3]}"

labels=[short(os.path.basename(f)) for f in files]
x=np.arange(len(files)); w=0.26
fig,ax=plt.subplots(figsize=(12.5,4.6),constrained_layout=True)
bars=ax.bar(x-w,R_rand,w,color=C_CMP[0],edgecolor="white",label=f"随机策略（均值 {R_rand.mean():.1f}）")
ax.bar(x,R_my,w,color=C_CMP[1],edgecolor="white",label=f"贪心启发式（均值 {R_my.mean():.1f}）")
ax.bar(x+w,R_ppo,w,color=C_CMP[2],edgecolor="white",hatch="//",label=f"PPO 本次训练（均值 {R_ppo.mean():.1f}）")
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=8)
ax.set_ylabel("Cmax / 下界（对数轴，越低越好）",fontsize=12,fontweight="bold")
ax.set_title("各实例最终排程质量对比",fontsize=12.5)
ax.grid(axis="y",color="#EBEBEB",linestyle="--",lw=0.7,zorder=0)
for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.8); sp.set_color("#7A7A7A"); sp.set_zorder(3)
ax.tick_params(direction="out",length=4,width=0.8)
ax.legend(loc="upper right",frameon=True,edgecolor="#999999",fontsize=9.5)
fig.savefig("figures/fig2_instance_comparison.png",dpi=300)
plt.close(fig)
print("saved figures/fig2_instance_comparison.png")

# ---------- 4. Gantt charts (setup + processing segments) ----------
def gantt_segments(env,info):
    segs=[]; 
    for m,jobs in enumerate(info["solution"]):
        t=0.0; last=-1
        for j in jobs:
            su=env.s[m,last,j] if last>=0 else 0.0
            p=env.p[m,j]
            if su>0: segs.append((m,t,t+su,j,True))
            segs.append((m,t+su,t+su+p,j,False))
            t+=su+p; last=j
    return segs

show=[13,5]  # I_6_3_S_1-124_3 (PPO beats myopic), I_200_10_S_1-9_10 (largest)
fig,axes=plt.subplots(len(show),1,figsize=(12.5,7.2),constrained_layout=True,
                      gridspec_kw={"height_ratios":[1,3]})
cmap=plt.get_cmap("tab20")
for ax,si in zip(axes,show):
    env=envs[si]
    r,info=ppo_greedy(env)
    myo=myopic(env)
    segs=gantt_segments(env,info)
    for m,t0,t1,j,setup in segs:
        if setup:
            ax.add_patch(plt.Rectangle((t0,m-0.36),t1-t0,0.72,facecolor="#c8c8c8",
                                       edgecolor="white",lw=0.4,hatch="///"))
        else:
            ax.add_patch(plt.Rectangle((t0,m-0.36),t1-t0,0.72,facecolor=cmap(j%20),
                                       edgecolor="white",lw=0.4))
    cmax=info["makespan"]
    ax.axvline(cmax,color=C_REF,lw=1.4,linestyle="--")
    ax.set_ylim(-0.6,env.m-0.4+ (0 if env.m>1 else 0))
    ax.set_yticks(range(env.m)); ax.set_ylabel("机器",fontsize=11,fontweight="bold")
    ax.set_xlim(0,cmax*1.03)
    ax.set_title(f"{os.path.basename(files[si])}   PPO Cmax={cmax:.0f}（{r:.2f}×下界）  "
                 f"贪心启发式={myo:.2f}×下界",fontsize=11)
    ax.grid(axis="x",color="#EBEBEB",lw=0.6,zorder=0)
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.8)
    ax.tick_params(direction="out",length=3,width=0.7)
    if si==show[-1]: ax.set_xlabel("时间",fontsize=12,fontweight="bold")
axes[0].legend(handles=[Patch(facecolor="#c8c8c8",hatch="///",edgecolor="white",label="换型时间 setup"),
                        Patch(facecolor=cmap(0),label="作业加工（颜色=作业编号）")],
               loc="upper right",frameon=True,edgecolor="#999999",fontsize=9)
fig.savefig("figures/fig3_gantt.png",dpi=300)
plt.close(fig)
print("saved figures/fig3_gantt.png")
print(f"training speed: {int(np.mean(speed))} env-steps/s avg")
