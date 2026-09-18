import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({
    "font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
    "font.family":"sans-serif",
    "axes.unicode_minus":False,
})

# palette (matches project figures, colour-blind-safe tint fills)
C_NET="#5B0DAD"; C_NETF="#EFE6F8"      # agent network: purple
C_ENV="#2E8F94"; C_ENVF="#E3F4F5"      # environment / data: teal
C_GAE="#3D78C2"; C_GAEF="#E8F0FB"      # GAE: blue
C_UPD="#C0392B"; C_UPDF="#FBEAE7"      # PPO update: red
C_EVAL="#707070"; C_EVALF="#F0F0F0"    # evaluation: gray
C_ARR="#444444"

fig,ax=plt.subplots(figsize=(13.5,8.37),constrained_layout=True)
ax.set_xlim(0,100); ax.set_ylim(0,62); ax.set_aspect("equal"); ax.axis("off")

def box(x1,y1,x2,y2,face,edge,title,lines,title_fs=10.5,body_fs=8.8,lh=2.2):
    ax.add_patch(FancyBboxPatch((x1,y1),x2-x1,y2-y1,
        boxstyle="round,pad=0.35,rounding_size=1.1",
        facecolor=face,edgecolor=edge,lw=1.6,zorder=2))
    cx=(x1+x2)/2
    ax.text(cx,y2-1.7,title,ha="center",va="center",fontsize=title_fs,
            fontweight="bold",color=edge,zorder=4)
    for i,t in enumerate(lines):
        ax.text(cx,y2-3.6-i*lh,t,ha="center",va="center",fontsize=body_fs,
                color="#222222",zorder=4)

def subbox(x1,y1,x2,y2,text,edge,fs=8.6):
    ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,facecolor="white",
        edgecolor=edge,lw=1.0,zorder=3))
    ax.text((x1+x2)/2,(y1+y2)/2,text,ha="center",va="center",fontsize=fs,
            color="#222222",zorder=4)

def arrow(p1,p2):
    ax.annotate("",xy=p2,xytext=p1,
        arrowprops=dict(arrowstyle="-|>",color=C_ARR,lw=1.7,
                        shrinkA=0,shrinkB=0),zorder=3)

def polyline(pts,end=None,label=None,lx=None,ly=None,fs=8.5):
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    ax.plot(xs,ys,color=C_ARR,lw=1.7,zorder=3,solid_capstyle="round")
    if end is not None:
        ax.annotate("",xy=end,xytext=pts[-1],
            arrowprops=dict(arrowstyle="-|>",color=C_ARR,lw=1.7,
                            shrinkA=0,shrinkB=0),zorder=3)
    if label:
        ax.text(lx,ly,label,ha="center",va="center",fontsize=fs,color="#333333",
                bbox=dict(fc="white",ec="none",alpha=0.9),zorder=4)

# ---------------- title ----------------
ax.text(50,60.3,"PPO 求解无关并行机调度（UP-MSP）算法机制图",
        ha="center",va="center",fontsize=14.5,fontweight="bold")
ax.text(50,57.9,"对应 train.py · env_upmsp.py · ppo_model.py 的实现",
        ha="center",va="center",fontsize=9.2,color="#666666")

# ---------------- upper band: interaction loop ----------------
box(2,37,15,50,C_ENVF,C_ENV,"实例库",
    ["instances/*.txt","15 个实例","n∈[6,200]","m∈[2,30]  加工 p + 换型 s"],body_fs=8.2)

box(19,36,38,50,C_ENVF,C_ENV,"环境 UPMSPEnv",
    ["统一 pad 至 200×30","观测 6690 维（含成本矩阵）","action_mask() 动作掩码",
     "step(): r = −ΔCmax"],body_fs=8.4)

# agent network with three heads
box(44,33,68,52,C_NETF,C_NET,"策略网络 ppo_model.PPO（自回归分解）",[],title_fs=10)
subbox(46,45.8,66,49.2,"MLP 主干：6690 → 256 → 256",C_NET)
subbox(46,42.0,66,45.0,"Job 头：256 → 200 选作业 π(j)\n掩码已完工作业 → Categorical",C_NET,fs=8.0)
subbox(46,37.8,66,41.2,"Machine 头：读该作业成本切片+负载\n→ 30 选机器 π(m|j)",C_NET,fs=8.0)
subbox(46,34.2,66,37.2,"Critic 头：状态价值 V(s_t)   π(a)=π(j)·π(m|j)",C_NET,fs=8.0)
ax.text(66.4,50.9,"GPU",ha="center",va="center",fontsize=7.2,color="white",
        bbox=dict(boxstyle="round,pad=0.25",fc=C_NET,ec="none"),zorder=5)

box(76,36,95,50,C_EVALF,C_EVAL,"评估与部署",
    ["每 50 轮：15 实例贪心 argmax","保存最优 weights/ppo_upmsp.pth",
     "test.py 排程 · 甘特图"],body_fs=8.4)

# ---------------- lower band: training data flow ----------------
box(64,6,86,20,C_ENVF,C_ENV,"Rollout 缓冲区",
    ["15 实例 lockstep 同步采集","256 步 × 15 实例 = 3840 样本/轮",
     "GPU 批量前向 ≈ 5100 步/秒"],body_fs=8.4)
ax.text(84.4,18.6,"GPU",ha="center",va="center",fontsize=7.2,color="white",
        bbox=dict(boxstyle="round,pad=0.25",fc=C_ENV,ec="none"),zorder=5)

box(40,6,58,20,C_GAEF,C_GAE,"GAE 优势估计",
    ["δ_t = r_t + γV(s_{t+1}) − V(s_t)","γ = 1.0（回合制任务）, λ = 0.95",
     "优势 Â 标准化, 回报 R = Â + V"],body_fs=8.4)

box(15,6,34,20,C_UPDF,C_UPD,"PPO 参数更新",
    ["L = clip 代理 + 0.5·价值损失 − β·熵","ε = 0.2, 熵系数 0.01→0.001 退火",
     "4 epochs × minibatch 1024","Adam 3e-4 线性退火 · 梯度裁剪"],body_fs=8.0)

# ---------------- legend ----------------
box(2,6,13,20,"white","#BBBBBB","图例",[],title_fs=9.5)
leg=[("智能体网络",C_NET),("环境/数据",C_ENV),
     ("GAE",C_GAE),("PPO 更新",C_UPD),("评估部署",C_EVAL)]
for i,(t,c) in enumerate(leg):
    y=15.6-i*2.5
    ax.add_patch(Rectangle((3.2,y-0.8),1.9,1.6,facecolor=c,edgecolor="none",zorder=4))
    ax.text(5.7,y,t,ha="left",va="center",fontsize=8,color="#222222",zorder=4)

# ---------------- arrows ----------------
arrow((15,43),(19,43))
ax.text(17,44.4,"加载",ha="center",va="center",fontsize=7.5,color="#333333",zorder=4)

arrow((38,47.5),(44,47.5))
ax.text(41,49.4,"观测 s_t\n+ 动作掩码",ha="center",va="center",fontsize=8,
        bbox=dict(fc="white",ec="none",alpha=0.9),zorder=4)

arrow((44,40.8),(38,40.8))
ax.text(41,39.3,"动作 a_t\n=(作业 j, 机器 m)",ha="center",va="center",fontsize=8,
        bbox=dict(fc="white",ec="none",alpha=0.9),zorder=4)

arrow((68,48),(76,48))
ax.text(72,49.4,"贪心评估",ha="center",va="center",fontsize=8,
        bbox=dict(fc="white",ec="none",alpha=0.9),zorder=4)

arrow((58,33),(72,20))
ax.text(67,27.6,"s, a, logπ_old, V",ha="center",va="center",fontsize=8.2,
        bbox=dict(fc="white",ec="none",alpha=0.9),zorder=4)

polyline([(30,50),(30,54.5),(96.5,54.5),(96.5,13)],end=(86,13),
         label="奖励 r_t = −ΔCmax / 下界 + 终端 −Cmax（done）",lx=62,ly=55.7,fs=8.5)

polyline([(20,20),(20,31.5),(47,31.5)],end=(47,33),
         label="θ ← θ（最大化 clip 目标）",lx=32,ly=32.8,fs=8.5)

arrow((64,13),(58,13))
arrow((40,13),(34,13))
ax.text(37,14.4,"Â, R",ha="center",va="center",fontsize=8.2,
        bbox=dict(fc="white",ec="none",alpha=0.9),zorder=4)

# ---------------- stage captions ----------------
ax.text(41,53.2,"① 交互采样（每个环境步）",ha="center",va="center",
        fontsize=8.5,color="#888888",style="italic")
ax.text(46,3.0,"② 经验收集 → 优势估计 → 参数更新（每 3840 样本一轮）",
        ha="center",va="center",fontsize=8.5,color="#888888",style="italic")

fig.savefig("figures/fig4_ppo_mechanism.png",dpi=300)
try:
    fig.savefig("figures/fig4_ppo_mechanism.pdf")
    print("saved figures/fig4_ppo_mechanism.png and .pdf")
except OSError as e:
    print(f"saved figures/fig4_ppo_mechanism.png (pdf skipped: {e.strerror})")
