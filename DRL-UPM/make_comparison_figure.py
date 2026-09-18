import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

plt.rcParams.update({
    "font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
    "font.family":"sans-serif",
    "axes.unicode_minus":False,
    "mathtext.fontset":"dejavusans",
})

fig,ax=plt.subplots(figsize=(16.6,9.48),constrained_layout=True)
ax.set_xlim(0,100); ax.set_ylim(0,57); ax.set_aspect("equal"); ax.axis("off")

DARK="#555555"
def rbox(x,y,w,h,fc,ec,lw=1.6,dashed=False,rs=1.0,z=2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,
        boxstyle=f"round,pad=0.30,rounding_size={rs}",
        facecolor=fc,edgecolor=ec,lw=lw,
        linestyle=(0,(4,3)) if dashed else "-",zorder=z))
def inbox(x,ytop,w,h,title,lines,tfs=8.8,fs=7.8,ec="#9aa3ad",tc="#333"):
    rbox(x,ytop-h,w,h,"white",ec,lw=1.0,rs=0.7,z=3)
    cx=x+w/2
    if title:
        ax.text(cx,ytop-1.05,title,ha="center",va="center",
                fontsize=tfs,fontweight="bold",color=tc,zorder=4)
        if lines:
            ax.text(cx,ytop-2.0-(h-2.3)/2,"\n".join(lines),
                    ha="center",va="center",fontsize=fs,color="#333",zorder=4,
                    linespacing=1.5)
    elif lines:
        ax.text(cx,ytop-h/2,"\n".join(lines),ha="center",va="center",
                fontsize=fs,color="#333",zorder=4,linespacing=1.5)
def varrow(x,y1,y2,c=DARK):
    ax.add_patch(FancyArrowPatch((x,y1),(x,y2),arrowstyle="-|>",
        mutation_scale=13,color=c,lw=1.6,zorder=3))
def harrow(x1,x2,y,c=DARK):
    ax.add_patch(FancyArrowPatch((x1,y),(x2,y),arrowstyle="-|>",
        mutation_scale=15,color=c,lw=2.0,zorder=3))
def polyline(pts,end=None,c=DARK,lw=1.6):
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    ax.plot(xs,ys,color=c,lw=lw,zorder=3,solid_capstyle="round")
    if end is not None:
        ax.add_patch(FancyArrowPatch(pts[-1],end,arrowstyle="-|>",
            mutation_scale=13,color=c,lw=lw,zorder=3))

# ---------------- title banner ----------------
rbox(22,51.2,56,4.4,"#D6EAF8","#2E86C1",lw=1.8,rs=1.2)
ax.text(50,54.15,"PPO 策略头结构改造对比",ha="center",va="center",
        fontsize=15.5,fontweight="bold",color="#1A5276")
ax.text(50,52.25,"扁平 6000 维动作头  →  自回归分解双头   ｜   同超参 · 同 2000 轮训练的单变量对照",
        ha="center",va="center",fontsize=9.6,color="#566573")

# ---------------- columns ----------------
COLS=[(1.2,"#EAF4FD","#5DADE2","#2E86C1","1. 决策问题","每步选一个 (作业, 机器) 对"),
      (20.9,"#FDEDEC","#EC7063","#C0392B","2. 改造前：扁平动作头","全部 6000 对直接竞争"),
      (40.6,"#F4ECF7","#A569BD","#7D3C98","3. 改造后：自回归分解","先选作业，再按作业选机器"),
      (60.3,"#EAFAF1","#58D68D","#1E8449","4. 训练与实现","数学形式与不变量"),
      (80.0,"#FEF5E7","#F5B041","#CA6F1E","5. 对照实验结果","同预算严格对照")]
for x,fc,ec,tc,t,s in COLS:
    rbox(x,4.2,17.5,45.4,fc,ec,lw=1.7,dashed=True,rs=1.1)
    ax.text(x+8.75,47.8,t,ha="center",va="center",fontsize=12.3,
            fontweight="bold",color=tc)
    ax.text(x+8.75,46.0,s,ha="center",va="center",fontsize=8.2,color="#777")

for x1,x2 in [(18.7,20.9),(38.4,40.6),(58.1,60.3),(77.8,80.0)]:
    harrow(x1,x2,26.5)

# ---------------- col 1: problem ----------------
x=2.2; w=15.5
inbox(x,44.0,w,6.2,"状态 s_t ∈ R^6690",
      ["作业特征 3×200：完成·最快/平均加工","机器特征 3×30：负载·计数·有效",
       "成本矩阵 200×30：加工+换型"],fs=7.4)
inbox(x,35.2,w,4.6,None,
      ["合法动作 ≤ 200×30 = 6000","非法动作 masked（−inf）"],fs=8.0)
inbox(x,29.6,w,4.2,None,["a_t = (j, m)","作业 j 分配到机器 m"],fs=8.0)
ax.text(x+0.4,22.2,
        "· 动作空间随实例规模变化\n   12 → 6000\n"
        "· 一个网络（pad 统一）\n   服务全部 15 个实例",
        ha="left",va="top",fontsize=7.9,color="#333",linespacing=1.6)

# ---------------- col 2: before ----------------
x=21.9; w=15.5
inbox(x,43.0,w,2.6,None,["状态 s_t"],fs=8.2)
varrow(x+w/2,40.4,39.6)
inbox(x,39.6,w,3.4,None,["MLP 主干","6690 → 256 → 256"],fs=7.9)
varrow(x+w/2,36.2,35.4)
inbox(x,35.4,w,4.6,None,["Linear 256 → 6000","6000 维 logits + mask","同一向量读出全部动作对"],fs=7.7)
varrow(x+w/2,30.8,30.0)
inbox(x,30.0,12.9,2.9,None,["a = (j, m)"],fs=8.2)
rbox(35.4,30.0,3.6,2.9,"#C0392B","#C0392B",lw=1.2,rs=0.7,z=3)
ax.text(37.2,31.45,"瓶颈",ha="center",va="center",fontsize=8.4,
        fontweight="bold",color="white",zorder=4)
rbox(x,20.0,w,5.4,"white","#9aa3ad",lw=1.0,rs=0.7,z=3)
ax.text(x+w/2,22.7,r"$\pi_{old}(a\,|\,s)=\mathrm{Cat}(6000\ \mathrm{logits})$",
        ha="center",va="center",fontsize=9.5,color="#333",zorder=4)
ax.text(x+0.4,18.6,
        "· 6000 对的偏好由同一个\n   256 维向量线性读出\n"
        "· 成对相对比较难以表达\n· 与逐步最优动作吻合率\n   7.6%（大实例 0~2%）\n"
        "· 平均 Cmax/LB = 12.14",
        ha="left",va="top",fontsize=7.9,color="#333",linespacing=1.6)

# ---------------- col 3: after ----------------
x=40.6
rbox(41.6,40.6,15.5,3.6,"white","#9aa3ad",lw=1.0,rs=0.7,z=3)
ax.text(49.35,42.4,r"$\pi(a\,|\,s)=\pi(j\,|\,s)\cdot\pi(m\,|\,s,j)$",
        ha="center",va="center",fontsize=10,color="#333",zorder=4)
cx=43.8; cw=13.3
inbox(cx,38.9,cw,2.5,None,["状态 s_t"],fs=8.2)
varrow(cx+cw/2,36.4,35.6)
inbox(cx,35.6,cw,3.0,None,["MLP 主干 → 256"],fs=7.9)
varrow(cx+cw/2,32.6,31.8)
inbox(cx,31.8,cw,3.6,None,["Job 头  256 → 200","选作业 π(j) · 掩码已完工"],fs=7.7)
varrow(cx+cw/2,28.2,27.4)
inbox(cx,27.4,cw,4.3,None,["Machine 头 ⊕ [cost_slice, 负载]","→ 30 选机器 π(m|j)"],fs=7.7)
polyline([(43.8,37.65),(42.3,37.65),(42.3,24.55)],end=(43.8,24.55),c="#7D3C98")
ax.text(41.35,31.1,"读该作业成本切片（30 维）",ha="center",va="center",
        fontsize=7.0,color="#7D3C98",rotation=90)
inbox(44.9,20.0,12.2,3.0,None,["a = j×30 + m"],fs=8.2)
ax.text(x+0.4,16.3,
        "· 机器头直接读取该作业的\n   加工+换型成本（30 维）\n"
        "· 归纳偏置：先定优先级，\n   再为该作业选最优机器\n"
        "· 分解可表示任意联合分布\n   → 表达力不降",
        ha="left",va="top",fontsize=7.9,color="#333",linespacing=1.6)

# ---------------- col 4: training ----------------
x=61.3; w=15.5
rbox(x,38.6,w,5.6,"white","#9aa3ad",lw=1.0,rs=0.7,z=3)
ax.text(x+w/2,41.4,r"$\log\pi(a)=\log\pi(j)+\log\pi(m\,|\,j)$",
        ha="center",va="center",fontsize=9.2,color="#333",zorder=4)
ax.text(x+w/2,39.6,"联合动作的对数概率（两段之和）",
        ha="center",va="center",fontsize=7.4,color="#666",zorder=4)
rbox(x,32.4,w,5.4,"white","#9aa3ad",lw=1.0,rs=0.7,z=3)
ax.text(x+w/2,34.6,r"$\mathcal{H}=\mathcal{H}(j)+\mathbb{E}_{j\sim\pi}[\mathcal{H}(m\,|\,j)]$",
        ha="center",va="center",fontsize=9.2,color="#333",zorder=4)
ax.text(x+w/2,33.4,"熵正则：精确作业熵 + MC 估计",
        ha="center",va="center",fontsize=7.4,color="#666",zorder=4)
rbox(x,26.0,w,5.4,"white","#9aa3ad",lw=1.0,rs=0.7,z=3)
ax.text(x+w/2,28.2,r"$\mathcal{L}=\mathcal{L}^{CLIP}+0.5\,\mathcal{L}^{VF}-\beta\mathcal{H}$",
        ha="center",va="center",fontsize=9.2,color="#333",zorder=4)
ax.text(x+w/2,27.0,"PPO 损失形式不变（clip ε=0.2）",
        ha="center",va="center",fontsize=7.4,color="#666",zorder=4)
inbox(x,24.4,w,3.4,None,["checkpoint 增加 \"arch\":\"factored\"","采样 / 贪心推理均改为两段式"],fs=7.6)
ax.text(x+0.4,18.0,
        "· 超参数、γ=1.0、GAE、\n   2000 轮全部保持不变\n· 严格单变量对照实验",
        ha="left",va="top",fontsize=7.9,color="#333",linespacing=1.6)

# ---------------- col 5: results ----------------
x=81.0; w=15.5
inbox(x,44.0,w,7.4,"平均 Cmax / 下界",
      ["12.14  →  8.62  （+29%）","15 / 15 实例全部改善","最差实例 31.3 → 19.8",
       "小实例 1.19~1.72 ≈ GA"],tfs=8.6,fs=8.0,tc="#CA6F1E")
inbox(x,34.6,w,6.4,"模仿探针诊断",
      ["逐步最优作业决策","91% 可从特征线性恢复","→ 信息瓶颈已解除"],tfs=8.6,fs=8.0,tc="#CA6F1E")
inbox(x,26.6,w,6.6,"剩余差距来源",
      ["makespan max 结构的","不可逆放大（一次错位","永久抬高完工时间）",
       "GA 每实例 3.6 万次搜索"],tfs=8.6,fs=7.8,tc="#CA6F1E")
ax.text(x+0.4,17.6,
        "· 五方法排名：GA 3.08 ＜\n   贪心 3.60 ＜ SPT 4.18 ＜\n   PPO 8.62 ＜ 随机 27.7",
        ha="left",va="top",fontsize=7.9,color="#333",linespacing=1.6)

# ---------------- bottom banner ----------------
varrow(10,4.2,3.3); varrow(89,4.2,3.3)
rbox(6,0.4,88,2.9,"#E8DAEF","#8E44AD",lw=1.6,rs=1.0)
ax.text(50,1.85,
        "唯一变量是策略头结构：同等训练预算下 12.14 → 8.62（+29%）；"
        "剩余与 GA 的差距来自 max 不可逆放大与搜索范式差，而非信息瓶颈",
        ha="center",va="center",fontsize=10.3,fontweight="bold",color="#4A235A")

fig.savefig("figures/fig6_structure_comparison.png",dpi=220)
try:
    fig.savefig("figures/fig6_structure_comparison.pdf")
    print("saved figures/fig6_structure_comparison.png and .pdf")
except OSError as e:
    print(f"saved figures/fig6_structure_comparison.png (pdf skipped: {e.strerror})")
