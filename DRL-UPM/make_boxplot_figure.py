import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
    "font.family":"sans-serif",
    "axes.unicode_minus":False,
})

C_REF="#3D78C2"
METHODS=[("随机策略","ratio_random","#C8C8C8"),
         ("SPT 规则","ratio_SPT","#2E8F94"),
         ("PPO（本次训练）","ratio_PPO","#C0392B"),
         ("贪心启发式","ratio_myopic","#707070"),
         ("GA 遗传算法","ratio_GA","#3D78C2")]

rows=[]
with open("results/evaluation_results.csv",encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r["instance"]!="MEAN":
            rows.append(r)
print(f"loaded {len(rows)} instances")

data={key:np.array([float(r[key]) for r in rows]) for _,key,_ in METHODS}
myo=data["ratio_myopic"]

def draw(ax,values,ref=None):
    pos=np.arange(len(METHODS))
    bp=ax.boxplot(values,positions=pos,patch_artist=True,widths=0.55,
                  medianprops=dict(color="black",lw=1.5),
                  whiskerprops=dict(color="#555555",lw=1.1),
                  capprops=dict(color="#555555",lw=1.1),
                  flierprops=dict(marker="o",markersize=3.5,
                                  markerfacecolor="#999999",markeredgecolor="none"))
    for patch,(_,_,c) in zip(bp["boxes"],METHODS):
        patch.set_facecolor(c); patch.set_alpha(0.8); patch.set_edgecolor("#333333")
    rng=np.random.default_rng(0)
    for i,(_,key,c) in enumerate(METHODS):
        x=np.full(len(data[key]),i)+rng.uniform(-0.13,0.13,len(data[key]))
        ax.plot(x,data[key],"o",ms=3.6,color="#222222",alpha=0.65,zorder=5)
    if ref is not None:
        ax.axhline(ref,color=C_REF,lw=1.5,linestyle="--",zorder=4)
        ax.text(len(METHODS)-0.45,ref*1.08,f"基准 = {ref:.1f}",fontsize=9,
                color=C_REF,ha="right")
    ax.set_yscale("log")
    ax.set_xticks(pos)
    ax.set_xticklabels([m[0] for m in METHODS],fontsize=9.5)
    ax.grid(axis="y",color="#EBEBEB",linestyle="--",lw=0.7,zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.9); sp.set_zorder(3)
    ax.tick_params(direction="out",length=4,width=0.8)

fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),constrained_layout=True)
draw(axes[0],list(data.values()))
axes[0].set_ylabel("Cmax / 下界（对数轴，越低越好）",fontsize=11,fontweight="bold")
axes[0].set_title("(a) 绝对排程质量分布（15 个实例）",fontsize=11.5)

rel=[data[key]/myo for _,key,_ in METHODS]
draw(axes[1],rel,ref=1.0)
axes[1].set_ylabel("Cmax / 贪心启发式（对数轴）",fontsize=11,fontweight="bold")
axes[1].set_title("(b) 相对贪心启发式的比率",fontsize=11.5)

fig.savefig("figures/fig5_boxplot.png",dpi=300)
fig.savefig("figures/fig5_boxplot.pdf")
print("saved figures/fig5_boxplot.png and .pdf")
for name,key,_ in METHODS:
    v=data[key]
    print(f"{name:<12} median {np.median(v):7.3f}  mean {v.mean():7.3f}  "
          f"min {v.min():7.3f}  max {v.max():7.3f}")
