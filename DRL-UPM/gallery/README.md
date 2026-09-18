# DRL-UPM 可视化报告图集

浏览器打开 `index.html` 查看完整报告（图片已内嵌，整个文件夹可随意拷贝分享）。

## 内容映射

| 文件 | 内容 | 生成脚本 |
|---|---|---|
| fig4_ppo_mechanism.png | PPO 求解算法机制图（自回归分解架构） | make_mechanism_figure.py |
| fig1_training_trend.png | 训练奖励/评估趋势 | make_figures.py |
| fig5_boxplot.png | 五方法箱线图（绝对+相对贪心） | make_boxplot_figure.py |
| fig2_instance_comparison.png | 分实例柱状对比 | make_figures.py |
| fig3_gantt.png | 排程甘特图（含换型时间） | make_figures.py |
| fig6_structure_comparison.png | 新旧策略头机制对比图 | make_comparison_figure.py |
| evaluation_results.csv/.md | 五方法结果表（Cmax+比率） | evaluate.py |

## 关键结论

- 自回归分解策略头使 PPO 平均 Cmax/LB 从 12.14 → 8.62（+29%，15/15 实例改善）
- 小实例（6 作业）PPO 1.19~1.72，与 GA（1.12~1.61）同一水平
- GA 3.08 仍为全场最优；差距根源：makespan max 结构的不可逆放大 + RL 信号弱 + 搜索范式差
- 训练：2000 轮 / RTX 3080 / 约 50 分钟 / 3500 env-steps/s

## 复现

    conda activate secenv
    python train.py && python evaluate.py && python test.py
    python make_figures.py && python make_boxplot_figure.py && python make_mechanism_figure.py
    python make_gallery.py
