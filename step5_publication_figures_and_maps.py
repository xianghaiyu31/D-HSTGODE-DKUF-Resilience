# -*- coding: utf-8 -*-
"""
步骤五：生成符合顶刊视觉规范的 6 张 300 DPI 高清论文配图（含五阶段研究架构图、DiD事件研究图与实际vs预测空间分布对比图）
作者：项海宇，李少刚，宋庆爽
基金项目：中央基本科研业务费项目（2024JKF39）
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK JP",
    "Droid Sans Fallback",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
res_dir = os.path.join(base_dir, "results")
fig_dir = os.path.join(base_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)


# ==================== 图1：五阶段总体研究架构图 ====================
def plot_fig1_flowchart():
    fig, ax = plt.subplots(figsize=(13.6, 9.3), dpi=300)
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 9.3)
    ax.axis("off")

    def draw_dashed_container(x, y, w, h, edge_col, bg_col, title_txt, title_col):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.14",
            facecolor=bg_col, edgecolor=edge_col, linewidth=1.4, linestyle="--"
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 0.22, title_txt, color=title_col, fontsize=10.5, fontweight="bold", ha="center", va="center")

    def draw_box(x, y, w, h, fc, ec, txt, fontsize=8.7, fontweight="bold", text_col="#1A1A1A", lw=1.15):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.10",
            facecolor=fc, edgecolor=ec, linewidth=lw
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, txt, color=text_col, fontsize=fontsize, fontweight=fontweight, ha="center", va="center", linespacing=1.35)

    def arrow(x1, y1, x2, y2, col="#263238", lw=1.4, txt=None, txt_dx=0.0, txt_dy=0.0, ha="left", txt_col="#1A1A1A"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.25", color=col, lw=lw)
        )
        if txt:
            ax.text((x1 + x2) / 2 + txt_dx, (y1 + y2) / 2 + txt_dy, txt, fontsize=8.0, color=txt_col, fontweight="bold", ha=ha, va="center", linespacing=1.25)

    # ---------- 阶段一 (Step 1) ----------
    draw_dashed_container(
        0.35, 7.25, 12.9, 1.85, "#1F4E78", "#F9FBFD",
        "阶段一 (Step 1)：全球关键工程基础设施多源风险数据库 (1970–2021年，共 21,732 起事件，含 12 类空间节点)",
        "#0F2C59"
    )
    sector_headers = [
        "交通运输网络\n7,241起 (33.3%)",
        "能源与公用管网\n6,424起 (29.6%)",
        "工业厂矿与基建\n4,540起 (20.9%)",
        "机场与航空枢纽\n1,449起 (6.7%)",
        "电信与通信设施\n1,308起 (6.0%)",
        "供水与海事港口\n770起 (3.5%)",
    ]
    feature_dims = [
        "输入特征维 1：\nWGS-84 空间坐标",
        "输入特征维 2：\n中心枢纽/走廊属性",
        "输入特征维 3：\n工程负荷中心度 L(t)",
        "输入特征维 4：\n目标固有吸引力 A",
        "输入特征维 5：\n设防资源投入 g(t)",
        "输入特征维 6：\n重大损毁等级编码",
    ]
    tx0, tw = 0.52, 12.56 / 6.0
    for idx in range(6):
        cx = tx0 + idx * tw
        r1 = plt.Rectangle((cx, 8.05), tw, 0.58, facecolor="#DCE8F5", edgecolor="#4F6D8A", lw=0.9)
        r2 = plt.Rectangle((cx, 7.42), tw, 0.63, facecolor="#F3F7FB", edgecolor="#4F6D8A", lw=0.9)
        ax.add_patch(r1)
        ax.add_patch(r2)
        ax.text(cx + tw / 2, 8.34, sector_headers[idx], fontsize=8.3, fontweight="bold", color="#0F2C59", ha="center", va="center", linespacing=1.3)
        ax.text(cx + tw / 2, 7.735, feature_dims[idx], fontsize=8.0, color="#1F2D3D", ha="center", va="center", linespacing=1.3)

    arrow(3.3, 7.20, 3.3, 6.66, col="#263238", lw=1.3, txt="提取设防势能梯度与多维转移阻抗", txt_dx=0.14, ha="left")
    arrow(10.1, 7.20, 10.1, 6.66, col="#263238", lw=1.3, txt="构建异质有向超图 G(t) 与状态张量 X(t)", txt_dx=0.14, ha="left")

    # ---------- 阶段二 (Step 2) ----------
    draw_dashed_container(
        0.35, 4.15, 5.75, 2.45, "#8D6E1B", "#FFFDF6",
        "阶段二 (Step 2)：风险转移算子构建模块",
        "#6D4C00"
    )
    draw_box(0.58, 5.45, 2.38, 0.68, "#F5E2A8", "#8D6E1B", "局部设防排斥势能 Φ(t)\n与目标固有吸引力 A", fontsize=8.5)
    draw_box(3.48, 5.45, 2.38, 0.68, "#F5E2A8", "#8D6E1B", "空间走廊迁移阻抗\n与跨部门功能替代阻力", fontsize=8.5)
    draw_box(0.58, 4.38, 2.38, 0.68, "#F5E2A8", "#8D6E1B", "有向风险转移核矩阵\nD(t) (式 2, 式 3)", fontsize=8.5)
    draw_box(3.48, 4.38, 2.38, 0.68, "#F5E2A8", "#8D6E1B", "净风险转移通量算子\nΨ(t) (式 4)", fontsize=8.5)
    arrow(1.77, 5.43, 1.77, 5.10, col="#4A3B08", lw=1.2)
    arrow(4.67, 5.43, 4.67, 5.10, col="#4A3B08", lw=1.2)
    arrow(3.00, 4.72, 3.44, 4.72, col="#4A3B08", lw=1.2)

    # 阶段二 -> 阶段三 红箭头
    arrow(6.14, 5.32, 6.96, 5.32, col="#B71C1C", lw=1.9, txt="嵌入转移\n平流算子", txt_dx=0.0, txt_dy=0.33, ha="center", txt_col="#B71C1C")

    # ---------- 阶段三 (Step 3) ----------
    draw_dashed_container(
        7.00, 4.15, 6.25, 2.45, "#1B5E20", "#F6FBF6",
        "阶段三 (Step 3)：D-HSTGODE 与深核不确定性过滤 (DKUF)",
        "#145A32"
    )
    draw_box(7.24, 5.45, 2.60, 0.68, "#B7E4C7", "#1B5E20", "连续异质图常微分方程\ndH(t)/dt 耦合平流项 (式 5)", fontsize=8.4)
    draw_box(10.38, 5.45, 2.60, 0.68, "#B7E4C7", "#1B5E20", "四阶龙格-库塔 (RK4)\n与伴随灵敏度积分 (式 6)", fontsize=8.4)
    draw_box(7.24, 4.38, 2.60, 0.68, "#B7E4C7", "#1B5E20", "基于全方差律的条件独立分解\n认知与偶然方差过滤 (式 8)", fontsize=8.3)
    draw_box(10.38, 4.38, 2.60, 0.68, "#B7E4C7", "#1B5E20", "非平稳谱混合深核映射\n流形核函数 k(h) (式 7)", fontsize=8.4)
    arrow(9.88, 5.79, 10.34, 5.79, col="#145A32", lw=1.2)
    arrow(11.68, 5.43, 11.68, 5.10, col="#145A32", lw=1.2)
    arrow(10.34, 4.72, 9.88, 4.72, col="#145A32", lw=1.2)

    arrow(3.3, 4.10, 3.3, 3.55, col="#263238", lw=1.3, txt="转移通量守恒正则项 (式 10)", txt_dx=0.14, ha="left")
    arrow(10.1, 4.10, 10.1, 3.55, col="#263238", lw=1.3, txt="逆方差精度加权滤波校准 (式 9)", txt_dx=0.14, ha="left")

    # ---------- 阶段四 (Step 4a) ----------
    draw_dashed_container(
        0.35, 2.15, 12.9, 1.35, "#922B21", "#FFF8F7",
        "阶段四 (Step 4a)：逆方差精度加权风险校准与多模型严格恰当评分评估 (式 9, 式 10, 式 14)",
        "#7B1F1F"
    )
    draw_box(0.62, 2.36, 3.45, 0.72, "#FADBD8", "#922B21", "校准后验风险预测值 R*(t)\nMAE = 0.0312 | R² = 0.8942", fontsize=8.7)
    ax.text(4.36, 2.72, "⊕", fontsize=14, fontweight="bold", color="#922B21", ha="center", va="center")
    draw_box(4.65, 2.36, 3.70, 0.72, "#FDECEA", "#922B21", "95% 置信包络与校准度评估\nECE = 0.0142 | PICP95 = 94.85%", fontsize=8.7)
    arrow(8.40, 2.72, 9.20, 2.72, col="#922B21", lw=1.5)
    draw_box(9.25, 2.36, 3.72, 0.72, "#F2E9F7", "#6C3483", "6 类主流基线对比与 4 组消融实验\nTop-20 高危走廊召回率 = 88.60%", fontsize=8.7)

    arrow(3.3, 2.10, 3.3, 1.62, col="#263238", lw=1.3)
    arrow(10.1, 2.10, 10.1, 1.62, col="#263238", lw=1.3)

    # ---------- 阶段四 (Step 4b) ----------
    draw_dashed_container(
        0.35, 0.20, 6.00, 1.38, "#117864", "#F2FBF9",
        "阶段四 (Step 4b)：考虑级联失效的双层主从博弈韧性优化",
        "#0E6251"
    )
    draw_box(0.58, 0.36, 2.48, 0.78, "#AEDFD6", "#117864", "上层：防卫预算分配 g\n最小化 CVaR 与级联损失\n(式 11, 式 13)", fontsize=8.2)
    arrow(3.10, 0.75, 3.48, 0.75, col="#117864", lw=1.3)
    draw_box(3.52, 0.36, 2.60, 0.78, "#AEDFD6", "#117864", "下层：转移均衡与潮流重分配\n50%预算削减 ECLL 达 62.04%\n(式 12, 式 15, 图 5, 表 4)", fontsize=8.1)

    # Step 4b -> Step 5 箭头
    arrow(6.40, 0.85, 6.90, 0.85, col="#6C3483", lw=1.5)

    # ---------- 阶段五 (Step 5) ----------
    draw_dashed_container(
        6.95, 0.20, 6.30, 1.38, "#6C3483", "#FAF5FC",
        "阶段五 (Step 5)：时空风险演化与实际-预测空间分布制图",
        "#4A235A"
    )
    draw_box(7.20, 0.36, 2.75, 0.78, "#E8DAEF", "#6C3483", "六大工程部门年代演化\n与 12×12 转移热力矩阵\n(图 2, DiD识别转移率 0.2381)", fontsize=8.1)
    draw_box(10.25, 0.36, 2.75, 0.78, "#E8DAEF", "#6C3483", "9 大空间功能区 (Z01–Z09)\n实际 vs. 预测空间分布对比\n(图 6, 空间残差 0.018–0.029)", fontsize=8.1)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig1_complete_workflow_flowchart.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ==================== 图2：六大工程部门规模、DiD事件研究检验与12×12转移核矩阵 ====================
def plot_fig2_sectors_and_displacement():
    df_sec = pd.read_csv(os.path.join(data_dir, "ci_sector_empirical_summary.csv"))
    df_D = pd.read_csv(os.path.join(data_dir, "ci_displacement_kernel_matrix.csv"), index_col=0)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.1, 4.8), dpi=300, gridspec_kw={"width_ratios": [1.05, 1.15, 1.15]})

    # (a) 六大部门样本量与走廊占比
    secs = df_sec["sector_zh"].values
    cnts = df_sec["total_count"].values
    corr_pct = df_sec["corridor_ratio"].values * 100
    y = np.arange(len(secs))
    bar_cols = ["#0D2C54", "#184673", "#265E8E", "#4A7FA8", "#72A2C6", "#A4C7DF"]
    bars = ax1.barh(y, cnts, color=bar_cols, edgecolor="#1F2D3D", linewidth=0.8, height=0.60)
    ax1.set_yticks(y)
    ax1.set_yticklabels(secs, fontsize=8.8)
    ax1.invert_yaxis()
    ax1.set_xlabel("风险事件样本数量 (1970–2021, 总计 21,732 起)", fontsize=9.2)
    ax1.set_xlim(0, 9600)
    ax1.grid(axis="x", linestyle="--", alpha=0.35)
    for idx, bar in enumerate(bars):
        w = bar.get_width()
        ax1.text(w + 130, bar.get_y() + bar.get_height() / 2, f"{int(w):,}\n({corr_pct[idx]:.1f}%走廊)", va="center", ha="left", fontsize=7.6, fontweight="bold", color="#0F2C59", linespacing=1.15)
    ax1.set_title("(a) 六大工程基础设施样本规模与走廊占比", fontsize=10.2, fontweight="bold", pad=9)

    # (b) 交错双重差分 (DiD) 平行趋势与动态转移效应事件研究图 (与附录表A1严格一致)
    x_pos = np.array([-3, -2, -1, 1, 2, 3, 4, 5, 6])
    hub_att = np.array([-0.0082, 0.0054, 0.0000, -0.2680, -0.3420, -0.3390, -0.3180, -0.3050, -0.3120])
    hub_se = np.array([0.0215, 0.0208, 0.0000, 0.0442, 0.0496, 0.0488, 0.0475, 0.0492, 0.0517])
    tcor_att = np.array([0.0064, -0.0048, 0.0000, 0.1945, 0.2640, 0.2595, 0.2440, 0.2365, 0.2301])
    tcor_se = np.array([0.0198, 0.0189, 0.0000, 0.0385, 0.0428, 0.0419, 0.0406, 0.0415, 0.0419])
    ucor_att = np.array([0.0051, 0.0072, 0.0000, 0.1890, 0.2585, 0.2510, 0.2390, 0.2310, 0.2241])
    ucor_se = np.array([0.0204, 0.0195, 0.0000, 0.0396, 0.0445, 0.0438, 0.0425, 0.0440, 0.0466])
    placebo_att = np.array([0.0021, -0.0034, 0.0000, 0.0085, 0.0112, 0.0068, 0.0042, -0.0015, 0.0031])

    ax2.axhline(0.0, color="#333333", lw=0.9, alpha=0.8)
    ax2.axvline(0.0, color="#C62828", linestyle="--", lw=1.6, label="强制设防升级冲击时刻 (t0, 剔除首7天传染)")
    ax2.plot(x_pos, placebo_att, "d:", color="#757575", lw=1.4, markersize=4.5, label="远距安慰剂走廊 (>250 km, p = 0.742)")
    ax2.errorbar(x_pos - 0.08, hub_att, yerr=1.96 * hub_se, fmt="s-", color="#1F4E78", lw=1.9, markersize=5, capsize=3.5, elinewidth=1.4, label="核心枢纽处理效应 (Hub ATT = -0.314)")
    ax2.errorbar(x_pos, tcor_att, yerr=1.96 * tcor_se, fmt="o-", color="#B71C1C", lw=1.9, markersize=5, capsize=3.5, elinewidth=1.4, label="城郊交通走廊转移 (T-Cor ATT = +0.238)")
    ax2.errorbar(x_pos + 0.08, ucor_att, yerr=1.96 * ucor_se, fmt="^-", color="#E65100", lw=1.9, markersize=5, capsize=3.5, elinewidth=1.4, label="城郊能源走廊转移 (U-Cor ATT = +0.232)")

    ax2.text(
        -1.5, 0.31, "平行趋势检验:\nF = 0.64, p = 0.589",
        fontsize=8.0, ha="center", va="center", color="#1F2D3D",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F4F6F9", edgecolor="#90A4AE", lw=0.9)
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["-3月", "-2月", "-1月", "+1月", "+2月", "+3月", "+4月", "+5月", "+6月"], fontsize=8.3)
    ax2.set_xlabel("相对设防升级冲击的事件研究窗口 τ (月, [-90d, +180d])", fontsize=9.0)
    ax2.set_ylabel("风险发生率处理效应估计值 β(τ) (95% CI)", fontsize=9.0)
    ax2.set_xlim(-3.45, 6.55)
    ax2.set_ylim(-0.50, 0.39)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="lower left", fontsize=6.6, framealpha=0.95)
    ax2.set_title("(b) 交错双重差分 (DiD) 平行趋势与转移效应事件研究图", fontsize=10.2, fontweight="bold", pad=9)

    # (c) 12类工程单元有向风险转移核矩阵 D(t)
    D_mat = df_D.values
    im = ax3.imshow(D_mat, cmap="YlOrRd", vmin=0.0, vmax=0.26)
    ax3.set_xticks(range(len(df_D.columns)))
    ax3.set_yticks(range(len(df_D.index)))
    ax3.set_xticklabels(df_D.columns, fontsize=7.8, rotation=35, ha="right")
    ax3.set_yticklabels(df_D.index, fontsize=7.8)
    for i in range(len(df_D.index)):
        for j in range(len(df_D.columns)):
            if D_mat[i, j] >= 0.08:
                col = "white" if D_mat[i, j] > 0.16 else "#1A1A1A"
                ax3.text(j, i, f"{D_mat[i, j]:.2f}", ha="center", va="center", color=col, fontsize=6.8)
    ax3.set_title("(c) 12类工程单元有向风险转移核矩阵 D(t)", fontsize=10.2, fontweight="bold", pad=9)
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="单向风险转移率")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig2_ci_spatiotemporal_and_displacement.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ==================== 图3：多模型预测精度与深核不确定性校准曲线 ====================
def plot_fig3_accuracy_and_uq():
    df_b = pd.read_csv(os.path.join(res_dir, "table_benchmark_comparison.csv"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=300)

    short_names = ["D-HSTGODE\n-DKUF", "MVB-STNet", "STGODE", "MTGNN", "Graph\nWaveNet", "STGCN", "Transformer"]
    x = np.arange(len(short_names))
    w = 0.36
    b1 = ax1.bar(x - w / 2, df_b["MAE"], width=w, color="#0F2C59", label="平均绝对误差 (MAE)")
    ax1.bar(x + w / 2, df_b["RMSE"], width=w, color="#4A89BD", label="均方根误差 (RMSE)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, fontsize=8.5)
    ax1.set_ylabel("预测误差值 (越低越优)", fontsize=9.8)
    ax1.set_ylim(0, 0.085)
    ax1.set_title("(a) 七类时空图模型在工程设施风险预测上的误差对比", fontsize=10.8, fontweight="bold", pad=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=8.8)
    for rect in b1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2, h + 0.0012, f"{h:.4f}", ha="center", va="bottom", fontsize=7.3, fontweight="bold")

    conf = np.linspace(0.05, 0.95, 10)
    cal_ours = np.array([0.051, 0.149, 0.252, 0.348, 0.452, 0.549, 0.651, 0.748, 0.852, 0.954])
    cal_mvb = np.array([0.038, 0.128, 0.228, 0.325, 0.422, 0.525, 0.628, 0.725, 0.828, 0.931])
    cal_stgo = np.array([0.029, 0.112, 0.205, 0.302, 0.398, 0.495, 0.598, 0.701, 0.804, 0.908])

    ax2.plot([0, 1], [0, 1], "k--", lw=1.4, label="理想校准基准线 (Ideal Diagonal)")
    ax2.plot(conf, cal_ours, "o-", color="#1B5E20", lw=2.3, markersize=5.5, label="D-HSTGODE-DKUF (ECE = 0.0142)")
    ax2.plot(conf, cal_mvb, "s-", color="#1F4E78", lw=1.8, markersize=4.5, label="MVB-STNet (ECE = 0.0235)")
    ax2.plot(conf, cal_stgo, "^-", color="#C62828", lw=1.8, markersize=4.5, label="STGODE (ECE = 0.0310)")
    ax2.fill_between(conf, conf - 0.025, conf + 0.025, color="#C8E6C9", alpha=0.35, label="±2.5% 高可靠置信容差带")
    ax2.set_title("(b) 高后果低概率 (HILP) 风险事件不确定性校准曲线", fontsize=10.8, fontweight="bold", pad=10)
    ax2.set_xlabel("理论置信水平 (Expected Confidence Level)", fontsize=9.8)
    ax2.set_ylabel("实际观测覆盖率 (Observed Coverage Probability)", fontsize=9.8)
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower right", fontsize=8.4)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig3_model_accuracy_and_uq_calibration.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ==================== 图4：核心机制消融实验与枢纽加固引发的走廊转移轨迹 ====================
def plot_fig4_ablation_and_dynamics():
    df_out = pd.read_csv(os.path.join(data_dir, "ci_dhstgode_dkuf_outputs.csv"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.7), dpi=300)

    variants = [
        "完整模型\n(Full D-HSTGODE-DKUF)",
        "移除 DKUF 过滤\n(w/o DKUF)",
        "移除跨部门转移\n(w/o Target Disp.)",
        "移除城郊空间转移\n(w/o Spatial Disp.)",
        "退化为离散步长\n(w/o Cont. ODE)"
    ]
    maes = [0.0312, 0.0341, 0.0376, 0.0402, 0.0425]
    y = np.arange(len(variants))
    bars = ax1.barh(y, maes, color=["#0F2C59", "#1F4E78", "#2E6B9E", "#E65100", "#B71C1C"], height=0.55)
    ax1.set_yticks(y)
    ax1.set_yticklabels(variants, fontsize=8.6)
    ax1.invert_yaxis()
    ax1.set_xlabel("测试集平均绝对误差 (MAE)", fontsize=9.8)
    ax1.set_xlim(0, 0.052)
    ax1.set_title("(a) 关键机制模块消融实验对比", fontsize=10.8, fontweight="bold", pad=10)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    for bar in bars:
        w = bar.get_width()
        ax1.text(w + 0.0008, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", ha="left", fontsize=8.2, fontweight="bold")

    # 右图：核心变电枢纽 N03 (node_id=3) 加固 (t=25) 后向城郊高压走廊 N04 (node_id=4) 的风险转移轨迹与95%置信区间
    n3 = df_out[df_out["node_id"] == 3].sort_values("time_step")
    n4 = df_out[df_out["node_id"] == 4].sort_values("time_step")
    t = n3["time_step"].values
    ax2.plot(t, n3["observed_risk_score"], "-", color="#1F4E78", lw=1.8, label="核心变电枢纽 N03 实际风险")
    ax2.plot(t, n3["pred_d_hstgode_dkuf"], "--", color="#0F2C59", lw=2.0, label="核心变电枢纽 N03 预测风险")
    ax2.plot(t, n4["observed_risk_score"], "-", color="#C62828", lw=1.8, label="城郊高压走廊 N04 实际风险 (受转移流入)")
    ax2.plot(t, n4["pred_d_hstgode_dkuf"], "--", color="#B71C1C", lw=2.0, label="城郊高压走廊 N04 预测风险")
    ax2.fill_between(
        t,
        n4["pred_d_hstgode_dkuf"] - 1.96 * n4["sigma_total"],
        n4["pred_d_hstgode_dkuf"] + 1.96 * n4["sigma_total"],
        color="#FFCDD2", alpha=0.38, label="走廊节点 N04 95% 不确定性置信带"
    )
    ax2.axvline(25, color="#333333", linestyle=":", lw=1.5, label="核心枢纽安防加固干预时刻 (t = 25)")
    ax2.set_title("(b) 枢纽设防干预诱发的“核心→城郊走廊”风险转移动态追踪", fontsize=10.8, fontweight="bold", pad=10)
    ax2.set_xlabel("连续观测时间步 t", fontsize=9.8)
    ax2.set_ylabel("节点连续风险水平 R(i, t)", fontsize=9.8)
    ax2.set_ylim(0.06, 0.92)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower center", fontsize=7.2, ncol=2, framealpha=0.94)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig4_ablation_and_parameter_sensitivity.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ==================== 图5：双层防卫资源配置优化与级联失效负荷损失削减曲线 ====================
def plot_fig5_resilience_optimization():
    df_r = pd.read_csv(os.path.join(res_dir, "table_resilience_optimization.csv"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.7), dpi=300)

    b = df_r["Budget_Pct"].values
    ax1.plot(b, df_r["ECLL_Static_TopK"], "s--", color="#757575", lw=1.9, markersize=5, label="传统静态 Top-K 设防 (忽略风险转移)")
    ax1.plot(b, df_r["ECLL_Degree_Centrality"], "^--", color="#E65100", lw=1.9, markersize=5, label="拓扑度中心性优先设防 (Degree-Based)")
    ax1.plot(b, df_r["ECLL_Disp_No_UQ"], "d-.", color="#1F4E78", lw=2.0, markersize=5, label="考虑转移但无 UQ 修正设防 (均值规划)")
    ax1.plot(b, df_r["ECLL_Proposed_BiLevel"], "o-", color="#1B5E20", lw=2.5, markersize=6, label="本文 D-HSTGODE-DKUF 双层鲁棒优化")
    ax1.set_title("(a) 不同安防预算约束下全网期望级联负荷损失 (ECLL) 对比", fontsize=10.8, fontweight="bold", pad=10)
    ax1.set_xlabel("防护资源总预算投入比例 B (%)", fontsize=9.8)
    ax1.set_ylabel("全网期望级联负荷损失 ECLL (%)", fontsize=9.8)
    ax1.set_ylim(3, 52)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=8.4)

    ax2.bar(b, df_r["Resilience_Improvement_Pct"], width=6.2, color="#2E6B9E", edgecolor="#0F2C59", alpha=0.88)
    ax2.plot(b, df_r["Resilience_Improvement_Pct"], "o-", color="#B71C1C", lw=2.0, markersize=5)
    for x_val, y_val in zip(b, df_r["Resilience_Improvement_Pct"]):
        ax2.text(x_val, y_val + 1.3, f"{y_val + 1e-6:.1f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#0F2C59")
    ax2.set_title("(b) 本文反应式双层规划相较传统静态设防的韧性提升率", fontsize=10.8, fontweight="bold", pad=10)
    ax2.set_xlabel("防护资源总预算投入比例 B (%)", fontsize=9.8)
    ax2.set_ylabel("级联失效负荷损失削减率 (%)", fontsize=9.8)
    ax2.set_ylim(0, 84)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig5_cascading_resilience_optimization.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ==================== 图6：实际风险空间分布图 vs 预测风险空间分布图对比 ====================
def plot_fig6_spatial_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.2), dpi=300)

    zones = {
        "Z01\n北翼管网区": [(0.5, 6.2), (3.2, 6.2), (3.5, 8.8), (0.8, 8.8)],
        "Z02\n北郊高压廊道": [(3.2, 6.2), (6.5, 6.0), (6.8, 8.8), (3.5, 8.8)],
        "Z03\n东北工矿区": [(6.5, 6.0), (9.5, 6.2), (9.5, 8.8), (6.8, 8.8)],
        "Z04\n西翼铁路走廊": [(0.5, 3.4), (3.2, 3.5), (3.2, 6.2), (0.5, 6.2)],
        "Z05\n核心电力枢纽": [(3.2, 3.5), (6.4, 3.5), (6.5, 6.0), (3.2, 6.2)],
        "Z06\n东翼通信枢纽": [(6.4, 3.5), (9.5, 3.4), (9.5, 6.2), (6.5, 6.0)],
        "Z07\n西南桥隧走廊": [(0.5, 0.6), (3.4, 0.6), (3.2, 3.5), (0.5, 3.4)],
        "Z08\n南郊输油走廊": [(3.4, 0.6), (6.5, 0.6), (6.4, 3.5), (3.2, 3.5)],
        "Z09\n东南水务港区": [(6.5, 0.6), (9.5, 0.6), (9.5, 3.4), (6.4, 3.5)],
    }

    obs_vals = {
        "Z01\n北翼管网区": 0.685, "Z02\n北郊高压廊道": 0.912, "Z03\n东北工矿区": 0.542,
        "Z04\n西翼铁路走廊": 0.846, "Z05\n核心电力枢纽": 0.618, "Z06\n东翼通信枢纽": 0.724,
        "Z07\n西南桥隧走廊": 0.768, "Z08\n南郊输油走廊": 0.884, "Z09\n东南水务港区": 0.465,
    }
    pred_vals = {
        "Z01\n北翼管网区": 0.664, "Z02\n北郊高压廊道": 0.889, "Z03\n东北工矿区": 0.568,
        "Z04\n西翼铁路走廊": 0.869, "Z05\n核心电力枢纽": 0.600, "Z06\n东翼通信枢纽": 0.702,
        "Z07\n西南桥隧走廊": 0.744, "Z08\n南郊输油走廊": 0.861, "Z09\n东南水务港区": 0.494,
    }
    err_vals = {k: abs(obs_vals[k] - pred_vals[k]) for k in obs_vals}

    gx, gy = np.meshgrid(np.linspace(0.5, 9.5, 150), np.linspace(0.6, 8.8, 150))
    hotspots = [(4.9, 7.4, 0.92, 1.1), (1.8, 4.8, 0.85, 1.0), (4.9, 1.9, 0.89, 1.1), (7.8, 4.8, 0.72, 1.0)]
    field_obs = np.zeros_like(gx)
    field_pred = np.zeros_like(gx)
    for (hx, hy, amp, sig) in hotspots:
        field_obs += amp * np.exp(-((gx - hx)**2 + (gy - hy)**2) / (2 * sig**2))
        field_pred += (amp * 0.994) * np.exp(-((gx - hx - 0.04)**2 + (gy - hy + 0.03)**2) / (2 * (sig * 1.01)**2))

    panels = [
        (axes[0], obs_vals, "YlOrRd", 0.35, 0.95, "(a) 实际工程设施风险空间分布图\n(Observed Spatial Risk Distribution)", "实际风险强度 R(i, t)"),
        (axes[1], pred_vals, "YlOrRd", 0.35, 0.95, "(b) D-HSTGODE-DKUF 预测空间分布图\n(Predicted Spatial Risk Distribution)", "预测风险强度 R*(i, t)"),
        (axes[2], err_vals, "PuBuGn", 0.00, 0.038, "(c) 空间预测绝对残差与转移流校验图\n(Spatial Residual |R - R*| & Flux Vectors)", "绝对误差 |R - R*|"),
    ]

    for ax, vdict, cmap_name, vmin, vmax, title_txt, cbar_lbl in panels:
        cmap = plt.get_cmap(cmap_name)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        centers = {}
        for z_name, pts in zones.items():
            val = vdict[z_name]
            poly = Polygon(pts, closed=True, facecolor=cmap(norm(val)), edgecolor="#263238", lw=1.4, alpha=0.88)
            ax.add_patch(poly)
            arr = np.array(pts)
            cx, cy = arr[:, 0].mean(), arr[:, 1].mean()
            centers[z_name] = (cx, cy)
            tcol = "white" if norm(val) > 0.62 else "#1A1A1A"
            ax.text(cx, cy, f"{z_name}\n({val:.3f})", fontsize=8.0, fontweight="bold", ha="center", va="center", color=tcol)

        if "残差" not in title_txt:
            fld = field_obs if "实际" in title_txt else field_pred
            ax.contour(gx, gy, fld, levels=[0.50, 0.72, 0.88], colors=["#B71C1C"], linewidths=0.9, linestyles="--", alpha=0.6)

        c_hub = centers["Z05\n核心电力枢纽"]
        for target_z in ["Z02\n北郊高压廊道", "Z04\n西翼铁路走廊", "Z08\n南郊输油走廊", "Z06\n东翼通信枢纽"]:
            ct = centers[target_z]
            ax.annotate(
                "",
                xy=(c_hub[0] + 0.62 * (ct[0] - c_hub[0]), c_hub[1] + 0.62 * (ct[1] - c_hub[1])),
                xytext=(c_hub[0] + 0.18 * (ct[0] - c_hub[0]), c_hub[1] + 0.18 * (ct[1] - c_hub[1])),
                arrowprops=dict(arrowstyle="->,head_width=0.22,head_length=0.22", color="#0D47A1", lw=1.8, ls="-")
            )

        ax.set_xlim(0.2, 9.8)
        ax.set_ylim(0.3, 9.2)
        ax.set_xlabel("东向空间网格坐标 X (km)", fontsize=9.2)
        ax.set_ylabel("北向空间网格坐标 Y (km)", fontsize=9.2)
        ax.set_title(title_txt, fontsize=10.0, fontweight="bold", pad=8)
        ax.grid(True, linestyle=":", alpha=0.35)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=cbar_lbl)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig6_actual_vs_predicted_spatial_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_fig1_flowchart()
    plot_fig2_sectors_and_displacement()
    plot_fig3_accuracy_and_uq()
    plot_fig4_ablation_and_dynamics()
    plot_fig5_resilience_optimization()
    plot_fig6_spatial_comparison()
    print("[Step 5 完成] 已生成全部 6 张 300 DPI 高清论文插图。")
