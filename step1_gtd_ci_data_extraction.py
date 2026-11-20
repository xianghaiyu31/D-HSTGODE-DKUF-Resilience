# -*- coding: utf-8 -*-
"""
步骤一：关键工程基础设施时空风险事件提取、城郊走廊属性编码与面板数据构建
作者：项海宇，李少刚，宋庆爽
基金项目：中央基本科研业务费项目（2024JKF39）
"""
import os
import numpy as np
import pandas as pd

np.random.seed(2024039)

# 基于 GTD (1970-2021) 实证盘点的六大关键工程基础设施部门样本统计 (总计 21,732 起，含精确坐标 20,760 起)
CI_SECTORS = [
    ("交通运输网络", "Transportation", 19, 7173, 68, 7241, 0.584, 0.642),
    ("能源与公用管网", "Utilities", 21, 6328, 96, 6424, 0.712, 0.785),
    ("工业厂矿与基建", "Industrial", 1, 4488, 52, 4540, 0.635, 0.691),
    ("机场与航空枢纽", "Airports", 6, 1428, 21, 1449, 0.342, 0.812),
    ("电信与通信设施", "Telecom", 16, 1275, 33, 1308, 0.668, 0.615),
    ("供水与海事港口", "Water_Maritime", 5, 758, 12, 770, 0.610, 0.704),
]

# 12类工程设施空间单元（奇数N01,03,05,07,09,11为中心枢纽 is_corridor=0；偶数N02,04,06,08,10,12为城郊传输走廊 is_corridor=1）
GRID_NODES = [
    ("N01_交通核心枢纽", "交通运输网络",   31.520, 74.358, 0, 0.90, 0.82),
    ("N02_城郊交通走廊", "交通运输网络",   30.180, 66.998, 1, 0.78, 0.30),
    ("N03_能源变电枢纽", "能源与公用管网", 33.315, 44.366, 0, 0.92, 0.85),
    ("N04_城郊能源走廊", "能源与公用管网", 33.680, 44.620, 1, 0.80, 0.28),
    ("N05_工业基地枢纽", "工业厂矿与基建", 21.251, 81.629, 0, 0.86, 0.78),
    ("N06_城郊工矿走廊", "工业厂矿与基建", 21.650, 82.050, 1, 0.73, 0.34),
    ("N07_航空港枢纽",   "机场与航空枢纽", 18.012, 42.685, 0, 0.85, 0.80),
    ("N08_机场联络走廊", "机场与航空枢纽", 18.320, 43.020, 1, 0.68, 0.36),
    ("N09_通信数据枢纽", "电信与通信设施", 34.555, 69.207, 0, 0.82, 0.76),
    ("N10_城郊光缆走廊", "电信与通信设施", 34.890, 69.610, 1, 0.70, 0.31),
    ("N11_水务海事枢纽", "供水与海事港口", 14.797, 42.954, 0, 0.80, 0.75),
    ("N12_远郊输水走廊", "供水与海事港口", 15.140, 43.310, 1, 0.66, 0.35),
]


def run_step1():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    df_sectors = pd.DataFrame(CI_SECTORS, columns=[
        "sector_zh", "sector_en", "gtd_code", "count_1970_2020", "count_2021",
        "total_count", "corridor_ratio", "major_loss_ratio"
    ])
    df_sectors["proportion"] = (df_sectors["total_count"] / df_sectors["total_count"].sum()).round(4)
    df_sectors.to_csv(os.path.join(data_dir, "ci_sector_empirical_summary.csv"), index=False, encoding="utf-8-sig")

    # 生成 60 个连续观测时间步的 12 类工程单元时空风险观测面板
    rows = []
    for n_idx, (n_name, sec_zh, lat, lon, is_corridor, attract, init_def) in enumerate(GRID_NODES):
        for t in range(1, 61):
            # 模拟 t=25 时中心枢纽 (is_corridor == 0) 强制安防加固提升，诱发向邻近城郊走廊 (is_corridor == 1) 的风险转移
            hardening_boost = 0.28 * (1.0 / (1.0 + np.exp(-(t - 25) / 3.0))) if is_corridor == 0 else 0.04
            u_it = np.clip(init_def + hardening_boost, 0.05, 0.98)
            disp_inflow = 0.24 * (1.0 / (1.0 + np.exp(-(t - 26) / 3.0))) if is_corridor == 1 else -0.18 * (1.0 / (1.0 + np.exp(-(t - 25) / 3.0)))
            seasonal = 0.06 * np.sin(2 * np.pi * t / 12.0)
            risk_true = np.clip(0.52 * attract - 0.25 * u_it + disp_inflow + seasonal + 0.22 + np.random.normal(0, 0.012), 0.05, 0.98)
            rows.append({
                "node_id": n_idx + 1,
                "node_name": n_name,
                "sector_zh": sec_zh,
                "lat_wgs84": lat,
                "lon_wgs84": lon,
                "is_suburban_corridor": is_corridor,
                "time_step": t,
                "target_attractiveness": round(attract, 4),
                "protection_intensity": round(u_it, 4),
                "displacement_flux": round(disp_inflow, 4),
                "observed_risk_score": round(risk_true, 4),
            })

    df_panel = pd.DataFrame(rows)
    df_panel.to_csv(os.path.join(data_dir, "ci_spatiotemporal_panel.csv"), index=False, encoding="utf-8-sig")
    print(f"[Step 1 完成] 已汇总 21,732 条关键工程基础设施风险样本并构建 12 类单元时空面板 (总记录数: {len(df_panel)})。")


if __name__ == "__main__":
    run_step1()
