# -*- coding: utf-8 -*-
"""
主控一键执行脚本：依次执行步骤一至步骤五，生成全部数据、评测结果表与 6 张论文高清插图。
作者：项海宇，李少刚，宋庆爽
基金项目：中央基本科研业务费项目（2024JKF39）
"""
import step1_gtd_ci_data_extraction as s1
import step2_displacement_operator_matrix as s2
import step3_dhstgode_dkuf_model as s3
import step4_training_testing_and_resilience as s4
import step5_publication_figures_and_maps as s5

if __name__ == "__main__":
    print("=== 开始运行关键工程基础设施风险转移与韧性优化全流程实验 ===")
    s1.run_step1()
    s2.run_step2()
    s3.run_step3()
    s4.run_step4()
    s5.plot_fig1_flowchart()
    s5.plot_fig2_sectors_and_displacement()
    s5.plot_fig3_accuracy_and_uq()
    s5.plot_fig4_ablation_and_dynamics()
    s5.plot_fig5_resilience_optimization()
    s5.plot_fig6_spatial_comparison()
    print("=== 全流程运行测试完毕 ===")
