# 完整主流程 v3b 任务索引（静态提案，未激活）

共444项；10 source配置190轮、30域评价、4 pilot、400 ODR请求（370独立执行、30别名）。
顺序：现行9项 → 固定40 ODR → Table2/3/4的35项source任务 → B扫描120 → N扫描240。
资源/GPU测量、最终分析和材料受限核验仍在backlog，本表不代表整体目标完成。

前置栏使用可完全展开的组名：
- S3：eval_phi3_synthetic、eval_phi3_lightbox、eval_phi3_sunlamp
- S6：eval_phi6_synthetic、eval_phi6_lightbox、eval_phi6_sunlamp
- F40：下方列出的全部40项固定ODR验收。
每行前置均已按组名展开后与真实requires集合逐项核对；不会省略串行依赖。

| 序号 | 任务ID | 前置（组名可展开） | 用途 | 是否alias |
|---:|---|---|---|---|
| 1 | train_phi3_EHS | 无 | 独立source训练 20轮 | 否 |
| 2 | eval_phi3_synthetic | train_phi3_EHS | source全域评价 synthetic | 否 |
| 3 | eval_phi3_lightbox | eval_phi3_synthetic | source全域评价 lightbox | 否 |
| 4 | eval_phi3_sunlamp | eval_phi3_lightbox | source全域评价 sunlamp | 否 |
| 5 | pilot_phi6_EHS | eval_phi3_sunlamp | 工程pilot：32 attempts / checkpoint16；独立CPU验收 | 否 |
| 6 | train_phi6_EHS | pilot_phi6_EHS | 独立source训练 20轮 | 否 |
| 7 | eval_phi6_synthetic | train_phi6_EHS | source全域评价 synthetic | 否 |
| 8 | eval_phi6_lightbox | eval_phi6_synthetic | source全域评价 lightbox | 否 |
| 9 | eval_phi6_sunlamp | eval_phi6_lightbox | source全域评价 sunlamp | 否 |
| 10 | fixed_phi3_A_lightbox_s2021_N1024_B4 | eval_phi6_sunlamp；S3 | fixed φ3/A lightbox s2021 N1024/B4 | 否 |
| 11 | fixed_phi3_A_lightbox_s2022_N1024_B4 | fixed_phi3_A_lightbox_s2021_N1024_B4；S3 | fixed φ3/A lightbox s2022 N1024/B4 | 否 |
| 12 | fixed_phi3_A_lightbox_s2023_N1024_B4 | fixed_phi3_A_lightbox_s2022_N1024_B4；S3 | fixed φ3/A lightbox s2023 N1024/B4 | 否 |
| 13 | fixed_phi3_A_lightbox_s2024_N1024_B4 | fixed_phi3_A_lightbox_s2023_N1024_B4；S3 | fixed φ3/A lightbox s2024 N1024/B4 | 否 |
| 14 | fixed_phi3_A_lightbox_s2025_N1024_B4 | fixed_phi3_A_lightbox_s2024_N1024_B4；S3 | fixed φ3/A lightbox s2025 N1024/B4 | 否 |
| 15 | fixed_phi3_A_sunlamp_s2021_N1024_B4 | fixed_phi3_A_lightbox_s2025_N1024_B4；S3 | fixed φ3/A sunlamp s2021 N1024/B4 | 否 |
| 16 | fixed_phi3_A_sunlamp_s2022_N1024_B4 | fixed_phi3_A_sunlamp_s2021_N1024_B4；S3 | fixed φ3/A sunlamp s2022 N1024/B4 | 否 |
| 17 | fixed_phi3_A_sunlamp_s2023_N1024_B4 | fixed_phi3_A_sunlamp_s2022_N1024_B4；S3 | fixed φ3/A sunlamp s2023 N1024/B4 | 否 |
| 18 | fixed_phi3_A_sunlamp_s2024_N1024_B4 | fixed_phi3_A_sunlamp_s2023_N1024_B4；S3 | fixed φ3/A sunlamp s2024 N1024/B4 | 否 |
| 19 | fixed_phi3_A_sunlamp_s2025_N1024_B4 | fixed_phi3_A_sunlamp_s2024_N1024_B4；S3 | fixed φ3/A sunlamp s2025 N1024/B4 | 否 |
| 20 | fixed_phi6_A_lightbox_s2021_N4096_B1 | fixed_phi3_A_sunlamp_s2025_N1024_B4；S6 | fixed φ6/A lightbox s2021 N4096/B1 | 否 |
| 21 | fixed_phi6_A_lightbox_s2022_N4096_B1 | fixed_phi6_A_lightbox_s2021_N4096_B1；S6 | fixed φ6/A lightbox s2022 N4096/B1 | 否 |
| 22 | fixed_phi6_A_lightbox_s2023_N4096_B1 | fixed_phi6_A_lightbox_s2022_N4096_B1；S6 | fixed φ6/A lightbox s2023 N4096/B1 | 否 |
| 23 | fixed_phi6_A_lightbox_s2024_N4096_B1 | fixed_phi6_A_lightbox_s2023_N4096_B1；S6 | fixed φ6/A lightbox s2024 N4096/B1 | 否 |
| 24 | fixed_phi6_A_lightbox_s2025_N4096_B1 | fixed_phi6_A_lightbox_s2024_N4096_B1；S6 | fixed φ6/A lightbox s2025 N4096/B1 | 否 |
| 25 | fixed_phi6_A_sunlamp_s2021_N4096_B1 | fixed_phi6_A_lightbox_s2025_N4096_B1；S6 | fixed φ6/A sunlamp s2021 N4096/B1 | 否 |
| 26 | fixed_phi6_A_sunlamp_s2022_N4096_B1 | fixed_phi6_A_sunlamp_s2021_N4096_B1；S6 | fixed φ6/A sunlamp s2022 N4096/B1 | 否 |
| 27 | fixed_phi6_A_sunlamp_s2023_N4096_B1 | fixed_phi6_A_sunlamp_s2022_N4096_B1；S6 | fixed φ6/A sunlamp s2023 N4096/B1 | 否 |
| 28 | fixed_phi6_A_sunlamp_s2024_N4096_B1 | fixed_phi6_A_sunlamp_s2023_N4096_B1；S6 | fixed φ6/A sunlamp s2024 N4096/B1 | 否 |
| 29 | fixed_phi6_A_sunlamp_s2025_N4096_B1 | fixed_phi6_A_sunlamp_s2024_N4096_B1；S6 | fixed φ6/A sunlamp s2025 N4096/B1 | 否 |
| 30 | fixed_phi3_B1_lightbox_s2021_N1024_B4 | fixed_phi6_A_sunlamp_s2025_N4096_B1；S3 | fixed φ3/B1 lightbox s2021 N1024/B4 | 否 |
| 31 | fixed_phi3_B1_lightbox_s2022_N1024_B4 | fixed_phi3_B1_lightbox_s2021_N1024_B4；S3 | fixed φ3/B1 lightbox s2022 N1024/B4 | 否 |
| 32 | fixed_phi3_B1_lightbox_s2023_N1024_B4 | fixed_phi3_B1_lightbox_s2022_N1024_B4；S3 | fixed φ3/B1 lightbox s2023 N1024/B4 | 否 |
| 33 | fixed_phi3_B1_lightbox_s2024_N1024_B4 | fixed_phi3_B1_lightbox_s2023_N1024_B4；S3 | fixed φ3/B1 lightbox s2024 N1024/B4 | 否 |
| 34 | fixed_phi3_B1_lightbox_s2025_N1024_B4 | fixed_phi3_B1_lightbox_s2024_N1024_B4；S3 | fixed φ3/B1 lightbox s2025 N1024/B4 | 否 |
| 35 | fixed_phi3_B1_sunlamp_s2021_N1024_B4 | fixed_phi3_B1_lightbox_s2025_N1024_B4；S3 | fixed φ3/B1 sunlamp s2021 N1024/B4 | 否 |
| 36 | fixed_phi3_B1_sunlamp_s2022_N1024_B4 | fixed_phi3_B1_sunlamp_s2021_N1024_B4；S3 | fixed φ3/B1 sunlamp s2022 N1024/B4 | 否 |
| 37 | fixed_phi3_B1_sunlamp_s2023_N1024_B4 | fixed_phi3_B1_sunlamp_s2022_N1024_B4；S3 | fixed φ3/B1 sunlamp s2023 N1024/B4 | 否 |
| 38 | fixed_phi3_B1_sunlamp_s2024_N1024_B4 | fixed_phi3_B1_sunlamp_s2023_N1024_B4；S3 | fixed φ3/B1 sunlamp s2024 N1024/B4 | 否 |
| 39 | fixed_phi3_B1_sunlamp_s2025_N1024_B4 | fixed_phi3_B1_sunlamp_s2024_N1024_B4；S3 | fixed φ3/B1 sunlamp s2025 N1024/B4 | 否 |
| 40 | fixed_phi3_B2_lightbox_s2021_N1024_B4 | fixed_phi3_B1_sunlamp_s2025_N1024_B4；S3 | fixed φ3/B2 lightbox s2021 N1024/B4 | 否 |
| 41 | fixed_phi3_B2_lightbox_s2022_N1024_B4 | fixed_phi3_B2_lightbox_s2021_N1024_B4；S3 | fixed φ3/B2 lightbox s2022 N1024/B4 | 否 |
| 42 | fixed_phi3_B2_lightbox_s2023_N1024_B4 | fixed_phi3_B2_lightbox_s2022_N1024_B4；S3 | fixed φ3/B2 lightbox s2023 N1024/B4 | 否 |
| 43 | fixed_phi3_B2_lightbox_s2024_N1024_B4 | fixed_phi3_B2_lightbox_s2023_N1024_B4；S3 | fixed φ3/B2 lightbox s2024 N1024/B4 | 否 |
| 44 | fixed_phi3_B2_lightbox_s2025_N1024_B4 | fixed_phi3_B2_lightbox_s2024_N1024_B4；S3 | fixed φ3/B2 lightbox s2025 N1024/B4 | 否 |
| 45 | fixed_phi3_B2_sunlamp_s2021_N1024_B4 | fixed_phi3_B2_lightbox_s2025_N1024_B4；S3 | fixed φ3/B2 sunlamp s2021 N1024/B4 | 否 |
| 46 | fixed_phi3_B2_sunlamp_s2022_N1024_B4 | fixed_phi3_B2_sunlamp_s2021_N1024_B4；S3 | fixed φ3/B2 sunlamp s2022 N1024/B4 | 否 |
| 47 | fixed_phi3_B2_sunlamp_s2023_N1024_B4 | fixed_phi3_B2_sunlamp_s2022_N1024_B4；S3 | fixed φ3/B2 sunlamp s2023 N1024/B4 | 否 |
| 48 | fixed_phi3_B2_sunlamp_s2024_N1024_B4 | fixed_phi3_B2_sunlamp_s2023_N1024_B4；S3 | fixed φ3/B2 sunlamp s2024 N1024/B4 | 否 |
| 49 | fixed_phi3_B2_sunlamp_s2025_N1024_B4 | fixed_phi3_B2_sunlamp_s2024_N1024_B4；S3 | fixed φ3/B2 sunlamp s2025 N1024/B4 | 否 |
| 50 | pilot_table2_h | fixed_phi3_B2_sunlamp_s2025_N1024_B4 | 工程pilot：32 attempts / checkpoint16；独立CPU验收 | 否 |
| 51 | train_table2_h | pilot_table2_h | 独立source训练 10轮 | 否 |
| 52 | eval_table2_h_synthetic | train_table2_h | source全域评价 synthetic | 否 |
| 53 | eval_table2_h_lightbox | eval_table2_h_synthetic | source全域评价 lightbox | 否 |
| 54 | eval_table2_h_sunlamp | eval_table2_h_lightbox | source全域评价 sunlamp | 否 |
| 55 | pilot_table2_e | eval_table2_h_sunlamp | 工程pilot：32 attempts / checkpoint16；独立CPU验收 | 否 |
| 56 | train_table2_e | pilot_table2_e | 独立source训练 20轮 | 否 |
| 57 | eval_table2_e_synthetic | train_table2_e | source全域评价 synthetic | 否 |
| 58 | eval_table2_e_lightbox | eval_table2_e_synthetic | source全域评价 lightbox | 否 |
| 59 | eval_table2_e_sunlamp | eval_table2_e_lightbox | source全域评价 sunlamp | 否 |
| 60 | pilot_table2_eh | eval_table2_e_sunlamp | 工程pilot：32 attempts / checkpoint16；独立CPU验收 | 否 |
| 61 | train_table2_eh | pilot_table2_eh | 独立source训练 20轮 | 否 |
| 62 | eval_table2_eh_synthetic | train_table2_eh | source全域评价 synthetic | 否 |
| 63 | eval_table2_eh_lightbox | eval_table2_eh_synthetic | source全域评价 lightbox | 否 |
| 64 | eval_table2_eh_sunlamp | eval_table2_eh_lightbox | source全域评价 sunlamp | 否 |
| 65 | train_table3_base | eval_table2_eh_sunlamp | 独立source训练 20轮 | 否 |
| 66 | eval_table3_base_synthetic | train_table3_base | source全域评价 synthetic | 否 |
| 67 | eval_table3_base_lightbox | eval_table3_base_synthetic | source全域评价 lightbox | 否 |
| 68 | eval_table3_base_sunlamp | eval_table3_base_lightbox | source全域评价 sunlamp | 否 |
| 69 | train_table3_erase | eval_table3_base_sunlamp | 独立source训练 20轮 | 否 |
| 70 | eval_table3_erase_synthetic | train_table3_erase | source全域评价 synthetic | 否 |
| 71 | eval_table3_erase_lightbox | eval_table3_erase_synthetic | source全域评价 lightbox | 否 |
| 72 | eval_table3_erase_sunlamp | eval_table3_erase_lightbox | source全域评价 sunlamp | 否 |
| 73 | train_table3_flare | eval_table3_erase_sunlamp | 独立source训练 20轮 | 否 |
| 74 | eval_table3_flare_synthetic | train_table3_flare | source全域评价 synthetic | 否 |
| 75 | eval_table3_flare_lightbox | eval_table3_flare_synthetic | source全域评价 lightbox | 否 |
| 76 | eval_table3_flare_sunlamp | eval_table3_flare_lightbox | source全域评价 sunlamp | 否 |
| 77 | train_table4_phi0_bn | eval_table3_flare_sunlamp | 独立source训练 20轮 | 否 |
| 78 | eval_table4_phi0_bn_synthetic | train_table4_phi0_bn | source全域评价 synthetic | 否 |
| 79 | eval_table4_phi0_bn_lightbox | eval_table4_phi0_bn_synthetic | source全域评价 lightbox | 否 |
| 80 | eval_table4_phi0_bn_sunlamp | eval_table4_phi0_bn_lightbox | source全域评价 sunlamp | 否 |
| 81 | train_table4_phi3_gn | eval_table4_phi0_bn_sunlamp | 独立source训练 20轮 | 否 |
| 82 | eval_table4_phi3_gn_synthetic | train_table4_phi3_gn | source全域评价 synthetic | 否 |
| 83 | eval_table4_phi3_gn_lightbox | eval_table4_phi3_gn_synthetic | source全域评价 lightbox | 否 |
| 84 | eval_table4_phi3_gn_sunlamp | eval_table4_phi3_gn_lightbox | source全域评价 sunlamp | 否 |
| 85 | Bscan_phi3_A_lightbox_s2021_N1024_B1 | eval_table4_phi3_gn_sunlamp；S3；F40 | Bscan φ3/A lightbox s2021 N1024/B1 | 否 |
| 86 | Bscan_phi3_A_lightbox_s2021_N1024_B2 | Bscan_phi3_A_lightbox_s2021_N1024_B1；S3；F40 | Bscan φ3/A lightbox s2021 N1024/B2 | 否 |
| 87 | Bscan_phi3_A_lightbox_s2021_N1024_B4 | Bscan_phi3_A_lightbox_s2021_N1024_B2；S3；F40 | Bscan φ3/A lightbox s2021 N1024/B4 | 是 → fixed_phi3_A_lightbox_s2021_N1024_B4 |
| 88 | Bscan_phi3_A_lightbox_s2021_N1024_B8 | Bscan_phi3_A_lightbox_s2021_N1024_B4；S3；F40 | Bscan φ3/A lightbox s2021 N1024/B8 | 否 |
| 89 | Bscan_phi3_A_lightbox_s2021_N1024_B16 | Bscan_phi3_A_lightbox_s2021_N1024_B8；S3；F40 | Bscan φ3/A lightbox s2021 N1024/B16 | 否 |
| 90 | Bscan_phi3_A_lightbox_s2021_N1024_B32 | Bscan_phi3_A_lightbox_s2021_N1024_B16；S3；F40 | Bscan φ3/A lightbox s2021 N1024/B32 | 否 |
| 91 | Bscan_phi3_A_lightbox_s2022_N1024_B1 | Bscan_phi3_A_lightbox_s2021_N1024_B32；S3；F40 | Bscan φ3/A lightbox s2022 N1024/B1 | 否 |
| 92 | Bscan_phi3_A_lightbox_s2022_N1024_B2 | Bscan_phi3_A_lightbox_s2022_N1024_B1；S3；F40 | Bscan φ3/A lightbox s2022 N1024/B2 | 否 |
| 93 | Bscan_phi3_A_lightbox_s2022_N1024_B4 | Bscan_phi3_A_lightbox_s2022_N1024_B2；S3；F40 | Bscan φ3/A lightbox s2022 N1024/B4 | 是 → fixed_phi3_A_lightbox_s2022_N1024_B4 |
| 94 | Bscan_phi3_A_lightbox_s2022_N1024_B8 | Bscan_phi3_A_lightbox_s2022_N1024_B4；S3；F40 | Bscan φ3/A lightbox s2022 N1024/B8 | 否 |
| 95 | Bscan_phi3_A_lightbox_s2022_N1024_B16 | Bscan_phi3_A_lightbox_s2022_N1024_B8；S3；F40 | Bscan φ3/A lightbox s2022 N1024/B16 | 否 |
| 96 | Bscan_phi3_A_lightbox_s2022_N1024_B32 | Bscan_phi3_A_lightbox_s2022_N1024_B16；S3；F40 | Bscan φ3/A lightbox s2022 N1024/B32 | 否 |
| 97 | Bscan_phi3_A_lightbox_s2023_N1024_B1 | Bscan_phi3_A_lightbox_s2022_N1024_B32；S3；F40 | Bscan φ3/A lightbox s2023 N1024/B1 | 否 |
| 98 | Bscan_phi3_A_lightbox_s2023_N1024_B2 | Bscan_phi3_A_lightbox_s2023_N1024_B1；S3；F40 | Bscan φ3/A lightbox s2023 N1024/B2 | 否 |
| 99 | Bscan_phi3_A_lightbox_s2023_N1024_B4 | Bscan_phi3_A_lightbox_s2023_N1024_B2；S3；F40 | Bscan φ3/A lightbox s2023 N1024/B4 | 是 → fixed_phi3_A_lightbox_s2023_N1024_B4 |
| 100 | Bscan_phi3_A_lightbox_s2023_N1024_B8 | Bscan_phi3_A_lightbox_s2023_N1024_B4；S3；F40 | Bscan φ3/A lightbox s2023 N1024/B8 | 否 |
| 101 | Bscan_phi3_A_lightbox_s2023_N1024_B16 | Bscan_phi3_A_lightbox_s2023_N1024_B8；S3；F40 | Bscan φ3/A lightbox s2023 N1024/B16 | 否 |
| 102 | Bscan_phi3_A_lightbox_s2023_N1024_B32 | Bscan_phi3_A_lightbox_s2023_N1024_B16；S3；F40 | Bscan φ3/A lightbox s2023 N1024/B32 | 否 |
| 103 | Bscan_phi3_A_lightbox_s2024_N1024_B1 | Bscan_phi3_A_lightbox_s2023_N1024_B32；S3；F40 | Bscan φ3/A lightbox s2024 N1024/B1 | 否 |
| 104 | Bscan_phi3_A_lightbox_s2024_N1024_B2 | Bscan_phi3_A_lightbox_s2024_N1024_B1；S3；F40 | Bscan φ3/A lightbox s2024 N1024/B2 | 否 |
| 105 | Bscan_phi3_A_lightbox_s2024_N1024_B4 | Bscan_phi3_A_lightbox_s2024_N1024_B2；S3；F40 | Bscan φ3/A lightbox s2024 N1024/B4 | 是 → fixed_phi3_A_lightbox_s2024_N1024_B4 |
| 106 | Bscan_phi3_A_lightbox_s2024_N1024_B8 | Bscan_phi3_A_lightbox_s2024_N1024_B4；S3；F40 | Bscan φ3/A lightbox s2024 N1024/B8 | 否 |
| 107 | Bscan_phi3_A_lightbox_s2024_N1024_B16 | Bscan_phi3_A_lightbox_s2024_N1024_B8；S3；F40 | Bscan φ3/A lightbox s2024 N1024/B16 | 否 |
| 108 | Bscan_phi3_A_lightbox_s2024_N1024_B32 | Bscan_phi3_A_lightbox_s2024_N1024_B16；S3；F40 | Bscan φ3/A lightbox s2024 N1024/B32 | 否 |
| 109 | Bscan_phi3_A_lightbox_s2025_N1024_B1 | Bscan_phi3_A_lightbox_s2024_N1024_B32；S3；F40 | Bscan φ3/A lightbox s2025 N1024/B1 | 否 |
| 110 | Bscan_phi3_A_lightbox_s2025_N1024_B2 | Bscan_phi3_A_lightbox_s2025_N1024_B1；S3；F40 | Bscan φ3/A lightbox s2025 N1024/B2 | 否 |
| 111 | Bscan_phi3_A_lightbox_s2025_N1024_B4 | Bscan_phi3_A_lightbox_s2025_N1024_B2；S3；F40 | Bscan φ3/A lightbox s2025 N1024/B4 | 是 → fixed_phi3_A_lightbox_s2025_N1024_B4 |
| 112 | Bscan_phi3_A_lightbox_s2025_N1024_B8 | Bscan_phi3_A_lightbox_s2025_N1024_B4；S3；F40 | Bscan φ3/A lightbox s2025 N1024/B8 | 否 |
| 113 | Bscan_phi3_A_lightbox_s2025_N1024_B16 | Bscan_phi3_A_lightbox_s2025_N1024_B8；S3；F40 | Bscan φ3/A lightbox s2025 N1024/B16 | 否 |
| 114 | Bscan_phi3_A_lightbox_s2025_N1024_B32 | Bscan_phi3_A_lightbox_s2025_N1024_B16；S3；F40 | Bscan φ3/A lightbox s2025 N1024/B32 | 否 |
| 115 | Bscan_phi3_A_sunlamp_s2021_N1024_B1 | Bscan_phi3_A_lightbox_s2025_N1024_B32；S3；F40 | Bscan φ3/A sunlamp s2021 N1024/B1 | 否 |
| 116 | Bscan_phi3_A_sunlamp_s2021_N1024_B2 | Bscan_phi3_A_sunlamp_s2021_N1024_B1；S3；F40 | Bscan φ3/A sunlamp s2021 N1024/B2 | 否 |
| 117 | Bscan_phi3_A_sunlamp_s2021_N1024_B4 | Bscan_phi3_A_sunlamp_s2021_N1024_B2；S3；F40 | Bscan φ3/A sunlamp s2021 N1024/B4 | 是 → fixed_phi3_A_sunlamp_s2021_N1024_B4 |
| 118 | Bscan_phi3_A_sunlamp_s2021_N1024_B8 | Bscan_phi3_A_sunlamp_s2021_N1024_B4；S3；F40 | Bscan φ3/A sunlamp s2021 N1024/B8 | 否 |
| 119 | Bscan_phi3_A_sunlamp_s2021_N1024_B16 | Bscan_phi3_A_sunlamp_s2021_N1024_B8；S3；F40 | Bscan φ3/A sunlamp s2021 N1024/B16 | 否 |
| 120 | Bscan_phi3_A_sunlamp_s2021_N1024_B32 | Bscan_phi3_A_sunlamp_s2021_N1024_B16；S3；F40 | Bscan φ3/A sunlamp s2021 N1024/B32 | 否 |
| 121 | Bscan_phi3_A_sunlamp_s2022_N1024_B1 | Bscan_phi3_A_sunlamp_s2021_N1024_B32；S3；F40 | Bscan φ3/A sunlamp s2022 N1024/B1 | 否 |
| 122 | Bscan_phi3_A_sunlamp_s2022_N1024_B2 | Bscan_phi3_A_sunlamp_s2022_N1024_B1；S3；F40 | Bscan φ3/A sunlamp s2022 N1024/B2 | 否 |
| 123 | Bscan_phi3_A_sunlamp_s2022_N1024_B4 | Bscan_phi3_A_sunlamp_s2022_N1024_B2；S3；F40 | Bscan φ3/A sunlamp s2022 N1024/B4 | 是 → fixed_phi3_A_sunlamp_s2022_N1024_B4 |
| 124 | Bscan_phi3_A_sunlamp_s2022_N1024_B8 | Bscan_phi3_A_sunlamp_s2022_N1024_B4；S3；F40 | Bscan φ3/A sunlamp s2022 N1024/B8 | 否 |
| 125 | Bscan_phi3_A_sunlamp_s2022_N1024_B16 | Bscan_phi3_A_sunlamp_s2022_N1024_B8；S3；F40 | Bscan φ3/A sunlamp s2022 N1024/B16 | 否 |
| 126 | Bscan_phi3_A_sunlamp_s2022_N1024_B32 | Bscan_phi3_A_sunlamp_s2022_N1024_B16；S3；F40 | Bscan φ3/A sunlamp s2022 N1024/B32 | 否 |
| 127 | Bscan_phi3_A_sunlamp_s2023_N1024_B1 | Bscan_phi3_A_sunlamp_s2022_N1024_B32；S3；F40 | Bscan φ3/A sunlamp s2023 N1024/B1 | 否 |
| 128 | Bscan_phi3_A_sunlamp_s2023_N1024_B2 | Bscan_phi3_A_sunlamp_s2023_N1024_B1；S3；F40 | Bscan φ3/A sunlamp s2023 N1024/B2 | 否 |
| 129 | Bscan_phi3_A_sunlamp_s2023_N1024_B4 | Bscan_phi3_A_sunlamp_s2023_N1024_B2；S3；F40 | Bscan φ3/A sunlamp s2023 N1024/B4 | 是 → fixed_phi3_A_sunlamp_s2023_N1024_B4 |
| 130 | Bscan_phi3_A_sunlamp_s2023_N1024_B8 | Bscan_phi3_A_sunlamp_s2023_N1024_B4；S3；F40 | Bscan φ3/A sunlamp s2023 N1024/B8 | 否 |
| 131 | Bscan_phi3_A_sunlamp_s2023_N1024_B16 | Bscan_phi3_A_sunlamp_s2023_N1024_B8；S3；F40 | Bscan φ3/A sunlamp s2023 N1024/B16 | 否 |
| 132 | Bscan_phi3_A_sunlamp_s2023_N1024_B32 | Bscan_phi3_A_sunlamp_s2023_N1024_B16；S3；F40 | Bscan φ3/A sunlamp s2023 N1024/B32 | 否 |
| 133 | Bscan_phi3_A_sunlamp_s2024_N1024_B1 | Bscan_phi3_A_sunlamp_s2023_N1024_B32；S3；F40 | Bscan φ3/A sunlamp s2024 N1024/B1 | 否 |
| 134 | Bscan_phi3_A_sunlamp_s2024_N1024_B2 | Bscan_phi3_A_sunlamp_s2024_N1024_B1；S3；F40 | Bscan φ3/A sunlamp s2024 N1024/B2 | 否 |
| 135 | Bscan_phi3_A_sunlamp_s2024_N1024_B4 | Bscan_phi3_A_sunlamp_s2024_N1024_B2；S3；F40 | Bscan φ3/A sunlamp s2024 N1024/B4 | 是 → fixed_phi3_A_sunlamp_s2024_N1024_B4 |
| 136 | Bscan_phi3_A_sunlamp_s2024_N1024_B8 | Bscan_phi3_A_sunlamp_s2024_N1024_B4；S3；F40 | Bscan φ3/A sunlamp s2024 N1024/B8 | 否 |
| 137 | Bscan_phi3_A_sunlamp_s2024_N1024_B16 | Bscan_phi3_A_sunlamp_s2024_N1024_B8；S3；F40 | Bscan φ3/A sunlamp s2024 N1024/B16 | 否 |
| 138 | Bscan_phi3_A_sunlamp_s2024_N1024_B32 | Bscan_phi3_A_sunlamp_s2024_N1024_B16；S3；F40 | Bscan φ3/A sunlamp s2024 N1024/B32 | 否 |
| 139 | Bscan_phi3_A_sunlamp_s2025_N1024_B1 | Bscan_phi3_A_sunlamp_s2024_N1024_B32；S3；F40 | Bscan φ3/A sunlamp s2025 N1024/B1 | 否 |
| 140 | Bscan_phi3_A_sunlamp_s2025_N1024_B2 | Bscan_phi3_A_sunlamp_s2025_N1024_B1；S3；F40 | Bscan φ3/A sunlamp s2025 N1024/B2 | 否 |
| 141 | Bscan_phi3_A_sunlamp_s2025_N1024_B4 | Bscan_phi3_A_sunlamp_s2025_N1024_B2；S3；F40 | Bscan φ3/A sunlamp s2025 N1024/B4 | 是 → fixed_phi3_A_sunlamp_s2025_N1024_B4 |
| 142 | Bscan_phi3_A_sunlamp_s2025_N1024_B8 | Bscan_phi3_A_sunlamp_s2025_N1024_B4；S3；F40 | Bscan φ3/A sunlamp s2025 N1024/B8 | 否 |
| 143 | Bscan_phi3_A_sunlamp_s2025_N1024_B16 | Bscan_phi3_A_sunlamp_s2025_N1024_B8；S3；F40 | Bscan φ3/A sunlamp s2025 N1024/B16 | 否 |
| 144 | Bscan_phi3_A_sunlamp_s2025_N1024_B32 | Bscan_phi3_A_sunlamp_s2025_N1024_B16；S3；F40 | Bscan φ3/A sunlamp s2025 N1024/B32 | 否 |
| 145 | Bscan_phi3_B2_lightbox_s2021_N1024_B1 | Bscan_phi3_A_sunlamp_s2025_N1024_B32；S3；F40 | Bscan φ3/B2 lightbox s2021 N1024/B1 | 否 |
| 146 | Bscan_phi3_B2_lightbox_s2021_N1024_B2 | Bscan_phi3_B2_lightbox_s2021_N1024_B1；S3；F40 | Bscan φ3/B2 lightbox s2021 N1024/B2 | 否 |
| 147 | Bscan_phi3_B2_lightbox_s2021_N1024_B4 | Bscan_phi3_B2_lightbox_s2021_N1024_B2；S3；F40 | Bscan φ3/B2 lightbox s2021 N1024/B4 | 是 → fixed_phi3_B2_lightbox_s2021_N1024_B4 |
| 148 | Bscan_phi3_B2_lightbox_s2021_N1024_B8 | Bscan_phi3_B2_lightbox_s2021_N1024_B4；S3；F40 | Bscan φ3/B2 lightbox s2021 N1024/B8 | 否 |
| 149 | Bscan_phi3_B2_lightbox_s2021_N1024_B16 | Bscan_phi3_B2_lightbox_s2021_N1024_B8；S3；F40 | Bscan φ3/B2 lightbox s2021 N1024/B16 | 否 |
| 150 | Bscan_phi3_B2_lightbox_s2021_N1024_B32 | Bscan_phi3_B2_lightbox_s2021_N1024_B16；S3；F40 | Bscan φ3/B2 lightbox s2021 N1024/B32 | 否 |
| 151 | Bscan_phi3_B2_lightbox_s2022_N1024_B1 | Bscan_phi3_B2_lightbox_s2021_N1024_B32；S3；F40 | Bscan φ3/B2 lightbox s2022 N1024/B1 | 否 |
| 152 | Bscan_phi3_B2_lightbox_s2022_N1024_B2 | Bscan_phi3_B2_lightbox_s2022_N1024_B1；S3；F40 | Bscan φ3/B2 lightbox s2022 N1024/B2 | 否 |
| 153 | Bscan_phi3_B2_lightbox_s2022_N1024_B4 | Bscan_phi3_B2_lightbox_s2022_N1024_B2；S3；F40 | Bscan φ3/B2 lightbox s2022 N1024/B4 | 是 → fixed_phi3_B2_lightbox_s2022_N1024_B4 |
| 154 | Bscan_phi3_B2_lightbox_s2022_N1024_B8 | Bscan_phi3_B2_lightbox_s2022_N1024_B4；S3；F40 | Bscan φ3/B2 lightbox s2022 N1024/B8 | 否 |
| 155 | Bscan_phi3_B2_lightbox_s2022_N1024_B16 | Bscan_phi3_B2_lightbox_s2022_N1024_B8；S3；F40 | Bscan φ3/B2 lightbox s2022 N1024/B16 | 否 |
| 156 | Bscan_phi3_B2_lightbox_s2022_N1024_B32 | Bscan_phi3_B2_lightbox_s2022_N1024_B16；S3；F40 | Bscan φ3/B2 lightbox s2022 N1024/B32 | 否 |
| 157 | Bscan_phi3_B2_lightbox_s2023_N1024_B1 | Bscan_phi3_B2_lightbox_s2022_N1024_B32；S3；F40 | Bscan φ3/B2 lightbox s2023 N1024/B1 | 否 |
| 158 | Bscan_phi3_B2_lightbox_s2023_N1024_B2 | Bscan_phi3_B2_lightbox_s2023_N1024_B1；S3；F40 | Bscan φ3/B2 lightbox s2023 N1024/B2 | 否 |
| 159 | Bscan_phi3_B2_lightbox_s2023_N1024_B4 | Bscan_phi3_B2_lightbox_s2023_N1024_B2；S3；F40 | Bscan φ3/B2 lightbox s2023 N1024/B4 | 是 → fixed_phi3_B2_lightbox_s2023_N1024_B4 |
| 160 | Bscan_phi3_B2_lightbox_s2023_N1024_B8 | Bscan_phi3_B2_lightbox_s2023_N1024_B4；S3；F40 | Bscan φ3/B2 lightbox s2023 N1024/B8 | 否 |
| 161 | Bscan_phi3_B2_lightbox_s2023_N1024_B16 | Bscan_phi3_B2_lightbox_s2023_N1024_B8；S3；F40 | Bscan φ3/B2 lightbox s2023 N1024/B16 | 否 |
| 162 | Bscan_phi3_B2_lightbox_s2023_N1024_B32 | Bscan_phi3_B2_lightbox_s2023_N1024_B16；S3；F40 | Bscan φ3/B2 lightbox s2023 N1024/B32 | 否 |
| 163 | Bscan_phi3_B2_lightbox_s2024_N1024_B1 | Bscan_phi3_B2_lightbox_s2023_N1024_B32；S3；F40 | Bscan φ3/B2 lightbox s2024 N1024/B1 | 否 |
| 164 | Bscan_phi3_B2_lightbox_s2024_N1024_B2 | Bscan_phi3_B2_lightbox_s2024_N1024_B1；S3；F40 | Bscan φ3/B2 lightbox s2024 N1024/B2 | 否 |
| 165 | Bscan_phi3_B2_lightbox_s2024_N1024_B4 | Bscan_phi3_B2_lightbox_s2024_N1024_B2；S3；F40 | Bscan φ3/B2 lightbox s2024 N1024/B4 | 是 → fixed_phi3_B2_lightbox_s2024_N1024_B4 |
| 166 | Bscan_phi3_B2_lightbox_s2024_N1024_B8 | Bscan_phi3_B2_lightbox_s2024_N1024_B4；S3；F40 | Bscan φ3/B2 lightbox s2024 N1024/B8 | 否 |
| 167 | Bscan_phi3_B2_lightbox_s2024_N1024_B16 | Bscan_phi3_B2_lightbox_s2024_N1024_B8；S3；F40 | Bscan φ3/B2 lightbox s2024 N1024/B16 | 否 |
| 168 | Bscan_phi3_B2_lightbox_s2024_N1024_B32 | Bscan_phi3_B2_lightbox_s2024_N1024_B16；S3；F40 | Bscan φ3/B2 lightbox s2024 N1024/B32 | 否 |
| 169 | Bscan_phi3_B2_lightbox_s2025_N1024_B1 | Bscan_phi3_B2_lightbox_s2024_N1024_B32；S3；F40 | Bscan φ3/B2 lightbox s2025 N1024/B1 | 否 |
| 170 | Bscan_phi3_B2_lightbox_s2025_N1024_B2 | Bscan_phi3_B2_lightbox_s2025_N1024_B1；S3；F40 | Bscan φ3/B2 lightbox s2025 N1024/B2 | 否 |
| 171 | Bscan_phi3_B2_lightbox_s2025_N1024_B4 | Bscan_phi3_B2_lightbox_s2025_N1024_B2；S3；F40 | Bscan φ3/B2 lightbox s2025 N1024/B4 | 是 → fixed_phi3_B2_lightbox_s2025_N1024_B4 |
| 172 | Bscan_phi3_B2_lightbox_s2025_N1024_B8 | Bscan_phi3_B2_lightbox_s2025_N1024_B4；S3；F40 | Bscan φ3/B2 lightbox s2025 N1024/B8 | 否 |
| 173 | Bscan_phi3_B2_lightbox_s2025_N1024_B16 | Bscan_phi3_B2_lightbox_s2025_N1024_B8；S3；F40 | Bscan φ3/B2 lightbox s2025 N1024/B16 | 否 |
| 174 | Bscan_phi3_B2_lightbox_s2025_N1024_B32 | Bscan_phi3_B2_lightbox_s2025_N1024_B16；S3；F40 | Bscan φ3/B2 lightbox s2025 N1024/B32 | 否 |
| 175 | Bscan_phi3_B2_sunlamp_s2021_N1024_B1 | Bscan_phi3_B2_lightbox_s2025_N1024_B32；S3；F40 | Bscan φ3/B2 sunlamp s2021 N1024/B1 | 否 |
| 176 | Bscan_phi3_B2_sunlamp_s2021_N1024_B2 | Bscan_phi3_B2_sunlamp_s2021_N1024_B1；S3；F40 | Bscan φ3/B2 sunlamp s2021 N1024/B2 | 否 |
| 177 | Bscan_phi3_B2_sunlamp_s2021_N1024_B4 | Bscan_phi3_B2_sunlamp_s2021_N1024_B2；S3；F40 | Bscan φ3/B2 sunlamp s2021 N1024/B4 | 是 → fixed_phi3_B2_sunlamp_s2021_N1024_B4 |
| 178 | Bscan_phi3_B2_sunlamp_s2021_N1024_B8 | Bscan_phi3_B2_sunlamp_s2021_N1024_B4；S3；F40 | Bscan φ3/B2 sunlamp s2021 N1024/B8 | 否 |
| 179 | Bscan_phi3_B2_sunlamp_s2021_N1024_B16 | Bscan_phi3_B2_sunlamp_s2021_N1024_B8；S3；F40 | Bscan φ3/B2 sunlamp s2021 N1024/B16 | 否 |
| 180 | Bscan_phi3_B2_sunlamp_s2021_N1024_B32 | Bscan_phi3_B2_sunlamp_s2021_N1024_B16；S3；F40 | Bscan φ3/B2 sunlamp s2021 N1024/B32 | 否 |
| 181 | Bscan_phi3_B2_sunlamp_s2022_N1024_B1 | Bscan_phi3_B2_sunlamp_s2021_N1024_B32；S3；F40 | Bscan φ3/B2 sunlamp s2022 N1024/B1 | 否 |
| 182 | Bscan_phi3_B2_sunlamp_s2022_N1024_B2 | Bscan_phi3_B2_sunlamp_s2022_N1024_B1；S3；F40 | Bscan φ3/B2 sunlamp s2022 N1024/B2 | 否 |
| 183 | Bscan_phi3_B2_sunlamp_s2022_N1024_B4 | Bscan_phi3_B2_sunlamp_s2022_N1024_B2；S3；F40 | Bscan φ3/B2 sunlamp s2022 N1024/B4 | 是 → fixed_phi3_B2_sunlamp_s2022_N1024_B4 |
| 184 | Bscan_phi3_B2_sunlamp_s2022_N1024_B8 | Bscan_phi3_B2_sunlamp_s2022_N1024_B4；S3；F40 | Bscan φ3/B2 sunlamp s2022 N1024/B8 | 否 |
| 185 | Bscan_phi3_B2_sunlamp_s2022_N1024_B16 | Bscan_phi3_B2_sunlamp_s2022_N1024_B8；S3；F40 | Bscan φ3/B2 sunlamp s2022 N1024/B16 | 否 |
| 186 | Bscan_phi3_B2_sunlamp_s2022_N1024_B32 | Bscan_phi3_B2_sunlamp_s2022_N1024_B16；S3；F40 | Bscan φ3/B2 sunlamp s2022 N1024/B32 | 否 |
| 187 | Bscan_phi3_B2_sunlamp_s2023_N1024_B1 | Bscan_phi3_B2_sunlamp_s2022_N1024_B32；S3；F40 | Bscan φ3/B2 sunlamp s2023 N1024/B1 | 否 |
| 188 | Bscan_phi3_B2_sunlamp_s2023_N1024_B2 | Bscan_phi3_B2_sunlamp_s2023_N1024_B1；S3；F40 | Bscan φ3/B2 sunlamp s2023 N1024/B2 | 否 |
| 189 | Bscan_phi3_B2_sunlamp_s2023_N1024_B4 | Bscan_phi3_B2_sunlamp_s2023_N1024_B2；S3；F40 | Bscan φ3/B2 sunlamp s2023 N1024/B4 | 是 → fixed_phi3_B2_sunlamp_s2023_N1024_B4 |
| 190 | Bscan_phi3_B2_sunlamp_s2023_N1024_B8 | Bscan_phi3_B2_sunlamp_s2023_N1024_B4；S3；F40 | Bscan φ3/B2 sunlamp s2023 N1024/B8 | 否 |
| 191 | Bscan_phi3_B2_sunlamp_s2023_N1024_B16 | Bscan_phi3_B2_sunlamp_s2023_N1024_B8；S3；F40 | Bscan φ3/B2 sunlamp s2023 N1024/B16 | 否 |
| 192 | Bscan_phi3_B2_sunlamp_s2023_N1024_B32 | Bscan_phi3_B2_sunlamp_s2023_N1024_B16；S3；F40 | Bscan φ3/B2 sunlamp s2023 N1024/B32 | 否 |
| 193 | Bscan_phi3_B2_sunlamp_s2024_N1024_B1 | Bscan_phi3_B2_sunlamp_s2023_N1024_B32；S3；F40 | Bscan φ3/B2 sunlamp s2024 N1024/B1 | 否 |
| 194 | Bscan_phi3_B2_sunlamp_s2024_N1024_B2 | Bscan_phi3_B2_sunlamp_s2024_N1024_B1；S3；F40 | Bscan φ3/B2 sunlamp s2024 N1024/B2 | 否 |
| 195 | Bscan_phi3_B2_sunlamp_s2024_N1024_B4 | Bscan_phi3_B2_sunlamp_s2024_N1024_B2；S3；F40 | Bscan φ3/B2 sunlamp s2024 N1024/B4 | 是 → fixed_phi3_B2_sunlamp_s2024_N1024_B4 |
| 196 | Bscan_phi3_B2_sunlamp_s2024_N1024_B8 | Bscan_phi3_B2_sunlamp_s2024_N1024_B4；S3；F40 | Bscan φ3/B2 sunlamp s2024 N1024/B8 | 否 |
| 197 | Bscan_phi3_B2_sunlamp_s2024_N1024_B16 | Bscan_phi3_B2_sunlamp_s2024_N1024_B8；S3；F40 | Bscan φ3/B2 sunlamp s2024 N1024/B16 | 否 |
| 198 | Bscan_phi3_B2_sunlamp_s2024_N1024_B32 | Bscan_phi3_B2_sunlamp_s2024_N1024_B16；S3；F40 | Bscan φ3/B2 sunlamp s2024 N1024/B32 | 否 |
| 199 | Bscan_phi3_B2_sunlamp_s2025_N1024_B1 | Bscan_phi3_B2_sunlamp_s2024_N1024_B32；S3；F40 | Bscan φ3/B2 sunlamp s2025 N1024/B1 | 否 |
| 200 | Bscan_phi3_B2_sunlamp_s2025_N1024_B2 | Bscan_phi3_B2_sunlamp_s2025_N1024_B1；S3；F40 | Bscan φ3/B2 sunlamp s2025 N1024/B2 | 否 |
| 201 | Bscan_phi3_B2_sunlamp_s2025_N1024_B4 | Bscan_phi3_B2_sunlamp_s2025_N1024_B2；S3；F40 | Bscan φ3/B2 sunlamp s2025 N1024/B4 | 是 → fixed_phi3_B2_sunlamp_s2025_N1024_B4 |
| 202 | Bscan_phi3_B2_sunlamp_s2025_N1024_B8 | Bscan_phi3_B2_sunlamp_s2025_N1024_B4；S3；F40 | Bscan φ3/B2 sunlamp s2025 N1024/B8 | 否 |
| 203 | Bscan_phi3_B2_sunlamp_s2025_N1024_B16 | Bscan_phi3_B2_sunlamp_s2025_N1024_B8；S3；F40 | Bscan φ3/B2 sunlamp s2025 N1024/B16 | 否 |
| 204 | Bscan_phi3_B2_sunlamp_s2025_N1024_B32 | Bscan_phi3_B2_sunlamp_s2025_N1024_B16；S3；F40 | Bscan φ3/B2 sunlamp s2025 N1024/B32 | 否 |
| 205 | Nscan_phi3_A_lightbox_s2021_N4_B4 | Bscan_phi3_B2_sunlamp_s2025_N1024_B32；S3；F40 | Nscan φ3/A lightbox s2021 N4/B4 | 否 |
| 206 | Nscan_phi3_A_lightbox_s2021_N8_B4 | Nscan_phi3_A_lightbox_s2021_N4_B4；S3；F40 | Nscan φ3/A lightbox s2021 N8/B4 | 否 |
| 207 | Nscan_phi3_A_lightbox_s2021_N16_B4 | Nscan_phi3_A_lightbox_s2021_N8_B4；S3；F40 | Nscan φ3/A lightbox s2021 N16/B4 | 否 |
| 208 | Nscan_phi3_A_lightbox_s2021_N32_B4 | Nscan_phi3_A_lightbox_s2021_N16_B4；S3；F40 | Nscan φ3/A lightbox s2021 N32/B4 | 否 |
| 209 | Nscan_phi3_A_lightbox_s2021_N64_B4 | Nscan_phi3_A_lightbox_s2021_N32_B4；S3；F40 | Nscan φ3/A lightbox s2021 N64/B4 | 否 |
| 210 | Nscan_phi3_A_lightbox_s2021_N128_B4 | Nscan_phi3_A_lightbox_s2021_N64_B4；S3；F40 | Nscan φ3/A lightbox s2021 N128/B4 | 否 |
| 211 | Nscan_phi3_A_lightbox_s2021_N256_B4 | Nscan_phi3_A_lightbox_s2021_N128_B4；S3；F40 | Nscan φ3/A lightbox s2021 N256/B4 | 否 |
| 212 | Nscan_phi3_A_lightbox_s2021_N512_B4 | Nscan_phi3_A_lightbox_s2021_N256_B4；S3；F40 | Nscan φ3/A lightbox s2021 N512/B4 | 否 |
| 213 | Nscan_phi3_A_lightbox_s2021_N1024_B4 | Nscan_phi3_A_lightbox_s2021_N512_B4；S3；F40 | Nscan φ3/A lightbox s2021 N1024/B4 | 是 → fixed_phi3_A_lightbox_s2021_N1024_B4 |
| 214 | Nscan_phi3_A_lightbox_s2021_N2048_B4 | Nscan_phi3_A_lightbox_s2021_N1024_B4；S3；F40 | Nscan φ3/A lightbox s2021 N2048/B4 | 否 |
| 215 | Nscan_phi3_A_lightbox_s2021_N4096_B4 | Nscan_phi3_A_lightbox_s2021_N2048_B4；S3；F40 | Nscan φ3/A lightbox s2021 N4096/B4 | 否 |
| 216 | Nscan_phi3_A_lightbox_s2021_N8192_B4 | Nscan_phi3_A_lightbox_s2021_N4096_B4；S3；F40 | Nscan φ3/A lightbox s2021 N8192/B4 | 否 |
| 217 | Nscan_phi3_A_lightbox_s2022_N4_B4 | Nscan_phi3_A_lightbox_s2021_N8192_B4；S3；F40 | Nscan φ3/A lightbox s2022 N4/B4 | 否 |
| 218 | Nscan_phi3_A_lightbox_s2022_N8_B4 | Nscan_phi3_A_lightbox_s2022_N4_B4；S3；F40 | Nscan φ3/A lightbox s2022 N8/B4 | 否 |
| 219 | Nscan_phi3_A_lightbox_s2022_N16_B4 | Nscan_phi3_A_lightbox_s2022_N8_B4；S3；F40 | Nscan φ3/A lightbox s2022 N16/B4 | 否 |
| 220 | Nscan_phi3_A_lightbox_s2022_N32_B4 | Nscan_phi3_A_lightbox_s2022_N16_B4；S3；F40 | Nscan φ3/A lightbox s2022 N32/B4 | 否 |
| 221 | Nscan_phi3_A_lightbox_s2022_N64_B4 | Nscan_phi3_A_lightbox_s2022_N32_B4；S3；F40 | Nscan φ3/A lightbox s2022 N64/B4 | 否 |
| 222 | Nscan_phi3_A_lightbox_s2022_N128_B4 | Nscan_phi3_A_lightbox_s2022_N64_B4；S3；F40 | Nscan φ3/A lightbox s2022 N128/B4 | 否 |
| 223 | Nscan_phi3_A_lightbox_s2022_N256_B4 | Nscan_phi3_A_lightbox_s2022_N128_B4；S3；F40 | Nscan φ3/A lightbox s2022 N256/B4 | 否 |
| 224 | Nscan_phi3_A_lightbox_s2022_N512_B4 | Nscan_phi3_A_lightbox_s2022_N256_B4；S3；F40 | Nscan φ3/A lightbox s2022 N512/B4 | 否 |
| 225 | Nscan_phi3_A_lightbox_s2022_N1024_B4 | Nscan_phi3_A_lightbox_s2022_N512_B4；S3；F40 | Nscan φ3/A lightbox s2022 N1024/B4 | 是 → fixed_phi3_A_lightbox_s2022_N1024_B4 |
| 226 | Nscan_phi3_A_lightbox_s2022_N2048_B4 | Nscan_phi3_A_lightbox_s2022_N1024_B4；S3；F40 | Nscan φ3/A lightbox s2022 N2048/B4 | 否 |
| 227 | Nscan_phi3_A_lightbox_s2022_N4096_B4 | Nscan_phi3_A_lightbox_s2022_N2048_B4；S3；F40 | Nscan φ3/A lightbox s2022 N4096/B4 | 否 |
| 228 | Nscan_phi3_A_lightbox_s2022_N8192_B4 | Nscan_phi3_A_lightbox_s2022_N4096_B4；S3；F40 | Nscan φ3/A lightbox s2022 N8192/B4 | 否 |
| 229 | Nscan_phi3_A_lightbox_s2023_N4_B4 | Nscan_phi3_A_lightbox_s2022_N8192_B4；S3；F40 | Nscan φ3/A lightbox s2023 N4/B4 | 否 |
| 230 | Nscan_phi3_A_lightbox_s2023_N8_B4 | Nscan_phi3_A_lightbox_s2023_N4_B4；S3；F40 | Nscan φ3/A lightbox s2023 N8/B4 | 否 |
| 231 | Nscan_phi3_A_lightbox_s2023_N16_B4 | Nscan_phi3_A_lightbox_s2023_N8_B4；S3；F40 | Nscan φ3/A lightbox s2023 N16/B4 | 否 |
| 232 | Nscan_phi3_A_lightbox_s2023_N32_B4 | Nscan_phi3_A_lightbox_s2023_N16_B4；S3；F40 | Nscan φ3/A lightbox s2023 N32/B4 | 否 |
| 233 | Nscan_phi3_A_lightbox_s2023_N64_B4 | Nscan_phi3_A_lightbox_s2023_N32_B4；S3；F40 | Nscan φ3/A lightbox s2023 N64/B4 | 否 |
| 234 | Nscan_phi3_A_lightbox_s2023_N128_B4 | Nscan_phi3_A_lightbox_s2023_N64_B4；S3；F40 | Nscan φ3/A lightbox s2023 N128/B4 | 否 |
| 235 | Nscan_phi3_A_lightbox_s2023_N256_B4 | Nscan_phi3_A_lightbox_s2023_N128_B4；S3；F40 | Nscan φ3/A lightbox s2023 N256/B4 | 否 |
| 236 | Nscan_phi3_A_lightbox_s2023_N512_B4 | Nscan_phi3_A_lightbox_s2023_N256_B4；S3；F40 | Nscan φ3/A lightbox s2023 N512/B4 | 否 |
| 237 | Nscan_phi3_A_lightbox_s2023_N1024_B4 | Nscan_phi3_A_lightbox_s2023_N512_B4；S3；F40 | Nscan φ3/A lightbox s2023 N1024/B4 | 是 → fixed_phi3_A_lightbox_s2023_N1024_B4 |
| 238 | Nscan_phi3_A_lightbox_s2023_N2048_B4 | Nscan_phi3_A_lightbox_s2023_N1024_B4；S3；F40 | Nscan φ3/A lightbox s2023 N2048/B4 | 否 |
| 239 | Nscan_phi3_A_lightbox_s2023_N4096_B4 | Nscan_phi3_A_lightbox_s2023_N2048_B4；S3；F40 | Nscan φ3/A lightbox s2023 N4096/B4 | 否 |
| 240 | Nscan_phi3_A_lightbox_s2023_N8192_B4 | Nscan_phi3_A_lightbox_s2023_N4096_B4；S3；F40 | Nscan φ3/A lightbox s2023 N8192/B4 | 否 |
| 241 | Nscan_phi3_A_lightbox_s2024_N4_B4 | Nscan_phi3_A_lightbox_s2023_N8192_B4；S3；F40 | Nscan φ3/A lightbox s2024 N4/B4 | 否 |
| 242 | Nscan_phi3_A_lightbox_s2024_N8_B4 | Nscan_phi3_A_lightbox_s2024_N4_B4；S3；F40 | Nscan φ3/A lightbox s2024 N8/B4 | 否 |
| 243 | Nscan_phi3_A_lightbox_s2024_N16_B4 | Nscan_phi3_A_lightbox_s2024_N8_B4；S3；F40 | Nscan φ3/A lightbox s2024 N16/B4 | 否 |
| 244 | Nscan_phi3_A_lightbox_s2024_N32_B4 | Nscan_phi3_A_lightbox_s2024_N16_B4；S3；F40 | Nscan φ3/A lightbox s2024 N32/B4 | 否 |
| 245 | Nscan_phi3_A_lightbox_s2024_N64_B4 | Nscan_phi3_A_lightbox_s2024_N32_B4；S3；F40 | Nscan φ3/A lightbox s2024 N64/B4 | 否 |
| 246 | Nscan_phi3_A_lightbox_s2024_N128_B4 | Nscan_phi3_A_lightbox_s2024_N64_B4；S3；F40 | Nscan φ3/A lightbox s2024 N128/B4 | 否 |
| 247 | Nscan_phi3_A_lightbox_s2024_N256_B4 | Nscan_phi3_A_lightbox_s2024_N128_B4；S3；F40 | Nscan φ3/A lightbox s2024 N256/B4 | 否 |
| 248 | Nscan_phi3_A_lightbox_s2024_N512_B4 | Nscan_phi3_A_lightbox_s2024_N256_B4；S3；F40 | Nscan φ3/A lightbox s2024 N512/B4 | 否 |
| 249 | Nscan_phi3_A_lightbox_s2024_N1024_B4 | Nscan_phi3_A_lightbox_s2024_N512_B4；S3；F40 | Nscan φ3/A lightbox s2024 N1024/B4 | 是 → fixed_phi3_A_lightbox_s2024_N1024_B4 |
| 250 | Nscan_phi3_A_lightbox_s2024_N2048_B4 | Nscan_phi3_A_lightbox_s2024_N1024_B4；S3；F40 | Nscan φ3/A lightbox s2024 N2048/B4 | 否 |
| 251 | Nscan_phi3_A_lightbox_s2024_N4096_B4 | Nscan_phi3_A_lightbox_s2024_N2048_B4；S3；F40 | Nscan φ3/A lightbox s2024 N4096/B4 | 否 |
| 252 | Nscan_phi3_A_lightbox_s2024_N8192_B4 | Nscan_phi3_A_lightbox_s2024_N4096_B4；S3；F40 | Nscan φ3/A lightbox s2024 N8192/B4 | 否 |
| 253 | Nscan_phi3_A_lightbox_s2025_N4_B4 | Nscan_phi3_A_lightbox_s2024_N8192_B4；S3；F40 | Nscan φ3/A lightbox s2025 N4/B4 | 否 |
| 254 | Nscan_phi3_A_lightbox_s2025_N8_B4 | Nscan_phi3_A_lightbox_s2025_N4_B4；S3；F40 | Nscan φ3/A lightbox s2025 N8/B4 | 否 |
| 255 | Nscan_phi3_A_lightbox_s2025_N16_B4 | Nscan_phi3_A_lightbox_s2025_N8_B4；S3；F40 | Nscan φ3/A lightbox s2025 N16/B4 | 否 |
| 256 | Nscan_phi3_A_lightbox_s2025_N32_B4 | Nscan_phi3_A_lightbox_s2025_N16_B4；S3；F40 | Nscan φ3/A lightbox s2025 N32/B4 | 否 |
| 257 | Nscan_phi3_A_lightbox_s2025_N64_B4 | Nscan_phi3_A_lightbox_s2025_N32_B4；S3；F40 | Nscan φ3/A lightbox s2025 N64/B4 | 否 |
| 258 | Nscan_phi3_A_lightbox_s2025_N128_B4 | Nscan_phi3_A_lightbox_s2025_N64_B4；S3；F40 | Nscan φ3/A lightbox s2025 N128/B4 | 否 |
| 259 | Nscan_phi3_A_lightbox_s2025_N256_B4 | Nscan_phi3_A_lightbox_s2025_N128_B4；S3；F40 | Nscan φ3/A lightbox s2025 N256/B4 | 否 |
| 260 | Nscan_phi3_A_lightbox_s2025_N512_B4 | Nscan_phi3_A_lightbox_s2025_N256_B4；S3；F40 | Nscan φ3/A lightbox s2025 N512/B4 | 否 |
| 261 | Nscan_phi3_A_lightbox_s2025_N1024_B4 | Nscan_phi3_A_lightbox_s2025_N512_B4；S3；F40 | Nscan φ3/A lightbox s2025 N1024/B4 | 是 → fixed_phi3_A_lightbox_s2025_N1024_B4 |
| 262 | Nscan_phi3_A_lightbox_s2025_N2048_B4 | Nscan_phi3_A_lightbox_s2025_N1024_B4；S3；F40 | Nscan φ3/A lightbox s2025 N2048/B4 | 否 |
| 263 | Nscan_phi3_A_lightbox_s2025_N4096_B4 | Nscan_phi3_A_lightbox_s2025_N2048_B4；S3；F40 | Nscan φ3/A lightbox s2025 N4096/B4 | 否 |
| 264 | Nscan_phi3_A_lightbox_s2025_N8192_B4 | Nscan_phi3_A_lightbox_s2025_N4096_B4；S3；F40 | Nscan φ3/A lightbox s2025 N8192/B4 | 否 |
| 265 | Nscan_phi3_A_sunlamp_s2021_N4_B4 | Nscan_phi3_A_lightbox_s2025_N8192_B4；S3；F40 | Nscan φ3/A sunlamp s2021 N4/B4 | 否 |
| 266 | Nscan_phi3_A_sunlamp_s2021_N8_B4 | Nscan_phi3_A_sunlamp_s2021_N4_B4；S3；F40 | Nscan φ3/A sunlamp s2021 N8/B4 | 否 |
| 267 | Nscan_phi3_A_sunlamp_s2021_N16_B4 | Nscan_phi3_A_sunlamp_s2021_N8_B4；S3；F40 | Nscan φ3/A sunlamp s2021 N16/B4 | 否 |
| 268 | Nscan_phi3_A_sunlamp_s2021_N32_B4 | Nscan_phi3_A_sunlamp_s2021_N16_B4；S3；F40 | Nscan φ3/A sunlamp s2021 N32/B4 | 否 |
| 269 | Nscan_phi3_A_sunlamp_s2021_N64_B4 | Nscan_phi3_A_sunlamp_s2021_N32_B4；S3；F40 | Nscan φ3/A sunlamp s2021 N64/B4 | 否 |
| 270 | Nscan_phi3_A_sunlamp_s2021_N128_B4 | Nscan_phi3_A_sunlamp_s2021_N64_B4；S3；F40 | Nscan φ3/A sunlamp s2021 N128/B4 | 否 |
| 271 | Nscan_phi3_A_sunlamp_s2021_N256_B4 | Nscan_phi3_A_sunlamp_s2021_N128_B4；S3；F40 | Nscan φ3/A sunlamp s2021 N256/B4 | 否 |
| 272 | Nscan_phi3_A_sunlamp_s2021_N512_B4 | Nscan_phi3_A_sunlamp_s2021_N256_B4；S3；F40 | Nscan φ3/A sunlamp s2021 N512/B4 | 否 |
| 273 | Nscan_phi3_A_sunlamp_s2021_N1024_B4 | Nscan_phi3_A_sunlamp_s2021_N512_B4；S3；F40 | Nscan φ3/A sunlamp s2021 N1024/B4 | 是 → fixed_phi3_A_sunlamp_s2021_N1024_B4 |
| 274 | Nscan_phi3_A_sunlamp_s2021_N2048_B4 | Nscan_phi3_A_sunlamp_s2021_N1024_B4；S3；F40 | Nscan φ3/A sunlamp s2021 N2048/B4 | 否 |
| 275 | Nscan_phi3_A_sunlamp_s2021_N4096_B4 | Nscan_phi3_A_sunlamp_s2021_N2048_B4；S3；F40 | Nscan φ3/A sunlamp s2021 N4096/B4 | 否 |
| 276 | Nscan_phi3_A_sunlamp_s2021_N8192_B4 | Nscan_phi3_A_sunlamp_s2021_N4096_B4；S3；F40 | Nscan φ3/A sunlamp s2021 N8192/B4 | 否 |
| 277 | Nscan_phi3_A_sunlamp_s2022_N4_B4 | Nscan_phi3_A_sunlamp_s2021_N8192_B4；S3；F40 | Nscan φ3/A sunlamp s2022 N4/B4 | 否 |
| 278 | Nscan_phi3_A_sunlamp_s2022_N8_B4 | Nscan_phi3_A_sunlamp_s2022_N4_B4；S3；F40 | Nscan φ3/A sunlamp s2022 N8/B4 | 否 |
| 279 | Nscan_phi3_A_sunlamp_s2022_N16_B4 | Nscan_phi3_A_sunlamp_s2022_N8_B4；S3；F40 | Nscan φ3/A sunlamp s2022 N16/B4 | 否 |
| 280 | Nscan_phi3_A_sunlamp_s2022_N32_B4 | Nscan_phi3_A_sunlamp_s2022_N16_B4；S3；F40 | Nscan φ3/A sunlamp s2022 N32/B4 | 否 |
| 281 | Nscan_phi3_A_sunlamp_s2022_N64_B4 | Nscan_phi3_A_sunlamp_s2022_N32_B4；S3；F40 | Nscan φ3/A sunlamp s2022 N64/B4 | 否 |
| 282 | Nscan_phi3_A_sunlamp_s2022_N128_B4 | Nscan_phi3_A_sunlamp_s2022_N64_B4；S3；F40 | Nscan φ3/A sunlamp s2022 N128/B4 | 否 |
| 283 | Nscan_phi3_A_sunlamp_s2022_N256_B4 | Nscan_phi3_A_sunlamp_s2022_N128_B4；S3；F40 | Nscan φ3/A sunlamp s2022 N256/B4 | 否 |
| 284 | Nscan_phi3_A_sunlamp_s2022_N512_B4 | Nscan_phi3_A_sunlamp_s2022_N256_B4；S3；F40 | Nscan φ3/A sunlamp s2022 N512/B4 | 否 |
| 285 | Nscan_phi3_A_sunlamp_s2022_N1024_B4 | Nscan_phi3_A_sunlamp_s2022_N512_B4；S3；F40 | Nscan φ3/A sunlamp s2022 N1024/B4 | 是 → fixed_phi3_A_sunlamp_s2022_N1024_B4 |
| 286 | Nscan_phi3_A_sunlamp_s2022_N2048_B4 | Nscan_phi3_A_sunlamp_s2022_N1024_B4；S3；F40 | Nscan φ3/A sunlamp s2022 N2048/B4 | 否 |
| 287 | Nscan_phi3_A_sunlamp_s2022_N4096_B4 | Nscan_phi3_A_sunlamp_s2022_N2048_B4；S3；F40 | Nscan φ3/A sunlamp s2022 N4096/B4 | 否 |
| 288 | Nscan_phi3_A_sunlamp_s2022_N8192_B4 | Nscan_phi3_A_sunlamp_s2022_N4096_B4；S3；F40 | Nscan φ3/A sunlamp s2022 N8192/B4 | 否 |
| 289 | Nscan_phi3_A_sunlamp_s2023_N4_B4 | Nscan_phi3_A_sunlamp_s2022_N8192_B4；S3；F40 | Nscan φ3/A sunlamp s2023 N4/B4 | 否 |
| 290 | Nscan_phi3_A_sunlamp_s2023_N8_B4 | Nscan_phi3_A_sunlamp_s2023_N4_B4；S3；F40 | Nscan φ3/A sunlamp s2023 N8/B4 | 否 |
| 291 | Nscan_phi3_A_sunlamp_s2023_N16_B4 | Nscan_phi3_A_sunlamp_s2023_N8_B4；S3；F40 | Nscan φ3/A sunlamp s2023 N16/B4 | 否 |
| 292 | Nscan_phi3_A_sunlamp_s2023_N32_B4 | Nscan_phi3_A_sunlamp_s2023_N16_B4；S3；F40 | Nscan φ3/A sunlamp s2023 N32/B4 | 否 |
| 293 | Nscan_phi3_A_sunlamp_s2023_N64_B4 | Nscan_phi3_A_sunlamp_s2023_N32_B4；S3；F40 | Nscan φ3/A sunlamp s2023 N64/B4 | 否 |
| 294 | Nscan_phi3_A_sunlamp_s2023_N128_B4 | Nscan_phi3_A_sunlamp_s2023_N64_B4；S3；F40 | Nscan φ3/A sunlamp s2023 N128/B4 | 否 |
| 295 | Nscan_phi3_A_sunlamp_s2023_N256_B4 | Nscan_phi3_A_sunlamp_s2023_N128_B4；S3；F40 | Nscan φ3/A sunlamp s2023 N256/B4 | 否 |
| 296 | Nscan_phi3_A_sunlamp_s2023_N512_B4 | Nscan_phi3_A_sunlamp_s2023_N256_B4；S3；F40 | Nscan φ3/A sunlamp s2023 N512/B4 | 否 |
| 297 | Nscan_phi3_A_sunlamp_s2023_N1024_B4 | Nscan_phi3_A_sunlamp_s2023_N512_B4；S3；F40 | Nscan φ3/A sunlamp s2023 N1024/B4 | 是 → fixed_phi3_A_sunlamp_s2023_N1024_B4 |
| 298 | Nscan_phi3_A_sunlamp_s2023_N2048_B4 | Nscan_phi3_A_sunlamp_s2023_N1024_B4；S3；F40 | Nscan φ3/A sunlamp s2023 N2048/B4 | 否 |
| 299 | Nscan_phi3_A_sunlamp_s2023_N4096_B4 | Nscan_phi3_A_sunlamp_s2023_N2048_B4；S3；F40 | Nscan φ3/A sunlamp s2023 N4096/B4 | 否 |
| 300 | Nscan_phi3_A_sunlamp_s2023_N8192_B4 | Nscan_phi3_A_sunlamp_s2023_N4096_B4；S3；F40 | Nscan φ3/A sunlamp s2023 N8192/B4 | 否 |
| 301 | Nscan_phi3_A_sunlamp_s2024_N4_B4 | Nscan_phi3_A_sunlamp_s2023_N8192_B4；S3；F40 | Nscan φ3/A sunlamp s2024 N4/B4 | 否 |
| 302 | Nscan_phi3_A_sunlamp_s2024_N8_B4 | Nscan_phi3_A_sunlamp_s2024_N4_B4；S3；F40 | Nscan φ3/A sunlamp s2024 N8/B4 | 否 |
| 303 | Nscan_phi3_A_sunlamp_s2024_N16_B4 | Nscan_phi3_A_sunlamp_s2024_N8_B4；S3；F40 | Nscan φ3/A sunlamp s2024 N16/B4 | 否 |
| 304 | Nscan_phi3_A_sunlamp_s2024_N32_B4 | Nscan_phi3_A_sunlamp_s2024_N16_B4；S3；F40 | Nscan φ3/A sunlamp s2024 N32/B4 | 否 |
| 305 | Nscan_phi3_A_sunlamp_s2024_N64_B4 | Nscan_phi3_A_sunlamp_s2024_N32_B4；S3；F40 | Nscan φ3/A sunlamp s2024 N64/B4 | 否 |
| 306 | Nscan_phi3_A_sunlamp_s2024_N128_B4 | Nscan_phi3_A_sunlamp_s2024_N64_B4；S3；F40 | Nscan φ3/A sunlamp s2024 N128/B4 | 否 |
| 307 | Nscan_phi3_A_sunlamp_s2024_N256_B4 | Nscan_phi3_A_sunlamp_s2024_N128_B4；S3；F40 | Nscan φ3/A sunlamp s2024 N256/B4 | 否 |
| 308 | Nscan_phi3_A_sunlamp_s2024_N512_B4 | Nscan_phi3_A_sunlamp_s2024_N256_B4；S3；F40 | Nscan φ3/A sunlamp s2024 N512/B4 | 否 |
| 309 | Nscan_phi3_A_sunlamp_s2024_N1024_B4 | Nscan_phi3_A_sunlamp_s2024_N512_B4；S3；F40 | Nscan φ3/A sunlamp s2024 N1024/B4 | 是 → fixed_phi3_A_sunlamp_s2024_N1024_B4 |
| 310 | Nscan_phi3_A_sunlamp_s2024_N2048_B4 | Nscan_phi3_A_sunlamp_s2024_N1024_B4；S3；F40 | Nscan φ3/A sunlamp s2024 N2048/B4 | 否 |
| 311 | Nscan_phi3_A_sunlamp_s2024_N4096_B4 | Nscan_phi3_A_sunlamp_s2024_N2048_B4；S3；F40 | Nscan φ3/A sunlamp s2024 N4096/B4 | 否 |
| 312 | Nscan_phi3_A_sunlamp_s2024_N8192_B4 | Nscan_phi3_A_sunlamp_s2024_N4096_B4；S3；F40 | Nscan φ3/A sunlamp s2024 N8192/B4 | 否 |
| 313 | Nscan_phi3_A_sunlamp_s2025_N4_B4 | Nscan_phi3_A_sunlamp_s2024_N8192_B4；S3；F40 | Nscan φ3/A sunlamp s2025 N4/B4 | 否 |
| 314 | Nscan_phi3_A_sunlamp_s2025_N8_B4 | Nscan_phi3_A_sunlamp_s2025_N4_B4；S3；F40 | Nscan φ3/A sunlamp s2025 N8/B4 | 否 |
| 315 | Nscan_phi3_A_sunlamp_s2025_N16_B4 | Nscan_phi3_A_sunlamp_s2025_N8_B4；S3；F40 | Nscan φ3/A sunlamp s2025 N16/B4 | 否 |
| 316 | Nscan_phi3_A_sunlamp_s2025_N32_B4 | Nscan_phi3_A_sunlamp_s2025_N16_B4；S3；F40 | Nscan φ3/A sunlamp s2025 N32/B4 | 否 |
| 317 | Nscan_phi3_A_sunlamp_s2025_N64_B4 | Nscan_phi3_A_sunlamp_s2025_N32_B4；S3；F40 | Nscan φ3/A sunlamp s2025 N64/B4 | 否 |
| 318 | Nscan_phi3_A_sunlamp_s2025_N128_B4 | Nscan_phi3_A_sunlamp_s2025_N64_B4；S3；F40 | Nscan φ3/A sunlamp s2025 N128/B4 | 否 |
| 319 | Nscan_phi3_A_sunlamp_s2025_N256_B4 | Nscan_phi3_A_sunlamp_s2025_N128_B4；S3；F40 | Nscan φ3/A sunlamp s2025 N256/B4 | 否 |
| 320 | Nscan_phi3_A_sunlamp_s2025_N512_B4 | Nscan_phi3_A_sunlamp_s2025_N256_B4；S3；F40 | Nscan φ3/A sunlamp s2025 N512/B4 | 否 |
| 321 | Nscan_phi3_A_sunlamp_s2025_N1024_B4 | Nscan_phi3_A_sunlamp_s2025_N512_B4；S3；F40 | Nscan φ3/A sunlamp s2025 N1024/B4 | 是 → fixed_phi3_A_sunlamp_s2025_N1024_B4 |
| 322 | Nscan_phi3_A_sunlamp_s2025_N2048_B4 | Nscan_phi3_A_sunlamp_s2025_N1024_B4；S3；F40 | Nscan φ3/A sunlamp s2025 N2048/B4 | 否 |
| 323 | Nscan_phi3_A_sunlamp_s2025_N4096_B4 | Nscan_phi3_A_sunlamp_s2025_N2048_B4；S3；F40 | Nscan φ3/A sunlamp s2025 N4096/B4 | 否 |
| 324 | Nscan_phi3_A_sunlamp_s2025_N8192_B4 | Nscan_phi3_A_sunlamp_s2025_N4096_B4；S3；F40 | Nscan φ3/A sunlamp s2025 N8192/B4 | 否 |
| 325 | Nscan_phi6_A_lightbox_s2021_N4_B4 | Nscan_phi3_A_sunlamp_s2025_N8192_B4；S6；F40 | Nscan φ6/A lightbox s2021 N4/B4 | 否 |
| 326 | Nscan_phi6_A_lightbox_s2021_N8_B4 | Nscan_phi6_A_lightbox_s2021_N4_B4；S6；F40 | Nscan φ6/A lightbox s2021 N8/B4 | 否 |
| 327 | Nscan_phi6_A_lightbox_s2021_N16_B4 | Nscan_phi6_A_lightbox_s2021_N8_B4；S6；F40 | Nscan φ6/A lightbox s2021 N16/B4 | 否 |
| 328 | Nscan_phi6_A_lightbox_s2021_N32_B4 | Nscan_phi6_A_lightbox_s2021_N16_B4；S6；F40 | Nscan φ6/A lightbox s2021 N32/B4 | 否 |
| 329 | Nscan_phi6_A_lightbox_s2021_N64_B4 | Nscan_phi6_A_lightbox_s2021_N32_B4；S6；F40 | Nscan φ6/A lightbox s2021 N64/B4 | 否 |
| 330 | Nscan_phi6_A_lightbox_s2021_N128_B4 | Nscan_phi6_A_lightbox_s2021_N64_B4；S6；F40 | Nscan φ6/A lightbox s2021 N128/B4 | 否 |
| 331 | Nscan_phi6_A_lightbox_s2021_N256_B4 | Nscan_phi6_A_lightbox_s2021_N128_B4；S6；F40 | Nscan φ6/A lightbox s2021 N256/B4 | 否 |
| 332 | Nscan_phi6_A_lightbox_s2021_N512_B4 | Nscan_phi6_A_lightbox_s2021_N256_B4；S6；F40 | Nscan φ6/A lightbox s2021 N512/B4 | 否 |
| 333 | Nscan_phi6_A_lightbox_s2021_N1024_B4 | Nscan_phi6_A_lightbox_s2021_N512_B4；S6；F40 | Nscan φ6/A lightbox s2021 N1024/B4 | 否 |
| 334 | Nscan_phi6_A_lightbox_s2021_N2048_B4 | Nscan_phi6_A_lightbox_s2021_N1024_B4；S6；F40 | Nscan φ6/A lightbox s2021 N2048/B4 | 否 |
| 335 | Nscan_phi6_A_lightbox_s2021_N4096_B4 | Nscan_phi6_A_lightbox_s2021_N2048_B4；S6；F40 | Nscan φ6/A lightbox s2021 N4096/B4 | 否 |
| 336 | Nscan_phi6_A_lightbox_s2021_N8192_B4 | Nscan_phi6_A_lightbox_s2021_N4096_B4；S6；F40 | Nscan φ6/A lightbox s2021 N8192/B4 | 否 |
| 337 | Nscan_phi6_A_lightbox_s2022_N4_B4 | Nscan_phi6_A_lightbox_s2021_N8192_B4；S6；F40 | Nscan φ6/A lightbox s2022 N4/B4 | 否 |
| 338 | Nscan_phi6_A_lightbox_s2022_N8_B4 | Nscan_phi6_A_lightbox_s2022_N4_B4；S6；F40 | Nscan φ6/A lightbox s2022 N8/B4 | 否 |
| 339 | Nscan_phi6_A_lightbox_s2022_N16_B4 | Nscan_phi6_A_lightbox_s2022_N8_B4；S6；F40 | Nscan φ6/A lightbox s2022 N16/B4 | 否 |
| 340 | Nscan_phi6_A_lightbox_s2022_N32_B4 | Nscan_phi6_A_lightbox_s2022_N16_B4；S6；F40 | Nscan φ6/A lightbox s2022 N32/B4 | 否 |
| 341 | Nscan_phi6_A_lightbox_s2022_N64_B4 | Nscan_phi6_A_lightbox_s2022_N32_B4；S6；F40 | Nscan φ6/A lightbox s2022 N64/B4 | 否 |
| 342 | Nscan_phi6_A_lightbox_s2022_N128_B4 | Nscan_phi6_A_lightbox_s2022_N64_B4；S6；F40 | Nscan φ6/A lightbox s2022 N128/B4 | 否 |
| 343 | Nscan_phi6_A_lightbox_s2022_N256_B4 | Nscan_phi6_A_lightbox_s2022_N128_B4；S6；F40 | Nscan φ6/A lightbox s2022 N256/B4 | 否 |
| 344 | Nscan_phi6_A_lightbox_s2022_N512_B4 | Nscan_phi6_A_lightbox_s2022_N256_B4；S6；F40 | Nscan φ6/A lightbox s2022 N512/B4 | 否 |
| 345 | Nscan_phi6_A_lightbox_s2022_N1024_B4 | Nscan_phi6_A_lightbox_s2022_N512_B4；S6；F40 | Nscan φ6/A lightbox s2022 N1024/B4 | 否 |
| 346 | Nscan_phi6_A_lightbox_s2022_N2048_B4 | Nscan_phi6_A_lightbox_s2022_N1024_B4；S6；F40 | Nscan φ6/A lightbox s2022 N2048/B4 | 否 |
| 347 | Nscan_phi6_A_lightbox_s2022_N4096_B4 | Nscan_phi6_A_lightbox_s2022_N2048_B4；S6；F40 | Nscan φ6/A lightbox s2022 N4096/B4 | 否 |
| 348 | Nscan_phi6_A_lightbox_s2022_N8192_B4 | Nscan_phi6_A_lightbox_s2022_N4096_B4；S6；F40 | Nscan φ6/A lightbox s2022 N8192/B4 | 否 |
| 349 | Nscan_phi6_A_lightbox_s2023_N4_B4 | Nscan_phi6_A_lightbox_s2022_N8192_B4；S6；F40 | Nscan φ6/A lightbox s2023 N4/B4 | 否 |
| 350 | Nscan_phi6_A_lightbox_s2023_N8_B4 | Nscan_phi6_A_lightbox_s2023_N4_B4；S6；F40 | Nscan φ6/A lightbox s2023 N8/B4 | 否 |
| 351 | Nscan_phi6_A_lightbox_s2023_N16_B4 | Nscan_phi6_A_lightbox_s2023_N8_B4；S6；F40 | Nscan φ6/A lightbox s2023 N16/B4 | 否 |
| 352 | Nscan_phi6_A_lightbox_s2023_N32_B4 | Nscan_phi6_A_lightbox_s2023_N16_B4；S6；F40 | Nscan φ6/A lightbox s2023 N32/B4 | 否 |
| 353 | Nscan_phi6_A_lightbox_s2023_N64_B4 | Nscan_phi6_A_lightbox_s2023_N32_B4；S6；F40 | Nscan φ6/A lightbox s2023 N64/B4 | 否 |
| 354 | Nscan_phi6_A_lightbox_s2023_N128_B4 | Nscan_phi6_A_lightbox_s2023_N64_B4；S6；F40 | Nscan φ6/A lightbox s2023 N128/B4 | 否 |
| 355 | Nscan_phi6_A_lightbox_s2023_N256_B4 | Nscan_phi6_A_lightbox_s2023_N128_B4；S6；F40 | Nscan φ6/A lightbox s2023 N256/B4 | 否 |
| 356 | Nscan_phi6_A_lightbox_s2023_N512_B4 | Nscan_phi6_A_lightbox_s2023_N256_B4；S6；F40 | Nscan φ6/A lightbox s2023 N512/B4 | 否 |
| 357 | Nscan_phi6_A_lightbox_s2023_N1024_B4 | Nscan_phi6_A_lightbox_s2023_N512_B4；S6；F40 | Nscan φ6/A lightbox s2023 N1024/B4 | 否 |
| 358 | Nscan_phi6_A_lightbox_s2023_N2048_B4 | Nscan_phi6_A_lightbox_s2023_N1024_B4；S6；F40 | Nscan φ6/A lightbox s2023 N2048/B4 | 否 |
| 359 | Nscan_phi6_A_lightbox_s2023_N4096_B4 | Nscan_phi6_A_lightbox_s2023_N2048_B4；S6；F40 | Nscan φ6/A lightbox s2023 N4096/B4 | 否 |
| 360 | Nscan_phi6_A_lightbox_s2023_N8192_B4 | Nscan_phi6_A_lightbox_s2023_N4096_B4；S6；F40 | Nscan φ6/A lightbox s2023 N8192/B4 | 否 |
| 361 | Nscan_phi6_A_lightbox_s2024_N4_B4 | Nscan_phi6_A_lightbox_s2023_N8192_B4；S6；F40 | Nscan φ6/A lightbox s2024 N4/B4 | 否 |
| 362 | Nscan_phi6_A_lightbox_s2024_N8_B4 | Nscan_phi6_A_lightbox_s2024_N4_B4；S6；F40 | Nscan φ6/A lightbox s2024 N8/B4 | 否 |
| 363 | Nscan_phi6_A_lightbox_s2024_N16_B4 | Nscan_phi6_A_lightbox_s2024_N8_B4；S6；F40 | Nscan φ6/A lightbox s2024 N16/B4 | 否 |
| 364 | Nscan_phi6_A_lightbox_s2024_N32_B4 | Nscan_phi6_A_lightbox_s2024_N16_B4；S6；F40 | Nscan φ6/A lightbox s2024 N32/B4 | 否 |
| 365 | Nscan_phi6_A_lightbox_s2024_N64_B4 | Nscan_phi6_A_lightbox_s2024_N32_B4；S6；F40 | Nscan φ6/A lightbox s2024 N64/B4 | 否 |
| 366 | Nscan_phi6_A_lightbox_s2024_N128_B4 | Nscan_phi6_A_lightbox_s2024_N64_B4；S6；F40 | Nscan φ6/A lightbox s2024 N128/B4 | 否 |
| 367 | Nscan_phi6_A_lightbox_s2024_N256_B4 | Nscan_phi6_A_lightbox_s2024_N128_B4；S6；F40 | Nscan φ6/A lightbox s2024 N256/B4 | 否 |
| 368 | Nscan_phi6_A_lightbox_s2024_N512_B4 | Nscan_phi6_A_lightbox_s2024_N256_B4；S6；F40 | Nscan φ6/A lightbox s2024 N512/B4 | 否 |
| 369 | Nscan_phi6_A_lightbox_s2024_N1024_B4 | Nscan_phi6_A_lightbox_s2024_N512_B4；S6；F40 | Nscan φ6/A lightbox s2024 N1024/B4 | 否 |
| 370 | Nscan_phi6_A_lightbox_s2024_N2048_B4 | Nscan_phi6_A_lightbox_s2024_N1024_B4；S6；F40 | Nscan φ6/A lightbox s2024 N2048/B4 | 否 |
| 371 | Nscan_phi6_A_lightbox_s2024_N4096_B4 | Nscan_phi6_A_lightbox_s2024_N2048_B4；S6；F40 | Nscan φ6/A lightbox s2024 N4096/B4 | 否 |
| 372 | Nscan_phi6_A_lightbox_s2024_N8192_B4 | Nscan_phi6_A_lightbox_s2024_N4096_B4；S6；F40 | Nscan φ6/A lightbox s2024 N8192/B4 | 否 |
| 373 | Nscan_phi6_A_lightbox_s2025_N4_B4 | Nscan_phi6_A_lightbox_s2024_N8192_B4；S6；F40 | Nscan φ6/A lightbox s2025 N4/B4 | 否 |
| 374 | Nscan_phi6_A_lightbox_s2025_N8_B4 | Nscan_phi6_A_lightbox_s2025_N4_B4；S6；F40 | Nscan φ6/A lightbox s2025 N8/B4 | 否 |
| 375 | Nscan_phi6_A_lightbox_s2025_N16_B4 | Nscan_phi6_A_lightbox_s2025_N8_B4；S6；F40 | Nscan φ6/A lightbox s2025 N16/B4 | 否 |
| 376 | Nscan_phi6_A_lightbox_s2025_N32_B4 | Nscan_phi6_A_lightbox_s2025_N16_B4；S6；F40 | Nscan φ6/A lightbox s2025 N32/B4 | 否 |
| 377 | Nscan_phi6_A_lightbox_s2025_N64_B4 | Nscan_phi6_A_lightbox_s2025_N32_B4；S6；F40 | Nscan φ6/A lightbox s2025 N64/B4 | 否 |
| 378 | Nscan_phi6_A_lightbox_s2025_N128_B4 | Nscan_phi6_A_lightbox_s2025_N64_B4；S6；F40 | Nscan φ6/A lightbox s2025 N128/B4 | 否 |
| 379 | Nscan_phi6_A_lightbox_s2025_N256_B4 | Nscan_phi6_A_lightbox_s2025_N128_B4；S6；F40 | Nscan φ6/A lightbox s2025 N256/B4 | 否 |
| 380 | Nscan_phi6_A_lightbox_s2025_N512_B4 | Nscan_phi6_A_lightbox_s2025_N256_B4；S6；F40 | Nscan φ6/A lightbox s2025 N512/B4 | 否 |
| 381 | Nscan_phi6_A_lightbox_s2025_N1024_B4 | Nscan_phi6_A_lightbox_s2025_N512_B4；S6；F40 | Nscan φ6/A lightbox s2025 N1024/B4 | 否 |
| 382 | Nscan_phi6_A_lightbox_s2025_N2048_B4 | Nscan_phi6_A_lightbox_s2025_N1024_B4；S6；F40 | Nscan φ6/A lightbox s2025 N2048/B4 | 否 |
| 383 | Nscan_phi6_A_lightbox_s2025_N4096_B4 | Nscan_phi6_A_lightbox_s2025_N2048_B4；S6；F40 | Nscan φ6/A lightbox s2025 N4096/B4 | 否 |
| 384 | Nscan_phi6_A_lightbox_s2025_N8192_B4 | Nscan_phi6_A_lightbox_s2025_N4096_B4；S6；F40 | Nscan φ6/A lightbox s2025 N8192/B4 | 否 |
| 385 | Nscan_phi6_A_sunlamp_s2021_N4_B4 | Nscan_phi6_A_lightbox_s2025_N8192_B4；S6；F40 | Nscan φ6/A sunlamp s2021 N4/B4 | 否 |
| 386 | Nscan_phi6_A_sunlamp_s2021_N8_B4 | Nscan_phi6_A_sunlamp_s2021_N4_B4；S6；F40 | Nscan φ6/A sunlamp s2021 N8/B4 | 否 |
| 387 | Nscan_phi6_A_sunlamp_s2021_N16_B4 | Nscan_phi6_A_sunlamp_s2021_N8_B4；S6；F40 | Nscan φ6/A sunlamp s2021 N16/B4 | 否 |
| 388 | Nscan_phi6_A_sunlamp_s2021_N32_B4 | Nscan_phi6_A_sunlamp_s2021_N16_B4；S6；F40 | Nscan φ6/A sunlamp s2021 N32/B4 | 否 |
| 389 | Nscan_phi6_A_sunlamp_s2021_N64_B4 | Nscan_phi6_A_sunlamp_s2021_N32_B4；S6；F40 | Nscan φ6/A sunlamp s2021 N64/B4 | 否 |
| 390 | Nscan_phi6_A_sunlamp_s2021_N128_B4 | Nscan_phi6_A_sunlamp_s2021_N64_B4；S6；F40 | Nscan φ6/A sunlamp s2021 N128/B4 | 否 |
| 391 | Nscan_phi6_A_sunlamp_s2021_N256_B4 | Nscan_phi6_A_sunlamp_s2021_N128_B4；S6；F40 | Nscan φ6/A sunlamp s2021 N256/B4 | 否 |
| 392 | Nscan_phi6_A_sunlamp_s2021_N512_B4 | Nscan_phi6_A_sunlamp_s2021_N256_B4；S6；F40 | Nscan φ6/A sunlamp s2021 N512/B4 | 否 |
| 393 | Nscan_phi6_A_sunlamp_s2021_N1024_B4 | Nscan_phi6_A_sunlamp_s2021_N512_B4；S6；F40 | Nscan φ6/A sunlamp s2021 N1024/B4 | 否 |
| 394 | Nscan_phi6_A_sunlamp_s2021_N2048_B4 | Nscan_phi6_A_sunlamp_s2021_N1024_B4；S6；F40 | Nscan φ6/A sunlamp s2021 N2048/B4 | 否 |
| 395 | Nscan_phi6_A_sunlamp_s2021_N4096_B4 | Nscan_phi6_A_sunlamp_s2021_N2048_B4；S6；F40 | Nscan φ6/A sunlamp s2021 N4096/B4 | 否 |
| 396 | Nscan_phi6_A_sunlamp_s2021_N8192_B4 | Nscan_phi6_A_sunlamp_s2021_N4096_B4；S6；F40 | Nscan φ6/A sunlamp s2021 N8192/B4 | 否 |
| 397 | Nscan_phi6_A_sunlamp_s2022_N4_B4 | Nscan_phi6_A_sunlamp_s2021_N8192_B4；S6；F40 | Nscan φ6/A sunlamp s2022 N4/B4 | 否 |
| 398 | Nscan_phi6_A_sunlamp_s2022_N8_B4 | Nscan_phi6_A_sunlamp_s2022_N4_B4；S6；F40 | Nscan φ6/A sunlamp s2022 N8/B4 | 否 |
| 399 | Nscan_phi6_A_sunlamp_s2022_N16_B4 | Nscan_phi6_A_sunlamp_s2022_N8_B4；S6；F40 | Nscan φ6/A sunlamp s2022 N16/B4 | 否 |
| 400 | Nscan_phi6_A_sunlamp_s2022_N32_B4 | Nscan_phi6_A_sunlamp_s2022_N16_B4；S6；F40 | Nscan φ6/A sunlamp s2022 N32/B4 | 否 |
| 401 | Nscan_phi6_A_sunlamp_s2022_N64_B4 | Nscan_phi6_A_sunlamp_s2022_N32_B4；S6；F40 | Nscan φ6/A sunlamp s2022 N64/B4 | 否 |
| 402 | Nscan_phi6_A_sunlamp_s2022_N128_B4 | Nscan_phi6_A_sunlamp_s2022_N64_B4；S6；F40 | Nscan φ6/A sunlamp s2022 N128/B4 | 否 |
| 403 | Nscan_phi6_A_sunlamp_s2022_N256_B4 | Nscan_phi6_A_sunlamp_s2022_N128_B4；S6；F40 | Nscan φ6/A sunlamp s2022 N256/B4 | 否 |
| 404 | Nscan_phi6_A_sunlamp_s2022_N512_B4 | Nscan_phi6_A_sunlamp_s2022_N256_B4；S6；F40 | Nscan φ6/A sunlamp s2022 N512/B4 | 否 |
| 405 | Nscan_phi6_A_sunlamp_s2022_N1024_B4 | Nscan_phi6_A_sunlamp_s2022_N512_B4；S6；F40 | Nscan φ6/A sunlamp s2022 N1024/B4 | 否 |
| 406 | Nscan_phi6_A_sunlamp_s2022_N2048_B4 | Nscan_phi6_A_sunlamp_s2022_N1024_B4；S6；F40 | Nscan φ6/A sunlamp s2022 N2048/B4 | 否 |
| 407 | Nscan_phi6_A_sunlamp_s2022_N4096_B4 | Nscan_phi6_A_sunlamp_s2022_N2048_B4；S6；F40 | Nscan φ6/A sunlamp s2022 N4096/B4 | 否 |
| 408 | Nscan_phi6_A_sunlamp_s2022_N8192_B4 | Nscan_phi6_A_sunlamp_s2022_N4096_B4；S6；F40 | Nscan φ6/A sunlamp s2022 N8192/B4 | 否 |
| 409 | Nscan_phi6_A_sunlamp_s2023_N4_B4 | Nscan_phi6_A_sunlamp_s2022_N8192_B4；S6；F40 | Nscan φ6/A sunlamp s2023 N4/B4 | 否 |
| 410 | Nscan_phi6_A_sunlamp_s2023_N8_B4 | Nscan_phi6_A_sunlamp_s2023_N4_B4；S6；F40 | Nscan φ6/A sunlamp s2023 N8/B4 | 否 |
| 411 | Nscan_phi6_A_sunlamp_s2023_N16_B4 | Nscan_phi6_A_sunlamp_s2023_N8_B4；S6；F40 | Nscan φ6/A sunlamp s2023 N16/B4 | 否 |
| 412 | Nscan_phi6_A_sunlamp_s2023_N32_B4 | Nscan_phi6_A_sunlamp_s2023_N16_B4；S6；F40 | Nscan φ6/A sunlamp s2023 N32/B4 | 否 |
| 413 | Nscan_phi6_A_sunlamp_s2023_N64_B4 | Nscan_phi6_A_sunlamp_s2023_N32_B4；S6；F40 | Nscan φ6/A sunlamp s2023 N64/B4 | 否 |
| 414 | Nscan_phi6_A_sunlamp_s2023_N128_B4 | Nscan_phi6_A_sunlamp_s2023_N64_B4；S6；F40 | Nscan φ6/A sunlamp s2023 N128/B4 | 否 |
| 415 | Nscan_phi6_A_sunlamp_s2023_N256_B4 | Nscan_phi6_A_sunlamp_s2023_N128_B4；S6；F40 | Nscan φ6/A sunlamp s2023 N256/B4 | 否 |
| 416 | Nscan_phi6_A_sunlamp_s2023_N512_B4 | Nscan_phi6_A_sunlamp_s2023_N256_B4；S6；F40 | Nscan φ6/A sunlamp s2023 N512/B4 | 否 |
| 417 | Nscan_phi6_A_sunlamp_s2023_N1024_B4 | Nscan_phi6_A_sunlamp_s2023_N512_B4；S6；F40 | Nscan φ6/A sunlamp s2023 N1024/B4 | 否 |
| 418 | Nscan_phi6_A_sunlamp_s2023_N2048_B4 | Nscan_phi6_A_sunlamp_s2023_N1024_B4；S6；F40 | Nscan φ6/A sunlamp s2023 N2048/B4 | 否 |
| 419 | Nscan_phi6_A_sunlamp_s2023_N4096_B4 | Nscan_phi6_A_sunlamp_s2023_N2048_B4；S6；F40 | Nscan φ6/A sunlamp s2023 N4096/B4 | 否 |
| 420 | Nscan_phi6_A_sunlamp_s2023_N8192_B4 | Nscan_phi6_A_sunlamp_s2023_N4096_B4；S6；F40 | Nscan φ6/A sunlamp s2023 N8192/B4 | 否 |
| 421 | Nscan_phi6_A_sunlamp_s2024_N4_B4 | Nscan_phi6_A_sunlamp_s2023_N8192_B4；S6；F40 | Nscan φ6/A sunlamp s2024 N4/B4 | 否 |
| 422 | Nscan_phi6_A_sunlamp_s2024_N8_B4 | Nscan_phi6_A_sunlamp_s2024_N4_B4；S6；F40 | Nscan φ6/A sunlamp s2024 N8/B4 | 否 |
| 423 | Nscan_phi6_A_sunlamp_s2024_N16_B4 | Nscan_phi6_A_sunlamp_s2024_N8_B4；S6；F40 | Nscan φ6/A sunlamp s2024 N16/B4 | 否 |
| 424 | Nscan_phi6_A_sunlamp_s2024_N32_B4 | Nscan_phi6_A_sunlamp_s2024_N16_B4；S6；F40 | Nscan φ6/A sunlamp s2024 N32/B4 | 否 |
| 425 | Nscan_phi6_A_sunlamp_s2024_N64_B4 | Nscan_phi6_A_sunlamp_s2024_N32_B4；S6；F40 | Nscan φ6/A sunlamp s2024 N64/B4 | 否 |
| 426 | Nscan_phi6_A_sunlamp_s2024_N128_B4 | Nscan_phi6_A_sunlamp_s2024_N64_B4；S6；F40 | Nscan φ6/A sunlamp s2024 N128/B4 | 否 |
| 427 | Nscan_phi6_A_sunlamp_s2024_N256_B4 | Nscan_phi6_A_sunlamp_s2024_N128_B4；S6；F40 | Nscan φ6/A sunlamp s2024 N256/B4 | 否 |
| 428 | Nscan_phi6_A_sunlamp_s2024_N512_B4 | Nscan_phi6_A_sunlamp_s2024_N256_B4；S6；F40 | Nscan φ6/A sunlamp s2024 N512/B4 | 否 |
| 429 | Nscan_phi6_A_sunlamp_s2024_N1024_B4 | Nscan_phi6_A_sunlamp_s2024_N512_B4；S6；F40 | Nscan φ6/A sunlamp s2024 N1024/B4 | 否 |
| 430 | Nscan_phi6_A_sunlamp_s2024_N2048_B4 | Nscan_phi6_A_sunlamp_s2024_N1024_B4；S6；F40 | Nscan φ6/A sunlamp s2024 N2048/B4 | 否 |
| 431 | Nscan_phi6_A_sunlamp_s2024_N4096_B4 | Nscan_phi6_A_sunlamp_s2024_N2048_B4；S6；F40 | Nscan φ6/A sunlamp s2024 N4096/B4 | 否 |
| 432 | Nscan_phi6_A_sunlamp_s2024_N8192_B4 | Nscan_phi6_A_sunlamp_s2024_N4096_B4；S6；F40 | Nscan φ6/A sunlamp s2024 N8192/B4 | 否 |
| 433 | Nscan_phi6_A_sunlamp_s2025_N4_B4 | Nscan_phi6_A_sunlamp_s2024_N8192_B4；S6；F40 | Nscan φ6/A sunlamp s2025 N4/B4 | 否 |
| 434 | Nscan_phi6_A_sunlamp_s2025_N8_B4 | Nscan_phi6_A_sunlamp_s2025_N4_B4；S6；F40 | Nscan φ6/A sunlamp s2025 N8/B4 | 否 |
| 435 | Nscan_phi6_A_sunlamp_s2025_N16_B4 | Nscan_phi6_A_sunlamp_s2025_N8_B4；S6；F40 | Nscan φ6/A sunlamp s2025 N16/B4 | 否 |
| 436 | Nscan_phi6_A_sunlamp_s2025_N32_B4 | Nscan_phi6_A_sunlamp_s2025_N16_B4；S6；F40 | Nscan φ6/A sunlamp s2025 N32/B4 | 否 |
| 437 | Nscan_phi6_A_sunlamp_s2025_N64_B4 | Nscan_phi6_A_sunlamp_s2025_N32_B4；S6；F40 | Nscan φ6/A sunlamp s2025 N64/B4 | 否 |
| 438 | Nscan_phi6_A_sunlamp_s2025_N128_B4 | Nscan_phi6_A_sunlamp_s2025_N64_B4；S6；F40 | Nscan φ6/A sunlamp s2025 N128/B4 | 否 |
| 439 | Nscan_phi6_A_sunlamp_s2025_N256_B4 | Nscan_phi6_A_sunlamp_s2025_N128_B4；S6；F40 | Nscan φ6/A sunlamp s2025 N256/B4 | 否 |
| 440 | Nscan_phi6_A_sunlamp_s2025_N512_B4 | Nscan_phi6_A_sunlamp_s2025_N256_B4；S6；F40 | Nscan φ6/A sunlamp s2025 N512/B4 | 否 |
| 441 | Nscan_phi6_A_sunlamp_s2025_N1024_B4 | Nscan_phi6_A_sunlamp_s2025_N512_B4；S6；F40 | Nscan φ6/A sunlamp s2025 N1024/B4 | 否 |
| 442 | Nscan_phi6_A_sunlamp_s2025_N2048_B4 | Nscan_phi6_A_sunlamp_s2025_N1024_B4；S6；F40 | Nscan φ6/A sunlamp s2025 N2048/B4 | 否 |
| 443 | Nscan_phi6_A_sunlamp_s2025_N4096_B4 | Nscan_phi6_A_sunlamp_s2025_N2048_B4；S6；F40 | Nscan φ6/A sunlamp s2025 N4096/B4 | 否 |
| 444 | Nscan_phi6_A_sunlamp_s2025_N8192_B4 | Nscan_phi6_A_sunlamp_s2025_N4096_B4；S6；F40 | Nscan φ6/A sunlamp s2025 N8192/B4 | 否 |

## F40完整展开

1. fixed_phi3_A_lightbox_s2021_N1024_B4
2. fixed_phi3_A_lightbox_s2022_N1024_B4
3. fixed_phi3_A_lightbox_s2023_N1024_B4
4. fixed_phi3_A_lightbox_s2024_N1024_B4
5. fixed_phi3_A_lightbox_s2025_N1024_B4
6. fixed_phi3_A_sunlamp_s2021_N1024_B4
7. fixed_phi3_A_sunlamp_s2022_N1024_B4
8. fixed_phi3_A_sunlamp_s2023_N1024_B4
9. fixed_phi3_A_sunlamp_s2024_N1024_B4
10. fixed_phi3_A_sunlamp_s2025_N1024_B4
11. fixed_phi6_A_lightbox_s2021_N4096_B1
12. fixed_phi6_A_lightbox_s2022_N4096_B1
13. fixed_phi6_A_lightbox_s2023_N4096_B1
14. fixed_phi6_A_lightbox_s2024_N4096_B1
15. fixed_phi6_A_lightbox_s2025_N4096_B1
16. fixed_phi6_A_sunlamp_s2021_N4096_B1
17. fixed_phi6_A_sunlamp_s2022_N4096_B1
18. fixed_phi6_A_sunlamp_s2023_N4096_B1
19. fixed_phi6_A_sunlamp_s2024_N4096_B1
20. fixed_phi6_A_sunlamp_s2025_N4096_B1
21. fixed_phi3_B1_lightbox_s2021_N1024_B4
22. fixed_phi3_B1_lightbox_s2022_N1024_B4
23. fixed_phi3_B1_lightbox_s2023_N1024_B4
24. fixed_phi3_B1_lightbox_s2024_N1024_B4
25. fixed_phi3_B1_lightbox_s2025_N1024_B4
26. fixed_phi3_B1_sunlamp_s2021_N1024_B4
27. fixed_phi3_B1_sunlamp_s2022_N1024_B4
28. fixed_phi3_B1_sunlamp_s2023_N1024_B4
29. fixed_phi3_B1_sunlamp_s2024_N1024_B4
30. fixed_phi3_B1_sunlamp_s2025_N1024_B4
31. fixed_phi3_B2_lightbox_s2021_N1024_B4
32. fixed_phi3_B2_lightbox_s2022_N1024_B4
33. fixed_phi3_B2_lightbox_s2023_N1024_B4
34. fixed_phi3_B2_lightbox_s2024_N1024_B4
35. fixed_phi3_B2_lightbox_s2025_N1024_B4
36. fixed_phi3_B2_sunlamp_s2021_N1024_B4
37. fixed_phi3_B2_sunlamp_s2022_N1024_B4
38. fixed_phi3_B2_sunlamp_s2023_N1024_B4
39. fixed_phi3_B2_sunlamp_s2024_N1024_B4
40. fixed_phi3_B2_sunlamp_s2025_N1024_B4
