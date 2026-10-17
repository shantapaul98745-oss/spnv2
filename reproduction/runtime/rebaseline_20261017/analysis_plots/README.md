# 科研图与论文差距准备

只拥有analysis_plots，不改analysis v1、source、ODR或pipeline；无训练、推理或GPU调用。所有CLI以唯一guarded CPU screen运行，CUDA_VISIBLE_DEVICES为空，OMP/OPENBLAS线程数2，Matplotlib Agg。

## CLI

- run_plots.py --snapshot PLAN/analysis/runs/已独立验收ID --run-id 唯一ID：核实快照独立验收与各文件SHA、当前源final/3MAT/CSV；生成完整扫描点清单、固定seed2021分布、Table6参数图、若已独立验收则GPU纯forward图、论文对照缺口与pending父任务。
- verify_plots.py --run-id ID：重新核对原CSV/MAT/final SHA，实际MAT均值与分母、独立统计每点5seed均值/样本标准差、预期点数、直方图/失败转移、PNG及论文PDF/提取页证据。
- cpu_tests.py --run-id 唯一ID：明确MOCK的有限图表布局/统计/错证据回归，不产生正式GPU或扫描测值。

## 真实图边界

实际ODR清单为Bscan phi3 A/B2×两域×6点，Nscan phi3/phi6 A×两域×12点；不会虚增Nscan B2。七指标heat_pose、effi_pose、final_pose、H/E各自eR/eT共56条曲线、504个点。每点必须2021—2025五观测流已接受且该指标非空，同一固定source SHA；图的误差条为五观测流均值的样本标准差ddof1，不代表五次source训练。整条曲线不齐只出pending，不画假的完整曲线。alias通过已验收快照映射canonical，不能增加独立实验数。

固定协议seed2021仅同ID双方final有效交集画delta直方图；所有ID分别计both valid/valid→reject/reject→valid/both reject。零有效不生成假的直方图。H/mixed均值属于现有H接受子集；图注明与论文Table4起all-samples口径不同。扩展observer字段独立预留，原MAT丢弃的姿态不恢复、不补0，不以E替代H；只有raw-numeric N/N覆盖才可能另标all-samples。

Table6四模型参数图来自resources/reports/table6_success_20261017四模型独立验收，whole SPNv2、EfficientDet backbone、EfficientNet与各head严格区分。GPU图只接受queue_revisions的spnv2-table6-forward-benchmark-v1及CPU独立验收schema和0退出，核实30个原始正有限时延、source/配置/三域证据/逐模型SHA。单位ms、images/s、MiB；全EHS纯forward，输入已驻留GPU，排除加载/预处理/H2D/PnP，不能称端到端或ODR更新时延。没有文件就pending，不从参数推算速度/显存。

## 论文证据

自有paper/Park2024SPNv2.pdf，原PDF SHA见reference/paper_reference.json。仅保存相关页PNG、表格数值和有限摘录，不复制全文文字。Table1页5，Table2/3页8，Table4页9，固定ODR Table5页10，Table6页13均做页面视觉核对；未知分母、未知Table5跨seed汇总与曲线逐点数值保持unknown/null。Table1表/图的GlassBlur与附近正文Gaussian blur差异保留。论文Table4起H旋转+E平移all-samples；冻结导出已丢弃被拒绝H姿态，当前不可宣称同分母复现。Table6仅encoder，论文CPU速度环境不同，不与未来GPU直接算加速比。

根模型统一Git提交；cache/仅Matplotlib缓存。现阶段真实扫描和分布待source/ODR完成；真实Table6只有参数图可生成。
