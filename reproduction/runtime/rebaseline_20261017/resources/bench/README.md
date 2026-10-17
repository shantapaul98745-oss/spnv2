# Table6资源测量准备（冻结代码，GPU未执行）

仅准备四个已有EHS架构的统一forward workload，未启动GPU、未改活跃队列或训练。
四模型顺序为 table4_phi0_bn、phi3_bn_ehs、table4_phi3_gn、phi6_gn_ehs。
必须先有各自独立source final及三域验收，共16项父队列accepted任务；
还必须有40项固定ODR真实验收和adaptation_audit，不能以历史模型或新ODR替代。

## 入口和调度

run_benchmark.py --report <new report> --artifacts <new directory>：
只能在守护screen、固定项目TORCH_HOME、两CPU线程、seed2021和CUBLAS设置下运行。
它先检查所有接受证明，再调用父级真实gpu_available及同一资源spec；
在唯一串行GPU任务创建CUDA上下文前检查一次。四模型之间不会把自己的
CUDA上下文错误当作外部任务放行或新增PID豁免。独占输出和本地锁防止重复执行。

pending_task.json是非execution schema，activation_allowed=false。
建议插在eval_table4_phi3_gn_sunlamp之后、首个Bscan之前；主模型统一审阅接入。
该定义显式依赖16项source/域接受任务和40项真实fixed任务，verify_gpu=false。
本任务不发布此定义，也不提交推送Git。

## 精确测量口径

768×512、物理batch1、固定CUDA驻留零张量、FP32、eval/no_grad、禁用AMP和TF32。
选择现有heatmap/efficientpose/segmentation全部三头，包含整个backbone和所有EHS头；
正式config和权重文件不修改。仅在构建架构后strict load已接受final，
逐项比较真实tensor与参数inventory，不测新随机初始化权重。

每模型固定10次warmup和30次实测；每次使用CUDA events并synchronize。
原始latencies_ms全部保留。统计为均值、中位数、p05/p90/p95、极值、总体标准差，
吞吐=1000/mean_ms（images/s）。分位数是(n-1)*q线性插值。
GPU allocated/reserved baseline和peak以bytes记录；peak在有限性/完整性检查前捕获。
模型/input为GPU驻留baseline，计入forward工作量峰值；不是整卡占用量。

不包含数据加载、预处理、H2D传输、PnP/姿态后处理，不称端到端速度或论文精度。
只测整网forward；encoder/整网/ODR更新参数数目沿用已验真resources参数证据，
并重新核对实际loaded模型inventory。没有新增phi0或phi3GN ODR实验。

odr_resources.py只读取已accepted固定40的真实adaptation_audit。
其adaptation_seconds包含实际audited_entry计时范围内的audit/save等开销，
不能与pure-forward时延直接混合；原始source/implementation/audit/acceptance SHA、
N/B、次数、峰值内存及ODR参数分母均保留，并与当前接受source SHA/静态scope交叉核对。

## 稳定未来API（当前尚不存在GPU结果）

总报告：
reports/table6_forward_four_models_20261017.json
schema=spnv2-table6-forward-benchmark-v1，measurements给出每模型{name,path,sha256}。
sources按name提供cfg/final/接受证明/参数inventory和哈希；
device/runtime位于总报告。原始模型报告在
artifacts/table6_forward_four_models_20261017/{name}.json。
每模型含30个latencies_ms、statistics、memory_bytes、parameter_counts、
source_final_sha256、strict_reload、model_digest前后、输入/模式/完整三头协议。

verify_benchmark.py --report <GPU report> --output <new CPU acceptance report>
须在独立CUDA_VISIBLE_DEVICES=''屏幕运行；检查所有文件/SHA/接受任务、
独立NumPy统计、CPU strict reload源模型、参数分母、有限值和ODR实际资源证据。
稳定验收路径为reports/table6_forward_four_models_20261017_verified.json，
schema=spnv2-table6-benchmark-cpu-verification-v1。
顶层report_sha256绑定总报告，models[].raw_measurement_sha256绑定各原始模型JSON，
models[].source_final_sha256绑定final。验收没有summary字段，统计位于models列表。

分析程序仅可在两报告status=passed、原报告exitcode及sidecar均为0、
总报告/各raw/final/proof等哈希全部吻合且统一条件成立后绘图。
缺少任何实际结果时保持pending，不用CPUfixture或静态参数替代GPU数据。

## CPU验收

11/11小型CPU合同测试通过，包括独立统计参考、坏计数/非有限值、错误hash、
参数分母/shape、ODR源模型与scope交叉绑定和非执行pending schema。
报告：reports/table6_bench_cpu_20261017_155647_599988.json。
screen3427543.table6_bench_cpu_20261017_155647_599988，原始日志exitcode0。
早期报告目录缺失导致的失败日志与旧pending定义已保留在logs/history，
没有把失败轮次说成通过。全部CPU测试使用CUDA_VISIBLE_DEVICES=''，
无GPU query/forward，也不证明真实硬件测量已经完成。
