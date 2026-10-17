# 最终证据汇总：第一阶段已核验，完整交付工具仍在实现

本阶段包括requirements、evidence、run_extended、verify_extended与CPU合同测试。
requirements核对10配置190轮、30域评价、400个ODR请求及别名、依赖顺序与五种观测流。
扩展汇总保留原接受子集，独立核对逐图指标、真实CSV的ID顺序、分母和单位，生成370组同ID前后配对及74组五观测流均值/样本标准差。
不存在的预测保持None；五个实验已经完成但有效预测为零时，单列数值限制，不伪造零误差、不将其当作尚未运行。

验证：15组范围合同、12组汇总合同通过；另对已冻结入口的5种source模式及42个ODR MOCK记录做独立接口核验。
所有模拟只说明CPU工程兼容，不能作为GPU训练或科学结果。真实当前快照0/430新评价被验收，因此汇总保持pending。

完整final_reporting/preparation_manifest.json尚未发布READY：科研图补充入口、最终审计仍待完成和主模型验收。stage_preparation_manifest.json只是此阶段的代码与验证记录，不授权队列切换。
当前训练、原损失、优化器和AMP协议均未修改。CLI在独立screen且隐藏CUDA中运行：

    python -B -s run_extended.py --snapshot PLAN/analysis/runs/NAME --run-id UNIQUE
    python -B -s verify_extended.py --run-id UNIQUE

--allow-incomplete仅用于工程验收；正式命令必须省略，并要求430条真实证据齐全。
