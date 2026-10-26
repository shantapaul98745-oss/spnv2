# 存储管理修订 2026-10-27

此目录是后续实验使用的保存程序版本，原odr003冻结writer和科学producer不改。
`durable_state.py`基于冻结writer，只添加：构造时必需存储策略；写入前峰值预算检查；原子发布后校验并滚动清理。`storage_policy.py`实现保护引用、磁盘用量和删除审计。模型状态内容、数学、RNG、日志计费与恢复验证未放宽。
五项零模型测试见 targeted_validation.json。历史清理证据位于 spnv2_runs/storage_audits/retention_20261027；旧协议/proof/预算不改写。
迁移中实验尚未接入这个writer；目标派发前必须完成明确存储/路径修订、依赖pins及针对性验收，不能因仓库提交就声称已部署。
