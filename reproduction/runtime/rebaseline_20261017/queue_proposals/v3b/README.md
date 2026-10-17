# 完整主流程 v3b（静态，未激活）

本目录重建 v3 提案，更新全部400项ODR任务对 runner.py 的完整SHA绑定，
对应已验收并推送的提交 aa8e26fd717135f86b37b1dc2ce69febf144acf8。
runner SHA为72385d6ca9f9fc4a2e7005a57761e9393a93922921226bbb254402639b9a7c47。
旧 v3 保留不变，现行 queue_v2.json 的9项结构前缀也完全不改。

完整提案为 full_main_v3_proposal.json，444项：
现9项 → 固定40 ODR（含A/B1/B2）→ Table2/3/4 source35项 → 扫描360项。
计数仍是10配置190轮、30 source域评价、4 pilot，以及400 ODR请求
（370独立执行、30别名）。资源测量、最终分析与材料受限核验仍在backlog。

用于Git归档的 proposal.json.gz 是完整JSON的确定性gzip：
mtime=0、无原文件名header、压缩等级9。原始12,433,032字节压缩为175,367字节；
已验证重复压缩字节相同，解压字节及SHA与原始文件完全相同。
proposal_index.json记录原始/gzip/总览摘要和中文说明；
archive_verification.json记录压缩与444行可逆依赖表的验收证据。
task_overview.md有444项ID、前置、用途和alias简表；S3/S6/F40组可完整展开，
每行展开后与真实requires集合一致。

10组原静态检查及6类负向依赖测试全部通过，661项资产SHA吻合。
gzip摘要、解压结果、444行/30 alias总览也在独立CPU屏幕中重新核验。
报告：reports/main_v3b_proposal_cpu_20261017_153210_2f79c3.json，
对应logs保留完整stdout/stderr及退出码0。

本任务没有发布队列、切换管理器、运行GPU、修改实现代码或提交推送Git。
静态通过不等于真实CUDA ODR已验收。主模型统一审阅、归档与后续切换。
