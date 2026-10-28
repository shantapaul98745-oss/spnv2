Current explicit approval2026-10-28: only pending-window BN mean accumulators learner.learning/window_base.bn_mean and corresponding bn_features channel_mean use a uniform fixedatol1e-4, rtol1e-5. No automatic tuning. All other BN moments/true running buffers/model/optimizer/gradients, original input/decision/RNG/internal rollback/GT rules unchanged. Effective authority index.numeric_acceptance/revision_v12. Reuse30accepted windows and finalize saved w84 complete768state/both128panels with zero model replay, retaining all three real failures and charges. This overrides only the two older BN mean values below.

Current explicit approval2026-10-28: only confidence in complete typed pose-prediction records is audited by exact original float32 heat_peaks q25 linear recomputation, then normalized only in comparison copies. Original peak inputs, actual G50 score/threshold/decisions and all other fields/standards remain unchanged. Preserve raw confidence/records and None availability. Effective authority index.numeric_acceptance/revision_v11. Reuse30accepted pairs; resume w84 from complete704checkpoint, keeping both prior real failures and their77+222physicalforwards.

Current explicit approval2026-10-28: only heat_peaks in full typed pose-prediction records uses its11-vector L2 norm with unchanged atol1e-6/rtol1e-5. Preserve shape/dtype/finiteness/availability/order and every actual threshold/decision; no other field or tolerance changes. Effective authority index.numeric_acceptance/revision_v10. Reuse30accepted pairs; w84 has no saved candidate/panel, resume from actual complete704checkpoint and charge necessary unpersisted replay plus the prior77failedforwards.

Current explicit approval2026-10-28: only learner.learning/window_base.bn_mean (pending update-window mean accumulator) uses atol1e-5; corresponding bn_features channel_mean uses atol2e-5; both rtol1e-5. bn_mean_sq/cache mean-square/true running buffers/model/optimizer/gradients and every other numerical/discrete/RNG/internal rollback/GT rule stay unchanged. Runtime authority index.numeric_acceptance/revision_v9. Reuse29 accepted windows, finalize saved w54 complete512state and both128panels with zero model replay; preserve both actual failures and all costs. This overrides older values for these two mean fields only.

Current explicit approval2026-10-28: cross-GPU rotation matrices, quaternions and derived0..180degree angles use fixed0.01degree acceptance. Use raw matrix Frobenius chord and normalized sign-equivalent quaternion chord with float64 arithmetic, no SVD projection; shape/dtype/finiteness and valid rotations remain required. Translation/box, model/optimizer/gradients, BN, exact actual decisions/RNG/same-run rollback and GT metrics remain unchanged. Effective authority index.numeric_acceptance/revision_v8. Reuse29 accepted pairs; resume w54 from complete448 checkpoint, all prior failures and charges retained.

Current explicit approval2026-10-28: only the derived observation G50 gate_score is verified by exact recomputation from its own raw arrival using the unchanged hash-bound gate code/frozen calibration. Comparison ignores only that redundant derived score after self-consistency succeeds on each side. Raw input acceptance, actual decisions/reasons, thresholds, parameter/gradient/BN/RNG and same-run rollback remain unchanged. This supersedes earlier raw gate_score cross-GPU comparison only. Reuse28 accepted windows and all saved w31 panels/checkpoint; charge only necessary unpersisted tail.

Current explicit approval2026-10-28: only typed cross-GPU geometric pose predictions use object norm <=1e-6+1e-5*norm(expected), with matrix Frobenius/vector Euclidean and normalized sign-equivalent quaternion distance. Raw evidence/metrics stay unchanged. Model/optimizer/gradients retain pointwise atol1e-6/rtol1e-5; BN mean atol1e-5, mean-square atol1e-6; all discrete/RNG/same-run rollback/isolation exact. Reuse28 accepted pairs and saved w31 panels/checkpoint; recompute only unpersisted tail. This overrides older geometric pointwise statements below. Guard is always hourly, never15minutes.

Current explicit approval2026-10-28: only cross-GPU BN temporary-cache channel-mean atol is1e-5 (rtol1e-5). Mean-square and all other numeric tolerances remain1e-6/1e-5; all discrete/RNG/same-run restoration requirements remain unchanged. Effective authority is index.numeric_acceptance. Reuse saved panels/checkpoints, no replay.

Current additional G50-P approval2026-10-27: only cross-GPU BN temporary feature caches compare per-channel mean/mean-square under unchanged atol1e-6/rtol1e-5, exact key/shape/dtype/finiteness. Same-run complete cache rollback remains bitwise exact; all other numerical/discrete/RNG/model/GT/forward limits unchanged. Reuse saved panels/checkpoints without new model evaluation.

Current G50-P scientific acceptance amendment (user2026-10-27): cross-GPU floats use fixed abs1e-6+rel1e-5*abs(expected), finite values only; discrete decisions/IDs/order/RNG and same-run full rollback/branch isolation remain exact. Effective acceptance is original frozen protocol plus index.numeric_acceptance. Preserve prior27 strict acceptances and all failure costs.

## Current G50-P shared-GPU resource amendment (user2026-10-27)

For this existing diagnostic only, other GPU jobs and nonzero utilization do not block admission. Use the bound shared_gpu_20261027 resource policy: two fresh snapshots must each show at least6144MiB free GPU memory and40960MiB host MemAvailable. The estimates are4GiB+2GiB GPU safety and32GiB+8GiB host safety, based on existing records/configuration, never a model probe. Retain four project locks, exact process registration, single budget writer,300-second resource waits, hourly chat guard,40,000 physical forwards and unchanged strict science/GT/frozen scope. Do not signal others or restart healthy scientific workers. This supersedes the old blanket-foreign-compute wait rule only for this diagnostic.

# Project instructions

## 持续有效：存储保留与磁盘治理（2026-10-27用户授权）

- 所有新实验启动前必须有存储保留策略和峰值磁盘预估，列出恢复点、最终成果、冻结/验收引用和必要失败现场；无策略不得开始保存。
- 普通完整恢复检查点默认滚动保留最近两个校验有效版本；冻结诊断、验收、迁移明确引用的版本先登记保护，不能因较旧而删。使用 `reproduction/storage_management_20261027/durable_state.py` 与同目录 `storage_policy.py` 的保存后保留机制；原冻结实现保持原字节，迁移/运行接入必须另做版本绑定和受影响路径验证，不能热改健康worker。
- 新检查点完整落盘和校验后，持有写锁清理超出策略、无引用的旧版本；先写路径、大小、用途、已有哈希及删除依据，保留删除回执。当前最新两个恢复点、原始权重、最终成果、不可替代失败现场及协议/版本/日志/账本/退出/proof不得破坏。
- 优先复用不可变状态；硬链接或去重只用于后续绝不原地写入的文件，写入必须产生新文件并原子替换，不污染分支或历史。不得为节省空间重算已成功持久化的必要成果。
- 阶段验收后及时清理无必要中间产物。失败保留解释和复现所需的最小现场及真实费用，不永久保留全部完整副本；依赖不明单列保留，不阻塞其他安全清理。
- 临时文件、缓存使用本项目专用系统临时/用户缓存目录。正常退出清理；异常遗留按身份与引用核对，不碰他人文件。项目外层不堆安装临时目录。
- 每次保存写入磁盘用量；预计新状态使峰值超过计划时停止该写入并记录，先定位增长和清理，不自动扩容或改变科学预算。阶段交接和守护巡检报告明显增长。
- 这些限制仅管理存储，不改变数学/科学协议、严格确定性、GT/数据边界、统计口径、计算预算或历史验收。参考模板 `experiments/EXPERIMENT_PLAN_TEMPLATE.md`。

Follow the standing efficient-execution policy at the top of AGENT.md. The user-confirmed 2026-10-25 policy authorizes targeted diagnosis, necessary fixes, minimal reproduction/validation and safe resumption within the current scientific/data/budget boundaries. A scientific stop blocks invalid computation and downstream dispatch, not technical diagnosis. Preserve user changes and the existing screen, ownership, resource, version and acceptance requirements.

Research background is fixed in CURRENT_RESEARCH_SUMMARY.md in this project root. Read it only when research context or a changed scientific judgement requires it; follow its references selectively. Do not routinely load the complete innovation catalogue, old reports, rule history or models.

For the current G50-P diagnostic, authoritative live state is ../spnv2_runs/plans/g50p_risk_diagnostic_20261025/status.json and monitoring/guard_cursor.json in that plan. Follow the cursor to the current controller and only new outcome/wrapper-exit/proof/budget/checkpoint metadata. Reuse unchanged accepted evidence; do not repeat handoff checks or reopen closed queues. This entry supersedes historical instructions to read the entire AGENT.md and the old global progress report on every check.

Keep the unique spnv2-g50-p heartbeat hourly during prolonged resource waits or healthy long tasks; restore15-minute checks near stage transitions or required acceptance, then return to hourly. The server's300-second resource checks and automatic resumption continue. Do not change the unrelated spnv2 literature heartbeat or create/resume goal mode.

The direct2026-10-26 user amendment removes only this diagnostic cumulative14400s active-time stop. Read current index.protocol and index.budget_policy for effective limits; preserve old protocol/identity/ledger/proofs.40000 physical forwards and all science/data/ownership/resource/acceptance boundaries remain. Track measured time separately from historical conservative charges.
