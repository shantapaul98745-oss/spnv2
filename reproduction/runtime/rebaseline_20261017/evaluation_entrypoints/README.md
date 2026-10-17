# 同次评价扩展入口

只新增evaluation_entrypoints；不修改冻结source、tasks、ODR、pipeline、analysis、科研图、observer或bench。source/ODR原接受子集和MAT不变。所有CPU验收走唯一guarded screen，CUDA_VISIBLE_DEVICES为空。实际未来GPU评价仍由原parent gate和原serial lock控制，本入口不绕过、不杀进程、不重启worker。

## CLI

- capture_source.py --base-entry PLAN/tasks/source_artifacts.py evaluate 原全部参数：原source入口照常运行，仅在实际engine.inference.do_valid上下文包原solver，原forward一次/样本。结束后独立build_extension。旧输出/已有capture不覆盖。
- capture_source.py --base-entry PLAN/tasks/source_artifacts.py verify_eval 原全部参数：先原CPU验收，再扩展独立CPU。accepted source report仅加稳定extension指针，其余字段相同；再次原verifier重写base后确定性恢复同一指针，最终字节一致；重复extra验收失败会恢复原有accepted报告字节。
- run_odr.py --task ID [--execute]：加载原run_task module，仅覆写该module.job为deepcopy任务的实际child wrapper命令，common.job不动；保留base_command计划，原process record记录实际命令。固定40额外扩展proof齐全之前拒绝非fixed，原base barrier也照常执行。
- capture_odr.py --task ID：仅供上述parent调用。对应line实际inference同次observer，运行原audited_entry，audit.task仍原manifest task；适应权重先保存的原合同不变。
- verify_odr.py --task ID：先原完整独立CPU verifier，再extra CPU；已存在base report字节必须幂等。alias无forward/capture，仅重验canonical extension，确定性保存canonical proof SHA。

## 稳定旁证接口

source accepted report顶层extension，以及ODR evaluation_entrypoints/reports/ID.json顶层extension统一为schema=spnv2-evaluation-extension-link-v1，含report={path,sha256}、verification={path,sha256}、context_sha256、applicability、observer_status。source另带entrypoint_verification={path,sha256}；ODR proof自身无需自指。alias顶层canonical_task、canonical_proof={path,sha256}，extension与canonical完全相同，无自己的capture。

source wrapper proof schema为spnv2-source-evaluation-extension-entrypoint-verification-v1；ODR为spnv2-odr-evaluation-extension-entrypoint-verification-v1；status均passed才可接受。E-only为not_applicable_no_heatmap，observer_status=not_applicable、零H记录，H指标保持null。

context绑定原base entry、实际配置、source final/ODR适应前已保存权重、source验收、冻结source/line manifest、observer实现与本入口协议SHA。所有新report/proof同内容幂等，不同内容拒绝。包装函数恢复do_valid/solver，隔离并恢复engine/config/common等通用模块与sys.path/sys.argv。

## CPU证据范围

CPU tests使用明确MOCK目录，实际调用冻结inference、原source verify_eval与observer独立CPU数学；source GPU模型构建和ODR适应/base-state-verifier进程边界替换为mock。原run_task.py代码不改，用CPU进程桩检查child命令、进程记录、重复verify与alias/fixed40；不能称真实训练/ODR/科学结果。实际GPU回路需未来已接受source后经父队列执行。

observer已绑定修复dtype/GT norm阈值边界的release-v2（19项CPU验收）；入口以六组完整CPU整合验收绑定当前protocol。API字段不变，旧18-pass历史保留在observer自身history。候选队列仍由root审阅接入，不在本目录激活。Git由root统一归档。
