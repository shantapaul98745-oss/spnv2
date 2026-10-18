# φ3临近完成的接续候选（尚未启用）

本目录只准备下一段49项：原9项保序、仅6项未启动source评价加同次推理observer、原计划固定ODR40。完整190轮及后续扫描/资源/证据仍需按阶段接续。

proposal.json.gz为无时间戳的完整候选压缩副本；解压SHA见preflight.json。运行资产仍在服务器原路径，runtime_map.json给出逐文件SHA。prepare_proposal.py只写候选，不修改生产status/queue，不发送信号或启动任何实验。

CPU预检通过，退出0；首次兼容错误的源码和日志服务器留存，不覆盖。已通过的observer/entrypoint CPU证据仅说明工程准备，不是正式科学结果。等待用户允许一次仅manager的受控重启；训练保持原PID/starttime/screen。获允许后必须重核候选SHA、六项仍未启动和原worker身份，使用--revise-pending-from及六个明确--pending-task，不能复用历史--extend-from。

原接受子集成绩保持，新增H数值/混合指标单列分母，缺失不补零、不用E替代H，只有实际N/N覆盖才称all-samples。源final固定、原GPU串行门槛与独立验收均保留。


## 已启用记录

用户随后明确允许一次manager-only切换；14:38:42已启用相同SHA候选。生产绑定为PLAN/queue_observed_stage_20261018.json，训练未重启。证明见accepted_activation.json（原路径在runtime_map.json）。prepare与activate均为一次性脚本，不得重复执行；故障恢复只带新manifest，不重用修订参数。
