# SPNv2 项目工作约定

## 持续有效：精简检查与自主推进（用户确认于2026-10-25）

这些是跨对话的执行要求，不因换聊天或交接而失效。它们优先于旧项目记录中已经被后续授权取代的重复审批和全盘复查要求；不扩大当前实验范围。最新用户指令和当前有效实验协议仍优先。

- **授权内默认执行。** 已授权的技术定位、必要修复、针对性验证和安全续接由执行者负责，不逐项请示，不等待用户反复说“继续”。只有需要改变科学协议、数据边界、预算或确实超出既有授权时，才请求用户决定。
- **检查必须服务于决策。** 每项新增检查都须对应具体风险及会受结果影响的下一步决定；已有协议明确要求的准入和验收属于必要检查。不得仅以“保险起见”为由增加检查层级，也无需逐项向用户播报检查理由。
- **证据复用，增量核验。** 已验收且相关代码、输入和环境绑定未变的证据直接复用；新对话只做一次有界交接核验，之后按游标核新增结果。不能因换聊天或阶段切换重读全部模型、重跑已完成实验或重新审计全实现。
- **最小必要验证。** 启动时执行既定资源、身份和预算准入；结束时核新增退出、结果及协议要求的独立验收和严格一致性。额外检查只由实际变化、失败或未解决的具体疑点触发；通过后立即推进，不继续堆叠无关模拟测试。
- **停止异常计算，不停止技术排障。** 出现轨迹不一致、NaN/Inf或数据越界时，停止受影响计算及下游派发并保留证据；同时自主定位原因。先利用已有证据；若具体差异未保存，可在原授权、资源门和剩余预算内做限定范围的最小复现，保存首个差异字段。不得盲目重试、近似放行或重算已成功持久化的计算。修复只验证受影响路径，严格通过后安全续接；需要改变研究条件才询问用户。
- **不把未知原因当作被动等待理由。** 有可执行的排障步骤就继续；若确实只能等待资源或外部信息，明确唯一阻塞项并复用已有调查结果，不反复进行同一无变化检查。
- **保留真实边界。** 科学规则、确定性要求、GT边界、冻结名单、统计分母、预算和失败成本不因精简流程而放宽。健康worker不热修改、不重启；不干扰他人计算，不用模型试跑探测资源。
- **安静执行。** 服务器独立执行长任务，聊天仅做必要的单次巡检；只报告实质进展、阶段完成、明确异常、预算停止或需要用户决定的事项。目标模式及守护归属按最新授权，不因本规则自动创建、恢复或改绑。

## 阅读入口与范围

- 每次接手先读本文件及 `../paper/SPNv2_复现实时进度.md`，随后对照实际配置、源码快照、状态JSON、权重和日志核验。文档不是实时事实的替代品。
- 代码：`/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2`；资产：同级 `spnv2_runs`；文档：同级 `paper`；数据：`/home/DATA/Downloads/DataSet/speedplus`。
- 环境：`/home/ps/anaconda3/envs/SPNv2/bin/python`。运行设置 `PYTHONNOUSERSITE=1 PYTHONUTF8=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg`，独立配置缓存。
- 只使用本项目的代码、资产和记录。保留用户修改及相关论文；不复制、修改或操作其他项目、环境和进程。用户已于2026-10-17明确授权每项代码修改验证后提交并推送到本项目origin/main；由主模型统一执行，遵循下方版本记录规范。

## 已确认的独立补齐目标

- 保留历史 φ3-BN、φ6-GN 的20轮模型和ODR/B/N结果；已有batch4直接更新基线不重训。
- 新 φ3 使用物理batch4、累积4；新 φ6 使用物理batch1、累积4。φ3前向BN统计仍只看4张，不称四卡SyncBN等价复现。
- 独立实现累积：保持局部loss归一；完整窗口后统一反缩放、裁剪、step和scale更新。尾批在前向前排除，跨epoch不累积。
- 新source训练种子2021；768×512；AdamW LR1e-3、WD1e-4、AMP、clip1.0、15/18轮衰减；H-only10轮，其余20轮。不因累积线性放大学习率。
- 训练顺序：φ3-EHS20 → φ6-EHS20 → 固定ODR → Table2 H10/E20/EH20 → Table3基础/加Erase/加Flare各20 → Table4 φ0-BN/φ3-GN各20 → 扫描、分析及剩余材料。合计10个source配置190轮。
- 每个模型独立初始化。主结果固定final，不按目标域成绩选模；ImageNet初始化、GN初始化和本地style差异必须披露。
- 新source各做synthetic11994、lightbox6740、sunlamp2791全域评价。hE/hH/混合、单位、拒绝数、有效分母和稳定样本ID分清；缺失不填零。
- 固定ODR：φ3 N1024/B4，φ6 N4096/B1，两域2021—2025五种子；φ3 A/B1/B2分线。五种子是ODR观测流随机性，不是五次source训练。
- B扫描120项：φ3 A/B2×两域×五种子×六个B；N扫描240项：两架构均B4×两域×五种子×12个N。仅相同source哈希和协议可复用，旧source结果不混入。
- H/E/EH无分割头，不安排熵最小化ODR。PRISMA25仅在图像、标签、相机核实后做离线hH评价；缺失明确标受限。创新训练后置。

## 实验执行与验收

- 所有实验（含CPU验收、GPU试跑、训练、推理、ODR、扫描）默认在独立screen中运行；简短只读检查可直接执行。
- 每项使用唯一会话、run与日志，记录完整stdout/stderr、退出码、命令、配置、哈希及恢复信息。使用本项目 `spnv2_runs/scripts/run_in_screen_guarded.sh`，不复活旧会话或覆盖历史日志。
- GPU串行；启动前重查资源与进程归属。screen存在或退出码0都不足以证明实验已验收。
- 先通过累积数值、尾批、AMP、恢复及真实三头短测，再长训练。NaN/Inf、计数/恢复错误、重载或结果验收失败时停止后续队列；预热后连续16个AMP跳步也停止。不得自动换参、换种子重跑。
- checkpoint仅在完整更新边界保存，含模型、优化器、scaler、随机状态、采样位置、配置/初始化/数据哈希；保留原始日志与失败现场。
- 实时进度MD每60秒原子刷新；阶段转换、checkpoint、评价完成和异常立即更新。写入失败记录告警，禁止推进下一任务，不因此自动终止正常训练。

## 多智能体工作流

- 主模型按复杂度判断，不默认创建子智能体；简单任务直接完成。
- 只对复杂或可独立并行的重复工作分派；禁止并发修改重叠文件。主模型统一验收、整合和交付。
- 按任务难度选择可用模型，兼顾质量、耗时、成本；关键算法与审查使用强推理模型，整理和机械检查可用轻量模型。

## 记录与持续更新

- 每次重要决策、实现变更、验收结果或阶段切换后同步更新本文件中的事实索引和进度文件；禁止把计划标成已完成。
- 本轮目录：`../spnv2_runs/plans/rebaseline_20261017/`。实施前代码和现有未提交改动已完整保存在其 `backups/preimplementation/`；原HEAD为 `0d7922c5791d08c52c812bdc6154272452c1411c`。
- 历史Table2状态为 `failed_or_interrupted/train_H`，三个旧pilot完成，正式H输出无checkpoint；不能直接重新启动旧队列。新协议需新的pilot及新目录。
- 临时交接文档已按用户要求删除，不重建。当前事实以本轮状态JSON、事件和实际产物为准。

## 最新执行决策（2026-10-17 北京时间）

- 按用户最新要求，删除真实 batch16 可行性短测；本轮不运行、不排队、不将其作为复现启动门槛。正式 φ3 仍为物理 batch4 × 累积4，φ6仍为物理 batch1 × 累积4。更新后的完整计划见 `../spnv2_runs/plans/rebaseline_20261017/EXECUTION_PLAN.md`。
- 最终CPU算法/恢复/评价摘要验收13/13通过；真实φ3三头全增强512样本短测通过，32个更新尝试、22次成功、10次AMP初始跳步。
- GPU中断恢复：从第16个窗口恢复后，权重、优化器、scaler、随机状态、样本ID及计数与连续运行逐项完全相同。显存峰值约9.64GiB，短测吞吐约9.61张/秒；它们是工程短测证据，不是正式复现精度。
- 独立只读源码快照已冻结在本轮 `source/`，核心代码经SHA256核对；正式训练独立初始化，不继承短测权重。正式训练状态须查询实时进度与实际进程，不能根据本段判定已启动。

阶段1执行入口已建立（2026-10-17）：`spnv2_runs/plans/rebaseline_20261017/queue.json`，管理器screen为`rebaseline_queue_20261017_phi3`，启动验收凭据为本轮`reports/preflight_start_phi3.json`。队列按GPU实际资源自动等待并串行推进φ3训练及三域评价；本阶段结束不代表190轮完整计划完成。接手先核验status.json中的管理器PID/starttime及任务证据，禁止另起重复队列。资源是否空闲及训练是否启动只看实时状态。

## 执行守护与已启动事实（2026-10-17）

- φ3-BN EHS正式20轮训练已由服务器队列于北京时间14:26:45启动，物理batch4×累积4；screen为`spnv2_source_phi3_ga4_20261017`。训练与管理器PID均按/proc启动标识复核存活，严禁重复启动或重启健康训练。最新epoch/batch以实时MD和原始日志为准。
- 用户要求持续保持目标执行并紧盯GPU，空闲且验收通过就抓紧启动下一项。服务器队列在资源等待时约每5秒检查GPU，每60秒刷新进度，已证明能在聊天目标记录缺失时独立启动并维持实验。
- 本聊天只保留现有守护任务`spnv2-2`，当前每小时单次巡检，按预计结束/阶段交接适配界面支持的间隔。目标模式暂停不取消服务器实验和定时巡检；定时任务不得创建或恢复目标模式，目标记录缺失也不创建。用户若明确暂停或取消实验，遵守其最新指令。
- 目标卡片的显示不是训练存活证据；完整190轮及全部后续验收前，不将阶段完成标记为整体目标完成。

## 简洁进度查看入口（2026-10-17）

- 用户日常看进度优先在远程终端运行 `bash /home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/progress.sh`；同屏每5秒刷新，显示轮次、本轮百分比、loss、更新/AMP计数、实际进程核验、GPU整卡利用率和异常。`--once`仅查看一次。
- 查看器只读状态和日志，不启动、停止或修改实验；Ctrl+C仅退出此查看器。原始SPNV2_EVENT与样本ID仍完整保留供审计，不把原始刷屏日志作为默认用户入口。

## Git 版本记录规范（用户2026-10-17最新授权）

- 每完成一项代码修改并通过适当验证，主模型统一 commit 并推送到本项目 `origin/main`；不等到全部实验结束。子智能体提供准确文件清单、动机、作用、测试证据，由主模型验收后提交。
- 每条提交写清“改动位置、原因、作用、验证结果”；同步维护根目录 `CODE_CHANGES.md`，让用户无需阅读完整代码即可了解改动。
- 先把本轮实施前保存的8处旧修改单独归档，再记录本轮实现；不将旧改动混称为本轮新增。
- 仓库外的本项目实验驱动、配置及验收代码按原路径映射到仓库 `reproduction/`，保持逐文件SHA256一致并记录映射；实际运行入口仍为原资产目录，禁止搬动或重启健康训练。
- 使用明确文件清单提交。数据集、权重、原始日志、每60秒变化的进度状态及相关论文PDF保留在服务器资产中；不因代码同步而批量上传这些资产。
- 推送失败必须如实记录本地已提交/远端未同步状态，保留待推送提交；不得把本地commit说成GitHub已更新，不强推覆盖远端历史。

- 2026-10-17追加队列已验收接续：当前绑定本轮`queue_v2.json`（9项），原φ3四项不变，追加φ6 pilot→独立20轮→三域评价。仅重启管理器，φ3训练PID/starttime保持不变；后续普通恢复用`--manifest queue_v2.json`，不得再次使用旧扩展参数。修订证据见`revisions/manager_cutover_v2_a/accepted.json`。

## 论文评价分母与后续安全接续（2026-10-17）

- 已核对本项目Park2024SPNv2.pdf第9页Sec5.4：Table4起主结果采用全部样本的H旋转与E平移；Table2/3另按接受子集规则。当前冻结评价在H拒绝后丢弃其数值姿态，故不能把现有接受子集成绩直接称为论文全部样本同口径。
- 后续补独立observer旁证：不改冻结训练/推理源码、不增加forward、不改变原接受子集；保留拒绝前仍有限的H姿态，另报扩展分母。H缺失不填零、不用E代替；覆盖N/N才称all-samples。该observer当前仍在实现验收，尚未进入运行队列。
- 管理器新增显式的“只包装未启动source评价”修订能力：必须逐项列名，保留原解释器/参数、asset哈希、依赖、输出、GPU规则；训练/已启动/已验收任务禁止修改。新11项CPU（含真实CPU子进程无重复启动）与原59项队列回归均通过。
- 当前实际管理器仍绑定queue_v2.json；本次只更新磁盘代码和审计能力，未重启管理器或健康训练。v3b提案属于历史准备记录，后续将另建完整修订并重新绑定当前实现哈希。

- Table6资源测量工具已通过主模型代码审阅及11项CPU合同验收；实际GPU测量尚未运行。四个已验收EHS final统一FP32、batch1、768×512、10次预热＋30次CUDA Event计时，记录纯整网forward，不声称端到端速度或精度。排在Table4所有模型及12域评价、固定40项ODR验收后，扫描前；ODR资源只读已有固定40审计，不额外跑ODR。历史pending任务定义仍是准备快照，启用前必须重绑当前runner等SHA。

- 科研图工具analysis_plots v1已通过CPU/mock及真实快照独立核验，主模型审阅代码/参考SHA和实际静态参数图。该v1只解释原接受子集：56指标系列/504扫描点当前全pending；未来扩展分母必须独立接入，不能从旧NaN MAT重建。原论文页面渲染只保留服务器阅读证据，Git归档实现与结构化数值，未复制论文全文/页面图。

- 扩展评价v2及同次前向source/ODR入口已通过CPU工程核验并归档；保持原训练/原MAT，修复HIL阈值处原dtype运算阶段差异，原失败证据保留。完整final_reporting仍未READY，只有范围合同与扩展汇总阶段通过，当前队列保持v2。
- AMP跳步按完整累积窗口计：φ3每次跳过4个小批/16次图像处理，这组梯度不产生优化器更新，当前不会立即重放。不得把跳过的窗口算成成功更新；前向仍可能更新BN运行统计，不能声称模型全部状态未变。记录attempts/success/skips，不擅自重训或改协议。


## 用户最新优先级：结果对标与待机（2026-10-17）

- 对用户的主要汇报聚焦：已完成哪项复现、是否与论文同口径、平移/旋转/综合位姿的论文值与本次值、绝对差和相对差，以及尚存训练条件/统计分母差异。不能在结果未出前承诺追平论文。
- 更新尝试、成功更新、AMP跳步等过程计数继续保存在后台原始记录中，不作为日常进展汇报重点。只有它们构成真实异常、影响实验有效性或用户主动询问时再解释。
- 下一阶段实验若已准备好，就待机监测当前真实进程及GPU，完成并验收后及时串行接续；不为填满等待时间主动扩展工具、泛化审计或反复增加测试。
- 只开展保证实验正确、完成必要接续和论文同口径结果所必需的工作。待机期间不改健康训练，不临时增添实验或更换参数。整体190轮及后续复现范围仍然保留。
- 本次已只读核对当前9项队列及全部未来任务的已绑定文件哈希，均无漂移：φ3完成后先三域评价，再φ6短测与独立正式训练/三域评价。此处为接续准备事实，不代表正式结果已出。
- 非当前接续必需的final_reporting审计/补充绘图/v4工具开发已停止继续分派，现有文件保留。待阶段完成、发生异常或有必要处理事项时再行动和汇报。


## 现行运行模式：服务器持续执行，聊天定时巡检（2026-10-17）

- 本节取代此前“持续保持目标模式/目标缺失自动恢复”的要求。完整范围和验收标准不变；本次确认服务器独立机制后按用户明确要求暂停目标模式，停止聊天连续续轮，不停止实验或现有定时守护。
- 现有spnv2-2守护已保留原聊天和ACTIVE、每小时设置；单次只看新增状态，处理已授权且当前必要的事项后结束，不循环等待、不反复全盘扫描、不主动扩展工具。可以按实际阶段自主更新同一任务的完整提示及界面支持的间隔，无需逐次询问，不新增重复任务。
- 长健康训练每小时；约2小时内预计结束、交接或待验收时适当加密（优先界面支持的15分钟），恢复长训练后回每小时。全部计划逐项验收完成才停用守护并汇报。
- 已核验当前训练和管理器分别脱离SSH运行于screen，进度/失败阻止/验收/GPU串行接续由服务器承担。不得重启或重复创建健康训练和队列。服务器/管理器故障仍由巡检依持久证据安全续接，不声称已经设置开机自动恢复。
- 当前只自动接好9项至φ6三域评价；已验收observer尚未包装六个source评价（须在φ3正式评价前接入）。其余8配置150轮、固定ODR、扫描/资源、最终汇总仍需按阶段必要接续，不能把历史v3b或未发布v4当已生效完整队列。详见`../spnv2_runs/plans/rebaseline_20261017/SERVER_EXECUTION_HANDOFF.md`及同目录monitoring/guard_cursor.json。
- 定时提示版本在monitoring/spnv2-2_prompt_20261017.txt；每次新增阶段/接续/验收变化更新游标、当前下一动作和缺口，保留全计划引用和未完成阶段。不要通过新建/恢复目标模式推进。


## 2026-10-18 接续准备：候选已验收，尚未启用

- 14:30巡检时φ3-BN EHS为第19/20轮、batch4544/11988，训练和管理器精确身份健康；当前仍绑定9项queue_v2，未停止、重启或重复创建任何生产进程。
- `queue_proposals/observed_stage_20261018/proposal.json` 为49项候选：保留原φ3训练、φ6 pilot及φ6训练定义，严格只包装六项未启动source评价，再追加已授权固定ODR40；不改变训练协议、原评价参数/分母或forward次数。新增旁证单列有效分母；N/N才可称全部样本。完整10配置190轮、剩余150轮/扫描/资源/案例/最终验收范围不变。
- 独立screen中CPU候选构建与纯修订合同检查通过，291个实际资产SHA核对；旧ODR提案唯一runner绑定漂移更新为已验证现行版本。首版因早期φ3任务未设置asset_sha256字段而失败，修正兼容读取后重新通过，原失败日志与源码保留；未执行GPU评价或ODR，未改生产state/manifest。
- 已核查当前manager无热加载接口，HUP也会停止它。启用候选必须一次仅管理器的受控重启，训练保持原PID/starttime和screen；这与用户最新“不重启健康队列”限制冲突，必须先取得这一次明确例外。不得以修改当前manifest、状态或热补丁绕过；获允许后先重验六项仍未启动，再按既有排他锁与纯修订流程接续，不能触碰已启动评价。
- 同一个spnv2-2已用应用工具改为ACTIVE/15分钟并逐字读回；未新建定时任务、未恢复目标模式。进入后续长健康训练后回每小时。当前待用户确认上述管理器例外；若评价已开始，先保留实际证据并汇报，不擅自补跑或回写任务。


## 最新接续状态：已按用户一次性授权启用（2026-10-18 14:38）

本节取代上方“候选未启用/等待授权”的当前状态。用户明确同意“允许这一次仅切换管理器，训练继续”。14:38:42已完成manager-only受控切换并验收，训练PID3111276/starttime517862167前后不变，batch5892→5896，未重启训练或其screen。新manager PID1851843/starttime526573814，screen `rebaseline_queue_observed_20261018`；日志`logs/queue_manager_observed_20261018.log`。精确身份仅作交接，今后须读实时状态重核。

当前绑定`queue_observed_stage_20261018.json`，49项，canonical SHA256 `a6f82b1efc62ebb6223e1cfec4919e44de35424b53709230f39b0655ea3e27a5`。六个原source评价均在未启动状态完成严格wrapper绑定；原训练/pilot逐字段保持，原评价参数、旧结果及同次推理次数保持；追加原计划固定ODR40。三域评价同时生成原接受子集与扩展H数值/混合旁证，各自列清分母，只有实际N/N覆盖才可对标论文全部样本口径。尚无正式评价/ODR结果，不提前承诺论文差距。

切换证据：`revisions/manager_observed_cutover_20261018_b/accepted.json`，正式修订审计`revisions/2026-10-18T14-38-42+08-00_1851843_a6f82b1efc62ebb6.json`。首次切换脚本因服务器Python无pidfd接口在发送信号前失败，旧manager/worker未受影响；保留原失败记录，改用既有PID/starttime核验方法后成功，只有一次实际manager切换。

普通故障恢复只使用`--manifest queue_observed_stage_20261018.json`，禁止重用历史`--extend-from`或`--revise-pending-from/--pending-task`参数。一次性健康manager切换授权已经使用；不得重复运行一次性脚本、停止/重启健康训练或队列。目标模式仍暂停。现有spnv2-2保持ACTIVE/15分钟，实际注册完整提示见`monitoring/spnv2-2_prompt_active49_20261018.txt`，已逐字读回验证；后续长健康训练恢复每小时。

接下来由服务器自动：φ3自然结束→成功退出、final严格重载和训练证据验收→synthetic11994/lightbox6740/sunlamp2791评价及原/扩展结果验收→φ6 GPU pilot/恢复验收→独立20轮与三域评价→固定ODR40。任何异常按原规则保留现场并阻止下游，不擅自重训换参。

尚未接入当前49项的全部缺口继续保留：其余8套source150轮及24域评价、Table2三个pilot、Table6资源、B120/N240扫描、案例、PRISMA25材料核验、完整扩展绘图/结果汇总与最终逐项验收。完整范围仍为10套190轮和400 ODR请求/370实际+30严格别名；49项完成不是全部完成。


## 最新阶段：φ3训练及synthetic正式评价通过（2026-10-18 16:48核验）

φ3-BN EHS独立20轮于16:36:40完成、16:36:49通过验收；main/verify均退出0，final严格重载并与最终checkpoint一致。固定final SHA256为7692e3cb2ac9c1890ff3881aa42f70d0cc81175740e4c8795466e7eda01e011a；原训练进程自然退出，不再要求其PID存活。权重与验收路径见本轮reports/source_phi3_verified.json，未重训或选其他checkpoint。

synthetic全11994样本于16:45:20验收通过，H拒绝0、11994/11994覆盖；hE平移0.0589231m、hH旋转1.0348671度、混合分数0.02852483。同次推理扩展report、独立proof和entrypoint proof哈希及passed状态已读核；无需重复GPU forward。小型每轮validation的1199样本不当作这次正式11994结果。

16:48正在评价lightbox，后续sunlamp再经各自原/扩展验收，由既有49项队列进入φ6 pilot/独立20轮/三域评价/固定ODR40。主模型不等待这些任务跑完，不重复扫描或增加额外测试。目标域论文差距尚待真实结果；不拿synthetic数字代替。守护保持同一spnv2-2每15分钟、目标模式暂停；完整190轮及所有其余阶段/约束不变，不能将φ3训练完成称为整体完成。

阶段凭据：monitoring/phi3_source_synthetic_accepted_20261018.json；最新完整守护提示：monitoring/spnv2-2_prompt_phi3_evaluation_20261018.txt。实时阶段与PID以status.json和进度MD为准；本节替代之前“φ3仍训练中/没有正式评价”的当前状态。


## 最新阶段：φ3三域通过，φ6正式训练中（2026-10-18 17:16核验）

本节替代之前“目标域评价进行中/待φ6 pilot”的当前状态。lightbox6740于16:49:57、sunlamp2791于16:51:56通过原结果与同次推理扩展独立验收；各main/verify退出0，report/proof/entrypoint实际SHA吻合。与既有synthetic11994验收合计，φ3三域全部通过。

全部样本H旋转+E平移结果（平移m/旋转°/分数）：lightbox 0.279724950/9.881825158/0.217882592，sunlamp 0.291407326/16.331912979/0.335371050，覆盖分别6740/6740、2791/2791；原接受子集6730与2782、几何拒绝10与9另表保留。

重要参考修正：对应论文Table4 source应为lightbox0.175/6.479/0.142、sunlamp0.225/11.065/0.230；旧截图0.142/5.624/0.122与0.182/9.601/0.198来自Table5 ODR后。当前source不与ODR结果作同阶段比较。结果明显未追平论文；保留数据，不自动调参/重训，不将差距全部归因于累积。详细绝对/相对差和口径见monitoring/PHI3_SOURCE_COMPARISON_20261018.md。

φ6 GPU pilot及严格恢复验收16:53:41通过；现有49项队列16:53:44自动启动φ6-GN EHS独立20轮，物理1×累积4，不继承pilot权重。17:16为第1/20轮；worker2584192/start527383895，screen spnv2_train_phi6_ehs_qv2_20261017，日志logs/train_phi6_ehs_qv2_20261017.log。manager1851843/start526573814保持健康，GPU99%，心跳/MD新鲜。身份只作交接，下次仍读实时状态。

下一步保持健康训练；自然完成后成功退出、final严格重载/数据计数验收，再三域原/扩展评价，各自通过后固定ODR40。不重跑φ3或pilot，不重复manager切换。原spnv2-2已恢复ACTIVE每小时单次巡检，应用提示逐字读回一致；目标仍暂停。阶段凭据monitoring/phi3_targets_phi6_start_20261018.json，最新完整提示monitoring/spnv2-2_prompt_phi6_training_20261018.txt。

完整10配置190轮及全部评价/ODR/扫描/证据要求保持；当前队列之外8配置150轮、24域评价、Table2三pilot、资源、B/N扫描、案例、PRISMA25及最终汇总验收缺口保持。长训练期间只处理必要事项，不扩展无关工具。


## 用户授权的旧模型补评与一次性暂停（2026-10-18 18:59）

用户要求暂时暂停φ6，先把自己的旧batch4、不累积φ3固定final按统一入口完整补评，再继续φ6。此授权仅适用于这一次操作，不重训、不重启管理器，也不扩大为任意暂停权限。旧final SHA dbb9deb17c37ce48b01edc1d658ad1a64c09094cf88d28a3c79adb49fd4cad50；只评synthetic11994、lightbox6740、sunlamp2791。新累积φ3已有同入口完整结果，直接复用；本次不操作其他项目。

执行目录：../spnv2_runs/plans/old_source_reeval_20261018。旧权重已通过CPU安全加载、严格重载及有限性检查；8项全样本验收测试、12项暂停恢复保护测试及实际生产身份/锁核验通过。它们是工程验收，不是补评成绩。补评同次前向保存每张图像的E/H预测和拒绝标记，原接受子集另存，只有真实N/N才能称全样本，不填零或用E替代缺失H。

18:59:23已确认原φ6 PID2584192/start527383895处于T状态，GPU空闲且剩余6391MiB后开始旧模型synthetic补评。仅SIGSTOP原训练主进程，所有内存状态保留；manager1851843/start526573814和49项生产队列保持。独立screen控制器与watchdog先清理本次评价，再SIGCONT同一个训练进程。最长暂停30分钟，控制器失联90秒触发恢复；不运行退出重载或重新初始化。SIGSTOP不产生新的checkpoint，保留内存中的准确训练位置。暂停时长单独记录，不混入科学吞吐对比。

补评进度：../paper/SPNv2_旧模型补评进度.md；操作事实：该执行目录status.json、pause_intent.json、pause_confirmed.json、watchdog_ready.json和recovery.json。原生产MD仍由manager维护，暂停期间原batch暂时不前进是预期；禁止因此重启训练。完成或异常后必须核实原PID/start保持、训练日志重新前进及GPU工作恢复。

原spnv2-2守护已补充本次授权和识别规则，保持ACTIVE每小时；目标模式仍暂停。完整10配置190轮、原49项接续及所有尚未接入阶段、验收要求不变。旧模型补评不计作新增训练配置，不以此宣称完整复现完成。


## 最新状态：旧模型三域补评完成，φ6原进程已恢复（2026-10-18 19:17验收）

本节取代上方“旧补评进行中/φ6主动暂停”的当前状态。旧φ3固定final完整补评三域均通过main/verify退出0、原结果和同次前向扩展独立验收；全样本覆盖synthetic11994/11994、lightbox6740/6740、sunlamp2791/2791，原拒绝0/10/4另存。旧模型综合分数依次0.0266852493、0.2306220063、0.3292670081。与新累积φ3的同入口配置、代码、后处理、observer、样本顺序逐项一致；新模型结果直接复用，没有再推理。

完整对比：../spnv2_runs/plans/old_source_reeval_20261018/SOURCE_COMPARISON.md；机器证据reports/source_old_new_verified.json。Table4 source论文综合分数为0.027/0.142/0.230；新累积模型为0.0285248320/0.2178825917/0.3353710503。新模型lightbox综合误差比旧模型减少5.52%，sunlamp增大1.85%；相对论文新模型分别仍高53.44%和45.81%。不能因同口径统计已对齐而声称训练条件完全一致，也不能把差异全部归于累积。

旧历史补评分数与本次完整保存相差lightbox -0.000359588（-0.1557%）、sunlamp +0.000005885（+0.0018%），远小于与论文的差距。本次完整结果作为以后该固定旧final的统一对照，旧结果原样保留。

本次一次性暂停18:59:20→19:14:46，约926秒。控制器与独立watchdog退出0，先清理自有评价后继续原φ6 PID2584192/start527383895；19:17已核训练恢复后samples_seen21664，大于暂停前记录21204，manager1851843/start526573814及49项manifest保持，GPU恢复φ6计算。无重训、重载、回退或重新初始化。一次性授权已执行完毕，禁止重跑控制器或再次暂停健康训练。静态恢复凭据recovery.json及上述机器证据包含完整前后身份和事件。

原spnv2-2已恢复ACTIVE每小时单次巡检，完整提示见old_source_reeval_20261018/guard_prompt_complete.txt；目标模式仍暂停。用户要求“持续扫描、结束立即整理”针对本次补评，现已完成。后续继续φ6自然训练→严格final验收→三域完整评价→固定ODR40；完整10配置190轮及其余150轮/消融/扫描/资源/案例/材料核验/最终验收范围和缺口不变。


## 最新阶段：两套主模型及固定ODR40完成，Table2候选待授权（2026-10-22）

φ6独立20轮16:08:57退出0，16:09:12严格final验收；三域原/扩展结果16:48:41前全部通过。final SHA256=362c4c035371ac88ecb205a60e21f529e6601ff465b27e9ef81d45317934261d。扩展三域11994/6740/2791均真实N/N，原拒绝0/18/2另存。固定ODR40于22:03:22最后验收，40实际执行、80 main/verify退出0，各原/入口/独立扩展proof与逐样本哈希通过只读核验。详见本轮monitoring/PHI6_FIXED40_COMPARISON_20261022.md、phi6_fixed40_stage_20261022.json与fixed40_completion_20261022.json。Table5跨种子汇总方式和phi6论文B未明确，参考差不能称严格同协议差距。

当前49项phase_complete、active=null；manager1851843/start526573814健康待机，无活动训练worker属于预期。仅完成40/190 source轮；剩余8套150轮及24域评价、Table2三个新pilot、Table6、B120/N240、案例、PRISMA25材料和最终验收均保留。

Table2三个pilot的最小52项候选已独立CPU验收通过，目录queue_proposals/table2_pilots_20261022。原49项完全保留，仅追加H→E→EH，56个不同新增资产哈希匹配；canonical SHA256=976cf1bf4caf473d47b861e01b69a216399da0e492b6bcb9fe282134365e5adc。未执行GPU pilot、未激活候选、未改生产状态或发送信号。健康manager无热扩展接口，旧一次性切换授权已消耗；必须取得新的这一次manager-only授权后按activation_review.json复核锁/身份/证据并切换，不能另起重复队列绕过。

本次现有spnv2-2设为ACTIVE/15分钟，已应用读回；完整提示monitoring/spnv2-2_prompt_stage49_complete_20261022.txt。目标模式仍暂停，全部完整计划通过最终验收前守护不能停用。待授权且状态无变化时结束单次巡检，不重复播报、重跑验收或反复请求相同授权。


## 最新接续：用户本次授权的52项管理器切换已验收（2026-10-22 23:29）

用户明确允许这一次仅切换管理器，按已验收52项候选启动H/E/EH三个pilot。控制器独立screen退出0，23:29:53切换验收通过，23:30:59独立读回通过；原49项定义与执行记录逐字段保持，没有重跑。旧manager1851843/start526573814正常退出释放锁，新manager3305820/start564320727独占pipeline.lock，独立screen rebaseline_queue_table2_pilots_20261022。H pilot已实际启动并使用GPU；E、EH由服务器依次成功退出、独立验收并检查GPU后串行接续。实时任务/PID从status.json读取，不要求已完成pilot的旧worker继续存活。

当前正式manifest为queue_proposals/table2_pilots_20261022/proposal.json，52项，canonical SHA256=976cf1bf4caf473d47b861e01b69a216399da0e492b6bcb9fe282134365e5adc。切换证据revisions/manager_table2_pilots_cutover_20261022/accepted.json和independent_readback.json，正式修订revisions/2026-10-22T23-29-51+08-00_3305820_976cf1bf4caf473d.json。普通故障恢复仅用新--manifest；绝不能重用一次性--extend-from、旧observer修订参数或控制器。该次授权已执行完毕，不能任意重复重启健康队列。

现有spnv2-2继续ACTIVE/15分钟单次巡检，完整提示monitoring/spnv2-2_prompt_table2_active52_20261022.txt；目标模式保持暂停。下一步逐项核验3个pilot的main/verify与严格恢复/计数/有限性，之后按原计划准备Table2正式H10/E20/EH20及三域。其余8套150轮/24域评价、Table6、B120/N240、案例/PRISMA25材料与最终交付仍未接入；52项阶段不是190轮完成。无变化不循环等待或重复验收。

本次记录时新增pilot状态：pilot_table2_h=accepted, pilot_table2_e=accepted, pilot_table2_eh=accepted。已accepted的项目均main/verify退出0，真实RTX4060Ti工程报告SHA与独立proof匹配，完整连续/恢复状态树相等；它们是工程短测，不是正式精度。机器记录monitoring/table2_pilots_cutover_20261022.json。


## Table2正式64项候选已验收、未启用（2026-10-22 23:44）

原52项与三个真实GPU pilot全部accepted，manager3305820/start564320727健康phase_complete。新目录queue_proposals/table2_formal_20261022已通过独立build/verify screen退出0，canonical SHA256=f53581b9add1460ae4f5582bcbda3c84d647c901ffd946950aa343183b60b4a2；verified.json SHA256=126545de404173a6fc38b468a9d009cfa95b9cf91f80dde261a3280f3c38fe60。原52项完全保留，仅追加H10/E20/EH20及各synthetic11994/lightbox6740/sunlamp2791三域，共50轮和9项评价；85个不同新增资产SHA通过，新输出均为空。训练独立初始化4×4、原参数/冻结source保持，不继承pilot权重。9项评价同次前向wrapper，H-only只用H，E-only无H适用项，EH混合；Table2原接受子集逐头主口径与扩展分表，缺失None，不填零或E代替H。

当前仅候选，不曾启动正式Table2训练或改变生产状态/manager。主模型审查构建/独立验收源码，独立核对原52前缀、候选与proof哈希及主指标分母逻辑。等待新的这一次manager-only切换授权；之前52项pilot切换授权已消耗，不能任意重启健康manager。候选activation_review.json列明受控接续；当前正常恢复仍只用52项manifest。

原两笔待推送记录已通过GitHub连接器等树同步并读回，服务器对齐bc2e770e，旧本地SHA保留archive/local-pending-before-connector-20261022分支。完整190轮仍只有40轮正式source完成，150轮尚未执行；本候选不等于已完成50轮。其余100轮、24域评价中未执行项、Table6、B/N扫描、案例/PRISMA及最终验收不缩减。守护ACTIVE/15分钟单次，目标暂停，未变化不反复索取同一授权。


## 2026-10-23 current authorized research stage

User confirmed ODR003 exploration v2 and standing manager-only switches. Remaining150 source epochs are deferred optional, not prerequisites. Official offline phi3 source identity and full GPU correctness with zero tolerance passed. See reproduction/odr003_exploratory_20261023/EXPERIMENT_PLAN.md and IMPLEMENTATION_STATUS.md. Runtime plans/odr003_exploratory_20261023/status.json governs the new phase; original52 accepted tasks must never rerun. Pure synthetic bbox decoder warmup addresses the independently diagnosed original TorchScript cold/hot arithmetic difference; original A update behavior remains frozen. Goal mode stays paused.


## 2026-10-23 - Confirmed G50-P round-two implementation

User confirmation authorizes G50-P, G50-RW and R-P within maximum18 complete ODR runs, after the fixed eight engineering trajectories pass. Historical150 source-training epochs remain deferred; goal mode stays paused. The current plan is ../spnv2_runs/plans/odr003_round2_20261023/EXPERIMENT_PLAN.md and confirmation.json. A shared GPU may be used when idle; no extra scheduler authorization is needed. Any other live compute process blocks launch. Never signal or interfere with other users processes. Only exact per-run launch identity, PID/starttime/boot/command/nonce and exclusive locks establish ownership. Waiting resource checks occur at least30 minutes apart.

Complete recoverable checkpoints, not log commits, determine durable scientific progress. Unpersisted computation may replay and is separately charged; persisted commits never replay. Local227 CPU synthetic/mock contracts passed. Real SPNv2/GPU engineering and independent proof are still pending; no second-round efficacy result exists. Source/A/G50/PnP/data boundaries and fixed rules stay unchanged. Implementation and current state are in the independent round-two plan; never restart old completed queues.


## 2026-10-23 G50-P metadata writer repair

The first bounded guard detected a registry trust failure from this host umask0002 and mode0664. The registry remained empty; no worker/model run had started. The writer now explicitly creates0600 metadata; strict trust validation remains. Local230 contracts passed. The eight engineering definitions, IDs, model/data/scientific rules are unchanged; current status/index and scoped revisions record the authorized manager-only deployment. Other users GPU tasks remain untouched.


## 2026-10-24 G50-P latest execution policy and preprocessing repair

Latest user authorization requires autonomous delivery,300-second waiting resource checks and private immediate resource requests without ordinary manager restart. The first real reference failed at first image_read, before any model forward; old failure and complete zero checkpoint remain intact. The RNG correction and guarded independent complete-state import have passed;312 local contracts passed. Eight logical engineering trajectories are preserved and must pass real independent proof before automatic P6 then count-matched RW/RP(max12) and independent delivery. Current actual processes/stage come from the new plan status and guard cursor, not historical manager IDs or GPU snapshots. Target/goal mode remains paused.

## 2026-10-24 auditor binding repair

All eight fixed GPU engineering main trajectories completed, preserving their original task/code/checkpoint identities. Independent CPU acceptance then stopped on an auditor schema mismatch: prepared artifacts include informational bytes in addition to the same path and SHA. Both engineering and complete-run auditors now compare strict path/SHA identity, validate optional byte metadata, and keep their existing file hashes and GT boundaries. An explicit auditor-only revision binds original index/code, the failed audit and unchanged eight producer tasks; missing producers cannot be relaunched. Current335 local CPU synthetic/mock contracts pass, including21 new metadata/tamper contracts and2 no-replay queue cases. Real engineering acceptance remains pending until a separately admitted CPU process verifies the saved states; no additional model forward is required. Evidence: revisions/auditor_binding_repair_20261024/. Scientific thresholds, seeds, source, streams and18-run cap are unchanged.

## 2026-10-24 confirmed deterministic G50-P execution

User explicitly approved30 complete ODR (A6/G506/P6/RW6/RP6) plus two512-image O evaluations after strict revalidation of the same eight128-observation engineering trajectories. Previous nondeterministic results and all failures remain preserved and are not current same-protocol baselines. Strict deterministic FP32 disables benchmark/TF32, enables fail-closed deterministic algorithms and persists backend/version receipts on every attempt/recovery. Frozen source/A update mathematics/PnP/thresholds/data/seeds and original512/511 reference IDs remain.373 local CPU synthetic/mock contracts passed with identical before/after code maps; metadata-only deployment rechecked all6 reference lists. Current bound index887e230c74c5ae7c51ba69b61e4d69f2ff93effb443cd0b7a4beeb602db9253e; real GPU engineering is still required. Resource wait300s, private immediate recheck, exact ownership and shared-host protections remain. Server automatic chain is engineering8 independent proof -> O2/A6/G506 proofs -> P6 -> verified-count RW/RPmax12 -> independent summary. Goal paused; old150 source epochs deferred. See reproduction/odr003_round2_20261023/ and runtime status, never infer acceptance from main exit0.


## 2026-10-25 G50-P update-risk diagnostic v1.0

The completed thirty-ODR plus two Offline study remains closed. This separate diagnostic authorizes A: existing 793 candidate/117 rollback records with zero model forwards; B: exactly 48 frozen paired windows, at most 40000 physical forwards and 14400 active seconds including all reconstruction, verification, state transfers and durable IO. Six 128-image panels exclude each stream's 1024 adaptation images. Frozen input SHA: 6608ffca250fe9733e1cd244e39eac84146ecd564b097ea101c7d2087f2d63e5. First included pilots are lightbox2021 candidate1 and sunlamp2021 candidate9; strict scientific reconstruction and independent receipts are required before the other 46. No complete ODR or new method experiment is authorized.

Implementation in reproduction/g50p_risk_diagnostic_20261025 adds isolated reconstruction, full-window-base rollback, state-pure branch evaluation, independent pair verification, fixed poststratified reports and durable global cost accounting. It imports the unchanged original 50-file strict deterministic producer 3ef5cbd32f16bcfcb650f4b5d7a04a69e718aa47; the archived performance candidate is not deployed. Shared-host two-snapshot admission, three old pipeline locks plus the new lock, exact nonce/process identity, and fail-closed budget enforcement remain mandatory. Code-only revisions cannot reset the stable project/frozen-input budget identity.

Local validation: A11 pure-record checks; B11 then22 synthetic correctness/recovery checks, plus3 receipt and2 trace-proof metadata checks; budget8, entry5, manager9 and weighted metadata2 checks. All passed; previous failed harness attempts remain recorded. Conservatively charge all necessary local validation 4748 physical forwards upper (including repeats and2 unknown dispatches) and360 active seconds upper to B before server work. This is not real SPNv2/GPU pilot acceptance. Real A analysis and B pilots are pending at publication; runtime status and independently verified receipts, not this source commit, establish progress. No old result/checkpoint/task binding is changed. Budget exhaustion yields actual partial results and ends the finite stage. Goal mode stays paused.


## 2026-10-25 diagnostic import-bootstrap correction

Experiment A extraction and analysis actually exited0 with passed zero-forward proofs. The first B window stopped before any model forward because prediction imported utils before the verified source core path was installed. Both paired runner and independent verifier now use the original producer order: import non-prediction modules, verify bindings/configure core paths, then import prediction. Frozen producer50, science,48 window selections and128-image panels are unchanged. Two targeted import-order mock tests passed with0 model construction/forward (3.453s); manager metadata contracts separately validate preserved A receipts and versioned partial delivery. These bounded pure-metadata tests remain inside the original360s local-validation conservative upper charge; no initial receipt or budget was reset.

The failed actual attempt retains its original task/code/index/exit/failure data and11.825699580833316s active-time upper charge. The same lightbox2021 window1 receives only a new technical run identity ending_importfix1; it is not a new window/ODR. Old A tasks/proofs stay byte-identical and are reused through explicit hash-bound preprocessing receipts. New pending B tasks reside in tasks/import_bootstrap_20261025. A complete original index and changed-code snapshots are preserved under revisions/import_bootstrap_20261025/before. A new manager is allowed only after exact prior process exits and locks were verified; state migration retains all prior attempts and the original project/frozen-input budget. Real two-window acceptance remains pending, never replaced by these local tests.


## 2026-10-25 diagnostic scripted-rotation recovery

A and the first lightbox2021/w1 pair have actual passed proofs and must be reused. The second pilot stopped at the first observation after parent64. Bounded evidence identifies the original scripted rotation decoder cold/hot arithmetic as the sole five-float divergence; [1,6] runtime priming restores exactly the original operator output without model forward or RNG changes. A zero-forward targeted real-device check and preserved-pair metadata guard passed. Production repair has its own revision, original failures/code/tasks/costs retained, first accepted pair explicitly preserved, and second logical window technical ID ending_rotfix1. Actual main/verify/pilot acceptance remains determined by runtime status and receipts, not this preparation note. The current scientific producer,48 frozen windows,128-image panels,strict equality and40000/14400 caps stay unchanged.


## 2026-10-25 diagnostic pilot admission template repair

Both included pilot pairs have actual main and independent verification passed. Overall pilot admission is still pending: a stale audit-template implementation binding caused a pre-import exit1. The versioned pilot/summary dispatcher candidate preserves both accepted pairs and all failures/costs, and requires fresh admitted zero-forward metadata validation before deployment. A dedicated server repair controller waits for foreign computation to leave, validates, and safely launches the existing fixed queue; real pilot_admitted status remains authoritative. No new window or scientific rule is introduced. See runtime revisions/admission_template_fix_20261025/controller_status.json and monitoring/guard_cursor.json.
