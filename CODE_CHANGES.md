# SPNv2 代码改动记录

本文件用中文说明每项修改的位置、原因、作用与验证。Git 提交中的逐行差异是代码变化的最终依据。

## 2026-10-17：补录复现前已有状态

- 来源：本轮实施前 13:50:38 保存的独立备份，原 HEAD 为 `0d7922c5791d08c52c812bdc6154272452c1411c`。
- 范围：`core/dataset/SPEEDPLUSDataset.py`、`core/dataset/transforms/randomsunflare.py`、`core/engine/inference.py`、`core/engine/trainer.py`、`core/utils/utils.py`、两个 `experiments/offline_train_full_config_*` 配置和 `tools/train.py`，共8处。
- 原因：这些修改在当前独立复现实施前已存在；单独归档，以便后续准确比较本轮新增内容。
- 作用：保留历史状态，不把旧修改归因于本轮实现；不改变任何运行中的实验文件。
- 验证：8个文件逐个与实施前备份的 SHA256 一致。历史功能的原始动机和完整验收没有在本记录中重新推断。

## 2026-10-17：梯度累积、严格恢复与评价口径

| 改动位置 | 为什么改 | 改后作用 |
|---|---|---|
| `core/config/default.py` | 原训练循环没有累积和可审计恢复开关 | 新增累积次数等配置；默认累积1兼容旧配置 |
| `core/engine/trainer.py` | 单卡要按既定物理batch累积，且更新/AMP计数需准确 | 局部loss除以累积次数；完整窗口才反缩放、裁剪、更新；保留原始loss与成功/跳步计数；有限值和连续溢出保护 |
| `core/dataset/build.py`、`core/utils/training_state.py` | 尾批和中断恢复会影响样本顺序与随机增强 | 前向前剔除不完整窗口；按epoch和样本设种子；保存配置、初始化、CSV/几何哈希、路径大小时间清单及完整训练状态；边界恢复并严格核对 |
| `tools/train.py` | 旧结果不能覆盖，正式训练必须独立初始化并有恢复证据 | 独占输出目录；保存initial/边界checkpoint/final；定期检查点与进度事件；严格显式恢复 |
| `core/engine/inference.py` | 热图拒绝预测不能当成0误差，评价需要完整分母 | 输出各头物理指标、单位、有效数、拒绝数和评价事件；无有效预测保留缺失值 |
| `AGENT.md`、`AGENTS.md` | 长实验需要统一边界和交接入口 | 记录独立复现计划、screen/GPU串行、验收停止条件、多智能体分工与本次Git同步要求 |

- 验证：最终CPU算法/边界/恢复/评价摘要13项通过；评价文件独立验收4项通过；真实φ3三头768×512、batch4×累积4短测32个更新窗口通过；第16窗口恢复至32后模型、优化器、AMP、随机状态、样本顺序及计数逐项完全一致。
- 限制：以上是实现和短测验收，不代表20轮复现精度已完成。物理batch4的BN不等同论文四卡SyncBN。正式训练独立初始化，不继承pilot权重。
- 运行安全：当前φ3使用已冻结的独立源码快照，Git同步不修改或重启该训练。

## 后续记录方式

每项经过验证的代码改动都追加“具体文件、原因、作用、验证、限制”，并附相应Git提交。运行日志和实时进度仍在服务器资产目录持续更新。

## 2026-10-17：实验队列、配置、ODR与简洁进度工具

以下文件在仓库 `reproduction/runtime/` 按实际资产目录结构归档；`reproduction/archive_manifest.json` 逐项记录真实运行文件与副本的SHA256。

| 改动位置 | 为什么改 | 改后作用与验收 |
|---|---|---|
| `pipeline/runner.py`、`task_entry.py`、`progress.py` | 长实验需串行启动、精确判断进程、不能只看screen或退出码 | 资源门槛、PID/启动标识、退出证据与独立验收；MD原子刷新，失败停止后续；队列23项CPU回归通过 |
| `pipeline/revision.py`、`runner.py`及对应测试 | 后续实验需要追加，同时当前健康训练必须继续 | 原队列前缀不可变、独占锁和审计修订；新增资产哈希启动/验收前复核；18项修订、5项接入、7项接续/资产门槛通过 |
| `queue_v2.json`、`queue_proposals/*.py` | 将完整计划逐步变成可核验的真实命令和依赖 | 当前原φ3四项保持，追加φ6 pilot/独立正式20轮/三域评价；静态9组审计通过。切换只更换管理器，训练worker PID与启动标识未变 |
| `configs/`、`protocol_matrix.json`、`EXECUTION_PLAN.md` | 防止训练配置漏项、混用输出或偏离用户决策 | 10正式配置共190轮；每模型三域完整评价；真实batch16短测已从计划删除 |
| `pilots/gpu_smoke.py`、`verify_pilot.py`、`evidence.py`及四份配置 | φ6和H/E/EH正式训练前需要独立短测与恢复证据 | 使用真实生产训练循环，32窗口及第16窗口恢复；CPU证据校验16项通过。φ6和H/E/EH实际GPU短测尚待对应队列执行，不能称已通过 |
| `tasks/source_artifacts.py`、`verify_heads.py`、`audit_checkpoint.py` | 成功退出不等于权重/全域结果可用 | 独立重载、初始化/样本流/优化器/AMP/RNG与完整分母核验；评价导出4项通过；φ3实际第500窗口checkpoint独立CPU审计通过 |
| `odr/common.py`、`generate.py`、`run_task.py` | 固定ODR和扫描必须独立、串行且协议一致 | 400请求/370实际执行/30严格别名；统一GPU门槛及缓存；固定40项全部验收前不允许扫描 |
| `odr/audited_entry.py`、`evidence.py`、`verify_result.py`、`verify_heads.py` | ODR要证明实际观测流、参数范围和评价结果 | 标准进度事件、观测ID/唯一数、适应前后权重、输入证据、完整优化器/RNG、独立CPU重载及零有效预测处理；15项原算法数学测试+7组接入回归通过；真实GPU ODR尚未执行 |
| `odr/B1_only_change.patch`、`B2_only_change.patch`及SHA清单 | A/B1/B2要可单独复核，不能混线 | B1只限定骨干归一化仿射更新，B2另固定BN动量0.1；每项从对应新source final重新开始 |
| `resources/scripts/` | 补齐Table6参数量并核实官方初始化 | 四架构整网/骨干/编码器/ODR参数计数独立CPU验收；B0旧URL哈希迁移已由官方新URL及360张量一致性核验；GPU资源测量仍待执行 |
| `runtime/shared/`、`reproduction/sync_runtime_archive.py` | 原始审计日志过密、仓库外代码不易审阅 | 只读简洁页每5秒同屏刷新；代码副本逐文件一致并映射真实路径；不控制或重启训练 |

上述内容是代码与工程验收记录，不代表10套source、ODR和扫描的正式精度已完成。正式结果仍需完整训练、评价与验收。原始失败日志和历史版本保留在服务器对应资产目录。

## 2026-10-17：简洁进度兼容ODR事件时间

- 位置：`reproduction/runtime/shared/compact_progress.py`（实际运行文件为资产目录同名文件）。
- 原因：训练事件使用Unix秒，ODR事件允许ISO日期；旧查看器直接转数字会在进入ODR时读取失败。
- 作用：统一解析两种时间，保持新旧事件比较和心跳新鲜度准确；无效时间显示未知，不控制实验。
- 验证：Unix秒、带时区ISO、UTC Z及无效输入直接核验通过；当前真实φ3简洁页渲染通过。训练源码、参数及进程未改动。

## 2026-10-17：完整队列的进度刷新频率

- 位置：`reproduction/runtime/rebaseline_20261017/pipeline/runner.py` 的事件汇总，以及 `tests/test_progress_cadence.py`。
- 原因：完整任务提案有444项，旧逻辑每次参数更新都会强制重写整份状态；这超出“普通进度每60秒”的原约定，会产生大量重复I/O。
- 作用：普通更新、观测和评价进度每60秒刷新完整状态；阶段变化、checkpoint、恢复、评价完成和异常仍立即刷新。原始日志与逐事件审计全部保留，内存中的计数每次照常更新。
- 验证：6项新频率/即时事件测试通过，23项既有队列回归通过，均在独立CPU screen退出0。
- 生效边界：管理器在下一次安全接续后加载该实现；不改变训练源码、模型参数、样本流或训练worker。

## 2026-10-17：完整主流程提案及可读版本归档

- 位置：`reproduction/runtime/rebaseline_20261017/queue_proposals/v3b/`；归档同步器和README。
- 原因：主流程包含10训练配置190轮、30域评价、4个后续pilot和400项ODR/扫描请求，需要在继续运行现φ3时核验后续依赖，并让用户能查看所有安排。
- 作用：444项提案保持当前9项队列定义不变；40固定ODR验收后进入Table2/3/4，最后扫描。400请求中370为实际执行、30为严格复用验收。`task_overview.md` 提供全部任务和前置关系；完整12.4MB JSON压缩为175367字节gzip，提供原文/压缩/解压SHA256及661项资产清单。
- 验证：10组静态检查、6类错误依赖负向测试及661资产SHA核验通过；gzip重复生成一致、解压后逐字节匹配原JSON。
- 边界：当前是已验收提案，尚未激活；资源测量和最终分析还需单独接入验收。4个后续pilot不含已完成的φ3工程短测。未改变当前训练。
- 归档器新增已验收资源工具的显式开关，避免默认复制仍在准备的bench目录；gzip代码副本与服务器文件严格一致。

## 2026-10-17：最终证据矩阵、严格配对与姿态投影案例

- 位置：`reproduction/runtime/rebaseline_20261017/analysis/`；归档器同步分析实现清单。
- 原因：不能把新旧权重、接受子集变化、观测流种子或缺失预测混在一起解释复现差距。
- 作用：从独立验收报告及3MAT逐图证据生成30个source域评价＋400项ODR请求的完整矩阵，逐项列单位/有效分母/拒绝数；首次通过验收的final SHA固定，历史结果单列。
- 严格配对：同一ID且双方均有效才计算改善/退化；拒绝→有效、有效→拒绝及双方拒绝另列；各自接受子集均值与配对交集均值分表。30别名不增算独立实验；五种子只表示同一source的观测流随机性。
- 指标定义：仅H为hR+hT，仅E为eR+eT；双头才为hR+eT，CSV保留实际HIL归一化公式和单位。
- 案例与限制：从真实姿态预测投影3D点，不冒充MAT中不存在的原始预测2D关键点；检查展示图像元数据及相机/关键点哈希。PRISMA仅核实项目指定候选位置和逐ID材料清单，当前缺可核验材料，明确受限。
- 验证：最终6组CPU数学/破坏/投影回归通过；真实材料快照及独立验收均退出0。目前430个新结果位置已验收0，26项历史独立，核心/含PRISMA完成标志均为false；没有用模拟值填正式结果。
- 保留：两次中间语法错误的失败日志在原资产目录保留，最终修复后重新验收；代码验收不代表正式复现完成。资源测量和后续科研图另行接入。

## 2026-10-17：只包装未启动评价的队列修订

- 位置：pipeline/pending_observers.py、runner.py、tests/test_pending_observers.py；AGENT.md与执行计划记录论文第9页的评价分母。
- 原因：补齐原接受子集之外的数值姿态，需要为尚未开始的source评价添加旁证入口，同时持续运行中的20轮训练及原代码/结果必须保留。
- 作用：显式列出所有待包装评价；仅允许增加capture_source.py外围入口及其校验入口，原解释器、全部参数、旧asset SHA、依赖、GPU规则、会话和输出均保留。已开始/已验收/有历史或异常证据的任务一律不能修订。
- 审计：在管理器独占锁下保存原/新定义、状态及哈希；恢复时核验同一PID/starttime，不能再启动一次。普通append-only恢复行为保持兼容。
- 验证：新增11项CPU验收通过（含真实CPU子进程身份接续且无重复启动/无GPU调用）；原队列23、追加18、绑定5、接续7、刷新6共59项回归全部通过，所有screen退出0。
- 使用状态：代码已准备，当前queue_v2管理器与φ3训练均未重启；observer和完整后续队列另行验收接入。真实batch16仍不安排。

## 2026-10-17：四模型资源测量与独立CPU核验

- 位置：resources/bench的bench_contract、run_benchmark、verify_benchmark、odr_resources、test_benchmark、prepare_pending，以及静态schema/README/准备清单。
- 原因：论文参数量、整网显存/耗时及ODR更新开销属于不同分母，不能用训练速度或随机模型计时冒充完整模型资源结果。
- 作用：只在四个EHS final、12个完整域评价及40个固定ODR都验收后，串行测四网FP32/batch1整网EHS forward；10预热＋30个CUDA Event样本，统计均值/中位/分位/吞吐和显存基线/峰值。固定GPU输入，明确排除数据读取、传输及PnP后处理。
- 参数与ODR：实际参数逐名/shape/numel比对已验真清单；固定40审计的实际适应时间、显存、更新数和affine参数范围独立汇总，不新增φ0或φ3GN ODR。
- 验证：11项CPU统计/模式/计数/文件破坏/参数分母合同通过；主模型检查所有6个代码SHA、报告SHA和screen退出码0。未来GPU结果另须严格final重载、参数范围与独立NumPy统计验收。
- 限制：当前没有GPU实测数值；首轮测试汇总父目录错误的失败现场及修复前准备记录保留。pending定义是未激活历史快照，完整队列启用前会重新固定当前实现SHA，不把准备通过写成实验完成。

## 2026-10-17：可核验科研图与论文数值参考

- 位置：analysis_plots的common/render/run_plots/verify_plots/CPU测试、Poppler只读提取脚本、结构化paper_reference与实现清单；Git不复制论文页面PNG、整篇文字、模拟图或运行产物。
- 原因：B/N曲线必须绑定同一新source与完整五观测流，缺失结果不能补点；论文Table2/3接受子集与Table4起all-samples不能直接混比。
- 作用：准备56条指标系列、504个扫描统计点，五观测流均值及样本标准差；固定seed2021的8组逐ID改善/退化分布和失败转移，按交集分母标注。只从验收证据取值，未齐点保持pending而不画曲线。
- Table6：已真实生成四模型静态参数图；未来GPU图必须读取严格验收的四组30次原始延迟及显存，不把CPU/mock测值冒充GPU结果。
- 论文参考：148个Table2/3/4指标行及4个固定ODR域记录保留页码、单位、PDF与页面SHA；未知精确分母/seed聚合明确未知。主模型视觉核验第9页all-samples原文和实际静态参数图，扩展observer结果尚待独立接入。
- 验证：3组CPU/mock统计、布局、零有效值与破坏合同通过；真实生成CLI及独立verifier退出0，代码/引用SHA经主模型核对。当前504点均pending、正式配对0、实际新MAT验收0，仅1张静态参数PNG，未宣称正式扫描完成。

## 2026-10-17：同次前向的扩展评价与逐图独立汇总

- 位置：extended_evaluation v2、evaluation_entrypoints、final_reporting第一阶段及归档器。当前运行中的冻结训练源码、损失、AMP与优化器未改。
- 原因：原MAT丢弃几何拒绝的H姿态，不能由NaN结果恢复论文后续表格的全部样本口径；扩展统计还必须保留原float32/float64运算阶段，避免HIL阈值边界改变。
- 作用：仅观察原求解器的同一次调用，原返回值和MAT保持不变，另存拒绝前有限姿态/缺失原因；source/ODR入口绑定final/config/原报告与CPU独立证明，alias复用canonical、零额外前向，扫描前同时核验固定40项原证据和扩展证据。
- 汇总：核对10配置190轮/30域/400请求的完整范围；逐图ID、单位、分母、source与observer SHA独立重查，370组同ID改善/退化及几何拒绝转移另列、74组五观测流统计。零有效预测均值为None，不能填零、不能混入未运行状态。
- 验证：observer19组CPU含真实GT阈值边界；入口6组整合+4组合同+独立5 source/42 ODR MOCK证据链；范围15组和汇总12组合同均通过。主模型核对代码/原入口/报告/退出码SHA，并验收独立接口复查；所有MOCK均为工程测试，无真实GPU科学结果。
- 限制：完整科研图补充和最终审计仍在实施，只有stage_preparation_manifest，不发布完整READY、不切换当前9项队列。未把未验收v4提案或代理正在写的文件归档。真实快照仍0/430新域评价验收，不能因CPU验收通过宣称实验完成。


## 2026-10-17：按用户要求聚焦论文结果，已备好接续时待机

- 位置：AGENT.md项目协作规则；本次没有修改训练、评价或队列代码。
- 原因：用户关心最终指标能否同口径对标论文及具体差值，不需要持续汇报优化器过程计数，也不希望等待时额外扩展工具。
- 作用：过程计数后台保留，面向用户报告论文值/本次值/绝对与相对差/口径限制；下一项准备好后待机监测，仅在完成、真实异常或必要接续时处理。
- 验证：只读核对当前训练/管理器精确PID及启动标识、实时日志、GPU；9项队列所有未来任务的绑定文件哈希均无漂移。整体190轮范围不变，未重新启动或停止训练。


## 2026-10-17：服务器持续执行，聊天改为每小时单次巡检

- 位置：AGENT.md、现行EXECUTION_PLAN、SERVER_EXECUTION_HANDOFF与现有spnv2-2提示快照/路径映射。本次不改训练、评价、队列代码或实验参数。
- 原因：用户要求停止聊天连续续轮，服务器继续既定实验；定时巡检只处理新增状态及当前必要事项，不能自动创建/恢复目标模式。
- 作用：保留同一ACTIVE每小时任务与原聊天；允许按实际阶段调整受支持间隔，单次完成即结束；保留10配置190轮及全部未完成阶段/验收约束和必要接续缺口，最终全部验收后才停用守护。
- 验证：训练/管理器精确PID与启动标识、SCREEN脱离SSH祖先、持续推进的服务器心跳/日志、现有原子刷新和异常拦截逻辑已读核；原22项队列及23+6回归日志退出0，未对生产注入故障或重跑。9项未来任务资产哈希无漂移。应用工具只更新spnv2-2，ACTIVE/每小时/原聊天/完整提示逐字读回一致，未新建任务。
- 明确缺口：当前只接至φ6三域评价；六source评价的已CPU验收observer尚未入队，须在φ3正式评价前接入；后续150轮/固定ODR/扫描/资源及完整最终汇总仍需分阶段接续。未宣称全部队列或开机恢复已就绪。


## 2026-10-18：准备同次推理评价与固定ODR接续候选（未启用）

- 位置：reproduction/queue_proposals/observed_stage_20261018，AGENT.md、交接文档及守护提示快照。
- 原因：φ3已接近20轮结束，旧评价会丢弃部分几何拒绝的H数值，影响按论文全部样本口径比较；需要在正式评价前接入已验收observer。
- 作用：只包装六个尚未启动的评价，保留原参数/旧输出/单次推理；另接原计划固定40项ODR，候选共49项，完整190轮范围不变。训练、pilot及生产队列没有改动。候选仅生成文件，不启用或控制进程。
- 验证：独立screen纯修订预检通过、291个实际资产哈希核对；原有wrapper CPU证据哈希一致。早期φ3无asset_sha256字段的兼容错误已修复，首次失败留档，修正版退出0。候选预检不代表真实评价/ODR完成。
- 启用条件：现有manager没有热加载，用户禁止重启健康队列，因此等待一次manager-only重启例外授权，训练原进程不动；获允许后重验六项仍未启动、原worker身份和资产绑定，再受控切换。不得覆盖运行中manifest或状态绕过限制。
- 监测：只更新原spnv2-2为15分钟并验证读回，进入下一长训练后恢复每小时；目标模式保持暂停。未上传权重、数据或原始训练日志。


## 2026-10-18：按一次性授权启用评价与固定ODR阶段队列

- 位置：queue_observed_stage_20261018.json、activate_authorized_b.py及切换验收记录，AGENT/执行计划/交接说明和现有守护提示。
- 原因与作用：用户允许这一次仅切换manager，使已验收的同次推理评价入口与原计划固定ODR40在φ3结束前生效。原6评价严格未启动包装；原训练/pilot、评价参数、原输出及全部原资产绑定保持。
- 验证：独立screen实际切换退出0，训练PID3111276/starttime517862167不变且batch5892→5896；49项绑定与6项未启动状态、排他锁、修订审计、新manager身份及心跳/MD新鲜度核对。原9项之后的范围和验收标准未缩小。
- 失败留档：首次尝试因Python无pidfd接口在发信号前失败，未停止旧manager；改用既有精确PID/starttime方法后成功，仅一次实际manager切换。代码/日志及两次意图记录均保留。
- 管理：普通恢复只带新manifest，不重复历史修订参数；一次性授权已使用，目标不恢复，当前原spnv2-2每15分钟巡检。没有重训、改变参数或新建重复队列。尚无正式评价/ODR成绩。


## 2026-10-18：记录φ3 source与synthetic正式验收（无实验代码修改）

- 位置：AGENT、交接文档、monitoring阶段凭据和原spnv2-2提示快照。
- 原因/作用：训练20轮与11994样本正式评价已完成，将当前阶段更新为目标域评价，再由现有队列进入φ6；保留完整190轮范围和所有后续缺口。
- 验证：main/verify成功退出、final严格重载/最终checkpoint一致性、final与扩展report/proof实际SHA、11994覆盖和零拒绝均已读核；没有重跑训练/评价/测试、没有改参数或重启队列。
- 守护：原任务仍15分钟且提示读回一致，目标继续暂停；不把本阶段标成完整复现完成，论文目标域差距待真实评价。


## 2026-10-18：记录φ3目标域结果、修正论文阶段参照并接续φ6（仅文档）

- 位置：AGENT.md、交接文档、monitoring的目标域凭据/对比表/守护提示及仓库镜像。
- 原因：两域评价与φ6 pilot已经验收，需把当前阶段更新为φ6长训练；核对发现旧截图数字属于Table5 ODR后，source应对比Table4对应行。
- 作用：同时记录论文与本次source、绝对/相对差、全样本覆盖与原接受子集；同一个spnv2-2改回每小时单次巡检。完整190轮及后续阶段保持，目标不恢复。
- 验证：两域与pilot的main/verify六个退出码均0；原/扩展独立proof通过且SHA实际匹配；lightbox6740/6740、sunlamp2791/2791；φ6正式进程身份、GPU和服务器刷新健康。论文已有结构化Table4/Table5参考由独立只读子智能体复核，主模型再次核对并计算差值；应用设置和完整提示逐字读回。
- 未修改实验代码、配置或参数，没有重跑、增加forward或重启健康进程。巡检游标/原始日志/数据/权重仍只保留在服务器。


## 2026-10-18：固定旧source完整补评，受保护暂停并续行φ6

- 位置：reproduction/old_source_reeval_20261018中的prepare/verify_comparison、三个独立配置、pause_controller及对应测试；AGENT与服务器交接说明。
- 原因：用户希望尽快把旧batch4不累积模型从历史拒绝样本补算变为同次完整保存，明确授权暂时暂停φ6后继续。
- 作用：旧权重原样重评三域；独立watchdog先验收再暂停，最多30分钟，失联或失败先清理自有评价进程再继续同一φ6进程，保留所有训练状态；不修改冻结训练/推理代码、生产队列或旧结果。
- 验证：旧final CPU strict reload与有限性通过；全样本gate8/8、暂停隔离/身份/清理顺序12/12，真实生产状态/锁与pidfd无作用自检通过。实际暂停已核原PID/start和GPU空闲；三域正式补评及恢复证据待自然产出后追加，不提前标完成。
- 运行路径与Git副本逐文件SHA256一致，见runtime_map.json；不提交模型、数据、原始日志或实时状态。


## 2026-10-18：完成旧source补评与同口径对比，确认φ6恢复

- 位置：reproduction/old_source_reeval_20261018/finalize_comparison.py、SOURCE_COMPARISON.md及reports/source_old_new_verified.json；AGENT与交接说明。
- 原因/作用：立即整理用户追加补评，严格核对旧/新模型同一评价代码、配置、数据顺序及全样本分母，给出论文绝对/相对差和历史补评差异；复用新模型结果，不新增GPU推理。
- 验证：三域main/verify六项退出0、原/扩展独立验收和N/N覆盖通过，CPU汇总及旧新对比退出0；原φ6同一PID/start恢复、后续样本增长、manager和队列绑定保持，controller/watchdog均退出0。已恢复小时巡检。
- 结果：lightbox新比旧综合误差减少5.52%，sunlamp增加1.85%；新相对论文仍高53.44%/45.81%。完整数值和逐样本证据路径入库；训练差异不作单因素因果归因。


## 2026-10-22 - Accepted main sources and fixed40; prepare Table2 pilots

Locations: reproduction/monitoring/phi6_fixed40_stage_20261022.json, PHI6_FIXED40_COMPARISON_20261022.md and fixed40_completion_20261022.json; reproduction/queue_proposals/table2_pilots_20261022/; AGENT.md and SERVER_EXECUTION_HANDOFF.md.

Reason/effect: archive accepted phi6 final/three-domain evidence and fixed40 actual runs; prepare only H/E/EH engineering pilots after stage49. Original49 definitions and evidence are preserved. Candidate52 stays inactive pending a new explicit manager-only authorization. Full190 epochs and remaining resource/scans/delivery remain incomplete.

Validation: actual source final/checkpoint/evaluation proof/per-sample hashes match; 40 ODR main/verify pairs exited0, independent proofs and N/N coverage checked read-only. Candidate build and independent CPU acceptance used distinct screens, exit0; existing validate_manifest/extend_state contracts passed, 49-prefix exact and 56 distinct new asset hashes matched. Main agent reviewed both scripts and independently compared prefix/hash/receipts. CPU acceptance is not a GPU pilot/scientific result. Logs/status/model/data remain outside Git. Production runner/manifest/processes were not changed.


## 2026-10-22 - Authorized Table2 pilot manager cutover

Location: reproduction/revisions/manager_table2_pilots_cutover_20261022/activate.py and acceptance records; monitoring/table2_pilots_cutover_20261022.json; AGENT.md and SERVER_EXECUTION_HANDOFF.md. Reason/effect: execute the user-authorized one-time transition from completed49 to accepted52 candidate, adding H/E/EH pilots only. Retain all original49 definitions/evidence and never rerun them. New independent manager screen uses existing GPU/exit/verification gates. Normal recovery uses only new --manifest, without one-time extension flags.

Validation: syntax compilation and independent static review before execution; controller own screen exit0; accepted receipt and independent readback confirm original49 exact records, old-manager exit/sole new lock owner, new manifest SHA and only pilot launch events. Actual pilot/independent proof statuses are recorded in the monitoring receipt; short pilots are not formal scientific results. No other process was signalled, no goal mode resumed. Raw before/after state, logs, checkpoints and data stay on server.


## 2026-10-22 - Minimal Table2 formal64 candidate (inactive)

Locations: reproduction/queue_proposals/table2_formal_20261022/ build_candidate.py, verify_candidate.py, proposal and evidence; AGENT.md and monitoring/SERVER_EXECUTION_HANDOFF.md. Reason/effect: append only H10/E20/EH20 and nine full source evaluations to completed52, preserving all52 definitions and execution evidence. Keep frozen train config/source/initialization, prepend same-forward observer only to new evaluation commands. H/E/EH modes and original accepted-subset versus extension denominators are explicit; no entropy ODR, missing values remain None. Candidate is not activated; a new explicit manager-only cutover authorization is required.

Validation: unique independent CPU build and verify screens exit0; existing validate_manifest/extend_state contracts, 52-prefix and execution evidence invariance, three train configs/independent final, nine original evaluation argv/output/GPU contracts, 85 actual asset SHAs, three CSV row counts, missing-head/None/full-coverage mode contracts and absent outputs passed. Main model reviewed both scripts and existing single-head entrypoints; independently checked candidate/proof hashes. No GPU run, production state write or process signal occurred. Prior pending commits were synchronized with exact identical trees through GitHub connector; original local commit identities retained in a backup branch.


# 003首轮探索实现与验收

用户已确认v2，授权有限探索：36次基础ODR，有合格趋势时最多18次随机预算对照。目标模式暂停；剩余150轮source消融保留为可选扩展。

官方φ3-BN EHS offline权重SHA256为d8f9c281d23adaeb77d03023aee8d40ae1cb9e18dd503539c4719bbe719514cc，已通过1550张量严格加载和有限性验证。来源展示采用作者发布入口与文件哈希。

数据边界固定：两域各512开发，lightbox6228/sunlamp2279预留；seed20261022整组划分；3条无放回N1024观测流使用2021/2022/2023。图像相似性筛查不能证明采集独立。适应和门控不读取GT。

CPU合同：风险20项、数据9项、门控状态7项、坐标/HIL5项、runner真实资产入口9项通过。真实GPU验收b的82项严格检查和实际状态文件独立CPU读回通过；ALL与冻结A逐图输出/损失/完整状态相同，mixed拒绝/尾部及恢复相同。首次验收a失败记录保留：首图未选中bbox出现6.103515625e-05差异；固定输入诊断证实原TorchScript解码器冷/热图差异。使用6次纯算子合成输入预热统一执行起点，零容差保持，不读取额外图像、不修改权重/优化器或原A数学。

实现位置：implementation/gate_core.py（固定共享分位门控）、state_adapter.py（保留原A更新与BN命名行为）、prediction.py（无GT评分和独立评价边界）、data_boundary.py（冻结图像清单）、risk_metrics.py（固定O/A参考与预设筛选）、run_exploration.py/verify_exploration.py（单任务及独立验收）、phase_tools.py（校准及汇总）。

先执行两域官方O开发评价和共享校准；实际校准SHA确定后，再绑定36次基础探索。不提前写入科学结果；每项main成功退出、严格重载、完整开发记录与独立proof均通过才推进。

候选首次构建因manager禁止绑定项目外数据集相机路径而拒绝，尚未发送任何进程信号。修正只让manager检查项目内资产，相机仍由运行入口和独立验收逐次核对SHA；原失败候选保留。


## 2026-10-23 ODR003 execution repair and frozen36-run stage

The independent003 plan uses an unchanged copy of the validated queue engine with an explicit plan-directory launcher holding both legacy and current plan locks. Historical52 completed tasks and failed preflight remain preserved. Fixed CUDA initialization before peak-memory accounting, verified without model forward. Repaired CPU confidence recomputation to preserve the original float32 quantile operation; all512 saved lightbox predictions were independently accepted without repeating forward. Both official Offline developer512 evaluations and shared calibration now accepted. Prepared and independently validated3->40 manifest adds36 paired-stream adaptations plus one summary; no conditional random runs yet. Implementations/configurations mapped under reproduction/odr003_exploratory_20261023; source weights, images, labels and raw tensors excluded.


## 2026-10-23 ODR003 exploration completed

All36 paired-stream base adaptations, both developer Offline evaluations, shared calibration and independently recomputed38-run summary passed. Neither shared G setting satisfies the preregistered cross-domain risk directions; conditional random controls not run. G50 lightbox mean positive degradation -21.93%, tail -30.19%; sunlamp mean -2.68%, tail +3.96%. Developer-only report, resource/coverage tables, paired-stream metrics and posthoc diagnostic cases exported; no reserved-GT evaluation or extra model forward. Saved data images/contact sheet, weights and full per-image raw summary remain local/server; Git contains code, scientific numeric plot, concise report and acceptance hashes. This completes the confirmed exploratory scope, not historical190-epoch reproduction.


## 2026-10-23 - G50-P transactional protection and shared-host engineering queue

Location: reproduction/odr003_round2_20261023/ mirrors the independent runtime plan byte-for-byte. guard_pools.py and frozen_backend.py implement past-only uniform/fragile reservoirs and HE diagnostics using the unchanged original decoder. protected_learning.py/protected_engine.py add full-window commit/rollback and original G50 gating. durable_state.py makes complete fsynced generations authoritative, separately charging replayed physical work. reference_driver.py preserves the original adapter as an independent engineering comparator. round2_protocol.py fixes the18-run matrix, RW probability control, RP exact budget and predeclared hold criteria. verify_round2.py/verify_engineering_phase.py independently reconstruct saved decisions, pools, counts and full recovery states. host_gate.py/host_snapshot.py/guarded_entry.py/queue_round2.py admit work only on an idle shared GPU, retain exact own-process identities and hold project locks; foreign processes are never signalled. Eight engineering tasks are prepared; complete experiments remain inactive templates until real engineering proof passes.

Validation:227 local CPU synthetic/mock tests passed, including original/disabled equivalence, partial/candidate recovery, durable-generation faults, scientific de-duplication and physical replay cost, pure scoring, pool ordering, fixed rules, RW endpoints/RP budgets, developer-evaluation resume/coverage, shared-resource admission and bounded queue recovery. No real SPNv2 forward or shared-host model validation occurred during this preparation. This is implementation evidence, not GPU acceptance or scientific efficacy. The confirmed plan and local proof hashes are archived; source weights, images, target labels, raw run logs and tensors remain outside Git. Existing original8 modifications, old checkpoints and related papers are preserved.


## 2026-10-23 - Private atomic registry metadata

Location: reproduction/odr003_round2_20261023/implementation/guarded_entry.py and test_private_json.py, re-bound eight engineering tasks/index and local contract proof. Read-only monitoring found that the shared host umask0002 created registry.json as0664 while the ownership collector correctly rejects group-writable trust records. The atomic writer now creates its temporary inode with O_EXCL and explicit0600 before any write, fsyncs and renames without weakening trust checks. Existing own registry permissions are repaired only after identity/schema/content verification. A necessary manager-only cutover can apply the new code bindings with no worker or scientific run to repeat.

Validation: three new local writer regressions plus the full230 local CPU synthetic/mock contracts passed; source, data, thresholds, eight engineering definitions, run IDs and all non-code task fields remain identical. No shared model forward or heavy validation was started for this repair. Previous code/manifest/status/permissions are retained in the scoped revision directory.


## 2026-10-24 - Isolated preprocessing RNG, complete checkpoint repair and autonomous continuation

Location: reproduction/odr003_round2_20261023/implementation/frozen_backend.py and preforward_repair.py. The first real engineering reference stopped during image_read because Albumentations1.1 probability-one Normalize/ToTensor composition still consumes Python RNG. No model forward or scientific observation/update occurred. The original complete zero checkpoint, actual exit1, uncertain non-model I/O and all source evidence are retained. Data RNG is now independent and checkpointed with per-image replay points; historical reconstruction does not advance the observation stream or mutate model RNG. A guarded independent CPU migration restored the actual entire learning/model/optimizer/RNG tree into a new execution directory with only identity and empty provider schema changed; strict source tensor equality and independent roundtrip passed. No new scientific trajectory was added.

queue_round2.py, queue_control.py and complete_phase_tools.py implement the user-authorized300-second wait cadence, exact privately addressed request/ack immediate recheck without restarting a healthy manager, and engineering8 -> acceptedP6 -> count-matched RW/RP(max12) -> independent summary/delivery. verify_complete.py and summarize_round2.py independently recheck real exits, full checkpoints, developer512 predictions, unchanged511/512 reference denominators, reused O/A/G50 identities, all improvements/regressions, actual RW budget differences, RP exact budget and costs. All complete-stage entry points were prepared before GPU engineering resumed. New model/schema/adaptation policies were not introduced.

Validation:312 local CPU synthetic/mock contracts passed; real guarded zero-checkpoint import screen exit0 with zero model forwards and unchanged actual learning-state tree; fourteen existing O/A/G50 report/code bindings and six paired baseline reuse records verified by metadata only, without repeated inference. Actual GPU engineering and scientific results remain gated and separately recorded. Frozen original source, data split, seeds, thresholds, maximum18 complete runs and research hold conditions stay unchanged; historical150 source epochs remain deferred.

## 2026-10-24: preserve completed engineering runs across auditor-only repair

Changed reproduction/odr003_round2_20261023/implementation/{artifact_binding,audit_revision,queue_round2,verify_engineering_phase,verify_complete}.py and scoped metadata/queue regression tests. Fixes false rejection of identical frozen path/SHA records with extra bytes metadata. Separates immutable producer code from repaired audit code; retains actual exit1, old task, reports/checkpoints and receipts, and prohibits rerunning the eight completed trajectories. No model/ODR scientific code changed. Validation:335 local CPU synthetic/mock contracts passed; shared-host real CPU acceptance is a distinct required step. Runtime and reproduction files match byte for byte; revision manifest binds preserved evidence.

## 2026-10-24: admitted read-only saved-state diagnosis

Added reproduction/odr003_round2_20261023/revisions/auditor_binding_repair_20261024/diagnostics/ after the repaired real CPU auditor exposed strict original/disabled state divergence. The isolated diagnostic retains fresh double resource sampling, three project locks, exact launch/nonce registration and private outcomes. It hashes/deserializes already-saved CPU states, compares initial/64-observation/final state and signals, and never constructs a model, forwards, reads target GT or produces acceptance. Four local CPU record tests passed; runtime copies are byte-identical. All eight GPU results and both failed verification attempts remain preserved; P has not started.

## 2026-10-24: user-approved deterministic execution and30-ODR baseline matrix

Location: reproduction/odr003_round2_20261023 implementation and revisions/deterministic_20261024. Reason: saved initial state and inputs matched while first GPU output differed before updates; strict equivalence/recovery blocked. Change: enforce deterministic FP32 backend before model work and recheck full backend/version on restore, retain original A/G50 reference updates, add source-only O checkpoint/evaluation, new A6/G506 and independent v2 acceptance, freeze original six reference lists, automate all32 outputs (30ODR+O2) and independently audit full summary. Historical eight runs, failed audits, CPU migration and65-second diagnostic are retained, costs separately bound; no source retraining or threshold/data changes. Validation:373 isolated Windows CPU synthetic/mock contracts passed with no code drift; two agent cross-reviews; metadata-only server validation verified50 implementation files,8 unique new engineering tasks and6 original reference-ID lists. All mapped runtime files have identical SHA. Scientific model results require actual independent GPU engineering and per-run proofs, not these local checks. A Windows/POSIX staging separator error failed before model work and was preserved and corrected; deployment subsequently passed without a model forward.


## 2026-10-24 — G50-P隔离内存候选，验证后不部署

- 在 reproduction/odr003_round2_20261023/performance_candidates/ 保留原版和仅优化上一代checkpoint False校验分支的流式候选；未修改50个当前绑定实现、任何任务/协议/检查点或健康worker。
- 完整P的guarded attempt2203.3268秒、确认操作385.6070秒、26次完整checkpoint213.0093秒；差额不全归因fsync。候选保留全部字节SHA/长度验证和持久化顺序。
- 7组存储合同、4组36图合成CPU科学状态/双中断恢复严格相等；文件fsync真实执行且次数一致。真实SPNv2/GPU新增forward0、ODR/O新增0。
- 固定16/64 MiB各版本3次完整save+fsync+recover，无可信净提速；64 MiB完整周期慢约0.36%，仅保存峰内存少约62 MiB。Windows不支持目录fsync，结果不代表Linux/GPU端到端验收。
- 本轮继续原producer 3ef5cbd3；候选仅归档，无安全混合版本准入且生产端到端收益未证实。细节、SHA及工程成本见候选REVIEW和结构化验收记录。


## 2026-10-25 G50-P update-risk diagnostic v1.0

The completed thirty-ODR plus two Offline study remains closed. This separate diagnostic authorizes A: existing 793 candidate/117 rollback records with zero model forwards; B: exactly 48 frozen paired windows, at most 40000 physical forwards and 14400 active seconds including all reconstruction, verification, state transfers and durable IO. Six 128-image panels exclude each stream's 1024 adaptation images. Frozen input SHA: 6608ffca250fe9733e1cd244e39eac84146ecd564b097ea101c7d2087f2d63e5. First included pilots are lightbox2021 candidate1 and sunlamp2021 candidate9; strict scientific reconstruction and independent receipts are required before the other 46. No complete ODR or new method experiment is authorized.

Implementation in reproduction/g50p_risk_diagnostic_20261025 adds isolated reconstruction, full-window-base rollback, state-pure branch evaluation, independent pair verification, fixed poststratified reports and durable global cost accounting. It imports the unchanged original 50-file strict deterministic producer 3ef5cbd32f16bcfcb650f4b5d7a04a69e718aa47; the archived performance candidate is not deployed. Shared-host two-snapshot admission, three old pipeline locks plus the new lock, exact nonce/process identity, and fail-closed budget enforcement remain mandatory. Code-only revisions cannot reset the stable project/frozen-input budget identity.

Local validation: A11 pure-record checks; B11 then22 synthetic correctness/recovery checks, plus3 receipt and2 trace-proof metadata checks; budget8, entry5, manager9 and weighted metadata2 checks. All passed; previous failed harness attempts remain recorded. Conservatively charge all necessary local validation 4748 physical forwards upper (including repeats and2 unknown dispatches) and360 active seconds upper to B before server work. This is not real SPNv2/GPU pilot acceptance. Real A analysis and B pilots are pending at publication; runtime status and independently verified receipts, not this source commit, establish progress. No old result/checkpoint/task binding is changed. Budget exhaustion yields actual partial results and ends the finite stage. Goal mode stays paused.


## 2026-10-25 diagnostic import-bootstrap correction

Experiment A extraction and analysis actually exited0 with passed zero-forward proofs. The first B window stopped before any model forward because prediction imported utils before the verified source core path was installed. Both paired runner and independent verifier now use the original producer order: import non-prediction modules, verify bindings/configure core paths, then import prediction. Frozen producer50, science,48 window selections and128-image panels are unchanged. Two targeted import-order mock tests passed with0 model construction/forward (3.453s); manager metadata contracts separately validate preserved A receipts and versioned partial delivery. These bounded pure-metadata tests remain inside the original360s local-validation conservative upper charge; no initial receipt or budget was reset.

The failed actual attempt retains its original task/code/index/exit/failure data and11.825699580833316s active-time upper charge. The same lightbox2021 window1 receives only a new technical run identity ending_importfix1; it is not a new window/ODR. Old A tasks/proofs stay byte-identical and are reused through explicit hash-bound preprocessing receipts. New pending B tasks reside in tasks/import_bootstrap_20261025. A complete original index and changed-code snapshots are preserved under revisions/import_bootstrap_20261025/before. A new manager is allowed only after exact prior process exits and locks were verified; state migration retains all prior attempts and the original project/frozen-input budget. Real two-window acceptance remains pending, never replaced by these local tests.


## 2026-10-25: bounded second-pilot trace-difference capture

Add an isolated technical probe under reproduction/g50p_risk_diagnostic_20261025/revisions/first_trace_difference_20261025. The production manager and accepted A/first-pair bindings stay unchanged. The probe reuses the exact host admission/nonce/locks/pidfd deadline and original global budget, restores the frozen parent at observation64, and permits at most one previously unpersisted model forward. It saves the first differing fields and complete produced state, then stops without scientific acceptance or downstream dispatch. Existing failed attempt and costs remain intact. Syntax validation passed; actual GPU evidence remains pending until this bounded probe executes. User changes to AGENT.md/AGENTS.md are retained separately.


## 2026-10-25: targeted scripted rotation recovery check

The first-difference probe exited0 and persisted one observation: only five E rotation matrix elements differ; complete parent state and other trace fields match. Add a one-forward correction check that primes the unchanged scripted rotation operator to the already-hot state of the frozen parent at observation64, captures its raw six-dimensional input, and runs a separate cold/hot operator control on that saved input with zero additional model forwards. Require the cold result to equal the preserved failure, hot result to equal the original producer, full observation strict equality, and unchanged scientific learning state. This fixes no production binding yet and admits no downstream experiment. Initial probe cost42.11670124065131s and1 forward remains in the original budget. Syntax validation passed; actual correction evidence is pending.


## 2026-10-25: restore the original scripted rotation runtime at diagnostic recovery

The failed sunlamp2021/w9 replay differs only in five E rotation matrix floats after restoring an exactly equal parent at observation64. A bounded1-forward probe persisted the first difference. The next bounded1-forward check captured its raw [1,6] tensor: a fresh original scripted rot_6d_to_matrix gives exactly the failed values, and its subsequent calls give exactly the frozen producer values. A [1,1,6] warmup failed because TorchScript specializes by input rank; this unsuccessful check and its25.367898621596396s charge are preserved.

The paired runner now restores the unchanged operator call phase with a fixed [1,6] dummy tensor and no model forward or RNG draw. Zero-observation parents remain cold. It also persists exact first-difference fields on future trajectory errors. The manager explicitly reuses the accepted first pair through its original task/index/code snapshot, actual launches/exits, report and independent proof; it cannot relaunch that preserved result. A remains bound and reused. The second logical window receives only a new technical ID ending_rotfix1. All48 selections, panels, original producer50, scientific equations, tolerances and global budget remain unchanged.

A zero-forward real-device check passed exact cold-failure/hot-producer comparisons on the saved raw input, RNG purity, zero-parent cold behavior, complete fixed task validation, preserved-pair lineage and rejection of an altered proof. Its first harness attempt stopped on a missing round2_protocol import before model/GPU work; that code, task, outcome and charged cost are retained; the corrected bounded checker exited0. Production two-pilot acceptance remains pending until the repaired main and independent verification actually pass. Candidate index SHA37cd266e0dd0dc3bc85afd2ce3184377d66c132fd89a03cc257b32f28f32e5f5; targeted proof SHA4b78a802f97897d66fcabb596c572c6d76043b64090ea0e45ff895f1ed8cbffb. Also archive the user-updated standing efficient-execution policy already present in AGENT.md/AGENTS.md.


## 2026-10-25: version audit dispatch after accepted pilot recovery

Both fixed paired pilots have actual main and independent verifier exit0/passed proofs. Pilot-wide admission then failed before host admission or model import because the audit template still carried pre-rotation-repair implementation hashes. Preserve the failed immutable task, wrapper exit1/log, absent owned outcome and all accepted science. Version the pilot/summary task and receipt paths, bind both templates to the current implementation, and reject stale templates at index validation. Add explicit preservation of the second accepted pair while leaving both original task/proof bindings intact. No paired model result is rerun.

The candidate and a zero-forward metadata checker pass syntax review. The real checker returned resource-wait75 before imports because another GPU computation was present; this is not validation success. A dedicated bounded repair controller uses the existing fresh resource gate/locks/nonce/budget before each necessary checker attempt, waits300s on resource75 only, stops on any actual failure, and deploys only after a passed bound proof. It then launches the normal fixed queue, which must still pass pilot admission before the remaining46. The controller never signals others or restarts healthy workers.

The pre-admission failed dispatch had no original budget allocation. A separate conservative control-plane receipt adds120s overhead upper plus actual settlement time and2s finalization allowance (122.03320651315153s total) to the same hash-chained budget via ordinary no-forward attempt events. This is an explicit conservative accounting debit, not a claim of measured model execution. Necessary admitted metadata validation is additionally charged normally. Limits/initial receipt/history are unchanged. Scientific producer, selections, panels, data and numerical rules are unchanged.


## 2026-10-26: explicitly remove cumulative diagnostic time stopping

User authorizes continuing the same frozen48-window G50-P diagnostic without the14400s cumulative active-time hard limit or its derived termination. Preserve original protocol/identity bytes,21 accepted pair proofs, A/pilot proofs, budget stop, all failed attempts and every prior charge. A hash-bound v2 budget policy is appended to the existing journal; the original journal prefix24913 events/SHA4ec7ff95935e86997fb90e9b2abe76afbe4ff19b79ac57c97c815e55e4dfe689 is not rewritten. Original reserved17150/confirmed12402 forwards and charged time14400 remain intact.40000 physical forwards remains hard.

Budget/entry remove time-based admission, check, close, final-receipt and watchdog termination only under this explicit policy. Unlimited attempts have no deadline/allocation upper; an unresolved execution remains unknown and blocks duplicates. Closed measured time and conservative additions are separate. The exact-pidfd watchdog only monitors exit when its deadline is null. Slow close-fsync exposed one leftover legacy2s stop in a new targeted test; it was fixed before any scientific execution. Eight final zero-model local contracts pass, including beyond-old-deadline execution, strict40000 cap, policy tamper rejection, unresolved accounting, slow-finalization accounting and no watchdog signal. Earlier validation failure/cost is retained.

Actual admitted server metadata validation exited0 and proof24f6cb2abd548f7b284bd57118cd4929f94e7a07758b449367935e3726b371dd passed with0 model forwards, no deadline and preserved real ledger/identity bytes. New manager preserves A across only the changed time infrastructure using old execution-code snapshots and retains all21 original pair bindings. Pilot admission is reused, not rerun. Pending tasks/templates bind the same scientific producer/inputs and amended time policy. Scientific code,math,loss,thresholds,pools,seeds/software and GT boundaries do not change.

Resume the stopped lightbox2023/w87 under technical ID ending_notimelimit1 from saved complete observation512/candidate77. Immutable state.bin is linked; journal/CURRENT/metadata copied independently. No persisted candidate/branch prediction exists. Candidate78 onward in the old diagnostic journal was not complete-checkpoint-persisted and must replay with new cost. Its prior191 forwards and212.285456080921s upper stay in the original ledger. Remaining26 untouched windows follow original order. Formal summary/final acceptance remains required after all48; physical-forward exhaustion instead yields actual partial scope.


## 2026-10-26: propagate amended policy to generated pair verifiers

The resumed lightbox2023/w87 main actually exited0 and persisted full exact trace/state and both128-image branches (599 additional physical forwards). Its generated independent verifier omitted budget_policy and was rejected by host_entry before admission/import/model work, wrapper exit1 with no owned outcome. Preserve that failed immutable task/log and cost; do not rerun the successful producer.

The manager now copies the current budget policy into dynamic verify_pair tasks and versions their identity/path by current index SHA. Add strict preserved_main lineage for the one completed producer awaiting verification: original task/index/code snapshots, exact report/decision/GT/purity/state claims, actual unique main exit0/outcome/launch/nonce/report bindings. This only selects independent verification; it never adds accepted. All21 old accepted pairs/A/pilot remain unchanged;26 unstarted definitions get only the new control-code binding. Budget/entry/verifier science code and time-unlimited policy remain byte-identical.

Actual admitted zero-forward metadata check passed: the real generated verifier passes the unchanged host validation; the original missing-policy variant still rejects; old failed task remains immutable; tampered preserved main rejects; accepted count stays21 and producer is not reexecuted. Proof SHA c01ceef119747af79eee2fb60e5e4e74baafa22345f6d067e61663b5bbdc796a. Real independent verification is still required after safe cutover. No mathematical/GT/window/panel/40000 limit change.


## 2026-10-26: tolerate the owned worker terminal-evidence handoff race

The manager exited1 after PermissionError reading its own worker proc/environ at the worker exit boundary. The sunlamp2021/w16 worker itself exited0 and saved both128-image panels plus the complete matched-state report. Preserve the crash log and actual terminal evidence; metadata reconciliation records that successful main without model replay and without independent acceptance.

Manager finish now immediately rechecks wrapper/outcome files when a proc identity probe encounters PermissionError, FileNotFoundError or ProcessLookupError. An unreadable identity is unverified, never treated as alive/dead/owned. If terminal evidence is incomplete, the same active attempt remains, with no downstream dispatch; a30-second terminal-handoff grace ends in explicit blocked if still incomplete. Complete terminal evidence uses the unchanged strict outcome/independent-verifier path and does not require an exited PID to stay alive. This is no cumulative scientific runtime deadline and sends no signal. Other exceptions remain visible.

Two focused local zero-model contracts passed: PermissionError with terminal arrival in the same call, and persistent unreadability with active retained then blocked. The actual shared-host metadata check returned resource-wait75 before imports because another GPU task was present; this is not site-validation success. A dedicated controller waits300s on resource75 only, requires the bound targeted proof before safe deployment, and then starts the ordinary queue to verify the saved main before further windows.26 accepted pairs plus the completed pending main are preserved; all science/data/budget-policy/40000-forward boundaries remain.


## 2026-10-27 存储保留修订
新增 versioned storage_management_20261027 writer：显式磁盘预估、最近两个有效恢复点、冻结引用pins、发布后清理与删除审计；5项零模型测试通过。更新AGENTS/AGENT及实验计划模板。原冻结科学实现和验收绑定未修改；实际清理和迁移记录在服务器storage_audits与migrations目录，目标运行接入仍需版本绑定。

实际清理三阶段已exit0：684个已审查无引用中间检查点，释放511051304960字节文件块；保留冻结父/匹配、最终状态和恢复点。约13.12GB中断/预处理恢复现场单列未决。归档三个受锁保护执行器与证据哈希清单，旧预算前缀不变，零模型前向，新增维护费用已单独追加同一账本。


## 2026-10-27 中山大学迁移与保留策略接入候选
新增 sysu_migration_20261027 版本化namespace/bootstrap/动态单任务配置、原manager最小派发修订和带源码围栏/预算前缀/资产校验的自动续接控制器；原host_entry资源/身份/预算/严格科学验收及producer50文件未改。存储适配器复用旧序列化器一次，state.bin字节一致，按授权root保留最新两个版本与pins。实际零模型namespace与独立screen退出0，补齐2171张冻结派生图像缓存并逐张SHA通过。传输v2以已验收不可变基底做无损差分，真实1.304GB对象编码1.213MB并还原至原始SHA，不改科学数据、不增加模型前向。源空闲控制器按精确PIDfd身份退出143，源派发锁由独立保护进程持有，旧账本最终前缀保留；目标科学计算仍待迁移/实际准入与跨GPU严格重合。所有源码和失败/验证日志在本项目migration目录留存。


## 2026-10-27: resume through verified window dependencies during background migration

Replace the all276-assets deployment prerequisite with exact per-window receipt/stat gates. Independently verify the already persisted sunlamp2021/w16 main first; keep original frozen order and wait300s for the next missing dependency before the unchanged host resource gate. Native controller publishes only verified immutable hardlinks without moving receiver staging/bases; mutable metadata is copied independently. Namespace repeats the task readiness/binding checks before heavy imports. Accepted26 producers and all source originals/proofs remain reused.59,036,584,274 bytes of accepted-state replicas can transfer later for reproducibility; no scientific bytes or acceptance tolerance change.

Versioned snapshots and mapping are reproduction/sysu_migration_20261027/pipelined_v1.11 focused zero-model contracts and independent receipt/publication checks passed. Old idle deployment controller exact identity was stopped with actual wrapper exit143; scientific workers and background transport were not restarted. Runtime switch still requires the real bound namespace validation and original admission; no target science acceptance is claimed by unit tests. Source ledger/fence and40000-forward cap remain intact; time stopping remains disabled, pending migration/control costs retained separately for one safe final settlement.


## 2026-10-27: capacity-based shared GPU admission for the existing diagnostic

Add version-bound shared_gpu_20261027 host gate/snapshot/entry and exact resource-amendment lineage. Replace foreign-process and total-used-memory rejection with6144MiB free GPU and40960MiB host MemAvailable thresholds, checked in two fresh snapshots. Retain ownership, duplicate protection, four project locks, fresh registration/readback, single dispatch and unchanged forward/time policy. Runtime conflicts continue identity checks without treating foreign arrivals or our own already allocated memory as a stop. No process-control operation is authorized by the gate.

GPU estimate4096MiB+safety2048MiB is conservative relative to the actual27 completed diagnostic reports and six original P reports (max PyTorch reserved2718MiB). Host32768MiB+safety8192MiB is a configuration-based planning estimate; no historicalRSS peak was found. Public proc/meminfo samples replace guesswork at admission, without model probing. Ten focused zero-model gate/entry tests pass. Original N host modules, all mathematical producer files, A/pilot/26 accepted records and saved w16 main remain unchanged; actual deployment validation and strict scientific acceptance must still be evidenced after safe idle-manager-only switch.


## 2026-10-27: CUDA runtime compatibility in the preserved migration namespace

Actual shared-capacity admission succeeded with two foreign GPU jobs and w16 independently exited0 (27 accepted). The next frozen w23 attempt exited1 before any model forward: CUDA driver initialization returned OS error304. Devices and real libcuda exposure were intact; the inherited proc mount was read-only. A bounded same-thread comm metadata comparison in the exact failed task namespace returned EROFS with old mounts and passed with a native proc RW bind; comm bytes, UID and full-command identity remained unchanged. No Torch/CUDA/model probe was used for that comparison.

Version the namespace/bootstrap/config extension separately to add only --bind /proc /proc, preserving native permissions, host PID view and all explicit read-only dataset/GT/scientific mounts. Original shared gate, manager/host_entry, mathematical code, policy6144/40960, forward40000/time-unlimited policy and all27 accepted records remain unchanged. Reconcile only the specifically proven zero-forward failed w23 attempt after actual exit/closed-budget checks; preserve its failure and cost, then use the normal admitted queue and strict science acceptance. The background immutable asset publisher is untouched.


## 2026-10-27: explicitly authorized cross-GPU numeric acceptance

User authorizes revised scientific acceptance. Freeze abs1e-6+rel1e-5 times expected magnitude, finite values only, only for cross-GPU floating trace/state comparisons. Types/shapes/IDs/order/decisions/counters/RNG remain exact; all same-run whole-window rollback and branch/provider isolation use the original exact comparator. No model mathematics/loss/optimizer/GT/statistical threshold changes. Versioned comparator and narrowly edited diagnostic runner/verifier are explicit execution overrides in an immutable acceptance amendment; original implementation files and27 accepted proofs stay intact. Reports/proofs state tolerance-based acceptance and do not claim bitwise equality.

Saved actual first difference passes the frozen rule; changed decisions/counters/RNG, nonfinite values and out-of-bound floats reject. AST checks confirm original model/runtime/branch/metric/budget functions unchanged. Reuse saved complete recovery points, preserve the failed physical forward and every previous cost, then ordinary two-snapshot shared-capacity entry resumes the remaining frozen windows.


### Numeric acceptance derived-angle defining domain
The first authorized run passed the old stop, then exposed acos amplification in rotation_disagreement while raw poses and the discrete decision remained equivalent. Preserve that failed run and all3 additional forwards. For this specifically defined derived angle, compare cos(degrees) in its dimensionless defining domain under the SAME abs1e-6/rel1e-5 bounds; raw rotations/quaternions still compare directly and decisions/RNG/internal rollback remain exact. No GT, model, gate threshold, loss or numeric bound is changed. Version1 bytes and policy remain archived; the new scientific acceptance revision is explicit.


### Coherent protection derived-record validation
Retain original abs1e-6/rel1e-5 primary bounds and exact decisions/RNG/isolation. Protection HE angle values use their cosine domain; redundant HE-delta rule records must independently reconstruct value, compared value/threshold and predicate exactly from their own pre/post inputs. Only after that check is the redundant representation normalized for cross-device comparison; persisted original diagnostics are unchanged. Forged derived values and changed decisions reject. This removes repeated rejection of the same acos round-off without widening primary bounds or changing any gate threshold or mathematical execution. Reports explicitly distinguish policy acceptance from bitwise/full-raw equality.


## 2026-10-27: admitted zero-forward saved-state diagnosis
The actual w23 run persisted both128-panel branches, candidate and complete192-observation checkpoint, then stopped under the fixed numeric policy. A registered, twice-admitted CPU-only inspector compared exactly those saved states without model construction,images,GT or model forwards. The first exceeding tensor is learner.learning.bn_features; maxabs2.288818359375e-5. Diagnostic-only omission of the two bn_features cache fields establishes all remaining state passes the current policy; this is not an acceptance waiver. Inspecting original adapter confirms these are forward-hook feature inputs used for BN mean/mean-square accumulation; actual model/optimizer/gradient/RNG/pool/provider fields passed. Further cache acceptance change requires explicit scientific decision. Preserve all saved panels and real failed cost before any recovery; no model replay. Initial inspector registry/output/JSON metadata errors are archived with real exits and costs; final inspector exits0.


## 2026-10-27: explicit BN-cache channel-statistics approval and saved-work completion
User specifically approves only cross-GPU BN temporary feature caches to compare channel means and mean-squares using unchanged atol1e-6/rtol1e-5. Cache key/shape/dtype/finiteness stay checked; verification accumulates moments in float64. Other parameters/optimizer/gradients/RNG/decisions and same-run exact rollback/isolation are unchanged.

The w23 GPU attempt actually completed candidate, two128-image panels and complete192-observation checkpoint, then failed only final cross-device cache comparison. A narrow finalizer reuses those immutable artifacts, original exact-isolation control-flow evidence and actual failed exits; it constructs no model, reads no images/GT and issues0 model forwards. Original failed costs are reconstructed read-only from the hash-chained ledger, kept as failed and independently audited; a fresh real finalization exit0 and fresh independent verifier remain required. No old outcome/exit is rewritten or refunded. Matched state reuses an immutable checkpoint hardlink, with only the pre-existing ledger_boundary normalization.


### All-layer BN statistics diagnosis after explicit approval
The new zero-model finalizer reached statistical comparison and correctly stopped on a BN channel mean outside the unchanged bound; no panel or model was rerun. One registered, admitted CPU-only inspector checked all290 cache tensor instances /105984 channels in the existing matching checkpoints. All mean-square statistics pass(maxboundfraction0.08548); maximum channel-mean absolute difference3.7535404171507025e-6, worst boundfraction2.08923. Fixed acceptance is not bypassed. A proposed change only to BN mean atol1e-5 is recorded unapplied and requires user authorization; other boundaries remain unchanged.


## 2026-10-28: explicit BN-mean-only atol approval
User authorizes only BN temporary-cache channel-mean absolute tolerance1e-5. Its relative tolerance stays1e-5; mean-squares and every other numeric/discrete/RNG/internal restoration boundary stay unchanged. Scope is selected by exact bn_features/channel_mean path, not global tensor tolerance. Targeted tests accept the approved mean difference while rejecting the same ordinary weight change, an out-of-bound mean-square, changed RNG and changed decision. Existing model work and two128-panel files remain immutable; reuse the saved zero-model finalizer and preserve its previous real failed exit/cost in the new recovery manifest.


## 2026-10-28：用户授权几何预测对象范数验收与w31尾段续接

- `reproduction/numeric_acceptance_20261027/numeric_compare.py`：仅完整姿态预测记录中的旋转矩阵、平移、四元数及框按对象范数比较，系数仍为atol1e-6、rtol1e-5；参数/梯度、BN统计、离散决定、RNG及同运行完整回滚不变。保存原始值，不修改指标或保护阈值。
- `pair_runner.py`、`recover_saved_main.py`及`resume_geometry_norm_20261028.py`：复用w31已完成两支128面板和完整283观察检查点，仅续接未持久化尾段。真实失败exit1和原账本费用保留，由独立验收审计原hash链，不伪造旧成功。恢复状态用不可变硬链接保护，原始科学代码未改。
- `versions/v5/`保存原执行代码；计划revision_v6/policy_v6及恢复manifest明确新旧绑定与回退入口，28个既有验收结果保持原proof。
- 定向零前向验证：真实首差异通过；超界几何、NaN、离散决定变化被拒；模型参数逐元素、RNG严格、BN两个统计容差均保留。正式运行仍需原资源/身份/预算准入及真实独立验收，不能把元数据测试称科学窗口通过。


## 2026-10-28：用户授权仅派生门控分数精确重算自洽

- `reproduction/numeric_acceptance_20261027/numeric_compare.py`：仅observation记录中的派生G50门控分数，先用各自原始arrival及冻结gate_core/calibration精确重算分数、阈值判断和原因；通过后仅在比较副本中排除冗余分数。原始值不改，原始输入仍按已批准规则比较，实际离散决定/阈值不放宽，模型、优化器、梯度、BN及RNG/同运行回滚标准不变。
- `resume_gate_score_20261028.py`：精确核对V6停止、真实exit1、无活动科学进程的原锁边界及连续账本，保留两个实际失败及全部费用；复用已保存283观察检查点与两支128面板，仅续接未持久化尾段。旧V6比较器和续接代码归档在`versions/v6/`。
- 定向零前向验证：保存的ECDF排名跳档首差异通过；伪造自身分数、修改接受决定、模型参数超界和RNG改变仍拒绝；BN容差保持。正式窗口验收以新main及独立verify真实退出/proof为准。计划revision_v7/policy_v7绑定实现、授权、旧证据及回退路径，28个已验收proof不改。


## 2026-10-28：用户授权旋转相关固定0.01度跨GPU验收

- `reproduction/numeric_acceptance_20261027/numeric_compare.py`：仅完整姿态预测的旋转矩阵及四元数、派生旋转角改用0.01度。矩阵使用原始Frobenius弦长`2sqrt(2)sin(theta/2)`，四元数使用归一化符号等价弦长`2sin(theta/4)`；float64比较避免近零角acos放大，不投影抹去矩阵误差。形状、dtype、有限性、正交/正定向及非零四元数核验；其他数值、实际离散决定/阈值、RNG、同运行回滚/分支隔离、GT指标均不改。
- `resume_rotation_degree_20261028.py`：按真实退出和原锁安全续接w54的完整448观察恢复点，保留旧33前向失败及成本。29个既有验收结果直接复用；使用已固定的初始化校验回执，避免后台传输刷新元数据影响运行绑定。
- `recover_saved_main.py`：恢复回执只列实际存在并复用的面板，w54在目标候选前失败、尚无面板，不虚称复用面板；真实失败成本仍从原hash链重建并独立审计。旧V7源代码在`versions/v7/`。
- 定向零前向验证：真实w54首差异通过；旋转/四元数0.0099度通过、0.0101度拒绝；符号等价、零四元数、反射/缩放矩阵、平移/参数/RNG及实际决定边界均核验。实际main/独立验收仍以真实退出和proof为准，不能用本测试冒充通过。


## 2026-10-28：w54已保存完整状态的零前向差异定位

- `reproduction/numeric_acceptance_20261027/inspect_w54_saved_state.py`：在原四pipeline锁、manager锁、两新鲜shared heavy-CPU资源快照、精确登记读回及原GlobalBudget零模型attempt内，比较w54保存的512观察完整状态与原冻结match；不加载图像/GT、不构造模型、不改变任何数值标准。修复诊断器对BN虚拟通道统计路径的只读解析，使用当前字段实际绝对容差，先持久化首差异避免详情构建失败丢现场。实际CPU诊断exit0，原账本结算真实CPU费用、前向0；首差异为当前更新窗口BN均值累加器，尚未获准改变其标准。
- `inspect_w54_state_categories.py`用于一次分类核对参数、buffers、优化器、梯度、BN累加器、缓存及其余科学状态，诊断结果不能当独立科学验收或删除失败依据；任何拟议容差仅作诊断预测、不生效。运行证据在D/launches/<实际nonce>/，旧model失败、完整检查点和两支128面板原样保留。

- 分类入口实际CPU验收已exit0，nonce135e7037db694ab8bfa6248d96c201fc，新增模型前向0。明确仅当前bn_mean累加器20个张量/86元素与窗口基点临时缓存通道均值1个张量/1通道超界；参数、buffers、优化器、梯度、BN二阶、RNG及其余完整科学状态均符合原标准。新的BN容差只提出给用户，未应用。两次CPU读取及全部科学失败费用留原账本，未退款/重置。

- `inspect_w54_failed_cache_moments.py`补核唯一在通道均值处提前退出的缓存层，避免把尚未访问的均方误称通过。真实admitted CPU nonce93d012b186d34c1b9dcd0a014413048c、exit0、前向0：该层均方最大现行界限比例0.05063，仍通过；均值仅1通道超界，在拟议atol2e-5下最大比例0.58548（未应用）。这使全部受影响类别明确为BN均值累加器与缓存通道均值，其他状态标准未改。


## 2026-10-28：用户授权两项BN均值容差并复用w54完整成果

- `numeric_acceptance_20261027/numeric_compare.py`仅对`learner.learning/window_base.bn_mean`窗口均值累加器采用atol1e-5、相同learning快照的BN缓存通道均值采用atol2e-5，rtol保持1e-5。BN二阶、缓存均方、真实running buffers、参数/梯度/优化器、其余数值及离散/RNG/同运行完整回滚和GT要求不变；不按字段名字模糊放宽。
- `resume_BN_two_means_20261028.py`在真实旧退出及原四锁/manager锁安全边界，绑定已保存512观察完整状态、候选和两支128面板，移除被完整恢复取代的旧448 partial提示，沿用零模型finalizer生成新main后执行真实独立验收。原两次失败、原标准、29个已验收proof及全部账本费用保留，不退款、不改写旧成功或失败。
- 旧V8比较器和实际分支执行代码归档于`versions/v8/`；完整状态只建立不可变硬链接，不复制模型或重算面板。定向零模型检查通过两项新界限及越界拒绝，并确认二阶、参数/running buffer、无关同名字段、RNG仍按原标准。真正科学验收以新main/verify实际退出和proof为准。


## 2026-10-28：用户授权仅heat_peaks向量范数验收

- `numeric_acceptance_20261027/numeric_compare.py`仅完整姿态预测记录中的11维heat_peaks改用L2整体范数，仍为atol1e-6/rtol1e-5；shape/dtype/有限性、顺序、实际阈值/离散决定均保持。其他预测字段、旋转0.01度、两BN均值、模型/优化器/梯度、RNG、同运行回滚/分支隔离和GT规则不改，原始记录不改。
- `resume_heat_peaks_20261028.py`核对V9真实退出及连续账本/原四锁，在安全边界将w84从实际完整704检查点续接，保留旧失败77物理前向费用。此窗口尚无候选或面板，不能虚称复用；30个已验收窗口直接复用、不重跑。旧V9比较器在`versions/v9/`。
- 定向零模型验证：真实首差异通过，大误差/错误维数/非有限值和离散决定改变仍拒绝，无关model同名向量不放宽；实际科学完成仍需新main和独立verify真实exit0/proof。版本化policy_v10/revision_v10与恢复manifest保留原任务、输入、父检查点、旧错误及回退链。


## 2026-10-28：用户授权confidence原公式自洽验收

- `numeric_acceptance_20261027/numeric_compare.py`仅对完整姿态预测中的confidence，核验其等于全部11个原FP32 heat_peaks按`np.quantile(...,.25,method='linear')`重算值；验证原输入float32回转无静默舍入后，只在比较副本归一化冗余confidence，不改原值。原G50重算先用未修改arrival执行，保持真实分数/阈值/决定自洽；None可用性、各点输入及其他标准不改，同名模型参数不放宽。
- `resume_confidence_20261028.py`在V10真实退出/原四锁/连续账本安全边界，保留w84两个旧失败及77+222前向费用，从实际704完整恢复点续接未持久化部分；尚无候选/面板，30个已验收结果直接复用。旧V10比较器在`versions/v10/`。
- 定向零前向验证：真实保存事件通过、原字典不变；伪造派生值、非原始float32峰值、修改决定、缺峰却有confidence和无关同名参数均拒绝。policy_v11/revision_v11保留旧标准、原任务输入/版本/失败及回退链；最终必须真实main及独立verify退出和proof通过。


## 2026-10-28：一次性完整状态分类入口及BN缓存双矩穷尽核对

- `numeric_acceptance_20261027/inspect_saved_state_categories.py`将既有分类入口参数化为`--run-id`，只允许当前blocked且未accepted的冻结窗口；核对当前index未变、完整恢复游标等于原match后才读取状态。沿用四pipeline锁/manager锁、两新鲜共享资源快照、精确身份读回和原账本零模型attempt，绝不构造模型、读取图像/GT或改变验收。
- 同一次读取中显式计算每层BN缓存的通道均值与均方，即使原比较器在均值处先返回失败，也不会漏掉第二项；保留原路径语义，报告各类失败、最大差/比例和仅诊断的最小必要绝对界限，不能将其作为放行或自动调容差。
- w84实际诊断nonce27c47b50f9fe4bd685020930787df358，真实exit0、模型前向0，完整768观察状态与两支128面板未改。仅当前bn_mean及当前/窗口基点的缓存通道均值超界；BN缓存均方、参数/优化器/梯度、真实运行buffers、RNG及其余科学状态通过。当前V11科学标准仍不变，新界限需要用户决定。
