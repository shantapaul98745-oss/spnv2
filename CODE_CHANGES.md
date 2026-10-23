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
