# 最终证据汇总：CPU实现与真实完成状态分离

本目录只生成和核验汇总产物，不启动训练、推理、ODR或GPU，不修改被读取的report/MAT、pipeline、ODR、源码和其他agent文件。所有运行须通过项目唯一guarded screen，CUDA_VISIBLE_DEVICES=''，OMP/OPENBLAS线程数2。

## CLI

- `run_analysis.py --run-id 唯一ID --cases`：读取当前实际配置/queue_v2、source_artifacts验收、ODR验收和3MAT，生成完整新矩阵、独立历史矩阵、中文MD、accepted均值、严格配对、五观测流统计、投影案例及PRISMA材料状态。每次输出到新的runs/ID，不覆盖旧快照。
- `verify_snapshot.py --run-id ID`：独立CPU核对产物哈希、430个预期记录位置、缺失值、历史隔离、canonical去重、五种子完整性、Markdown结构和完成标志。
- `cpu_tests.py --run-id 唯一ID`：仅tiny mock fixtures验证数学、状态/拒绝/错序/错误SHA/alias/投影；不是正式结果。
- `--prisma-manifest 项目内路径`：可选的schema=project-prisma25-v1清单，含labels/camera及各自SHA，labels列出25个唯一sample_id、项目内image路径/内容SHA、quaternion/translation。只有材料与真实离线hH验收均齐全才允许PRISMA完成。未提供时仅核实固定项目内候选路径，不联网或搜索其他项目。

完整解释器为`/home/ps/anaconda3/envs/SPNv2/bin/python -B -s -u`。父队列可把这些CPU命令放入项目run_in_screen_guarded.sh；本目录不创建后台管理器。

## 严格口径

source_pins.json可先为空；仅status=passed、配置/路径/SHA与实际final一致且code_sha256匹配冻结源码的source报告能首次建立pin；全部现有证据完整通过后才写入。已有pin逐次重验final路径与SHA，任何更换直接拒绝，不自动重置。source只取各实际配置的final，首次独立验收后SHA固定到source_pins.json；不按目标域成绩挑checkpoint。新source验收缺失时输出未完成/待验收与null，绝不借历史模型值。输入验收/3MAT/CSV/稳定ID在汇总时重新核对，产物记录SHA链。原始图像内容没有在上游ODR整体哈希，案例仍检查保存的path/size/mtime并记录实际展示图像内容SHA。相机/关键点与已验收ODR输入证据绑定。

每个指标明确head、单位、总样本、有效分母、拒绝数。H接受子集改变时，source/ODR各自accepted均值列在accepted_means.csv；逐图改善/退化只在相同ID且双方该指标有效的交集比较。final分数（H-only=hR+hT、E-only=eR+eT、双头=hR+eT）的交集均值/分位数与H拒绝→有效、有效→拒绝、双方拒绝分别保存；缺失从不填0或Inf。ID重排被拒绝，不暗中按位置配对。

A/B1/B2分别分组。五seed是同一source的ODR观测流随机性，不是五次source训练；未齐2021—2025不出五种子均值/标准差。30个alias必须核实canonical验收report SHA与所有协议字段；matrix保留请求身份，配对与重复统计只算实际canonical实验。历史26项来自自己的旧ledger/冻结final/3MAT，单列且不进入新source配对和五种子统计；旧φ3 seed2021观测清单缺失继续披露。

## 案例限制

实际MAT包含heat_q/heat_t、effi_R/effi_t、bbox，没有原始预测2D关键点。因此PNG仅为真实姿态预测的3D Tango点投影，明确不冒称热图检测点。source固定final；案例池固定为fixed任务seed2021、配置ID字典序轮转；样本按严格配对改善/退化、中位及失败转移选择，并保存依据。无对应有效预测/材料时给受限，不造预测。当前新source未完成，正式案例未产出；tests中的MOCK PNG只是工程投影验证。

## 完整标志

完整矩阵包括10套source×3域及400个ODR请求；30alias不增加370个实际实验。全部位置已验收、370个严格配对完成、74组观测流各齐5seed、真实案例齐且无证据缺损时，core_scope_complete才可为true。scientific_scope_complete与complete还要求PRISMA材料/离线hH均齐。候选材料路径全部核实不存在时，conditional_prisma_status=verified_limitation且verified_limitation=true，明确这是条件任务的已核实限制，不阻止核心范围标志；材料存在但证据不足或齐而未评则不得如此标记。总体完成仍由root审计其他外部任务。代码CPU测试通过不等于正式复现完成。

运行时依赖现有NumPy/SciPy/YAML/Pillow/Matplotlib；不安装或更新包。读取范围仅LuXinghai项目和既有speedplus数据根。此目录是本机运行驱动；Git版本归档由root统一提交推送。
