# 实验代码与配置版本归档

这里保存仓库外本项目实验代码、配置和验收工具的逐文件副本，便于在 GitHub 查看改动。`archive_manifest.json` 给出实际运行路径、仓库路径和 SHA256 对照。

## 查看顺序

1. 先看根目录 `CODE_CHANGES.md` 的中文改动说明。
2. `runtime/rebaseline_20261017/EXECUTION_PLAN.md` 是完整复现范围；当前自动队列为 `queue_v2.json`。
3. `pipeline/` 是串行队列、进程核验、原子进度和安全追加修订；`tasks/` 是权重和评价文件独立验收。
4. `configs/` 是10套正式训练及各域评价配置；`pilots/` 是独立短测与恢复核验。
5. `odr/` 是固定40项及B/N扫描的生成器、配置、A/B1/B2差异和验收代码；`resources/` 是参数统计与官方初始化来源核验代码。
6. `runtime/shared/compact_progress.py` 和 `progress.sh` 提供用户日常使用的简洁只读进度界面。

## 运行位置

运行入口仍在服务器 `/home/DATA/Downloads/UsrSourceCode/LuXinghai/spnv2_runs/`。这里的副本用于版本审阅；路径和源码哈希按原位置核验，不能直接把副本作为新实验入口。当前训练的源码快照保持冻结。

数据集、训练权重、原始日志、实时状态和论文PDF保存在原资产目录。生成的大型任务目录以生成器、全部配置和SHA256记录；目前实际激活的队列文件完整纳入版本。

每项代码修改验证后，运行 `python reproduction/sync_runtime_archive.py` 更新副本和清单，再由主模型连同 `CODE_CHANGES.md` 一起提交、推送。它只同步代码副本，不操作实验或Git分支。待分析工具验收后可加 `--include-analysis`。
