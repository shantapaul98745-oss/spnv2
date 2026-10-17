# SPNv2 代码改动记录

本文件用中文说明每项修改的位置、原因、作用与验证。Git 提交中的逐行差异是代码变化的最终依据。

## 2026-10-17：补录复现前已有状态

- 来源：本轮实施前 13:50:38 保存的独立备份，原 HEAD 为 `0d7922c5791d08c52c812bdc6154272452c1411c`。
- 范围：`core/dataset/SPEEDPLUSDataset.py`、`core/dataset/transforms/randomsunflare.py`、`core/engine/inference.py`、`core/engine/trainer.py`、`core/utils/utils.py`、两个 `experiments/offline_train_full_config_*` 配置和 `tools/train.py`，共8处。
- 原因：这些修改在当前独立复现实施前已存在；单独归档，以便后续准确比较本轮新增内容。
- 作用：保留历史状态，不把旧修改归因于本轮实现；不改变任何运行中的实验文件。
- 验证：8个文件逐个与实施前备份的 SHA256 一致。历史功能的原始动机和完整验收没有在本记录中重新推断。
