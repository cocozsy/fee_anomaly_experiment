# fee_anomaly_experiment

该目录仅保留“电费异常原因分析 + 教师学生蒸馏”实验所需最小内容。
monitor
.venv/bin/tensorboard --logdir checkpoints/env_self_v1/tb_logs --port 6006
[http://localhost:6006](http://localhost:6006)
运行后重点看：

训练日志：checkpoints/env_self_v1/train_live.log
进度图：experiments/figures/env_self_v1_progress_live.png
TensorBoard：[http://localhost](http://localhost):<你设置的端口>
一键并发监控（训练 + 进度图 + TensorBoard）

```bash
bash run_monitor.sh --output-dir checkpoints/env_self_v1
```

## 目录说明

- `code/`
  - `build_aligned_dataset.py`：将7张业务表对齐成客户-月建模样本
  - `models.py`：教师模型、学生模型、蒸馏损失
  - `train_distill.py`：训练与蒸馏主脚本
  - `TECHNICAL_REPORT.md`：完整技术报告
- `data/input_output_tables/`
  - 原始7张业务表（放置于当前仓库的 `data/input_output_tables/`）
- `data/aligned/`
  - `aligned_customer_month.csv`：已对齐样本
  - `aligned_metadata.json`：样本元信息

## 运行步骤

1. 构建对齐数据（可重建）

```bash
python code/build_aligned_dataset.py   --input_dir data/input_output_tables   --output_csv data/aligned/aligned_customer_month.csv   --output_meta data/aligned/aligned_metadata.json
```

1. 训练教师 + 蒸馏学生

```bash
python code/train_distill.py   --aligned_csv data/aligned/aligned_customer_month.csv   --output_dir checkpoints/fee_anomaly   --teacher_epochs 12   --student_epochs 16   --batch_size 512
```

