# CodeEMO 部署说明 (235)

> 本文件说明 235 服务器上 CodeEMO 项目的部署状态与使用方法。
> 上游项目: `/root/.openclaw/workspace-staging/pc-ceo_assistant/CodeEMO/`
> 部署时间: 2026-07-13

## 路径布局

| 路径 | 内容 |
|---|---|
| `/home/ubuntu/CodeEMO/` | 项目代码（不含 .venv） |
| `/home/ubuntu/CodeEMO/.venv/` | 独立 Python 虚拟环境 (25MB) |
| `/home/ubuntu/IDE_logs/` | 真实数据集 (1.3GB + 4.9KB) |
| `/tmp/IDE_logs/` → `/home/ubuntu/IDE_logs/` | 软链（保持项目原配置） |

## 环境

- **基础 Python**: `/home/ubuntu/campus-multimodal/py311` (Python 3.11.15)
- **PyTorch**: torch 2.11.0+cu130 (system-site-packages 复用)
- **GPU**: Tesla T4 16GB
- **已装包**: scikit-learn 1.9.0, pandas 3.0.3, numpy 1.26.4, scipy 1.17.1
- **未装**: mamba-ssm (本项目自实现 Mamba，无需该包)

## 数据集

- 28,588,309 条 IDE 事件
- 473 个学生
- 7 种事件类型: focus_gained, focus_lost, text_insert, text_remove, run, submit, text_paste
- 通过率: 33.6%

## 快速运行

```bash
ssh235 "cd /home/ubuntu/CodeEMO && \
  PYTHONPATH=/home/ubuntu/CodeEMO .venv/bin/python models/rf/train.py --folds 5"
```

或用统一入口:
```bash
ssh235 "cd /home/ubuntu/CodeEMO && \
  PYTHONPATH=/home/ubuntu/CodeEMO .venv/bin/python main.py --model rf"
```

可选模型: `rf`, `lstm`, `bilstm`, `transformer`, `mamba`, `mamba_gpu`, `all`

## 已验证 (迁移后 smoke test)

| 模型 | Acc | F1 | AUC | 状态 |
|---|---|---|---|---|
| RF (CPU) | 0.816 | 0.733 | 0.906 | ✓ |
| LSTM (GPU) | 0.831 | 0.780 | 0.897 | ✓ |

## 复现环境 (如 .venv 丢失)

```bash
ssh235 "
mkdir -p /home/ubuntu/CodeEMO
cd /home/ubuntu/CodeEMO
/home/ubuntu/campus-multimodal/py311/bin/python3.11 -m venv .venv --system-site-packages
.venv/bin/pip install --upgrade pip
.venv/bin/pip install scikit-learn
"
```

数据复现: `scp -i ~/.ssh/id_openclaw /tmp/IDE_logs/IDE_logs.csv /tmp/IDE_logs/passed.csv ubuntu@111.229.46.235:/home/ubuntu/IDE_logs/`
