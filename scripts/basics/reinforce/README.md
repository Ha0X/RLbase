# REINFORCE（策略梯度）— 最小项目

这是一个小巧、**可运行**的 PyTorch 项目，用于在 CartPole-v1 上学习 REINFORCE 算法。
包含基线选项（ value head ）以减少方差，也可以关闭以使用纯 REINFORCE。

## 快速开始

```bash
# （可选）创建虚拟环境
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 训练（纯 REINFORCE）
python main.py --env CartPole-v1 --total-steps 100000 --lr 3e-3 --gamma 0.99 --no-baseline

# 训练（REINFORCE + baseline / advantage）
python main.py --env CartPole-v1 --total-steps 150000 --lr 3e-3 --gamma 0.99 --baseline --entropy-coef 0.01

# 查看 tensorboard（可选）
tensorboard --logdir runs
```

## 文件说明

- `main.py` — CLI 入口；设置环境/智能体/训练循环
- `src/policy.py` — 策略网络（分类）；可选价值头作为基线
- `src/train.py` — 蒙特卡洛轨迹收集 + 回报/优势 + 更新
- `src/utils.py` — 辅助函数（随机种子、归一化、日志）
- `requirements.txt` — 依赖项
