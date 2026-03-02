# 基础知识

## 什么是强化学习？
智能体通过与环境交互，不断学习，完成特定目标

## 基本元素

状态 $s$ (state)
动作 $a$ (action)
策略 $\pi(a|s) = P(A_t=a | S_t=s)$

在策略$\pi$和状态$s$时，采取行动后的价值（value），用$v_π(s)$表示

奖励(reward)：$t$时刻智能体在状态 $S_t$ , 采取的动作为 $A_t$ , 对应的奖励 $R_{t+1}$ 会在 $t+1$ 时刻得到

环境的状态转化模型：在状态 $s$下采取动作 $a$ , 转到下一个状态 $s^′$ 的概率，记作$P_{s,s'}^a$

一句话来讲：在某个 $state$ 下，$agent$ 依据 $policy$，采取$action$ ，与$environment$ 交互，$agent$ 获得反馈 $reward$。
$agent$ 获得的 $reward$ 会指导 $policy$ 改进，在 $state$ 选择 $action$。循环往复，$policy$ 不断被优化。

## 马尔可夫性质

马尔可夫性质是指在给定当前状态的情况下，未来状态的条件概率分布仅依赖于当前状态，而与过去状态无关。

$ P(S_{t+1}|S_t)=P(S_t+1|S_t,S_{t-1},...S_1)$

## 状态价值与动作价值

### 状态价值
$$v_{\pi}(s) = \mathbb{E}_{\pi}(R_{t+1} + \gamma R_{t+2} + \gamma^2R_{t+3}+...|S_t=s)$$

### 动作价值
$$q_{\pi}(s,a) = \mathbb{E}_{\pi}(G_t|S_t=s, A_t=a) = \mathbb{E}_{\pi}(R_{t+1} + \gamma R_{t+2} + \gamma^2R_{t+3}+...|S_t=s,A_t=a)$$

### 转换关系
根据动作价值函数 $q_{\pi}(s,a)$ 和状态价值函数 $v_{\pi}(s)$ 的定义，我们容易得到他们之间的转化关系：

$$v_{\pi}(s) = \sum\limits_{a \in A} \pi(a|s)q_{\pi}(s,a)$$
状态价值函数是所有动作价值函数基于策略 $π$ 的期望。

## 强化学习的分类

### Model-Based 与 Model-Free
- Model-Based 意味着有环境的 explicit model，即 $P(S_{t+1},R|S_t ,a_t)$
- Model-Free 相对应地，无

### On-policy 与 Off-policy
RL算法中都需要做两件事：
1. 收集数据(Data Collection)：与环境交互，收集学习样本;
2. 学习(Learning)样本：学习收集到的样本中的信息，提升策略。

- off-policy：将收集数据当做一个单独的任务  Q-learning
The learning is from the data off the target policy
- on-policy 里面只有一种策略，它既为目标策略又为行为策略
On-Policy 强化学习需要通过当前的策略收集数据，并使用这些数据来改进当前策略。比如 SARSA 就是一个典型的 On-Policy 算法

### Online policy 与 Offline policy
- offline RL:离线强化学习。学习过程中，不与环境进行交互，只从dataset中直接学习，而dataset是采用别的策略收集的数据，并且采集数据的策略并不是近似最优策略。

- online RL:在线强化学习。学习过程中，智能体需要和环境进行交互。并且在线强化学习可分为on-policy RL和off-policy RL。on-policy采用的是当前策略搜集的数据训练模型，每条数据仅使用一次。off-policy训练采用的数据不需要是当前策略搜集的。

Offline RL 和Imitation Learning的区别：Off-line RL中数据包括奖励，IL中数据不包括奖励。Off-line RL不要求数据是近似最优策略的得到的，IL中的专家数据基于得到搜集专家数据的策略是近似最优策略的假设。

### 无监督强化学习
无监督强化学习中，智能体并没有明确的奖励信号。它依赖于自我探索、环境的固有特性，或者一些内在目标来指导其行为。这意味着没有外部的奖励指引，智能体的目标更多是自我生成目标、获取经验或增强与环境的交互

## DP（动态规划） (Model-Based)

### Policy Iteration
给定一个初始策略 $π$ ，可以得到基于该策略的价值函数 $v_π$ (用贝尔曼方程计算)
基于该价值函数又可以得到一个贪婪策略 ${\pi}^′$ = $ greedy(v_π)$
如此反复进行，价值函数和策略均得到迭代更新，并最终收敛得到最优价值函数 $v^∗$ 和最优策略 $π^∗$

### Value Iteration
直接迭代更新最优价值函数 $V^*$（用贝尔曼最优方程），无需显式维护策略，最终从收敛的 $V^*$ 中导出最优策略 $\pi^*$

## 蒙特卡洛法 (Model-Free)
通过采样若干经历完整的状态序列(episode)来估计状态的真实价值。
所谓的经历完整，就是这个序列必须是达到终点的。比如下棋问题分出输赢，驾车问题成功到达终点或者失败。

要求某一个状态的状态价值，只需要求出所有的完整序列中该状态出现时候的收获再取平均值即可近似求解预测：

$$G_t =R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3}+...  \gamma^{T-t-1}R_{T}$$
$Average(G_t),  s.t. S_t=S$
$V (s_t) = V(s_t) + \alpha·[ G_t − V(s_t) ]$

ϵ−贪婪法：使用1−ϵ的概率贪婪地选择目前认为是最大行为价值的行为，而用ϵ的概率随机的从所有可选行为中选择行为

## 时序差分方法(TD)
不需要等一个完整的回合结束才更新价值估计 (与MC不同)，自举利用当前的估计值来更新。
TD(0)：一步更新，在当前状态 $s_t$ 基础上，根据下一状态 $s_{t+1}$ 的估计来更新当前状态的价值。更新量由 当前价值估计 和 通过下一状态的价值估计 计算的误差（称为TD误差）决定：

$$V(s_t) \leftarrow V(s_t) + \alpha \Big( r_{t+1} + \gamma\ V(s_t+1)-V(s_t)\Big)$$

其中 $r_{t+1} + \gamma\ V(s_t+1)$ 为实际值，$V(s_t)$ 为预期值。

类似于 TD(0)，可以对 TD(1) 和 TD(2) 等进行扩展，主要是通过增加多步的预期来改善更新的准确性：

$$V(s_t)←V(s_t)+α(r_{t+1}+γ r_{t+2}+γ^2 V(s_{t+2})−V(st))$$

## Q-learning

### 核心思想

-  学习 动作价值函数 $Q(s,a)$：在状态 $s$ 下执行动作 $a$，并按最优策略走，能获得的期望回报。
-  最优 Q 函数满足 Bellman 最优方程：

$$Q^*(s,a) = \mathbb{E}\Big[ r + \gamma \max_{a'} Q^*(s',a') \Big]$$

### 更新公式
$$Q(s,a) \leftarrow Q(s,a) + \alpha \Big( r + \gamma \max_{a'} Q(s',a') - Q(s,a) \Big)$$

其中：
- $\alpha$：学习率  
- $r$：即时奖励  
- $\gamma$：折扣因子  
- $\max_{a'} Q(s',a')$：下一状态的最佳 Q 值估计  

### 对公式的直观理解
- 旧值：原有对 $Q(s,a)$ 的估计  
- 目标值：$r + \gamma \max_{a'} Q(s',a')$  
- 更新：往目标值方向移动一点，幅度由 $\alpha$ 决定  

## SARSA

### 核心思想
SARSA 是 On-Policy（基于当前策略） 的算法，这意味着它使用当前行为策略来选择动作并更新 Q 值。相比之下，Q-learning 是 Off-Policy（基于目标策略） 的算法，它使用最优策略（即当前 Q 值最大的动作）来更新 Q 值，即使这个动作在实际执行时并不一定是最优的。

### SARSA 与 Q-learning 的区别
- Q-learning 是 Off-Policy，它在更新 Q 值时，选择最大 Q 值的动作 来进行更新，而不考虑当前策略的行为。
  Q-learning 的更新公式是：

  $$Q(s,a) \leftarrow Q(s,a) + \alpha \Big( r + \gamma \max_{a'} Q(s',a') - Q(s,a) \Big)$$
  Q-learning 强调的是最优策略，并不依赖于当前的行为策略。
- SARSA 是 On-Policy，它在更新 Q 值时，使用 实际执行的动作 $a_{t+1}$ 来进行更新，而不是选择最大 Q 值的动作。
  SARSA 的更新公式：

  $$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \Big( r_{t+1} + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t) \Big)$$
  SARSA 更依赖于智能体当前的策略，因此它会受到探索策略（如 ϵ\epsilonϵ-greedy）影响，导致它更新的 Q 值更符合当前策略的行为。

## 从强化学习到深度强化学习

### 为什么我们需要神经网络
Q-learning 的表格形式（即通过 Q(s,a) 存储每个状态-动作对的 Q 值）在状态空间和动作空间很大时变得不可行，因为需要维护一个巨大的 Q 表。这个问题在高维连续空间中尤其严重

## DQN
$Q(s,a;\theta) \approx Q^*(s,a)$
通过训练，DQN 会更新网络参数 $θ$，使得它输出的 Q 值接近最优 Q 值

### 核心机制

#### 经验回放（Replay Buffer）
存储 transition信息 $(s, a, r, s^′)$，训练时随机采样，打破数据相关性 
用相邻的样本连续训练evaluate network会带来网络过拟合，泛化能力差的问题

#### 目标网络（Target Network）
- 复制一份 Q 网络作为目标网络 $\theta^-$，周期性更新  
- 目标值：$y = r + \gamma \max_{a'} Q(s',a'; \theta^-)$
目标网络的引入是为了避免Q网络参数频繁更新导致的训练不稳定问题。目标网络提供了稳定的目标值，使得 TD误差不会因目标值的快速变化而震荡

### 损失函数
$$L(\theta) = \Big( y - Q(s,a;\theta) \Big)^2$$

### 训练流程
1. 初始化 Q 网络和目标网络  
2. 建立经验回放池  
3. 每步：
  - $\epsilon$-贪心选择动作  
  - 执行动作，存储 $(s,a,r,s^′)$  
  - 从回放池采样 batch，计算目标值 $y$  
  - 更新 Q 网络参数 $\theta$  
  - 每隔 $N$ 步更新目标网络  
4. 循环直到收敛  

### 注意
Q网络：用于计算当前策略下每个状态-动作对的 Q 值，并通过梯度下降更新参数 θ。
目标网络（Target Network）：目标网络不参与梯度下降和反向传播，它仅用于计算目标值 y，即用于计算 Q-learning 中的 TD 目标。

```python
# 定义 Q 网络结构
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 初始化环境和网络
env = gym.make('CartPole-v1')
input_dim = env.observation_space.shape[0]  # 输入维度，CartPole 的状态空间维度为 4
output_dim = env.action_space.n  # 输出维度，CartPole 的动作空间有 2 个动作

# 创建 Q 网络和目标网络
q_network = QNetwork(input_dim, output_dim)
target_network = QNetwork(input_dim, output_dim)
target_network.load_state_dict(q_network.state_dict())  # 复制初始参数

# 设置优化器和损失函数
optimizer = optim.Adam(q_network.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 经验回放池
replay_buffer = deque(maxlen=10000)

# 超参数
batch_size = 64
gamma = 0.99  # 折扣因子
epsilon = 1.0  # 初始探索概率
epsilon_decay = 0.995  # 探索概率衰减
min_epsilon = 0.01  # 最小探索概率
target_update_interval = 1000  # 目标网络更新的间隔
learning_starts = 1000  # 训练开始的时间步数

# 训练过程
def train_step(batch):
    states, actions, rewards, next_states, dones = batch
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # 计算 Q 网络的 Q 值
    q_values = q_network(states)
    q_value = q_values.gather(1, actions.unsqueeze(1))  # 获取实际选择的动作的 Q 值

    # 计算目标 Q 值
    with torch.no_grad():
        next_q_values = target_network(next_states)
        next_q_value = next_q_values.max(1)[0]  # 获取最大 Q 值
        target = rewards + (gamma * next_q_value * (1 - dones))

    # 计算损失并更新 Q 网络
    loss = criterion(q_value.squeeze(), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 经验回放采样
def sample_experience(batch_size):
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

# 主训练循环
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # 探索或利用
        if random.random() < epsilon:
            action = env.action_space.sample()  # 探索
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = q_network(state_tensor).argmax().item()  # 利用 Q 网络的预测

        # 执行动作，获取奖励和下一状态
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        # 存储经验
        replay_buffer.append((state, action, reward, next_state, done))

        # 更新状态
        state = next_state

        # 如果经验回放池中的经验足够，进行训练
        if len(replay_buffer) >= learning_starts:
            batch = sample_experience(batch_size)
            train_step(batch)

    # 衰减 epsilon（探索概率）
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # 每隔 target_update_interval 步更新目标网络
    if episode % target_update_interval == 0:
        target_network.load_state_dict(q_network.state_dict())

    print(f"Episode {episode+1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

# 测试训练后的模型
for _ in range(10):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = q_network(state_tensor).argmax().item()  # 选择最优动作
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
    print(f"Test Episode Total Reward: {total_reward}")
```

## 基于策略的算法

基于策略的算法的目标是 直接学习一个策略，而不是通过学习 Q 值来选择动作。策略是智能体在给定状态下选择动作的 概率分布。

## REINFORCE
一种基本的 策略梯度 方法，通过 蒙特卡洛估计 来计算策略的梯度，进而更新策略。

## Actor-Critic框架
- Actor（执行者）：负责 “做决策”，输出动作策略。它根据当前环境状态，直接输出具体动作（确定性策略）或动作的概率分布（随机性策略），决定智能体下一步该做什么。
- Critic（评论家）：负责 "评好坏"，评估动作价值。它根据当前状态和 Actor 选择的动作，计算该动作带来的 "价值"（通常是 Q 值或状态价值 V），判断这个决策的长期收益高低。

### 双向优化

- 优化 Actor：用 Critic 给出的 TD Error 作为反馈信号，调整 Actor 的参数，让它未来更倾向于选择 Critic 评分高的动作。
- 优化 Critic：根据 TD Error 调整自身参数，减少 "预期价值" 与 "实际价值" 的差距，提升评估的准确性。

```python
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# 超参数
GAMMA = 0.99
LR = 1e-3
HIDDEN_SIZE = 128
MAX_EPISODES = 5000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Actor-Critic 网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 公共层
        self.fc1 = nn.Linear(state_dim, HIDDEN_SIZE)
        # Actor 分支：输出每个动作的概率
        self.actor = nn.Linear(HIDDEN_SIZE, action_dim)
        # Critic 分支：输出状态价值 V(s)
        self.critic = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.actor(x)
        probs = F.softmax(logits, dim=-1)  # 动作概率分布
        value = self.critic(x)             # 状态价值 V(s)
        return probs, value


def train():
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, action_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        done = False
        log_probs = []
        values = []
        rewards = []

        # 采样一条轨迹
        while not done:
            state_tensor = torch.tensor([state], dtype=torch.float32, device=DEVICE)
            probs, value = model(state_tensor)

            # 从分布中采样动作
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            # 保存对数概率、价值
            log_probs.append(dist.log_prob(action))
            values.append(value)

            # 与环境交互
            next_state, reward, done, truncated, _ = env.step(action.item())
            done = done or truncated

            rewards.append(reward)
            state = next_state

        # 计算返回 Gt（蒙特卡洛）
        # returns = []
        # G = 0
        # for r in reversed(rewards):
        #     G = r + GAMMA * G
        #     returns.insert(0, G)
        # returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        # 转为张量
        log_probs = torch.stack(log_probs)
        values = torch.cat(values).squeeze()

        # 一步TD目标：target_t = r_t + γ * V(s_{t+1})
        with torch.no_grad():
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)  # [T]
            next_values = torch.cat([values[1:], torch.zeros(1, device=DEVICE)])   # 末步的 V(s_{T+1})=0
            td_targets = rewards_t + GAMMA * next_values                           # [T]

        # Advantage: A_t = target_t - V(s_t)
        advantages = td_targets - values


        # 损失函数：Actor Loss + Critic Loss
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + critic_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印结果
        if (episode + 1) % 10 == 0:
            total_reward = sum(rewards)
            print(f"Episode {episode+1}: return={total_reward}, loss={loss.item():.3f}")
            
    env.close()


if __name__ == "__main__":
    train()
```

## SAC（Soft Actor-Critic）(最大熵)

### 核心目标
$$\pi^*(a|s) = \arg\max_\pi \mathbb{E}\left[ \sum_{t=0}^\infty \gamma^t \left( r_t + \alpha H(\pi(\cdot|s_t)) \right) \right]$$

- 最大化累积奖励的同时，最大化策略的熵：

  $$H(\pi(\cdot|s)) = -\mathbb{E}_{a\sim\pi}[ \log \pi(a|s) ]$$

  （鼓励探索）
- $\alpha$ 为熵权重，平衡奖励与探索

### 核心机制

#### 随机策略（Stochastic Policy）
- Actor 网络输出动作的概率分布（连续空间常用高斯分布）：$\pi(a|s;\phi)$，通过采样生成动作
- 相比确定性策略，随机策略天然支持熵最大化，提升探索效率和策略鲁棒性

#### 双 Q 网络（Dual Q-Networks）
- 两个独立的 Q 网络 $Q_1(s,a;\theta_1)$、$Q_2(s,a;\theta_2)$，评估状态-动作价值（含熵奖励）
- 抑制价值估计过高的偏差（类似 DQN 目标网络的改进，但更激进）

#### 目标 Q 网络（Target Q-Networks）
- 对应双 Q 网络的目标网络 $Q_1^-(s,a;\theta_1^-)$、$Q_2^-(s,a;\theta_2^-)$
- 目标值计算（含熵项）：

  $$y = r + \gamma \left( \min_{i=1,2} Q_i^-(s',a';\theta_i^-) - \alpha \log \pi(a'|s';\phi) \right)$$
- 采用指数移动平均（EMA）更新，保证目标稳定

#### 经验回放（Replay Buffer）
- 存储 transition 信息 $(s,a,r,s',\text{done})$，训练时随机采样，打破数据相关性
- 支持离线学习（Off-policy），提升样本效率

### 损失函数

#### Q 网络损失（Critic 优化）
- 最小化当前 Q 值与目标值的均方误差：

  $$L(\theta_1) = \mathbb{E}\left[ \left( y - Q_1(s,a;\theta_1) \right)^2 \right]$$

  $$L(\theta_2) = \mathbb{E}\left[ \left( y - Q_2(s,a;\theta_2) \right)^2 \right]$$

#### 策略网络损失（Actor 优化）
- 最大化 Q 值与熵的加权和（通过梯度上升实现，等价于最小化负损失）：

  $$L(\phi) = -\mathbb{E}_{s\sim D, a\sim\pi}\left[ \min_{i=1,2} Q_i(s,a;\theta_i) - \alpha \log \pi(a|s;\phi) \right]$$

#### 熵权重 $\alpha$ 优化（可选）
- 自动调整$\alpha$，使策略熵接近目标值 $H_{\text{target}}$：

  $$L(\alpha) = -\mathbb{E}_{s\sim D, a\sim\pi}\left[ \alpha \left( \log \pi(a|s;\phi) + H_{\text{target}} \right) \right]$$

### 训练流程
1. 初始化：
  - Actor 网络 $\pi(\phi)$、双 Q 网络 $Q_1(\theta_1)$、$Q_2(\theta_2)$
  - 目标 Q 网络 $Q_1^-、Q_2^-$（初始参数复制自 $Q_1、Q_2$）
  - 经验回放池 $D$、熵权重 $\alpha$
2. 每步交互：
  - 从当前状态 $s$，用 Actor 网络采样动作 $a \sim \pi(\cdot|s;\phi)$
  - 执行动作，获得奖励 $r$、下一状态 $s'$、终止信号 $\text{done}$
  - 将 $(s,a,r,s',\text{done})$ 存入 $D$
3. 每步训练（从 $D$ 采样批量数据）：
  - 采样 $a' \sim \pi(\cdot|s';\phi)$（带噪声增强探索）
  - 计算目标值 $y$（用目标 Q 网络和 $\log \pi(a'|s')$）
  - 用 $L(\theta_1)、L(\theta_2)$ 更新双 Q 网络
  - 用 $L(\phi)$ 更新 Actor 网络
  - 用 EMA 更新目标 Q 网络参数
  - （可选）用 $L(\alpha)$ 更新熵权重 $\alpha$
4. 循环直到收敛

### 注意
- Actor 网络负责生成随机策略，通过 Q 网络评估和熵项引导优化
- 双 Q 网络+目标网络抑制价值偏差，经验回放支持离线学习
- 最大熵机制使策略兼具"高奖励"和"高探索性"，在连续动作空间表现优异

## PPO

PPO是一个on-policy的强化学习算法，是目前应用最广泛的强化学习算法之一

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions import Normal

class Actor(nn.Module):
   """输出动作分布的均值和方差"""
   def __init__(self, state_dim, action_dim):
      super().__init__()
      self.net = nn.Sequential(
         nn.Linear(state_dim, 64),
         nn.Tanh(),
         nn.Linear(64,64),
         nn.Tanh(),
      )
      self.mu_head = nn.Linear(64, action_dim)
      self.log_std = nn.Parameter(torch.zeros(action_dim))

   def forward(self, state):
      x = self.net(state)
      mu = self.mu_head(x)
      std = self.log_std.exp()
      return mu, std

   def act(self, state):
      mu, std = self.forward(state)
      dist = Normal(mu, std)
      action = dist.sample()
      log_prob = dist.log_prob(action).sum(dim=-1) #多维下所有动作的log概率
      return action, log_prob

class Critic((nn.Module)):
   """估计V(s)"""
   def __init__(self,state_dim):
      super().__init__()
      self.net = nn.Sequential(
         nn.Linear(state_dim,64)
         nn.Linear(64,64),
         nn.Tanh(),
         nn.Linear(64,1)
      )

   def forward(self,state):
      return self.net(state).squeeze(-1)


class PPO:
    def __init__(self, state_dim, action_dim, clip_eps=0.2, gamma=0.99, lam=0.95, lr=3e-4):

        # 两个 Actor：当前 & 旧
        self.actor = Actor(state_dim, action_dim)
        self.actor_old = Actor(state_dim, action_dim)
        self.actor_old.load_state_dict(self.actor.state_dict())

        # 两个 Critic：当前 & target (可选稳定)
        self.critic = Critic(state_dim)
        self.critic_target = Critic(state_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma, self.lam, self.clip_eps = gamma, lam, clip_eps

   def compute_gae(self, rewards, values, dones):
        """计算 GAE 优势估计"""
        # 优势函数：这个动作比平均值好多少 而Q值难算 用TD改进--> GAE
        # δt=rt+γV(st+1)−V(st) “实际得到的奖励 + 未来估计” 与 “我之前预测的价值” 的差距。
        adv, gae = [], 0
        next_value = 0
        # [s1, s2, s3, s4] 计算顺序：s4 → s3 → s2 → s1（反向遍历）
        for r, v, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            delta = r + self.gamma * next_value * (1 - done) - v
            gae = delta + self.gamma * self.lam * (1 - done) * gae
            adv.insert(0, gae)
            next_value = v
        return torch.tensor(adv, dtype=torch.float32)

   def update(self, batch):
        """核心PPO更新"""
        states = torch.stack(batch['states'])
        actions = torch.stack(batch['actions'])
        old_logps = torch.stack(batch['logps']).detach()
        rewards = batch['rewards']
        dones = batch['dones']

        with torch.no_grad():
            values = self.critic(states)
            adv = self.compute_gae(rewards, values, dones)
            returns = adv + values
            adv = (adv - adv.mean()) / (adv.std() + 1e-5)  # 标准化

        for _ in range(10):  # K epochs
            mu, std = self.actor(states)
            dist = Normal(mu, std)
            logp = dist.log_prob(actions).sum(-1)
            ratio = torch.exp(logp - old_logps)

            # PPO裁剪目标
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            loss_actor = -torch.mean(torch.min(surr1, surr2))

            # Critic目标
            value_pred = self.critic(states)
            loss_critic = F.mse_loss(value_pred, returns)

            # 总损失
            loss = loss_actor + 0.5 * loss_critic

            self.opt_actor.zero_grad()
            self.opt_critic.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.opt_actor.step()
            self.opt_critic.step()

        # 同步旧策略
        self.actor_old.load_state_dict(self.actor.state_dict())


# 旧actor网络收集数据
"""
obs = env.reset()
for t in range(T):
    action, logp = agent.actor_old.act(torch.tensor(obs, dtype=torch.float32))
    next_obs, reward, done, _ = env.step(action.numpy())
    buffer.append((obs, action, logp, reward, done))
    obs = next_obs if not done else env.reset()
```

## 总结
### Q-learning 系列算法：
- DQN 和 SARSA 都属于基于 Q 值的强化学习算法，通过学习状态-动作值函数来选择最优动作。
- DQN 使用 深度神经网络 来近似 Q 函数，而 SARSA 使用的是传统的 表格 Q 值 更新，适用于离散动作空间。
- DQN 是 Off-Policy 算法，而 SARSA 是 On-Policy 算法。

### 基于策略的算法
- REINFORCE、PPO 和 A2C 都属于 基于策略的强化学习算法，它们直接优化策略，而不是通过值函数（Q 值）来间接优化。
- PPO 和 A2C 是 Actor-Critic 类算法，它们同时学习一个 策略网络（Actor）和一个 值函数网络（Critic），其中 A2C 是 同步版本，而 PPO 是 异步版本，并通过 目标函数修剪 来稳定训练。
- REINFORCE 是最基础的 策略梯度方法，它直接优化策略，但 方差较大，训练过程可能不稳定。

### Soft Q-learning 系列
- SAC 是 基于策略的强化学习算法，但是它使用了 最大熵原则 来增强探索性，结合了 Q 值估计 和 策略优化，适用于连续动作空间。

## Something else

### 蒙特卡洛树搜索
MCTS 基于 树结构 进行搜索，每个节点代表一个状态，每条边代表从一个状态到另一个状态的转移。树的根节点表示当前状态，其他节点表示可能的未来状态。
常常被用于 博弈论 和 棋类游戏（如 围棋、象棋）等领域，能够在有限的计算资源下有效地进行决策和策略优化

#### 1. 选择（Selection）
- 从当前树的根节点出发，选择一条路径逐步进入到树的 叶节点（即未展开的节点），直到达到 树的叶子。
- 在选择过程中，会根据某种 策略 选择子节点。常见的策略是 上置信界（Upper Confidence Bound，UCB），它平衡了 探索（未充分探索的节点）和 利用（已知的好节点）。

#### 2. 扩展（Expansion）
- 一旦选择到一个未完全展开的叶节点，算法就会在该节点上扩展（即添加一个或多个新的子节点）。
- 扩展通常是随机的，即选择一个可能的动作或状态进行扩展，生成一个新的节点。

#### 3. 模拟（Simulation）
- 从新扩展的节点开始，进行一系列 模拟（或称为回合模拟）。在模拟阶段，算法会通过 随机决策 进行模拟，直到游戏结束（或达到某个终止条件）。
- 模拟的目的是估计该节点的胜率或评估值，这个评估值通过模拟完成后得到的结果来衡量，通常是胜利的概率或奖励的累积。

#### 4. 回传（Backpropagation）
- 一旦模拟完成，算法就会回传结果：更新每个节点的统计数据（如胜率、访问次数等）。
- 从叶节点开始，沿着路径回到根节点，将模拟结果更新到路径上所有节点的统计信息中。这个过程是 反向传播，帮助树中各节点调整其评估值。

