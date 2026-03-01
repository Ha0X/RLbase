import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparams
GAMMA = 0.99
LR = 1e-3
HIDDEN_SIZE = 128
MAX_EPISODES = 5000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_SIZE)
        self.actor = nn.Linear(HIDDEN_SIZE, action_dim)
        self.critic = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        logits = self.actor(x)
        probs = F.softmax(logits, dim=-1)
        value = self.critic(x)
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

        while not done:
            state_tensor = torch.tensor([state], dtype=torch.float32, device=DEVICE)
            probs, value = model(state_tensor)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            values.append(value)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            rewards.append(float(reward))
            state = next_state

        log_probs = torch.stack(log_probs)
        values = torch.cat(values).squeeze()

        with torch.no_grad():
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
            next_values = torch.cat([values[1:], torch.zeros(1, device=DEVICE)])
            td_targets = rewards_t + GAMMA * next_values

        advantages = td_targets - values

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 10 == 0:
            total_reward = float(np.sum(rewards))
            print(f"Episode {episode+1}: return={total_reward}, loss={loss.item():.3f}")

    env.close()


def main():
    train()


if __name__ == "__main__":
    main()



