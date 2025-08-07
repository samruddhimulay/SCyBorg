import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime
from Environment import *
from gymnasium import ObservationWrapper
from gymnasium.spaces import MultiDiscrete
from SAC_MULTI_DISCRETE import *

class DiscreteToMultiDiscreteWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.action_space = MultiDiscrete([env.action_space.n])


class ReplayBuffer:
    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr, self.size = 0, 0

    def add(self, obs, next_obs, action, reward, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            observations=torch.tensor(self.obs_buf[idxs]).float().to(device),
            next_observations=torch.tensor(self.next_obs_buf[idxs]).float().to(device),
            actions=torch.tensor(self.actions[idxs]).long().to(device),
            rewards=torch.tensor(self.rewards[idxs]).float().to(device),
            dones=torch.tensor(self.dones[idxs]).float().to(device),
        )



# === Setup ===
env = DiscreteToMultiDiscreteWrapper(DefenderCyborgEnv())
obs_shape = env.observation_space.shape
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Training Parameters ===
total_timesteps = 100_000
learning_starts = 1000
update_freq = 2
batch_size = 64
target_update_freq = 250
gamma = 0.99
tau = 0.005
policy_lr = 3e-4
q_lr = 3e-4
alpha = 0.2
autotune = False

# === Logger ===
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(f"runs/sac_cyborg_{timestamp}")

# === Agent ===
class Args:
    gamma = gamma
    tau = tau
    alpha = alpha
    policy_lr = policy_lr
    q_lr = q_lr
    autotune = autotune
    target_entropy_scale = 1.0

sac_agent = SACMultiDiscrete(env, device, Args)
replay_buffer = ReplayBuffer(100_000, obs_shape)


obs = env.reset()
global_step = 0
best_reward = -np.inf
ep_return, ep_len = 0, 0
returns = []

for global_step in range(total_timesteps):
    if global_step < learning_starts:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            action_tensor, _, _ = sac_agent.actor.get_action(obs_tensor)
            action = int(action_tensor.item())


    next_obs, reward, terminated, truncated, info = env.step(int(action if np.isscalar(action) else action[0]))

    done = terminated or truncated

    replay_buffer.add(obs, next_obs, action, reward, done)

    obs = next_obs
    ep_return += reward
    ep_len += 1

    if done:
        writer.add_scalar("charts/episodic_return", ep_return, global_step)
        writer.add_scalar("charts/episodic_length", ep_len, global_step)
        obs= env.reset()
        returns.append(ep_return)
        ep_return, ep_len = 0, 0

    if global_step % 1000 == 0 and global_step > 0:
        print(f"[Step {global_step}] Recent Return: {ep_return:.2f}, Episode Length: {ep_len}, Total Episodes: {len(returns)}")
        
        if returns:
            mean_return = np.mean(returns[-10:])
            print(f"🔁 Last 10 Episodes Mean Return: {mean_return:.2f}")
            writer.add_scalar("charts/mean_return_last_10", mean_return, global_step)

    # Update SAC
    if global_step > learning_starts and global_step % update_freq == 0:
        batch = replay_buffer.sample(batch_size)
        losses = sac_agent.update(batch, writer, global_step)

    # Target network update
    if global_step > learning_starts and global_step % target_update_freq == 0:
        sac_agent.update_target_networks()

print("✅ Training complete.")
writer.close()
