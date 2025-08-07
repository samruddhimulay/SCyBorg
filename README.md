<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Cybersecurity Defense Simulation Environments

A collection of reinforcement learning environments for cybersecurity defense training, featuring three evolutionary versions with different reward philosophies and increasing sophistication in preventing reward hacking.

## 🎯 Overview

These environments simulate a cybersecurity scenario where an RL agent must defend a server and two services against probabilistic attacks. Each version represents an evolution in design philosophy, from traditional positive rewards (V1) to zero-optimal penalty-focused systems (V2, V3).

## 🏗️ Environment Architecture

### Core Structure

- **Observation Space**: 4-bit binary vector `[server_compromised, service1_down, service2_down, attack_detected]`
- **Action Space**: 5 discrete actions `{nop, patch, restore, monitor, scan}`
- **Episode Length**: 20 steps maximum (early termination on total failure)
- **Attack Model**: Probabilistic attacks with dynamic probability adjustment


### Actions Available

| Action | ID | Description | Cooldown |
| :-- | :-- | :-- | :-- |
| **No Operation** | 0 | Do nothing (incurs inaction penalty if threats exist) | None |
| **Patch** | 1 | Fix compromised server (requires detection) | None |
| **Restore** | 2 | Restore downed services (requires detection) | None |
| **Monitor** | 3 | Detect threats and set detection flag | 2 steps |
| **Scan** | 4 | Clear detection flag | 2 steps |

## 📋 Version Comparison

| Feature | V1 (Traditional) | V2 (Zero-Optimal) | V3 (Enhanced) |
| :-- | :-- | :-- | :-- |
| **Reward Philosophy** | Positive rewards for success | Zero-optimal (penalty-focused) | Enhanced zero-optimal |
| **Observation Noise** | 10% | 10% | 5% |
| **Max Reward** | ~120 | 0 | 0 |
| **Inaction Penalty** | -2 | -1 | -3 |
| **Primary Focus** | Reward maximization | Penalty minimization | Exploit mitigation |

## 🚀 Getting Started

### Installation

```bash
pip install gymnasium
pip install numpy
```


### Basic Usage

```python
import gymnasium as gym
from your_env_module import CyberSecurityEnv

# Initialize environment (replace with specific version)
env = CyberSecurityEnvV3()

# Reset environment
observation, info = env.reset()

# Run episode
done = False
total_reward = 0

while not done:
    # Choose action (random example)
    action = env.action_space.sample()
    
    # Step environment
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Episode reward: {total_reward}")
```


## 🔧 Version Details

### V1: Traditional Reward-Based

**Best for**: Learning basic RL concepts, positive reinforcement scenarios

- **Strengths**: Clear positive incentives, intuitive reward structure
- **Weaknesses**: Susceptible to reward hacking, detection cycle exploitation
- **Max Theoretical Reward**: ~120 (perfect defense + survival bonus)

```python
# V1 Key Rewards
patch_success = +15
restore_per_service = +10
monitor_success = +3
full_uptime_bonus = +3
perfect_survival = +20
```


### V2: Zero-Optimal Design

**Best for**: Understanding penalty-driven learning, avoiding reward hacking

- **Strengths**: Eliminates primary reward hacking vectors
- **Weaknesses**: Lacks positive feedback, may encourage excessive inaction
- **Max Theoretical Reward**: 0 (perfect penalty avoidance)

```python
# V2 All positive rewards set to 0
# Focus on penalty avoidance:
inaction_penalty = -1
breach_penalty = -1 (scaled by duration)
detection_ignored = -5
```


### V3: Enhanced Anti-Exploit

**Best for**: Advanced RL research, robust policy development

- **Strengths**: Addresses specific exploits, reduced observation noise
- **Weaknesses**: Complex penalty structure, potentially over-engineered
- **Max Theoretical Reward**: 0 (enhanced penalty avoidance)

```python
# V3 Enhanced penalties
inaction_penalty = -3
scan_unnecessary_penalty = -4
fix_without_scan_penalty = -3  # New
observation_noise = 0.05  # Reduced from 0.10
```


## 🎮 Environment States

### Observation Vector

```python
[server_compromised, service1_down, service2_down, attack_detected]
# Example: [1, 0, 1, 1] means server compromised, service2 down, attack detected
```


### Attack Dynamics

- **Server Compromise**: 15% base probability per step
- **Service Disruption**: 10% base probability per service per step
- **Dynamic Adjustment**: Probabilities reduce by 20% after 3 consecutive secure steps
- **Noise Simulation**: Observations corrupted with configurable probability


## 📊 Training Considerations

### Recommended Algorithms

- **V1**: PPO, A2C (responds well to positive rewards)
- **V2**: SAC, TD3 (handles sparse rewards better)
- **V3**: PPO with entropy regularization (navigates complex penalty landscape)


### Hyperparameter Suggestions

```python
# Example for V3 with PPO
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
n_epochs = 10
clip_range = 0.2
ent_coef = 0.01  # Important for exploration in penalty-heavy environments
```


## 🔍 Metrics and Evaluation

### Key Performance Indicators

- **Total Episode Reward**: Primary optimization target
- **Breach Count**: Number of security incidents per episode
- **Defense Success Rate**: Percentage of successful defensive actions
- **Episode Length**: Steps survived before termination
- **Action Distribution**: Balance of defensive vs passive actions


### Benchmark Performance

```python
# Rough performance targets (environment-specific)
V1_good_performance = total_reward > 50
V2_good_performance = total_reward > -20
V3_good_performance = total_reward > -30
```


## 🚨 Known Issues and Limitations

### Reward Hacking Vulnerabilities

- **V1**: Detection cycle exploitation, temporal gaming
- **V2**: Excessive nop abuse, detection flag manipulation
- **V3**: Probabilistic nop abuse, cooldown timing abuse


### Design Considerations

- Zero-optimal design may slow learning convergence
- High penalty variance can cause training instability
- Observation noise requires robust policy architectures


## 🤝 Contributing

We welcome contributions to improve these environments! Areas of interest:

- Alternative reward structures
- Additional attack patterns
- Performance optimizations
- Documentation improvements


## 📄 License

[Your chosen license here]

## 📚 Citation

If you use these environments in your research, please cite:

```bibtex
@misc{cybersec_envs_2025,
    title={Cybersecurity Defense Simulation Environments: V1-V3},
    year={2025},
    note={Reinforcement Learning Environments for Cybersecurity Defense Training}
}
```

**Note**: These environments are designed for research and educational purposes. They simulate cybersecurity scenarios but should not be used as substitutes for real security systems or training.

<div style="text-align: center">⁂</div>

[^1]: V3_Report.pdf

[^2]: V2_Report.pdf

[^3]: V1_Report.pdf

