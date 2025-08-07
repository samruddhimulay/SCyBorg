import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np

def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MultiDiscreteSoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.observation_space.shape
        
        # Feature extraction layers
        self.fc1 = layer_init(nn.Linear(np.array(obs_shape).prod(), 256))
        self.fc2 = layer_init(nn.Linear(256, 256))
        
        # Separate Q-heads for each discrete action dimension
        self.action_dims = envs.action_space.nvec
        self.q_heads = nn.ModuleList([
            layer_init(nn.Linear(256, dim)) for dim in self.action_dims
        ])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output Q-values for each action dimension
        q_values = [head(x) for head in self.q_heads]
        return q_values

class MultiDiscreteActor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.observation_space.shape
        
        # Feature extraction layers
        self.fc1 = layer_init(nn.Linear(np.array(obs_shape).prod(), 256))
        self.fc2 = layer_init(nn.Linear(256, 256))
        
        # Separate policy heads for each discrete action dimension
        self.action_dims = envs.action_space.nvec
        self.policy_heads = nn.ModuleList([
            layer_init(nn.Linear(256, dim)) for dim in self.action_dims
        ])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output logits for each action dimension
        logits = [head(x) for head in self.policy_heads]
        return logits

    def get_action(self, x):
        logits = self(x)
        
        # Create separate categorical distributions for each action dimension
        policy_dists = [Categorical(logits=logit) for logit in logits]
        actions = [dist.sample() for dist in policy_dists]
        
        # Calculate log probabilities and action probabilities for each dimension
        log_probs = [F.log_softmax(logit, dim=1) for logit in logits]
        action_probs = [F.softmax(logit, dim=1) for logit in logits]
        
        # Stack actions into a single tensor
        actions = torch.stack(actions, dim=1)
        
        return actions, log_probs, action_probs

class SACMultiDiscrete:
    def __init__(self, envs, device, args):
        self.args = args
    
        # Handle VecNormalize wrapper
        if hasattr(envs, 'venv'):
            # If it's wrapped with VecNormalize, get the underlying environment
            base_env = envs.venv
        else:
            base_env = envs
        
        # Get action dimensions from the base environment
        if hasattr(base_env, 'single_action_space'):
            action_space = base_env.single_action_space
        else:
            # Fallback: get from one of the environments
            action_space = base_env.action_space
        
        self.action_dims = action_space.nvec
        self.device = device
        self.args = args
        
        # Initialize networks
        self.actor = MultiDiscreteActor(envs).to(device)
        self.qf1 = MultiDiscreteSoftQNetwork(envs).to(device)
        self.qf2 = MultiDiscreteSoftQNetwork(envs).to(device)
        self.qf1_target = MultiDiscreteSoftQNetwork(envs).to(device)
        self.qf2_target = MultiDiscreteSoftQNetwork(envs).to(device)
        
        # Copy weights to target networks
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        
        # Optimizers
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), 
            lr=args.q_lr, eps=1e-4
        )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()), 
            lr=args.policy_lr, eps=1e-4
        )
        
        # Automatic entropy tuning for each action dimension
        if args.autotune:
            self.target_entropy = [-args.target_entropy_scale * torch.log(1 / torch.tensor(dim, dtype=torch.float32)) 
                                 for dim in self.action_dims]
            self.log_alpha = [torch.zeros(1, requires_grad=True, device=device) 
                            for _ in self.action_dims]
            self.alpha = [log_alpha.exp().item() for log_alpha in self.log_alpha]
            self.a_optimizers = [optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4) 
                               for log_alpha in self.log_alpha]
        else:
            self.alpha = [args.alpha] * len(self.action_dims)

    def update(self, data,writer, global_step):
        # CRITIC training
        with torch.no_grad():
            _, next_state_log_pi, next_state_action_probs = self.actor.get_action(data["next_observations"])
            qf1_next_target = self.qf1_target(data["next_observations"])
            qf2_next_target = self.qf2_target(data["next_observations"])
            
            # Calculate min Q-target for each action dimension
            min_qf_next_target = []
            for i, (log_pi, action_probs, q1, q2, alpha) in enumerate(
                zip(next_state_log_pi, next_state_action_probs, qf1_next_target, qf2_next_target, self.alpha)
            ):
                min_q = torch.min(q1, q2)
                target = action_probs * (min_q - alpha * log_pi)
                min_qf_next_target.append(target.sum(dim=1))
            
            # Sum targets across all action dimensions
            min_qf_next_target = torch.stack(min_qf_next_target, dim=1).sum(dim=1)
            next_q_value = data["rewards"].flatten() + (1 - data["dones"].flatten()) * self.args.gamma * min_qf_next_target

        # Get Q-values for taken actions
        qf1_values = self.qf1(data["observations"])
        qf2_values = self.qf2(data["observations"])
        
        qf1_a_values = []
        qf2_a_values = []
        for i, (q1, q2) in enumerate(zip(qf1_values, qf2_values)):
            action_idx = data["actions"][:, i].long()
            qf1_a_values.append(q1.gather(1, action_idx.unsqueeze(1)).squeeze(1))
            qf2_a_values.append(q2.gather(1, action_idx.unsqueeze(1)).squeeze(1))
        
        qf1_a_values = torch.stack(qf1_a_values, dim=1).sum(dim=1)
        qf2_a_values = torch.stack(qf2_a_values, dim=1).sum(dim=1)
        
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # ACTOR training
        _, log_pi, action_probs = self.actor.get_action(data["observations"])
        with torch.no_grad():
            qf1_values = self.qf1(data["observations"])
            qf2_values = self.qf2(data["observations"])
        
        actor_loss = 0
        for i, (log_p, act_probs, q1, q2, alpha) in enumerate(
            zip(log_pi, action_probs, qf1_values, qf2_values, self.alpha)
        ):
            min_qf_values = torch.min(q1, q2)
            actor_loss += (act_probs * ((alpha * log_p) - min_qf_values)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha (entropy coefficient) for each action dimension
        if self.args.autotune:
            for i, (log_p, act_probs, log_alpha, target_ent, optimizer) in enumerate(
                zip(log_pi, action_probs, self.log_alpha, self.target_entropy, self.a_optimizers)
            ):
                alpha_loss = (act_probs.detach() * (-log_alpha.exp() * (log_p + target_ent).detach())).mean()
                
                optimizer.zero_grad()
                alpha_loss.backward()
                optimizer.step()
                self.alpha[i] = log_alpha.exp().item()
        
        # TensorBoard logging
        writer.add_scalar("loss/qf1", qf1_loss.item(), global_step)
        writer.add_scalar("loss/qf2", qf2_loss.item(), global_step)
        writer.add_scalar("loss/q_total", qf1_loss.item() + qf2_loss.item(), global_step)
        writer.add_scalar("loss/actor", actor_loss.item(), global_step)

        if self.args.autotune:
            for i, alpha_val in enumerate(self.alpha):
                writer.add_scalar(f"entropy/alpha_{i}", alpha_val, global_step)

        return {
            'qf1_loss': qf1_loss.item(),
            'qf2_loss': qf2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha[0]  # Log first alpha for monitoring
        }

    def update_target_networks(self):
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
