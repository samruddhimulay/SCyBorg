import numpy as np
from gymnasium import spaces, Env


class DefenderCyborgEnv(Env):
    def __init__(self):
        # State: [compromised, service1_down, service2_down, attack_detected]
        self.observation_space = spaces.MultiBinary(4)
        # Actions: 0=nop, 1=patch, 2=restore, 3=monitor, 4=scan
        self.action_space = spaces.Discrete(5)
        self.state = np.array([0, 0, 0, 0])
        self.max_steps = 20
        self.current_step = 0
        
        # Attack probabilities - tune these for difficulty
        self.compromise_prob = 0.15
        self.service1_attack_prob = 0.10
        self.service2_attack_prob = 0.10
        
        # Reward parameters
        self.successful_defense_reward = 10
        self.unnecessary_action_penalty = -2
        self.security_maintenance_reward = 1
        self.service_disruption_penalty = -5
        
        # Episode tracking
        self.episode_count = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_security_breaches = 0
        self.episode_successful_defenses = 0
        self.episode_unnecessary_actions = 0
        self.episode_attacks_occurred = 0
        self.action_counts = {i: 0 for i in range(5)}
    
    def seed(self,seed):
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Prepare episode summary for info if not the first episode
        episode_info = {}
        if self.episode_count > 0:
            episode_info = self._get_episode_summary()
        
        # Reset episode tracking
        self.state = np.array([0, 0, 0, 0])
        self.current_step = 0
        self.episode_count += 1
        self.episode_rewards = []
        self.episode_security_breaches = 0
        self.episode_successful_defenses = 0
        self.episode_unnecessary_actions = 0
        self.episode_attacks_occurred = 0
        self.action_counts = {i: 0 for i in range(5)}
        
        return self.state.copy()

    def _simulate_attacks(self):
        """Simulate external attacks each step"""
        compromised, service1_down, service2_down, attack_detected = self.state
        attacks_this_step = 0
        
        # Random attacks occur independently
        if not compromised and np.random.rand() < self.compromise_prob:
            compromised = 1
            attacks_this_step += 1
            
        if not service1_down and np.random.rand() < self.service1_attack_prob:
            service1_down = 1
            attacks_this_step += 1
            
        if not service2_down and np.random.rand() < self.service2_attack_prob:
            service2_down = 1
            attacks_this_step += 1
            
        # Track attacks for logging
        self.episode_attacks_occurred += attacks_this_step
        
        # Reset attack detection flag (it needs to be actively maintained)
        attack_detected = 0
        
        return np.array([compromised, service1_down, service2_down, attack_detected]), attacks_this_step

    def _calculate_base_reward(self):
        compromised, service1_down, service2_down, _ = self.state
        reward = 0
        breach_count = 0

        if compromised:
            reward -= 5
            breach_count += 1
        if service1_down:
            reward -= 5
            breach_count += 1
        if service2_down:
            reward -= 5
            breach_count += 1

        if not any([compromised, service1_down, service2_down]):
            reward += 2  # Stronger reward for full system uptime

        self.episode_security_breaches += breach_count
        return reward, breach_count

    def step(self, action):
        # Apply external attacks first
        self.state, attacks_this_step = self._simulate_attacks()
        compromised, service1_down, service2_down, attack_detected = self.state
        
        reward = 0
        done = False
        truncated = False
        info = {'action_successful': False}
        
        # Track action usage
        self.action_counts[action] += 1

        # Handle defensive actions only
        # --- ACTION REWARD LOGIC ---
        if action == 0:  # nop
            info['action_taken'] = 'no_operation'
            
        elif action == 1:  # patch
            if compromised == 1:
                compromised = 0
                reward += 10
                info['action_successful'] = True
                info['action_taken'] = 'patch_successful'
                self.episode_successful_defenses += 1
            else:
                reward -= 2
                info['action_taken'] = 'patch_unnecessary'
                self.episode_unnecessary_actions += 1
                
        elif action == 2:  # restore
            if service1_down == 1:
                service1_down = 0
                reward += 10
                info['action_successful'] = True
                info['action_taken'] = 'restore_service1'
                self.episode_successful_defenses += 1
            elif service2_down == 1:
                service2_down = 0
                reward += 10
                info['action_successful'] = True
                info['action_taken'] = 'restore_service2'
                self.episode_successful_defenses += 1
            else:
                reward -= 2
                info['action_taken'] = 'restore_unnecessary'
                self.episode_unnecessary_actions += 1
                
        elif action == 3:  # monitor
            if compromised or service1_down or service2_down:
                attack_detected = 1
                reward += 5
                info['action_successful'] = True
                info['action_taken'] = 'monitor_threat_detected'
                self.episode_successful_defenses += 1
            else:
                reward -= 2
                info['action_taken'] = 'monitor_no_threat'
                self.episode_unnecessary_actions += 1

        elif action == 4:  # scan
            if attack_detected == 1:
                attack_detected = 0
                reward += 1  # optional: only if clearing a real flag
                info['action_taken'] = 'scan_cleared'
            else:
                reward -= 1  # discourage mindless scanning
                info['action_taken'] = 'scan_unnecessary'
                self.episode_unnecessary_actions += 1


        # Update state
        self.state = np.array([compromised, service1_down, service2_down, attack_detected])
        
        # Add base reward for current security state
        base_reward, breach_count = self._calculate_base_reward()
        reward += base_reward
        
        # Track rewards
        self.episode_rewards.append(reward)
        
        # Increment step counter
        self.current_step += 1
        self.total_steps += 1

        # Episode termination conditions
        if service1_down == 1 and service2_down == 1:
            done = True
            reward -= 20
            info['termination_reason'] = 'total_system_failure'
        elif self.current_step >= self.max_steps:
            done = True
            info['termination_reason'] = 'max_steps_reached'
            # Bonus for surviving full episode
            if not any([compromised, service1_down, service2_down]):
                reward += 15

        # Add all important metrics to info for InfoCallback
        info.update({
            # Step-level metrics
            'step_reward': reward,
            'base_reward': base_reward,
            'action_taken_id': action,
            'attacks_this_step': attacks_this_step,
            'server_compromised': float(compromised),
            'service1_down': float(service1_down),
            'service2_down': float(service2_down),
            'attack_detected': float(attack_detected),
            'security_score': 3 - breach_count,
            'active_breaches': breach_count,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            
            # Episode tracking (current values)
            'episode_successful_defenses': self.episode_successful_defenses,
            'episode_unnecessary_actions': self.episode_unnecessary_actions,
            'episode_attacks_occurred': self.episode_attacks_occurred,
            'episode_security_breaches': self.episode_security_breaches,
            'current_episode_reward': sum(self.episode_rewards),
            'action_counts': self.action_counts.copy(),
        })

        # Add episode summary if episode is done
        if done:
            info['episode_summary'] = self._get_episode_summary()

        return self.state.copy(), reward, done, truncated, info

    def _get_episode_summary(self):
        """Get episode-level statistics for InfoCallback"""
        if not self.episode_rewards:
            return {}
            
        episode_reward = sum(self.episode_rewards)
        episode_length = len(self.episode_rewards)
        
        # Defense efficiency
        total_defensive_actions = self.episode_successful_defenses + self.episode_unnecessary_actions
        defense_efficiency = (self.episode_successful_defenses / total_defensive_actions 
                            if total_defensive_actions > 0 else 0)
        
        # Action distribution
        total_actions = sum(self.action_counts.values())
        action_distribution = {}
        if total_actions > 0:
            action_names = ['nop', 'patch', 'restore', 'monitor', 'scan']
            for action_id, count in self.action_counts.items():
                action_distribution[action_names[action_id]] = count / total_actions
        
        # Reward composition analysis
        positive_rewards = [r for r in self.episode_rewards if r > 0]
        negative_rewards = [r for r in self.episode_rewards if r < 0]
        
        return {
            'total_reward': episode_reward,
            'episode_length': episode_length,
            'average_reward': episode_reward / max(episode_length, 1),
            'security_breaches': self.episode_security_breaches,
            'successful_defenses': self.episode_successful_defenses,
            'unnecessary_actions': self.episode_unnecessary_actions,
            'attacks_occurred': self.episode_attacks_occurred,
            'defense_efficiency': defense_efficiency,
            'action_distribution': action_distribution,
            'positive_reward_steps': len(positive_rewards),
            'negative_reward_steps': len(negative_rewards),
            'average_positive_reward': np.mean(positive_rewards) if positive_rewards else 0,
            'average_negative_reward': np.mean(negative_rewards) if negative_rewards else 0,
        }

    def render(self):
        compromised, service1_down, service2_down, attack_detected = self.state
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Server compromised: {bool(compromised)}")
        print(f"Service1 down: {bool(service1_down)}")
        print(f"Service2 down: {bool(service2_down)}")
        print(f"Attack detected: {bool(attack_detected)}")
        print(f"Security Score: {3 - sum(self.state[:3])}/3")
        print("Available actions: 0=nop, 1=patch, 2=restore, 3=monitor, 4=scan")
        print("---")

    def get_action_meanings(self):
        return {
            0: "No Operation",
            1: "Patch (fix compromised server)",
            2: "Restore (fix disrupted services)", 
            3: "Monitor (detect threats)",
            4: "Scan (clear detection flags)"
        }
