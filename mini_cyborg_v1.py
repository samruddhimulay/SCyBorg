import numpy as np
from gymnasium import Env, spaces


class DefenderCyborgEnv(Env):
    def __init__(self):
        # State: [compromised, service1_down, service2_down, attack_detected]
        self.observation_space = spaces.MultiBinary(4)
        # Actions: 0=nop, 1=patch, 2=restore, 3=monitor, 4=scan
        self.action_space = spaces.Discrete(5)
        self.state = np.array([0, 0, 0, 0])
        self.max_steps = 20
        self.current_step = 0

        # Base attack probabilities
        self.base_compromise_prob = 0.15
        self.base_service1_attack_prob = 0.10
        self.base_service2_attack_prob = 0.10
        self.current_compromise_prob = self.base_compromise_prob
        self.current_service1_attack_prob = self.base_service1_attack_prob
        self.current_service2_attack_prob = self.base_service2_attack_prob

        # Reward parameters (revised for exploit prevention)
        self.patch_reward = 15
        self.restore_per_service_reward = 10
        self.monitor_reward = 3
        self.monitor_enable_bonus = 2  # New: bonus if monitor enables a fix
        self.scan_reward = 5
        self.unnecessary_action_penalty = -3
        self.scan_unnecessary_penalty = -2
        self.base_persistent_breach_penalty = -8
        self.inaction_penalty = -2
        self.detection_ignored_penalty = -5
        self.full_uptime_reward = 3
        self.partial_uptime_reward = 1
        self.failure_penalty = -30
        self.perfect_survival_bonus = 20
        self.action_cost = -1  # Multi-objective efficiency penalty

        # Mitigation trackers
        self.consecutive_secure_steps = 0  # For dynamic probs
        self.breach_duration_consecutive = 0  # For graduated penalties
        self.cooldowns = {3: 0, 4: 0}  # Cooldown counters for actions 3 and 4
        self.observation_noise_prob = 0.10  # For uncertainty modeling
        self.detection_required_fix_attempts = (
            0  # New: track failed fixes due to no detection
        )

        # Episode tracking
        self.episode_count = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_security_breaches = 0
        self.episode_breach_duration = 0
        self.episode_successful_defenses = 0
        self.episode_unnecessary_actions = 0
        self.episode_attacks_occurred = 0
        self.action_counts = {i: 0 for i in range(5)}

    def reset(self, seed=None):
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
        self.episode_breach_duration = 0
        self.episode_successful_defenses = 0
        self.episode_unnecessary_actions = 0
        self.episode_attacks_occurred = 0
        self.action_counts = {i: 0 for i in range(5)}
        self.lingering_detection_penalty = -2  # Penalty for unnecessary detection flag
        self.clean_state_reward = (
            2  # Reward for maintaining attack_detected = 0 when safe
        )

        # Reset mitigations
        self.consecutive_secure_steps = 0
        self.breach_duration_consecutive = 0
        self.cooldowns = {3: 0, 4: 0}
        self.current_compromise_prob = self.base_compromise_prob
        self.current_service1_attack_prob = self.base_service1_attack_prob
        self.current_service2_attack_prob = self.base_service2_attack_prob
        self.detection_required_fix_attempts = 0

        return self._get_noisy_observation(), episode_info

    def _get_noisy_observation(self):
        """Add observation noise: 10% false positive/negative per binary state."""
        noisy_state = self.state.copy()
        for i in range(4):  # For each state dimension
            if np.random.rand() < self.observation_noise_prob:
                noisy_state[i] = 1 - noisy_state[i]  # Flip the bit
        return noisy_state

    def _simulate_attacks(self):
        """Simulate external attacks each step with dynamic probabilities."""
        compromised, service1_down, service2_down, attack_detected = self.state
        attacks_this_step = 0

        # Random attacks occur independently with current probs
        if not compromised and np.random.rand() < self.current_compromise_prob:
            compromised = 1
            attacks_this_step += 1

        if not service1_down and np.random.rand() < self.current_service1_attack_prob:
            service1_down = 1
            attacks_this_step += 1

        if not service2_down and np.random.rand() < self.current_service2_attack_prob:
            service2_down = 1
            attacks_this_step += 1

        # Track attacks for logging
        self.episode_attacks_occurred += attacks_this_step

        # Detection flag persists unless cleared (no auto-reset)
        if not (compromised or service1_down or service2_down):
            attack_detected = 0  # Reset only if no threats

        return np.array(
            [compromised, service1_down, service2_down, attack_detected]
        ), attacks_this_step

    def _update_dynamic_probs(self, breach_count):
        """Adjust attack probabilities based on consecutive secure steps."""
        if breach_count == 0:
            self.consecutive_secure_steps += 1
            if self.consecutive_secure_steps >= 3:
                # Reduce probs by 20% (multiplicative)
                self.current_compromise_prob = self.base_compromise_prob * 0.8
                self.current_service1_attack_prob = self.base_service1_attack_prob * 0.8
                self.current_service2_attack_prob = self.base_service2_attack_prob * 0.8
        else:
            self.consecutive_secure_steps = 0
            # Reset to base
            self.current_compromise_prob = self.base_compromise_prob
            self.current_service1_attack_prob = self.base_service1_attack_prob
            self.current_service2_attack_prob = self.base_service2_attack_prob

    def _calculate_base_reward(self):
        compromised, service1_down, service2_down, attack_detected = self.state
        reward = 0
        breach_count = sum([compromised, service1_down, service2_down])

        # Update consecutive breach duration
        if breach_count > 0:
            self.breach_duration_consecutive += 1
        else:
            self.breach_duration_consecutive = 0

        # Graduated penalties for persistent breaches (per issue)
        scaled_penalty = self.base_persistent_breach_penalty * (
            1 + self.breach_duration_consecutive / 5.0
        )
        reward += breach_count * scaled_penalty

        # Penalty for lingering detection when no threats exist
        if not breach_count and attack_detected:
            reward += self.lingering_detection_penalty  # New penalty

        # Reward for maintaining a clean state (no breaches, no detection)
        if not breach_count and not attack_detected:
            reward += self.clean_state_reward  # New reward

        # Uptime rewards
        if breach_count == 0:
            reward += self.full_uptime_reward
        elif breach_count < 3:
            reward += self.partial_uptime_reward

        # Track breaches
        self.episode_security_breaches += breach_count
        self.episode_breach_duration += 1 if breach_count > 0 else 0

        return reward, breach_count

    def step(self, action):
        # Decrement cooldowns
        for act in self.cooldowns:
            if self.cooldowns[act] > 0:
                self.cooldowns[act] -= 1

        # Apply external attacks first
        self.state, attacks_this_step = self._simulate_attacks()
        compromised, service1_down, service2_down, attack_detected = self.state

        reward = 0
        done = False
        truncated = False
        info = {"action_successful": False}

        # Track action usage
        self.action_counts[action] += 1

        # Multi-objective action cost (except no-op)
        if action != 0:
            reward += self.action_cost

        # Handle defensive actions
        if action == 0:  # nop
            info["action_taken"] = "no_operation"
            # Penalty for inaction if threats exist
            if compromised or service1_down or service2_down:
                reward += self.inaction_penalty

        elif action == 1:  # patch (requires detection)
            if attack_detected:  # Detection required
                if compromised:
                    compromised = 0
                    reward += self.patch_reward
                    info["action_successful"] = True
                    info["action_taken"] = "patch_successful"
                    self.episode_successful_defenses += 1
                else:
                    reward += self.unnecessary_action_penalty
                    info["action_taken"] = "patch_unnecessary"
                    self.episode_unnecessary_actions += 1
            else:  # Failed due to no detection
                reward += self.unnecessary_action_penalty
                info["action_taken"] = "patch_failed_no_detection"
                self.episode_unnecessary_actions += 1
                self.detection_required_fix_attempts += 1

        elif action == 2:  # restore (requires detection)
            if attack_detected:  # Detection required
                services_fixed = 0
                if service1_down:
                    service1_down = 0
                    services_fixed += 1
                if service2_down:
                    service2_down = 0
                    services_fixed += 1
                if services_fixed > 0:
                    reward += services_fixed * self.restore_per_service_reward
                    info["action_successful"] = True
                    info["action_taken"] = (
                        f"restore_successful_{services_fixed}_services"
                    )
                    self.episode_successful_defenses += services_fixed
                else:
                    reward += self.unnecessary_action_penalty
                    info["action_taken"] = "restore_unnecessary"
                    self.episode_unnecessary_actions += 1
            else:  # Failed due to no detection
                reward += self.unnecessary_action_penalty
                info["action_taken"] = "restore_failed_no_detection"
                self.episode_unnecessary_actions += 1
                self.detection_required_fix_attempts += 1

        elif action == 3:  # monitor (with cooldown)
            if self.cooldowns[3] > 0:
                reward += self.unnecessary_action_penalty
                info["action_taken"] = "monitor_on_cooldown"
                self.episode_unnecessary_actions += 1
            else:
                threats = compromised or service1_down or service2_down
                if threats and not attack_detected:
                    attack_detected = 1
                    reward += self.monitor_reward
                    # Bonus if this enables a fix (check if breaches exist)
                    if threats:
                        reward += self.monitor_enable_bonus
                    info["action_successful"] = True
                    info["action_taken"] = "monitor_threat_detected"
                    self.episode_successful_defenses += 1
                    self.cooldowns[3] = 2  # Set cooldown
                else:
                    reward += self.unnecessary_action_penalty
                    info["action_taken"] = "monitor_no_threat_or_already_detected"
                    self.episode_unnecessary_actions += 1

        elif action == 4:  # scan (with cooldown)
            if self.cooldowns[4] > 0:
                reward += self.scan_unnecessary_penalty
                info["action_taken"] = "scan_on_cooldown"
                self.episode_unnecessary_actions += 1
            else:
                if attack_detected:
                    attack_detected = 0
                    reward += self.scan_reward
                    info["action_successful"] = True
                    info["action_taken"] = "scan_cleared"
                    self.episode_successful_defenses += 1
                    self.cooldowns[4] = 2  # Set cooldown
                else:
                    reward += self.scan_unnecessary_penalty
                    info["action_taken"] = "scan_unnecessary"
                    self.episode_unnecessary_actions += 1

        # Update state
        self.state = np.array(
            [compromised, service1_down, service2_down, attack_detected]
        )

        # Add base reward for current security state
        base_reward, breach_count = self._calculate_base_reward()
        reward += base_reward

        # Update dynamic probabilities
        self._update_dynamic_probs(breach_count)

        # Track rewards
        self.episode_rewards.append(reward)

        # Increment step counter
        self.current_step += 1
        self.total_steps += 1

        # Episode termination conditions
        if service1_down and service2_down:
            done = True
            reward += self.failure_penalty
            info["termination_reason"] = "total_system_failure"
        elif self.current_step >= self.max_steps:
            done = True
            info["termination_reason"] = "max_steps_reached"
            # Bonus only for perfect security (zero breaches entire episode)
            if self.episode_security_breaches == 0:
                reward += self.perfect_survival_bonus

        # Add all important metrics to info for InfoCallback
        info.update(
            {
                # Step-level metrics
                "step_reward": reward,
                "base_reward": base_reward,
                "action_taken_id": action,
                "attacks_this_step": attacks_this_step,
                "server_compromised": float(compromised),
                "service1_down": float(service1_down),
                "service2_down": float(service2_down),
                "attack_detected": float(attack_detected),
                "security_score": 3 - breach_count,
                "active_breaches": breach_count,
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "episode_count": self.episode_count,
                "current_attack_probs": {
                    "compromise": self.current_compromise_prob,
                    "service1": self.current_service1_attack_prob,
                    "service2": self.current_service2_attack_prob,
                },
                "detection_required_fix_attempts": self.detection_required_fix_attempts,  # New metric
                # Episode tracking (current values)
                "episode_successful_defenses": self.episode_successful_defenses,
                "episode_unnecessary_actions": self.episode_unnecessary_actions,
                "episode_attacks_occurred": self.episode_attacks_occurred,
                "episode_security_breaches": self.episode_security_breaches,
                "episode_breach_duration": self.episode_breach_duration,
                "current_episode_reward": sum(self.episode_rewards),
                "action_counts": self.action_counts.copy(),
            }
        )

        # Add episode summary if episode is done
        if done:
            info["episode_summary"] = self._get_episode_summary()

        return self._get_noisy_observation(), reward, done, truncated, info

    def _get_episode_summary(self):
        """Get episode-level statistics for InfoCallback with temporal shaping."""
        if not self.episode_rewards:
            return {}

        episode_reward = sum(self.episode_rewards)
        episode_length = len(self.episode_rewards)

        # Temporal reward shaping: Weight last 5 steps 2x
        recent_steps = min(5, episode_length)
        recent_rewards = self.episode_rewards[-recent_steps:]
        early_rewards = (
            self.episode_rewards[:-recent_steps]
            if recent_steps < episode_length
            else []
        )
        recent_performance_score = (
            sum(early_rewards) + 2 * sum(recent_rewards)
        ) / episode_length

        # Defense efficiency
        total_defensive_actions = (
            self.episode_successful_defenses + self.episode_unnecessary_actions
        )
        defense_efficiency = (
            self.episode_successful_defenses / total_defensive_actions
            if total_defensive_actions > 0
            else 0
        )

        # Action distribution
        total_actions = sum(self.action_counts.values())
        action_distribution = {}
        if total_actions > 0:
            action_names = ["nop", "patch", "restore", "monitor", "scan"]
            for action_id, count in self.action_counts.items():
                action_distribution[action_names[action_id]] = count / total_actions

        # Reward composition analysis
        positive_rewards = [r for r in self.episode_rewards if r > 0]
        negative_rewards = [r for r in self.episode_rewards if r < 0]

        return {
            "total_reward": episode_reward,
            "episode_length": episode_length,
            "average_reward": episode_reward / max(episode_length, 1),
            "recent_performance_score": recent_performance_score,  # New: temporal shaping
            "security_breaches": self.episode_security_breaches,
            "breach_duration": self.episode_breach_duration,
            "successful_defenses": self.episode_successful_defenses,
            "unnecessary_actions": self.episode_unnecessary_actions,
            "attacks_occurred": self.episode_attacks_occurred,
            "defense_efficiency": defense_efficiency,
            "action_distribution": action_distribution,
            "positive_reward_steps": len(positive_rewards),
            "negative_reward_steps": len(negative_rewards),
            "average_positive_reward": np.mean(positive_rewards)
            if positive_rewards
            else 0,
            "average_negative_reward": np.mean(negative_rewards)
            if negative_rewards
            else 0,
            "detection_required_fix_attempts": self.detection_required_fix_attempts,  # New metric
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
            1: "Patch (fix compromised server, requires detection)",
            2: "Restore (fix all disrupted services, requires detection)",
            3: "Monitor (detect threats if not already flagged)",
            4: "Scan (clear detection flag if set)",
        }
