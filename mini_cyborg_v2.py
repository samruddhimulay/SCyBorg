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

        # Reward parameters (zero-optimal: no positives, penalties only)
        self.patch_reward = 0
        self.restore_per_service_reward = 0
        self.monitor_reward = 0
        self.monitor_enable_bonus = 0
        self.scan_reward = 0
        self.unnecessary_action_penalty = -1
        self.scan_unnecessary_penalty = -1
        self.base_persistent_breach_penalty = -1
        self.inaction_penalty = -1
        self.detection_ignored_penalty = -5  # Stronger to force scanning
        self.full_uptime_reward = 0
        self.partial_uptime_reward = 0
        self.failure_penalty = -5
        self.perfect_survival_bonus = 0
        self.action_cost = -1
        self.lingering_detection_penalty = -5  # Stronger for uncleared flags

        # Mitigation trackers
        self.consecutive_secure_steps = 0
        self.breach_duration_consecutive = 0
        self.cooldowns = {3: 0, 4: 0}
        self.observation_noise_prob = 0.10
        self.detection_required_fix_attempts = 0

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

        episode_info = {}
        if self.episode_count > 0:
            episode_info = self._get_episode_summary()

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

        self.consecutive_secure_steps = 0
        self.breach_duration_consecutive = 0
        self.cooldowns = {3: 0, 4: 0}
        self.current_compromise_prob = self.base_compromise_prob
        self.current_service1_attack_prob = self.base_service1_attack_prob
        self.current_service2_attack_prob = self.base_service2_attack_prob
        self.detection_required_fix_attempts = 0

        return self._get_noisy_observation(), episode_info

    def _get_noisy_observation(self):
        noisy_state = self.state.copy()
        for i in range(4):
            if np.random.rand() < self.observation_noise_prob:
                noisy_state[i] = 1 - noisy_state[i]
        return noisy_state

    def _simulate_attacks(self):
        compromised, service1_down, service2_down, attack_detected = self.state
        attacks_this_step = 0

        if not compromised and np.random.rand() < self.current_compromise_prob:
            compromised = 1
            attacks_this_step += 1

        if not service1_down and np.random.rand() < self.current_service1_attack_prob:
            service1_down = 1
            attacks_this_step += 1

        if not service2_down and np.random.rand() < self.current_service2_attack_prob:
            service2_down = 1
            attacks_this_step += 1

        self.episode_attacks_occurred += attacks_this_step

        if not (compromised or service1_down or service2_down):
            attack_detected = 0

        return np.array(
            [compromised, service1_down, service2_down, attack_detected]
        ), attacks_this_step

    def _update_dynamic_probs(self, breach_count):
        if breach_count == 0:
            self.consecutive_secure_steps += 1
            if self.consecutive_secure_steps >= 3:
                self.current_compromise_prob = self.base_compromise_prob * 0.8
                self.current_service1_attack_prob = self.base_service1_attack_prob * 0.8
                self.current_service2_attack_prob = self.base_service2_attack_prob * 0.8
        else:
            self.consecutive_secure_steps = 0
            self.current_compromise_prob = self.base_compromise_prob
            self.current_service1_attack_prob = self.base_service1_attack_prob
            self.current_service2_attack_prob = self.base_service2_attack_prob

    def _calculate_base_reward(self):
        compromised, service1_down, service2_down, attack_detected = self.state
        reward = 0
        breach_count = sum([compromised, service1_down, service2_down])

        if breach_count > 0:
            self.breach_duration_consecutive += 1
        else:
            self.breach_duration_consecutive = 0

        scaled_penalty = self.base_persistent_breach_penalty * (
            1 + self.breach_duration_consecutive / 5.0
        )
        reward += breach_count * scaled_penalty

        if attack_detected and breach_count > 0:
            reward += self.detection_ignored_penalty

        if not breach_count and attack_detected:
            reward += self.lingering_detection_penalty

        if breach_count == 0:
            reward += self.full_uptime_reward
        elif breach_count < 3:
            reward += self.partial_uptime_reward

        self.episode_security_breaches += breach_count
        self.episode_breach_duration += 1 if breach_count > 0 else 0

        return reward, breach_count

    def step(self, action):
        for act in self.cooldowns:
            if self.cooldowns[act] > 0:
                self.cooldowns[act] -= 1

        self.state, attacks_this_step = self._simulate_attacks()
        compromised, service1_down, service2_down, attack_detected = self.state

        reward = 0
        done = False
        truncated = False
        info = {"action_successful": False}

        self.action_counts[action] += 1

        if action != 0:
            reward += self.action_cost

        if action == 0:
            info["action_taken"] = "no_operation"
            if compromised or service1_down or service2_down:
                reward += self.inaction_penalty

        elif action == 1:
            if attack_detected:
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
            else:
                reward += self.unnecessary_action_penalty
                info["action_taken"] = "patch_failed_no_detection"
                self.episode_unnecessary_actions += 1
                self.detection_required_fix_attempts += 1

        elif action == 2:
            if attack_detected:
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
            else:
                reward += self.unnecessary_action_penalty
                info["action_taken"] = "restore_failed_no_detection"
                self.episode_unnecessary_actions += 1
                self.detection_required_fix_attempts += 1

        elif action == 3:
            if self.cooldowns[3] > 0:
                reward += self.unnecessary_action_penalty
                info["action_taken"] = "monitor_on_cooldown"
                self.episode_unnecessary_actions += 1
            else:
                threats = compromised or service1_down or service2_down
                if threats and not attack_detected:
                    attack_detected = 1
                    reward += self.monitor_reward
                    if threats:
                        reward += self.monitor_enable_bonus
                    info["action_successful"] = True
                    info["action_taken"] = "monitor_threat_detected"
                    self.episode_successful_defenses += 1
                    self.cooldowns[3] = 2
                else:
                    reward += self.unnecessary_action_penalty
                    info["action_taken"] = "monitor_no_threat_or_already_detected"
                    self.episode_unnecessary_actions += 1

        elif action == 4:
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
                    self.cooldowns[4] = 2
                else:
                    reward += self.scan_unnecessary_penalty
                    info["action_taken"] = "scan_unnecessary"
                    self.episode_unnecessary_actions += 1

        self.state = np.array(
            [compromised, service1_down, service2_down, attack_detected]
        )

        base_reward, breach_count = self._calculate_base_reward()
        reward += base_reward

        self._update_dynamic_probs(breach_count)

        self.episode_rewards.append(reward)

        self.current_step += 1
        self.total_steps += 1

        if service1_down and service2_down:
            done = True
            reward += self.failure_penalty
            info["termination_reason"] = "total_system_failure"
        elif self.current_step >= self.max_steps:
            done = True
            info["termination_reason"] = "max_steps_reached"
            if self.episode_security_breaches == 0:
                reward += self.perfect_survival_bonus

        info.update(
            {
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
                "detection_required_fix_attempts": self.detection_required_fix_attempts,
                "episode_successful_defenses": self.episode_successful_defenses,
                "episode_unnecessary_actions": self.episode_unnecessary_actions,
                "episode_attacks_occurred": self.episode_attacks_occurred,
                "episode_security_breaches": self.episode_security_breaches,
                "episode_breach_duration": self.episode_breach_duration,
                "current_episode_reward": sum(self.episode_rewards),
                "action_counts": self.action_counts.copy(),
            }
        )

        if done:
            info["episode_summary"] = self._get_episode_summary()

        return self._get_noisy_observation(), reward, done, truncated, info

    def _get_episode_summary(self):
        if not self.episode_rewards:
            return {}

        episode_reward = sum(self.episode_rewards)
        episode_length = len(self.episode_rewards)

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

        total_defensive_actions = (
            self.episode_successful_defenses + self.episode_unnecessary_actions
        )
        defense_efficiency = (
            self.episode_successful_defenses / total_defensive_actions
            if total_defensive_actions > 0
            else 0
        )

        total_actions = sum(self.action_counts.values())
        action_distribution = {}
        if total_actions > 0:
            action_names = ["nop", "patch", "restore", "monitor", "scan"]
            for action_id, count in self.action_counts.items():
                action_distribution[action_names[action_id]] = count / total_actions

        positive_rewards = [r for r in self.episode_rewards if r > 0]
        negative_rewards = [r for r in self.episode_rewards if r < 0]

        return {
            "total_reward": episode_reward,
            "episode_length": episode_length,
            "average_reward": episode_reward / max(episode_length, 1),
            "recent_performance_score": recent_performance_score,
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
            "detection_required_fix_attempts": self.detection_required_fix_attempts,
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
